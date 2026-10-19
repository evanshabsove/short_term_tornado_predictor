import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tornado_predictor.ensemble import (
    DEFAULT_BUNDLE_MANIFEST_PATH,
    EnsemblePrediction,
    compute_bundle_id,
    file_sha256,
    load_bundle,
    predict_ensemble,
)
from tornado_predictor.inference import predict_proba
from tornado_predictor.unet import TornadoUNet

FEATURES = ["a", "b", "c"]
RECIPE = {"alpha": 0.25, "ema_decay": 0.9999}


def _write_bundle(tmp_path: Path, n_members: int = 3, member_features=None, base_channels: int = 4) -> Path:
    """A tiny, fully synthetic bundle: n_members random-weight U-Nets,
    each saved in the same checkpoint format the training script writes."""
    members = []
    for i in range(n_members):
        torch.manual_seed(100 + i)
        model = TornadoUNet(in_channels=len(FEATURES), base_channels=base_channels)
        names = (member_features or {}).get(i, FEATURES)
        path = tmp_path / f"member_{i}.pt"
        torch.save(
            {
                "model_class": "TornadoUNet",
                "model_state_dict": model.state_dict(),
                "model_hyperparameters": {"in_channels": len(FEATURES), "base_channels": base_channels},
                "feature_names": names,
            },
            path,
        )
        members.append({"path": path.name, "sha256": file_sha256(path), "seed": 100 + i})
    manifest = {
        "bundle_id": compute_bundle_id([m["sha256"] for m in members], RECIPE),
        "recipe": RECIPE,
        "feature_names": FEATURES,
        "members": members,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _random_input(n=None, h=10, w=12, seed=0):
    rng = np.random.default_rng(seed)
    shape = (len(FEATURES), h, w) if n is None else (n, len(FEATURES), h, w)
    return rng.normal(size=shape).astype("float32")


def test_mean_and_spread_match_manual_computation(tmp_path):
    bundle = load_bundle(_write_bundle(tmp_path))
    X = _random_input()
    pred = predict_ensemble(bundle, X, feature_names=FEATURES)

    per_member = np.stack([predict_proba(m, X) for m in bundle.models])
    np.testing.assert_allclose(pred.mean, per_member.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(pred.spread, per_member.std(axis=0, ddof=0), atol=1e-6)
    assert isinstance(pred, EnsemblePrediction)
    assert pred.mean.shape == (10, 12) and pred.spread.shape == (10, 12)
    assert pred.spread.max() > 0  # random members genuinely disagree


def test_batch_equals_per_sample(tmp_path):
    bundle = load_bundle(_write_bundle(tmp_path))
    X = _random_input(n=3)
    batch = predict_ensemble(bundle, X, feature_names=FEATURES)
    assert batch.mean.shape == (3, 10, 12)
    for i in range(3):
        single = predict_ensemble(bundle, X[i], feature_names=FEATURES)
        np.testing.assert_allclose(batch.mean[i], single.mean, atol=1e-5)
        np.testing.assert_allclose(batch.spread[i], single.spread, atol=1e-5)


def test_single_member_has_zero_spread(tmp_path):
    bundle = load_bundle(_write_bundle(tmp_path, n_members=1))
    pred = predict_ensemble(bundle, _random_input(), feature_names=FEATURES)
    np.testing.assert_array_equal(pred.spread, 0.0)


def test_feature_order_mismatch_raises(tmp_path):
    bundle = load_bundle(_write_bundle(tmp_path))
    with pytest.raises(ValueError, match="feature order"):
        predict_ensemble(bundle, _random_input(), feature_names=["b", "a", "c"])


def test_wrong_channel_count_raises(tmp_path):
    bundle = load_bundle(_write_bundle(tmp_path))
    with pytest.raises(ValueError, match="channels"):
        predict_ensemble(bundle, np.zeros((2, 10, 12), dtype="float32"), feature_names=FEATURES)


def test_bad_ndim_raises(tmp_path):
    bundle = load_bundle(_write_bundle(tmp_path))
    with pytest.raises(ValueError, match="must be"):
        predict_ensemble(bundle, np.zeros((10, 12), dtype="float32"), feature_names=FEATURES)


def test_sha256_mismatch_raises(tmp_path):
    manifest_path = _write_bundle(tmp_path)
    (tmp_path / "member_1.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="sha256"):
        load_bundle(manifest_path)


def test_members_with_different_feature_order_raise(tmp_path):
    manifest_path = _write_bundle(tmp_path, member_features={1: ["b", "a", "c"]})
    with pytest.raises(ValueError, match="feature order"):
        load_bundle(manifest_path)


def test_tampered_bundle_id_raises(tmp_path):
    manifest_path = _write_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_id"] = "000000000000"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bundle_id"):
        load_bundle(manifest_path)


def test_empty_manifest_raises(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"members": [], "feature_names": FEATURES, "recipe": RECIPE, "bundle_id": "x"}))
    with pytest.raises(ValueError, match="no members"):
        load_bundle(manifest_path)


def test_bundle_id_depends_on_members_and_recipe():
    base = compute_bundle_id(["aa", "bb"], RECIPE)
    assert base == compute_bundle_id(["bb", "aa"], RECIPE)  # order-independent
    assert base != compute_bundle_id(["aa", "cc"], RECIPE)
    assert base != compute_bundle_id(["aa", "bb"], {**RECIPE, "alpha": 0.5})


@pytest.mark.skipif(not DEFAULT_BUNDLE_MANIFEST_PATH.exists(), reason="production bundle not built yet (models/production/)")
def test_real_production_bundle_loads_and_predicts():
    bundle = load_bundle()
    assert len(bundle.models) >= 3
    assert len(bundle.feature_names) == 14
    X = np.zeros((14, 81, 138), dtype="float32")
    pred = predict_ensemble(bundle, X, feature_names=bundle.feature_names)
    assert pred.mean.shape == (81, 138)
    assert np.isfinite(pred.mean).all() and ((pred.mean >= 0) & (pred.mean <= 1)).all()
