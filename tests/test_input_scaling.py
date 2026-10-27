import numpy as np
import pytest
import torch
import xarray as xr

from tornado_predictor.inference import apply_input_scaling, assert_feature_order_matches
from tornado_predictor.training import DenseGridDataset, standardize_channels
from tornado_predictor.unet import TornadoUNet


def _nc(tmp_path, name, scale_big=1.0e5, offset=9.6e4, seed=0, n=6):
    """cape ~ N(400, 300) (left raw by the experiment), pres ~ huge scale, const = constant channel."""
    rng = np.random.default_rng(seed)
    shape = (n, 4, 5)
    ds = xr.Dataset(
        {
            "cape_mean": (("sample", "row", "col"), rng.normal(400, 300, shape).astype("float32")),
            "pres_sfc_mean": (("sample", "row", "col"), (offset + rng.normal(0, 3e3, shape)).astype("float32")),
            "const": (("sample", "row", "col"), np.full(shape, 7.0, dtype="float32")),
            "label": (("sample", "row", "col"), np.zeros(shape, dtype=np.int8)),
        },
        coords={"sample": np.arange(n), "row": np.arange(4), "col": np.arange(5)},
    )
    path = tmp_path / name
    ds.to_netcdf(path)
    return str(path)


def test_standardize_channels_zscores_only_the_named_channels(tmp_path):
    ds = DenseGridDataset(_nc(tmp_path, "train.nc"))
    cape_before = ds.X[:, ds.feature_names.index("cape_mean")].copy()

    stats = standardize_channels(ds, ["pres_sfc_mean"])

    p = ds.X[:, ds.feature_names.index("pres_sfc_mean")]
    assert abs(float(p.mean())) < 1e-4 and float(p.std()) == pytest.approx(1.0, abs=1e-4)
    assert np.array_equal(ds.X[:, ds.feature_names.index("cape_mean")], cape_before)  # base channels stay raw
    assert stats["names"] == ["pres_sfc_mean"] and len(stats["mean"]) == len(stats["std"]) == 1


def test_test_set_is_scaled_with_the_training_statistics_not_its_own(tmp_path):
    train = DenseGridDataset(_nc(tmp_path, "train.nc", offset=9.6e4, seed=0))
    test = DenseGridDataset(_nc(tmp_path, "test.nc", offset=9.9e4, seed=1))  # deliberately shifted distribution

    stats = standardize_channels(train, ["pres_sfc_mean"])
    standardize_channels(test, ["pres_sfc_mean"], stats=stats)

    t = test.X[:, test.feature_names.index("pres_sfc_mean")]
    expected_shift = (9.9e4 - 9.6e4) / stats["std"][0]
    assert float(t.mean()) == pytest.approx(expected_shift, rel=0.05)  # would be ~0 if it had used its own mean


def test_constant_channel_does_not_divide_by_zero(tmp_path):
    ds = DenseGridDataset(_nc(tmp_path, "c.nc"))
    standardize_channels(ds, ["const"])
    assert np.isfinite(ds.X).all()
    assert float(np.abs(ds.X[:, ds.feature_names.index("const")]).max()) == 0.0


def test_apply_input_scaling_replays_stats_and_is_a_noop_for_old_checkpoints(tmp_path):
    train = DenseGridDataset(_nc(tmp_path, "train.nc"))
    stats = standardize_channels(train, ["pres_sfc_mean"])

    scored = DenseGridDataset(_nc(tmp_path, "test.nc", seed=1))
    raw = scored.X.copy()
    apply_input_scaling(scored, {"feature_names": scored.feature_names})  # a checkpoint with no input_scaling
    assert np.array_equal(scored.X, raw)

    apply_input_scaling(scored, {"feature_names": scored.feature_names, "input_scaling": stats})
    i = scored.feature_names.index("pres_sfc_mean")
    assert not np.array_equal(scored.X[:, i], raw[:, i])
    assert np.array_equal(np.delete(scored.X, i, axis=1), np.delete(raw, i, axis=1))
    expected = (raw[:, i] - stats["mean"][0]) / stats["std"][0]
    assert np.allclose(scored.X[:, i], expected, rtol=1e-5)


def test_scaling_survives_a_checkpoint_roundtrip_and_order_guard_still_applies(tmp_path):
    train = DenseGridDataset(_nc(tmp_path, "train.nc"))
    stats = standardize_channels(train, ["pres_sfc_mean"])
    ckpt = {"feature_names": list(train.feature_names), "input_scaling": stats}
    path = tmp_path / "ck.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, weights_only=False)

    ds = DenseGridDataset(_nc(tmp_path, "t.nc", seed=2))
    assert_feature_order_matches(ds.feature_names, loaded)
    apply_input_scaling(ds, loaded)
    i = ds.feature_names.index("pres_sfc_mean")
    assert abs(float(ds.X[:, i].mean())) < 0.5  # same distribution as train -> near zero after the train scaling

    with pytest.raises(ValueError):
        assert_feature_order_matches(list(reversed(ds.feature_names)), loaded)


def test_unet_trains_finitely_on_a_hugely_scaled_channel_only_after_scaling(tmp_path):
    """Why the scaling exists: a ~1e5-scale channel makes the first-layer activations enormous."""
    raw = DenseGridDataset(_nc(tmp_path, "raw.nc", n=2))
    scaled = DenseGridDataset(_nc(tmp_path, "scaled.nc", n=2))
    standardize_channels(scaled, ["pres_sfc_mean"])

    torch.manual_seed(0)
    model = TornadoUNet(in_channels=len(raw.feature_names), base_channels=4)

    def first_layer_scale(ds):
        x = torch.from_numpy(ds.X[:1])
        with torch.no_grad():
            return float(model(x).abs().max())

    assert first_layer_scale(raw) > 100 * first_layer_scale(scaled)
