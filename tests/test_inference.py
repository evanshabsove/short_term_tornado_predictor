from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from tornado_predictor.inference import (
    assert_feature_order_matches,
    load_checkpoint,
    plot_sample,
    predict_proba,
    sanity_check_against_labels,
)
from tornado_predictor.model import TornadoCNN

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "models" / "tornado_cnn_pilot.pt"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_pilot.nc"


def _save_synthetic_checkpoint(tmp_path, in_channels=3, hidden_channels=4, feature_names=None):
    feature_names = feature_names or ["a", "b", "c"]
    torch.manual_seed(0)
    model = TornadoCNN(in_channels=in_channels, hidden_channels=hidden_channels)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss_history": [0.5, 0.3, 0.1],
            "training_hyperparameters": {"epochs": 3},
            "model_hyperparameters": {"in_channels": in_channels, "hidden_channels": hidden_channels},
            "feature_names": feature_names,
        },
        path,
    )
    return path, model


# --- load_checkpoint ---


def test_load_checkpoint_reconstructs_identical_model(tmp_path):
    path, original_model = _save_synthetic_checkpoint(tmp_path)
    loaded_model, checkpoint = load_checkpoint(str(path))

    assert not loaded_model.training  # eval mode
    assert checkpoint["feature_names"] == ["a", "b", "c"]

    x = torch.randn(1, 3, 5, 6)
    with torch.no_grad():
        original_model.eval()
        assert torch.allclose(loaded_model(x), original_model(x))


# --- assert_feature_order_matches ---


def test_assert_feature_order_matches_passes_on_exact_match():
    assert_feature_order_matches(["a", "b", "c"], {"feature_names": ["a", "b", "c"]})


def test_assert_feature_order_matches_raises_on_mismatch():
    with pytest.raises(ValueError):
        assert_feature_order_matches(["a", "c", "b"], {"feature_names": ["a", "b", "c"]})


# --- predict_proba ---


def test_predict_proba_shape_and_range():
    model = TornadoCNN(in_channels=3, hidden_channels=4)
    model.eval()
    X = np.random.default_rng(0).normal(size=(3, 5, 6)).astype("float32")

    proba = predict_proba(model, X)

    assert proba.shape == (5, 6)
    assert (proba >= 0).all() and (proba <= 1).all()


# --- sanity_check_against_labels ---


def test_sanity_check_against_labels_computes_expected_stats():
    proba = np.array([[0.1, 0.9], [0.2, 0.8]])
    label = np.array([[0, 1], [0, 0]])

    stats = sanity_check_against_labels(proba, label)

    assert stats["n_positive_cells"] == 1
    assert stats["mean_proba_positive"] == pytest.approx(0.9)
    assert stats["min_proba_positive"] == pytest.approx(0.9)
    assert stats["mean_proba_negative"] == pytest.approx((0.1 + 0.2 + 0.8) / 3)
    assert stats["max_proba_negative"] == pytest.approx(0.8)


def test_sanity_check_against_labels_handles_no_positives():
    proba = np.array([[0.1, 0.2], [0.3, 0.4]])
    label = np.zeros((2, 2))

    stats = sanity_check_against_labels(proba, label)

    assert stats["n_positive_cells"] == 0
    assert np.isnan(stats["mean_proba_positive"])
    assert np.isnan(stats["min_proba_positive"])


# --- plot_sample ---


def test_plot_sample_smoke():
    proba = np.random.default_rng(0).uniform(size=(5, 6))
    label = np.zeros((5, 6))
    label[2, 3] = 1

    ax, im = plot_sample(proba, label, title="test")

    assert ax.get_title() == "test"
    assert im.get_array().shape == (5, 6)


# --- against the real trained checkpoint ---


def test_against_real_checkpoint_and_pilot_dataset():
    if not CHECKPOINT_PATH.exists() or not DATASET_PATH.exists():
        pytest.skip("trained checkpoint or pilot dataset not present")

    from tornado_predictor.training import DenseGridDataset

    model, checkpoint = load_checkpoint(str(CHECKPOINT_PATH))
    dataset = DenseGridDataset(str(DATASET_PATH))
    assert_feature_order_matches(dataset.feature_names, checkpoint)

    for i in range(len(dataset)):
        X, y = dataset[i]
        proba = predict_proba(model, X)
        stats = sanity_check_against_labels(proba, y)

        assert stats["n_positive_cells"] > 0  # every pilot sample is "active"
        # The model was trained to near-zero mean loss on this exact data,
        # but that doesn't mean positive-cell probability approaches 1 --
        # alpha=0.25 down-weights the positive class's loss contribution
        # (see FocalLoss's docstring), which matters more at this
        # dataset's much more extreme per-sample imbalance (a handful of
        # positive cells among 11,178) than RetinaNet's original use case.
        # Empirically (see CLAUDE.md), positive cells land around
        # 0.26-0.34 while negatives land around 0.003-0.007 -- a strong,
        # consistent ~40-90x separation, not near-1.0 confidence.
        assert stats["mean_proba_positive"] > 0.15, f"sample {i}: model does not fit its own training labels"
        assert stats["mean_proba_positive"] > 10 * stats["mean_proba_negative"]
