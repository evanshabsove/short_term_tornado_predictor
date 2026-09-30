import json
import os

import numpy as np
import pytest
import torch
import xarray as xr

from tornado_predictor.evaluate import compute_metrics, evaluate_checkpoint, evaluate_model, get_proba_labels
from tornado_predictor.model import TornadoCNN

REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


def _save_synthetic_checkpoint(tmp_path, in_channels=3, hidden_channels=4, feature_names=None, with_val=True):
    feature_names = feature_names or ["a", "b", "c"]
    torch.manual_seed(0)
    model = TornadoCNN(in_channels=in_channels, hidden_channels=hidden_channels)
    path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "loss_history": [0.5, 0.3, 0.1],
        "training_hyperparameters": {"epochs": 3},
        "model_hyperparameters": {"in_channels": in_channels, "hidden_channels": hidden_channels},
        "feature_names": feature_names,
    }
    if with_val:
        checkpoint["val_loss_history"] = [0.4, 0.2, 0.05]
    torch.save(checkpoint, path)
    return path


def _save_synthetic_dataset(tmp_path, feature_names=("a", "b", "c"), n_samples=3, n_rows=4, n_cols=5, seed=0):
    """A tiny netCDF matching DenseGridDataset's expected schema (one
    data_var per feature plus "label", all (sample, row, col))."""
    rng = np.random.default_rng(seed)
    data_vars = {name: (("sample", "row", "col"), rng.normal(size=(n_samples, n_rows, n_cols)).astype("float32")) for name in feature_names}
    label = np.zeros((n_samples, n_rows, n_cols), dtype="float32")
    label[:, 0, 0] = 1  # at least one positive cell per sample
    data_vars["label"] = (("sample", "row", "col"), label)
    ds = xr.Dataset(data_vars)
    path = tmp_path / "dataset.nc"
    ds.to_netcdf(path)
    return path


class _FakeDataset:
    """Minimal duck-typed stand-in for DenseGridDataset -- get_proba_labels
    only needs len()/[i] -> (X, y), not the real netCDF-backed class."""

    def __init__(self, samples):
        self.samples = samples  # list of (X, y) numpy array pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# --- compute_metrics ---


def test_compute_metrics_perfect_separation_gives_auc_1():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.01, 0.02, 0.03, 0.9, 0.95])

    metrics = compute_metrics(y_true, y_score)

    assert metrics["auc_roc"] == pytest.approx(1.0)
    assert metrics["auc_pr"] == pytest.approx(1.0)


def test_compute_metrics_threshold_0_5_hand_checked():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.6, 0.9])
    # at 0.5: predicted positive = indices 1,3,4 (scores 0.6,0.6,0.9)
    # tp=2 (idx 3,4), fp=1 (idx 1), fn=1 (idx 2)

    metrics = compute_metrics(y_true, y_score)

    assert metrics["recall_at_0.5"] == pytest.approx(2 / 3)
    assert metrics["precision_at_0.5"] == pytest.approx(2 / 3)
    # correct at idx 0 (pred 0, true 0), 3, 4; wrong at idx 1 (pred 1, true 0), 2 (pred 0, true 1)
    assert metrics["acc_at_0.5"] == pytest.approx(3 / 5)


def test_compute_metrics_best_f1_threshold_picks_expected_point():
    # scores cleanly separate positives (>=0.7) from negatives (<=0.3)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_score)

    assert metrics["f1_at_best_f1"] == pytest.approx(1.0)
    assert metrics["precision_at_best_f1"] == pytest.approx(1.0)
    assert metrics["recall_at_best_f1"] == pytest.approx(1.0)
    assert 0.3 < metrics["best_f1_threshold"] <= 0.7


def test_compute_metrics_no_positives_returns_nan_not_raise():
    y_true = np.zeros(10)
    y_score = np.random.default_rng(0).uniform(size=10)

    metrics = compute_metrics(y_true, y_score)

    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["auc_pr"])
    assert np.isnan(metrics["cohens_d"])
    assert np.isnan(metrics["mean_proba_positive"])
    assert metrics["mean_proba_negative"] == pytest.approx(y_score.mean())


def test_compute_metrics_no_negatives_returns_nan_not_raise():
    y_true = np.ones(10)
    y_score = np.random.default_rng(0).uniform(size=10)

    metrics = compute_metrics(y_true, y_score)

    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["mean_proba_negative"])


def test_compute_metrics_include_curves_false_omits_curve_keys():
    y_true = np.array([0, 1])
    y_score = np.array([0.2, 0.8])

    metrics = compute_metrics(y_true, y_score, include_curves=False)

    assert "roc_curve" not in metrics
    assert "pr_curve" not in metrics


def test_compute_metrics_include_curves_true_is_json_serializable():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = compute_metrics(y_true, y_score, include_curves=True)

    assert "roc_curve" in metrics and "pr_curve" in metrics
    json.dumps(metrics)  # must not raise


# --- get_proba_labels ---


def test_get_proba_labels_flattens_and_concatenates_across_samples():
    model = TornadoCNN(in_channels=2, hidden_channels=3)
    model.eval()
    rng = np.random.default_rng(0)
    samples = [
        (rng.normal(size=(2, 4, 5)).astype("float32"), np.zeros((4, 5))),
        (rng.normal(size=(2, 4, 5)).astype("float32"), np.ones((4, 5))),
    ]
    dataset = _FakeDataset(samples)

    y_score, y_true = get_proba_labels(model, dataset)

    assert y_score.shape == (2 * 4 * 5,)
    assert y_true.shape == (2 * 4 * 5,)
    assert (y_score >= 0).all() and (y_score <= 1).all()
    assert y_true[:20].sum() == 0  # first sample's flattened labels are all-zero
    assert y_true[20:].sum() == 20  # second sample's flattened labels are all-one


# --- evaluate_model ---


def test_evaluate_model_returns_full_metric_keys():
    model = TornadoCNN(in_channels=2, hidden_channels=3)
    model.eval()
    rng = np.random.default_rng(0)
    samples = [
        (rng.normal(size=(2, 4, 5)).astype("float32"), (rng.uniform(size=(4, 5)) > 0.7).astype("float32"))
        for _ in range(3)
    ]
    dataset = _FakeDataset(samples)

    metrics = evaluate_model(model, dataset)

    for key in ["auc_roc", "auc_pr", "acc_at_0.5", "recall_at_0.5", "cohens_d", "roc_curve", "pr_curve"]:
        assert key in metrics


# --- evaluate_checkpoint (fully synthetic, offline) ---


def test_evaluate_checkpoint_end_to_end_synthetic(tmp_path):
    feature_names = ["a", "b", "c"]
    ckpt_path = _save_synthetic_checkpoint(tmp_path, in_channels=3, hidden_channels=4, feature_names=feature_names)
    ds_path = _save_synthetic_dataset(tmp_path, feature_names=feature_names)

    metrics = evaluate_checkpoint(str(ckpt_path), str(ds_path))

    assert metrics["final_train_loss"] == pytest.approx(0.1)
    assert metrics["final_val_loss"] == pytest.approx(0.05)
    assert metrics["n_cells"] == 3 * 4 * 5
    assert not np.isnan(metrics["auc_roc"])  # dataset was built with real positive cells


def test_evaluate_checkpoint_no_val_loss_history_gives_nan(tmp_path):
    feature_names = ["a", "b", "c"]
    ckpt_path = _save_synthetic_checkpoint(tmp_path, feature_names=feature_names, with_val=False)
    ds_path = _save_synthetic_dataset(tmp_path, feature_names=feature_names)

    metrics = evaluate_checkpoint(str(ckpt_path), str(ds_path))

    assert np.isnan(metrics["final_val_loss"])


def test_evaluate_checkpoint_raises_on_feature_order_mismatch(tmp_path):
    ckpt_path = _save_synthetic_checkpoint(tmp_path, feature_names=["a", "b", "c"])
    ds_path = _save_synthetic_dataset(tmp_path, feature_names=["a", "c", "b"])  # swapped order

    with pytest.raises(ValueError):
        evaluate_checkpoint(str(ckpt_path), str(ds_path))


# --- against a real artifact ---


def test_evaluate_checkpoint_against_real_split_demo_checkpoint():
    checkpoint_path = f"{REPO_ROOT}/models/tornado_cnn_split_demo.pt"
    dataset_path = f"{REPO_ROOT}/data/processed/training_dataset_val.nc"
    if not (os.path.exists(checkpoint_path) and os.path.exists(dataset_path)):
        pytest.skip("real split-demo checkpoint or val dataset not present")

    metrics = evaluate_checkpoint(checkpoint_path, dataset_path)

    # cross-check against a manual, independent computation of the same numbers
    from sklearn.metrics import roc_auc_score

    from tornado_predictor.inference import load_checkpoint
    from tornado_predictor.training import DenseGridDataset

    model, _ = load_checkpoint(checkpoint_path)
    dataset = DenseGridDataset(dataset_path)
    y_score, y_true = get_proba_labels(model, dataset)

    assert metrics["auc_roc"] == pytest.approx(roc_auc_score(y_true, y_score))
    assert metrics["mean_proba_positive"] == pytest.approx(y_score[y_true > 0].mean())
    assert metrics["n_cells"] == len(y_true)
