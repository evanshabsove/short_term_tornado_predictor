from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from tornado_predictor.model import TornadoCNN
from tornado_predictor.training import DenseGridDataset, FocalLoss, QuietMapDownsampler, train_model

PILOT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "training_dataset_pilot.nc"


def _tiny_synthetic_nc(tmp_path, n_samples=10, n_active=3):
    rng = np.random.default_rng(0)
    row, col = 5, 6
    label = np.zeros((n_samples, row, col), dtype=np.int8)
    for i in range(n_active):
        label[i, 1, 1] = 1  # first n_active samples are "active"
    ds = xr.Dataset(
        data_vars={
            "cape_mean": (("sample", "row", "col"), rng.normal(size=(n_samples, row, col)).astype("float32")),
            "cape_max": (("sample", "row", "col"), rng.normal(size=(n_samples, row, col)).astype("float32")),
            "label": (("sample", "row", "col"), label),
        },
        coords={"sample": np.arange(n_samples), "row": np.arange(row), "col": np.arange(col)},
    )
    path = tmp_path / "synthetic.nc"
    ds.to_netcdf(path)
    return path, n_active, n_samples - n_active


# --- FocalLoss ---


def test_focal_loss_is_scalar_and_finite():
    criterion = FocalLoss()
    logits = torch.randn(4, 5, 6)
    targets = (torch.rand(4, 5, 6) > 0.5).float()
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_focal_loss_penalizes_confident_wrong_predictions_more():
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.ones(1, 3, 3)

    confident_wrong = torch.full((1, 3, 3), -10.0)  # sigmoid ~= 0, target is 1
    confident_right = torch.full((1, 3, 3), 10.0)   # sigmoid ~= 1, target is 1

    assert criterion(confident_wrong, targets) > criterion(confident_right, targets)


def test_focal_loss_down_weights_easy_examples_vs_plain_bce():
    # An "easy" correct example (confident and right) should contribute
    # much less to focal loss than to plain BCE, since focal loss's
    # (1-p_t)^gamma term shrinks toward zero as p_t -> 1.
    logits = torch.full((1, 3, 3), 10.0)
    targets = torch.ones(1, 3, 3)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    focal = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
    assert focal < bce


# --- DenseGridDataset ---


def test_dense_grid_dataset_shapes_and_active_flag(tmp_path):
    path, n_active, n_quiet = _tiny_synthetic_nc(tmp_path)
    dataset = DenseGridDataset(str(path))

    assert len(dataset) == n_active + n_quiet
    assert dataset.feature_names == ["cape_mean", "cape_max"]
    assert dataset.is_active.sum() == n_active

    X, y = dataset[0]
    assert X.shape == (2, 5, 6)  # (channel, row, col)
    assert y.shape == (5, 6)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_dense_grid_dataset_against_real_pilot_dataset():
    if not PILOT_DATASET_PATH.exists():
        pytest.skip("pilot dataset not present")
    dataset = DenseGridDataset(str(PILOT_DATASET_PATH))

    assert len(dataset) == 6
    assert len(dataset.feature_names) == 14
    # the pilot dataset was deliberately built so every sample has a
    # positive cell -- confirms this is still true, and that
    # QuietMapDownsampler is currently a no-op on this dataset.
    assert dataset.is_active.all()

    for i in range(len(dataset)):
        X, y = dataset[i]
        assert X.shape == (14, 81, 138)
        assert y.shape == (81, 138)
        assert not np.isnan(X).any(), f"NaNs in features for sample {i}"
        assert not np.isnan(y).any(), f"NaNs in labels for sample {i}"
        assert set(np.unique(y)) <= {0.0, 1.0}


# --- QuietMapDownsampler ---


def test_downsampler_keeps_all_active_and_fraction_of_quiet():
    is_active = np.array([True, True, False, False, False, False, False, False, False, False])
    sampler = QuietMapDownsampler(is_active, quiet_keep_fraction=0.5, seed=0)

    indices = list(iter(sampler))
    assert len(indices) == len(sampler) == 2 + 4  # 2 active + 50% of 8 quiet

    active_idx = set(np.where(is_active)[0])
    assert active_idx <= set(indices)  # every active index is always present
    assert set(indices) - active_idx <= set(np.where(~is_active)[0])  # rest are quiet


def test_downsampler_redraws_quiet_maps_each_epoch():
    is_active = np.zeros(100, dtype=bool)
    sampler = QuietMapDownsampler(is_active, quiet_keep_fraction=0.1, seed=0)

    epoch_1 = set(iter(sampler))
    epoch_2 = set(iter(sampler))
    assert epoch_1 != epoch_2  # different random subset drawn each call


def test_downsampler_noop_on_real_pilot_dataset():
    if not PILOT_DATASET_PATH.exists():
        pytest.skip("pilot dataset not present")
    dataset = DenseGridDataset(str(PILOT_DATASET_PATH))
    sampler = QuietMapDownsampler(dataset.is_active, quiet_keep_fraction=0.2)

    # all 6 pilot samples are active -> sampler keeps all of them, every time
    assert len(sampler) == 6
    assert set(iter(sampler)) == set(range(6))


# --- train_model ---


def test_train_model_returns_expected_structure(tmp_path):
    path, n_active, n_quiet = _tiny_synthetic_nc(tmp_path, n_samples=6, n_active=3)
    dataset = DenseGridDataset(str(path))
    model = TornadoCNN(in_channels=2, hidden_channels=4)

    result = train_model(dataset, model, epochs=3, batch_size=2, quiet_keep_fraction=0.5, seed=0)

    assert set(result.keys()) == {"loss_history", "hyperparameters"}
    assert len(result["loss_history"]) == 3
    assert all(np.isfinite(loss) for loss in result["loss_history"])
    assert result["hyperparameters"]["epochs"] == 3


def test_train_model_is_deterministic_given_same_seed(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path, n_samples=6, n_active=3)

    def run():
        torch.manual_seed(0)
        dataset = DenseGridDataset(str(path))
        model = TornadoCNN(in_channels=2, hidden_channels=4)
        return train_model(dataset, model, epochs=3, batch_size=2, quiet_keep_fraction=0.5, seed=0)["loss_history"]

    assert run() == run()


def test_train_model_reduces_loss_on_real_pilot_dataset():
    if not PILOT_DATASET_PATH.exists():
        pytest.skip("pilot dataset not present")
    torch.manual_seed(0)
    dataset = DenseGridDataset(str(PILOT_DATASET_PATH))
    model = TornadoCNN(in_channels=len(dataset.feature_names), hidden_channels=16)

    result = train_model(dataset, model, epochs=30, batch_size=2, lr=1e-3, quiet_keep_fraction=0.2, seed=0)
    loss_history = result["loss_history"]

    assert len(loss_history) == 30
    assert all(np.isfinite(loss) for loss in loss_history)
    assert loss_history[-1] < loss_history[0], "expected the model to fit the 6-sample pilot dataset over 30 epochs"
