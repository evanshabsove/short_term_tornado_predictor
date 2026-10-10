from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from tornado_predictor.model import TornadoCNN
from tornado_predictor.training import DenseGridDataset, FocalLoss, QuietMapDownsampler, evaluate, train_model

PILOT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "training_dataset_pilot.nc"


def _tiny_synthetic_nc(tmp_path, n_samples=10, n_active=3, name="synthetic.nc"):
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
    path = tmp_path / name
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


def test_dense_grid_dataset_imputes_nan_features_to_zero(tmp_path):
    """0-2km/0-3km updraft helicity are NaN for runs before 2018-07-13
    (see features.py's module docstring) -- DenseGridDataset must
    impute these to 0.0 rather than propagating NaN into the model
    (which would silently poison loss/gradients). A no-op for every
    dataset without this gap, per test_dense_grid_dataset_shapes_and_active_flag
    above already confirming zero NaN for a normal dataset."""
    row, col = 5, 6
    ds = xr.Dataset(
        data_vars={
            "cape_mean": (("sample", "row", "col"), np.ones((3, row, col), dtype="float32")),
            "uh_0_2km_mean": (("sample", "row", "col"), np.full((3, row, col), np.nan, dtype="float32")),
            "uh_layers_available": (("sample", "row", "col"), np.zeros((3, row, col), dtype="float32")),
            "label": (("sample", "row", "col"), np.zeros((3, row, col), dtype=np.int8)),
        },
        coords={"sample": np.arange(3), "row": np.arange(row), "col": np.arange(col)},
    )
    path = tmp_path / "nan_synthetic.nc"
    ds.to_netcdf(path)

    dataset = DenseGridDataset(str(path))
    X, _ = dataset[0]

    assert not np.isnan(X).any()
    uh_channel = dataset.feature_names.index("uh_0_2km_mean")
    assert (X[uh_channel] == 0.0).all()
    cape_channel = dataset.feature_names.index("cape_mean")
    assert (X[cape_channel] == 1.0).all()  # untouched, real data unaffected by the imputation


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


# --- evaluate / val_dataset wiring ---


def test_evaluate_restores_prior_training_mode(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path)
    dataset = DenseGridDataset(str(path))
    model = TornadoCNN(in_channels=2, hidden_channels=4)
    criterion = FocalLoss()

    model.train()
    evaluate(dataset, model, criterion)
    assert model.training  # restored to train mode

    model.eval()
    evaluate(dataset, model, criterion)
    assert not model.training  # restored to eval mode


def test_evaluate_does_not_update_weights(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path)
    dataset = DenseGridDataset(str(path))
    model = TornadoCNN(in_channels=2, hidden_channels=4)
    criterion = FocalLoss()

    before = [p.clone() for p in model.parameters()]
    evaluate(dataset, model, criterion)
    after = list(model.parameters())

    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_train_model_with_val_dataset_reports_val_loss_history(tmp_path):
    train_path, _, _ = _tiny_synthetic_nc(tmp_path, n_samples=6, n_active=3, name="train.nc")
    val_path, _, _ = _tiny_synthetic_nc(tmp_path, n_samples=4, n_active=2, name="val.nc")

    train_dataset = DenseGridDataset(str(train_path))
    val_dataset = DenseGridDataset(str(val_path))
    model = TornadoCNN(in_channels=2, hidden_channels=4)

    result = train_model(train_dataset, model, val_dataset=val_dataset, epochs=3, batch_size=2, seed=0)

    assert "val_loss_history" in result
    assert len(result["val_loss_history"]) == 3
    assert all(np.isfinite(loss) for loss in result["val_loss_history"])


def test_train_model_without_val_dataset_omits_val_loss_history(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path)
    dataset = DenseGridDataset(str(path))
    model = TornadoCNN(in_channels=2, hidden_channels=4)

    result = train_model(dataset, model, epochs=2, batch_size=2, seed=0)

    assert "val_loss_history" not in result


# --- optional EMA of weights ---


def test_train_model_ema_is_off_by_default(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path, n_samples=6, n_active=3)
    torch.manual_seed(0)
    result = train_model(DenseGridDataset(str(path)), TornadoCNN(in_channels=2, hidden_channels=4), epochs=2, batch_size=2, quiet_keep_fraction=0.5, seed=0)
    assert "ema_state_dicts" not in result
    assert "ema_decays" not in result["hyperparameters"]


def test_train_model_ema_does_not_change_training_and_zero_decay_equals_final_weights(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path, n_samples=6, n_active=3)

    def run(**kw):
        torch.manual_seed(0)
        model = TornadoCNN(in_channels=2, hidden_channels=4)
        result = train_model(DenseGridDataset(str(path)), model, epochs=3, batch_size=2, quiet_keep_fraction=0.5, seed=0, **kw)
        return model, result

    m_plain, r_plain = run()
    m_ema, r_ema = run(ema_decays=(0.0, 0.99))
    assert r_plain["loss_history"] == r_ema["loss_history"]  # tracking the average must not alter training
    for k, v in m_plain.state_dict().items():
        assert torch.equal(v, m_ema.state_dict()[k])
    sd0 = r_ema["ema_state_dicts"][0.0]
    for k, v in m_ema.state_dict().items():  # decay 0 -> the average is just the latest weights
        assert torch.allclose(sd0[k], v)
    sd99 = r_ema["ema_state_dicts"][0.99]
    assert any(not torch.allclose(sd99[k], v) for k, v in m_ema.state_dict().items())  # a real average differs from the iterate
    fresh = TornadoCNN(in_channels=2, hidden_channels=4)
    fresh.load_state_dict(sd99)  # loadable into a fresh model


def test_train_model_ema_decay_one_stays_at_init(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path, n_samples=6, n_active=3)
    torch.manual_seed(0)
    model = TornadoCNN(in_channels=2, hidden_channels=4)
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = train_model(DenseGridDataset(str(path)), model, epochs=2, batch_size=2, quiet_keep_fraction=0.5, seed=0, ema_decays=(1.0,))
    for k, v in result["ema_state_dicts"][1.0].items():
        assert torch.allclose(v, init[k])
