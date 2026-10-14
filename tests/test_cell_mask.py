"""Tests for the cell-mask plumbing: masked FocalLoss, masked 'active' flags,
masked train_model/get_proba_labels, and the country-derived US-trusted mask."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tornado_predictor.cell_mask import (
    CA, MX, US, WATER, load_us_trusted_mask, resolve_cell_mask, us_trusted_mask,
)
from tornado_predictor.evaluate import get_proba_labels
from tornado_predictor.model import TornadoCNN
from tornado_predictor.training import DenseGridDataset, FocalLoss, masked_is_active, train_model

from test_evaluate import _FakeDataset
from test_training import _tiny_synthetic_nc

MASK_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "hrrr_coarse_grid_stride13_countries.nc"


# --- FocalLoss ---


def test_all_ones_mask_equals_unmasked_loss():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 6)
    targets = (torch.rand(3, 5, 6) > 0.7).float()
    crit = FocalLoss()
    assert torch.allclose(crit(logits, targets), crit(logits, targets, torch.ones(5, 6)))


def test_changes_inside_masked_cells_do_not_affect_loss_or_gradient():
    torch.manual_seed(0)
    mask = torch.ones(5, 6)
    mask[:, 4:] = 0  # last two columns excluded
    crit = FocalLoss()
    targets = (torch.rand(2, 5, 6) > 0.7).float()
    logits = torch.randn(2, 5, 6, requires_grad=True)
    base = crit(logits, targets, mask)
    base.backward()
    assert torch.all(logits.grad[..., 4:] == 0)

    t2 = targets.clone()
    t2[..., 4:] = 1 - t2[..., 4:]  # flip labels only in masked cells
    l2 = logits.detach().clone()
    l2[..., 4:] += 5.0  # and change predictions there
    assert torch.allclose(base, crit(l2, t2, mask))


def test_masked_loss_normalizes_by_total_cells():
    # one kept cell with loss L on a 1x2 grid -> L / 2 (total cells), not L / 1
    crit = FocalLoss()
    logits = torch.zeros(1, 1, 2)
    targets = torch.ones(1, 1, 2)
    full = crit(logits, targets)  # mean of two equal cell losses = per-cell loss
    masked = crit(logits, targets, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(masked, full / 2)


# --- masked_is_active ---


def test_is_active_ignores_positives_outside_mask(tmp_path):
    path, n_active, _ = _tiny_synthetic_nc(tmp_path)  # active samples have a positive at [1, 1]
    ds = DenseGridDataset(str(path))
    assert int(masked_is_active(ds, None).sum()) == n_active
    keep = np.ones((5, 6), bool)
    keep[1, 1] = False
    assert int(masked_is_active(ds, keep).sum()) == 0


# --- train_model ---


def test_train_model_with_all_true_mask_matches_no_mask(tmp_path):
    path, _, _ = _tiny_synthetic_nc(tmp_path)

    def run(mask):
        torch.manual_seed(0)
        model = TornadoCNN(in_channels=2, hidden_channels=4)
        r = train_model(DenseGridDataset(str(path)), model, epochs=2, batch_size=2, quiet_keep_fraction=0.5, seed=0, cell_mask=mask)
        return r["loss_history"], r

    base, _ = run(None)
    masked, r = run(np.ones((5, 6), bool))
    assert np.allclose(base, masked)
    assert r["hyperparameters"]["cell_mask_kept_cells"] == 30


# --- get_proba_labels ---


def test_get_proba_labels_with_cell_mask_keeps_only_valid_cells():
    torch.manual_seed(0)
    model = TornadoCNN(in_channels=2, hidden_channels=4)
    rng = np.random.default_rng(0)
    samples = [(rng.normal(size=(2, 4, 5)).astype("float32"), np.zeros((4, 5), "float32")) for _ in range(2)]
    samples[1][1][:] = 1.0
    ds = _FakeDataset(samples)
    keep = np.zeros((4, 5), bool)
    keep[:2] = True  # 10 of 20 cells
    s_full, y_full = get_proba_labels(model, ds)
    s, y = get_proba_labels(model, ds, cell_mask=keep)
    assert s.shape == (2 * 10,) and y.shape == (2 * 10,)
    assert np.allclose(s[:10], s_full[:20].reshape(4, 5)[:2].ravel())
    assert get_proba_labels(model, ds, cell_mask=np.ones((4, 5), bool))[0].shape == s_full.shape


# --- country-derived mask ---


def test_us_trusted_mask_from_codes():
    country = np.array([[WATER, US, CA, MX, 4]], dtype=np.int8)
    assert us_trusted_mask(country).tolist() == [[True, True, False, False, False]]


def test_resolve_cell_mask_names():
    assert resolve_cell_mask("none") is None and resolve_cell_mask(None) is None
    with pytest.raises(ValueError):
        resolve_cell_mask("nope")


@pytest.mark.skipif(not MASK_FILE.exists(), reason="country mask not built")
def test_real_us_trusted_mask_keeps_us_and_water_drops_canada_mexico():
    from tornado_predictor.grid import HrrrCoarseGrid

    grid = HrrrCoarseGrid.from_netcdf(MASK_FILE.with_name("hrrr_coarse_grid_stride13.nc"))
    valid = load_us_trusted_mask(MASK_FILE)
    assert valid.shape == (81, 138) and valid.dtype == bool

    def at(lon, lat):
        r, c = grid.assign_cells(lon, lat)
        return bool(valid[int(r), int(c)])

    assert at(-98.0, 38.5)  # Kansas
    assert at(-87.5, 47.7)  # Lake Superior (water kept)
    assert not at(-114.07, 51.05)  # Calgary
    assert not at(-100.3, 25.7)  # Monterrey
