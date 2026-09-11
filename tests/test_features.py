import numpy as np

from tornado_predictor.features import pool_field_to_grid
from tornado_predictor.grid import HRRR_NX, HRRR_NY, build_coarse_grid

GRID = build_coarse_grid()


def test_constant_field_pools_to_same_constant():
    field = np.full((HRRR_NY, HRRR_NX), 7.5)
    mean, max_ = pool_field_to_grid(field, GRID)
    assert mean.shape == (GRID.n_rows, GRID.n_cols)
    assert max_.shape == (GRID.n_rows, GRID.n_cols)
    assert np.allclose(mean, 7.5)
    assert np.allclose(max_, 7.5)


def test_spike_is_captured_by_max_but_diluted_in_mean():
    field = np.zeros((HRRR_NY, HRRR_NX))
    stride = GRID.stride
    # Put a spike at the first native pixel of block (2, 3).
    field[2 * stride, 3 * stride] = 1000.0
    mean, max_ = pool_field_to_grid(field, GRID)

    assert max_[2, 3] == 1000.0
    assert mean[2, 3] == 1000.0 / (stride * stride)

    # Neighboring blocks are unaffected.
    assert max_[2, 4] == 0.0
    assert mean[2, 4] == 0.0
    assert max_[1, 3] == 0.0


def test_mean_never_exceeds_max():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(HRRR_NY, HRRR_NX))
    mean, max_ = pool_field_to_grid(field, GRID)
    assert (mean <= max_ + 1e-12).all()
