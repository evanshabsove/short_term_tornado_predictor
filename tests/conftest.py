"""Shared fixtures for the forecast API tests. All are opt-in; nothing here is autouse."""

import numpy as np
import pandas as pd
import pytest

from tornado_predictor.forecast_api import TierConfig
from tornado_predictor.forecast_store import Forecast, ForecastStore
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.time_bins import bin_valid_time_range

BUNDLE_ID = "3c02381f1611"

# (bin, row, col, value): peaks that, after 3x3 max smoothing and the real tier set
# (Elevated >= 0.1224, High >= 0.1902), give 9 High cells in bin 0, 9 Elevated cells in
# each bin, and a strong score inside the Canadian band that must stay "no coverage".
DEFAULT_PEAKS = [(0, 40, 60, 0.25), (0, 50, 100, 0.15), (0, 4, 60, 0.30), (1, 45, 30, 0.13)]


@pytest.fixture(scope="session")
def grid():
    return build_coarse_grid()


@pytest.fixture(scope="session")
def country(grid):
    """Synthetic country codes: water everywhere, a Canadian band in the north,
    some US land, a Mexican block in the south-east, a few other-land cells."""
    c = np.zeros((grid.n_rows, grid.n_cols), dtype=np.int8)
    c[:10, :] = 2
    c[10:30, :20] = 1
    c[70:, 100:] = 3
    c[0:3, 0:3] = 4
    return c


@pytest.fixture(scope="session")
def tier_config():
    """The real tier set from the tracked tier-study results."""
    return TierConfig.load()


def mean_with_peaks(grid, peaks=DEFAULT_PEAKS, background=0.001) -> np.ndarray:
    m = np.full((2, grid.n_rows, grid.n_cols), background, dtype=np.float32)
    for b, r, c, v in peaks:
        m[b, r, c] = v
    return m


@pytest.fixture
def make_forecast(grid):
    def make(init="2026-10-19 01:00", mean=None, *, grid_id=None, bundle_id=BUNDLE_ID, **overrides) -> Forecast:
        init = pd.Timestamp(init)
        mean = mean_with_peaks(grid) if mean is None else mean
        kwargs = dict(
            init_time=init,
            mean=mean,
            spread=np.full(mean.shape, 0.002, dtype=np.float32),
            bundle_id=bundle_id,
            n_members=5,
            feature_names=["cape_mean", "cape_max"],
            bins=[
                {"bin_index": b, "fxx": 4 * b, "valid_start": lo, "valid_end": hi}
                for b, (lo, hi) in enumerate(bin_valid_time_range(init, b) for b in range(2))
            ],
            grid_id=grid.grid_id if grid_id is None else grid_id,
            trained_years={"first": 2014, "last": 2025},
            pull_seconds=31.0,
            generated_at=init + pd.Timedelta(minutes=63),
            git_commit="cafe123",
        )
        kwargs.update(overrides)
        return Forecast(**kwargs)

    return make


@pytest.fixture
def store_root(tmp_path):
    return tmp_path / "live"


@pytest.fixture
def store(store_root):
    return ForecastStore(store_root, "issued")
