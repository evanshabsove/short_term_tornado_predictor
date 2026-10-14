"""Spot checks on the saved country mask (scripts/08_canada_transfer/build_country_mask.py).

Skipped if the (gitignored) mask file has not been built.
"""

from pathlib import Path

import pytest
import xarray as xr

from tornado_predictor.grid import HrrrCoarseGrid

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
MASK = ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13_countries.nc"
WATER, US, CA, MX = 0, 1, 2, 3

pytestmark = pytest.mark.skipif(not (GRID.exists() and MASK.exists()), reason="grid/mask not built")


@pytest.fixture(scope="module")
def mask():
    grid = HrrrCoarseGrid.from_netcdf(GRID)
    with xr.open_dataset(MASK) as ds:
        yield grid, ds.load()


def _cell(grid, lon, lat):
    r, c = grid.assign_cells(lon, lat)
    return int(r), int(c)


@pytest.mark.parametrize(
    "name,lon,lat,expected",
    [
        ("Lake Superior", -87.5, 47.7, WATER),
        ("Lake Michigan", -87.0, 43.9, WATER),
        ("Lake Huron", -82.3, 44.8, WATER),
        ("Lake Erie", -81.2, 42.2, WATER),
        ("Lake Ontario", -77.8, 43.7, WATER),
        ("Gulf of Mexico", -90.0, 26.0, WATER),
        ("Kansas", -98.0, 38.5, US),
        ("Calgary", -114.07, 51.05, CA),
        ("Winnipeg", -97.14, 49.9, CA),
        ("Toronto", -79.4, 43.9, CA),
        ("Monterrey", -100.3, 25.7, MX),
    ],
)
def test_known_points(mask, name, lon, lat, expected):
    grid, ds = mask
    r, c = _cell(grid, lon, lat)
    assert int(ds.country[r, c]) == expected, name


def test_fractions_are_consistent(mask):
    _, ds = mask
    total = ds.frac_us + ds.frac_ca + ds.frac_mx + ds.frac_other_land + ds.frac_water
    assert float(abs(total - 1).max()) < 1e-3
    assert float(ds.frac_water.min()) >= 0


def test_lake_cell_is_water_by_area(mask):
    grid, ds = mask
    r, c = _cell(grid, -87.5, 47.7)
    assert float(ds.frac_water[r, c]) > 0.9
