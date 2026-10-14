"""
Per-cell validity masks for the coarse grid, derived from the country mask
(scripts/08_canada_transfer/build_country_mask.py).

SPC reports are US-only, so Canadian/Mexican cells are always labeled 0 even
when a real tornado occurred there -- "unknown", not "no tornado". The
US-trusted mask keeps US land and water cells (US reports do land on coastal
and lake cells) and drops Canada, Mexico and other land, so those cells can be
excluded from the loss and from metrics. Model INPUTS always keep the full grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_COUNTRY_MASK_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "hrrr_coarse_grid_stride13_countries.nc"
)

WATER, US, CA, MX, OTHER = 0, 1, 2, 3, 4  # codes of the `country` variable


def load_country(path: Path | str = DEFAULT_COUNTRY_MASK_PATH) -> np.ndarray:
    """(n_rows, n_cols) int8 country codes (0 water, 1 US, 2 CA, 3 MX, 4 other land)."""
    with xr.open_dataset(path) as ds:
        return ds["country"].values.astype(np.int8)


def us_trusted_mask(country: np.ndarray) -> np.ndarray:
    """True where SPC labels are meaningful: US land and water cells."""
    return np.isin(country, (WATER, US))


def load_us_trusted_mask(path: Path | str = DEFAULT_COUNTRY_MASK_PATH) -> np.ndarray:
    return us_trusted_mask(load_country(path))


def resolve_cell_mask(name: str | None, path: Path | str = DEFAULT_COUNTRY_MASK_PATH) -> np.ndarray | None:
    """CLI helper: 'none'/None -> no mask; 'us_trusted' -> load_us_trusted_mask()."""
    if name in (None, "none"):
        return None
    if name == "us_trusted":
        return load_us_trusted_mask(path)
    raise ValueError(f"unknown cell mask {name!r} (expected 'none' or 'us_trusted')")
