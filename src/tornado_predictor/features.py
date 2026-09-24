"""
Pulls severe-weather-relevant HRRR forecast fields for a given (init
time, forecast hour) via Herbie, and reduces each onto the ~40km grid
defined in grid.py.

Pooling reuses the grid's exact block alignment (every coarse cell is a
13x13 block of native HRRR pixels) instead of interpolating: each native
field is reshaped into blocks and reduced with mean and max. This is
zero-error relative to the native data and needs no extra geospatial
dependency.

Fields pulled (all verified against a live HRRR "sfc" product .idx
file): surface-based CAPE and CIN, 0-1km and 0-3km storm-relative
helicity, 0-6km bulk shear magnitude (computed from the native
VUCSH/VVCSH component fields -- HRRR has no native 0-8km shear field),
and 2m temperature/dewpoint. Each Herbie pull targets exactly one field
(rather than combining several per call) so the resulting xarray
Dataset always has exactly one data variable, sidestepping any
uncertainty about cfgrib's assigned variable names or Herbie's
multi-hypercube merge behavior.

This module extracts a single (init_time, fxx) snapshot. Aggregating
several fxx snapshots into one time_bins bin is a separate future step.
"""

from __future__ import annotations

import time

import numpy as np
import xarray as xr
from herbie import Herbie

from tornado_predictor.grid import HRRR_NX, HRRR_NY, HrrrCoarseGrid

# (canonical field name, Herbie search string). Exact level text
# confirmed against a live HRRR sfc .idx file -- note HLCY's "<top>-0"
# ordering (e.g. "1000-0", not "0-1000") and that a loose "CAPE:" would
# also match HRRR's other CAPE variants (mixed-layer, most-unstable).
FIELD_SPECS: list[tuple[str, str]] = [
    ("cape", "CAPE:surface"),
    ("cin", "CIN:surface"),
    ("srh_0_1km", "HLCY:1000-0 m above ground"),
    ("srh_0_3km", "HLCY:3000-0 m above ground"),
    ("shear_u_0_6km", "VUCSH:0-6000 m above ground"),
    ("shear_v_0_6km", "VVCSH:0-6000 m above ground"),
    ("t2m", "TMP:2 m above ground"),
    ("d2m", "DPT:2 m above ground"),
]

FIELD_UNITS: dict[str, str] = {
    "cape": "J kg-1",
    "cin": "J kg-1",
    "srh_0_1km": "m2 s-2",
    "srh_0_3km": "m2 s-2",
    "shear_0_6km": "m s-1",
    "t2m": "K",
    "d2m": "K",
}


def pull_hrrr_field(init_time, fxx: int, search_string: str, max_attempts: int = 3) -> np.ndarray:
    """One Herbie pull for a single field; returns its raw native-grid
    2D array. Asserts the shape matches HRRR's documented native grid --
    fail loud if a pull ever returns something unexpected (wrong
    product, cropped subset) rather than silently mis-pooling later.

    Retries up to max_attempts times with a short backoff -- for a
    large batch job (thousands of pulls over many hours), transient
    network blips and corrupted-cache reads (see CLAUDE.md's
    "corrupted local GRIB cache" incident) are expected occasionally;
    retrying is cheap and avoids failing an entire date over one flaky
    request. Does not distinguish transient failures from a genuinely
    missing date (e.g. before HRRR's archive starts) -- both raise
    after max_attempts, which the caller should treat as a real
    failure to record and move on from, not retry indefinitely."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            H = Herbie(init_time, model="hrrr", product="sfc", fxx=fxx)
            ds = H.xarray(search_string)
            da = next(iter(ds.data_vars.values()))
            array = da.values
            assert array.shape == (HRRR_NY, HRRR_NX), (
                f"unexpected shape {array.shape} for {search_string!r}, expected {(HRRR_NY, HRRR_NX)}"
            )
            return array
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(2 * (attempt + 1))
    raise last_error


def pull_all_fields(init_time, fxx: int) -> dict[str, np.ndarray]:
    """Pulls every field in FIELD_SPECS and combines the 0-6km shear
    components into a single magnitude field."""
    raw = {name: pull_hrrr_field(init_time, fxx, search) for name, search in FIELD_SPECS}
    shear_0_6km = np.hypot(raw.pop("shear_u_0_6km"), raw.pop("shear_v_0_6km"))
    return {**raw, "shear_0_6km": shear_0_6km}


def pool_field_to_grid(field: np.ndarray, grid: HrrrCoarseGrid) -> tuple[np.ndarray, np.ndarray]:
    """Block-pool a native-grid field onto grid's coarse cells, via
    mean and max over each stride x stride block. Cropping to
    n_rows*stride, n_cols*stride matches exactly the same leftover-pixel
    truncation grid.py's build_coarse_grid already applies -- this is
    what keeps pooled cells aligned index-for-index with
    grid.lat_center/lon_center."""
    n_rows, n_cols, stride = grid.n_rows, grid.n_cols, grid.stride
    cropped = field[: n_rows * stride, : n_cols * stride]
    blocks = cropped.reshape(n_rows, stride, n_cols, stride)
    return blocks.mean(axis=(1, 3)), blocks.max(axis=(1, 3))


def extract_features(init_time, fxx: int, grid: HrrrCoarseGrid) -> xr.Dataset:
    """Pulls all fields for (init_time, fxx) and pools each onto grid,
    returning a Dataset of <field>_mean/<field>_max variables over
    (row, col)."""
    init_time = str(init_time)
    fields = pull_all_fields(init_time, fxx)

    data_vars = {
        "lat": (("row", "col"), grid.lat_center, {"units": "degrees_north"}),
        "lon": (("row", "col"), grid.lon_center, {"units": "degrees_east"}),
    }
    for name, array in fields.items():
        mean, max_ = pool_field_to_grid(array, grid)
        units = FIELD_UNITS[name]
        data_vars[f"{name}_mean"] = (("row", "col"), mean, {"long_name": f"{name} (block mean)", "units": units})
        data_vars[f"{name}_max"] = (("row", "col"), max_, {"long_name": f"{name} (block max)", "units": units})

    valid_time = (np.datetime64(init_time) + np.timedelta64(fxx, "h")).astype("datetime64[s]")

    # netCDF attrs must be strings/numbers/arrays thereof, not nested
    # dicts -- record each field's search string under its own key.
    search_attrs = {f"search_{name}": search for name, search in FIELD_SPECS}

    return xr.Dataset(
        data_vars=data_vars,
        coords={"row": np.arange(grid.n_rows), "col": np.arange(grid.n_cols)},
        attrs={
            "title": "HRRR severe-weather fields pooled onto the coarse dense-grid",
            "init_time": init_time,
            "fxx": fxx,
            "valid_time": str(valid_time),
            **search_attrs,
        },
    )
