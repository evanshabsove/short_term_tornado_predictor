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
2m temperature/dewpoint, and 0-2km/0-3km/2-5km max updraft helicity
(MXUPHL). UH was added later than the original 7 fields, motivated by
literature (Sobash et al.'s "Explicit Forecasts of Low-Level Rotation"
line of work) identifying it as the single most predictive direct
model-diagnosed tornado/supercell surrogate in convection-allowing
model output -- more so than the purely thermodynamic/kinematic fields
this project started with. HRRR has no literal 0-1km MXUPHL layer (the
"UH01" some literature uses); 0-2km is the closest available layer,
confirmed by grepping a live .idx file, same spirit as the existing
0-6km-not-0-8km-shear precedent. Each Herbie pull targets exactly one
field (rather than combining several per call) so the resulting xarray
Dataset always has exactly one data variable, sidestepping any
uncertainty about cfgrib's assigned variable names or Herbie's
multi-hypercube merge behavior.

This module extracts a single (init_time, fxx) snapshot. Aggregating
several fxx snapshots into one time_bins bin is a separate future step.

**0-2km/0-3km updraft helicity are only available for part of the
archive.** Binary-searched live against real .idx files: absent on
2018-07-10, present on 2018-07-13 -- matching NOAA's documented HRRRv3
upgrade (2018-07-12). Only the 2-5km layer ("UH25", also the single
most literature-cited UH layer) is available for this project's full
~Sept 2014 - Sept 2025 archive. Per explicit user decision (not a
default silently chosen here), 0-2km/0-3km are pulled anyway and filled
with NaN for runs before the verified boundary, rather than restricting
the archive's date range or dropping the fields -- see
OPTIONAL_FIELD_NAMES and uh_layers_available_value. Consuming code
(training.DenseGridDataset) imputes these NaNs to 0.0 at model-input
time, not here -- this module's stored output stays raw/honest about
what is and isn't real data, consistent with dataset.py's existing
"storage isn't lossy, transformations happen explicitly downstream"
principle.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
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
    ("uh_0_2km", "MXUPHL:2000-0 m above ground"),
    ("uh_0_3km", "MXUPHL:3000-0 m above ground"),
    ("uh_2_5km", "MXUPHL:5000-2000 m above ground"),
]

# Names of fields added after the original 7 -- see module docstring.
# Used by augment_features.py to pull just the new fields for an
# already-built dataset without re-pulling everything.
NEW_FIELD_NAMES_2026_09: list[str] = ["uh_0_2km", "uh_0_3km", "uh_2_5km"]

# Fields not available for the full archive -- see module docstring.
# pull_field_value returns NaN (not an error) for these when a date's
# .idx doesn't have them, instead of raising/retrying.
OPTIONAL_FIELD_NAMES: set[str] = {"uh_0_2km", "uh_0_3km"}

# Verified live (binary search): absent 2018-07-10, present 2018-07-13.
# Matches NOAA's documented HRRRv3 upgrade (2018-07-12) -- this
# single-day precision gap (07-11/07-12 not directly checked) is
# immaterial: at most it misclassifies one calendar day's runs.
UH_EXTRA_LAYERS_AVAILABLE_FROM = pd.Timestamp("2018-07-13")

FIELD_UNITS: dict[str, str] = {
    "cape": "J kg-1",
    "cin": "J kg-1",
    "srh_0_1km": "m2 s-2",
    "srh_0_3km": "m2 s-2",
    "shear_0_6km": "m s-1",
    "t2m": "K",
    "d2m": "K",
    "uh_0_2km": "m2 s-2",
    "uh_0_3km": "m2 s-2",
    "uh_2_5km": "m2 s-2",
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


def field_available(init_time, fxx: int, search_string: str) -> bool:
    """Checks a live .idx (cheap -- the small index file, not a GRIB
    byte-range subset) for whether search_string matches anything,
    without attempting a full pull. Used to distinguish "this field
    genuinely doesn't exist for this date" (deterministic, no point
    retrying) from a transient network failure (which pull_hrrr_field
    already retries)."""
    H = Herbie(init_time, model="hrrr", product="sfc", fxx=fxx)
    idx = H.index_as_dataframe
    return bool(idx["search_this"].str.contains(search_string, regex=True).any())


def pull_field_value(init_time, fxx: int, name: str, search_string: str) -> np.ndarray:
    """Pulls one named field. For a field in OPTIONAL_FIELD_NAMES that
    isn't in this date/fxx's .idx, returns a NaN-filled array instead
    of raising -- see module docstring. Every other field still raises
    (via pull_hrrr_field's own retries) on failure, unchanged."""
    if name in OPTIONAL_FIELD_NAMES and not field_available(init_time, fxx, search_string):
        return np.full((HRRR_NY, HRRR_NX), np.nan, dtype="float32")
    return pull_hrrr_field(init_time, fxx, search_string)


def uh_layers_available_value(init_time) -> float:
    """1.0 if init_time's run can have real (not NaN) 0-2km/0-3km UH
    values, 0.0 otherwise -- an explicit missingness indicator feature
    (not just relying on the model to infer it from NaN patterns),
    always present regardless of whether this particular run actually
    has any NaN fields."""
    return float(pd.Timestamp(init_time) >= UH_EXTRA_LAYERS_AVAILABLE_FROM)


def pull_all_fields(init_time, fxx: int) -> dict[str, np.ndarray]:
    """Pulls every field in FIELD_SPECS and combines the 0-6km shear
    components into a single magnitude field."""
    raw = {name: pull_field_value(init_time, fxx, name, search) for name, search in FIELD_SPECS}
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

    # Single indicator variable (not split mean/max -- it's a per-run
    # constant, not a pooled physical field), always present regardless
    # of whether this run actually has any NaN fields.
    data_vars["uh_layers_available"] = (
        ("row", "col"),
        np.full((grid.n_rows, grid.n_cols), uh_layers_available_value(init_time)),
        {"long_name": "1.0 if 0-2km/0-3km updraft helicity are real (not NaN) for this run, else 0.0"},
    )

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
