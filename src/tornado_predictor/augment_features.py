"""
Adds new HRRR fields to an already-built scaled dataset's staged
per-run files, without re-pulling fields that were already
successfully extracted -- avoids a full ~30-45 hour rebuild
(build_scaled.py's pipeline deletes each date's raw HRRR GRIB cache
right after staging, so there's no way to add a field to an existing
build without *some* new network pulls; this keeps new-pull volume to
just the new fields, not all of them).

Reuses features.py's pull_field_value/pool_field_to_grid directly (both
already operate per-field, no changes needed there) and
dataset.representative_fxx for the same bin->fxx mapping the original
build used, so new fields are extracted at exactly the same
(init_time, fxx) as every other field in that sample -- guaranteed
alignment, not assumed. pull_field_value (not pull_hrrr_field directly)
is what makes 0-2km/0-3km UH's pre-2018-07-13 gap (see features.py's
module docstring) come through as NaN here too, not just in fresh
full builds -- the same optional-field handling applies on both paths.

Each augmented run is written to a NEW path, never overwriting the
source staged file in place: the existing v2 staging directory is the
source of truth for ~40 hours of already-correct work, and a bug in
the merge logic here should never be able to corrupt it. Callers (see
scripts/augment_dataset_fields.py) point src at the existing
data/interim/scaled_build_v2/ directory and dst at a fresh
data/interim/scaled_build_v3/ directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tornado_predictor.dataset import representative_fxx
from tornado_predictor.features import FIELD_UNITS, pool_field_to_grid, pull_field_value, uh_layers_available_value
from tornado_predictor.grid import HrrrCoarseGrid


def merge_new_fields(ds: xr.Dataset, new_fields: dict[str, tuple[np.ndarray, np.ndarray]]) -> xr.Dataset:
    """Pure merge step, separated from the network-pulling orchestration
    below so it's independently testable: assigns each
    name -> (mean_stack, max_stack) pair (already stacked across every
    sample in ds, in ds's own sample order) as new (sample, row, col)
    variables, alongside ds's existing variables unchanged."""
    new_vars = {}
    for name, (mean_stack, max_stack) in new_fields.items():
        units = FIELD_UNITS.get(name, "unknown")
        new_vars[f"{name}_mean"] = (("sample", "row", "col"), mean_stack, {"long_name": f"{name} (block mean)", "units": units})
        new_vars[f"{name}_max"] = (("sample", "row", "col"), max_stack, {"long_name": f"{name} (block max)", "units": units})
    return ds.assign(**new_vars)


def augment_one_run(
    src_path: Path,
    dst_path: Path,
    new_field_specs: list[tuple[str, str]],
    grid: HrrrCoarseGrid,
    hrrr_cache_root: Path,
) -> tuple[bool, str]:
    """Pulls new_field_specs for one already-staged run (src_path,
    unchanged) at the same (init_time, fxx) as every existing field in
    it, merges them in, and writes the result to dst_path (a new file).
    Returns (success, message)."""
    try:
        ds = xr.open_dataset(src_path)
        init_time = pd.Timestamp(ds["init_time"].values[0])
        bin_indices = [int(b) for b in ds["bin_index"].values]

        new_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, search in new_field_specs:
            mean_samples, max_samples = [], []
            for bin_index in bin_indices:
                fxx = representative_fxx(bin_index)
                array = pull_field_value(init_time, fxx, name, search)
                mean, max_ = pool_field_to_grid(array, grid)
                mean_samples.append(mean)
                max_samples.append(max_)
            new_fields[name] = (np.stack(mean_samples), np.stack(max_samples))

        augmented = merge_new_fields(ds, new_fields)

        # v2 staged files predate the concept of optional fields, so they
        # never have this indicator -- add it here too, same value/logic
        # extract_features uses for any freshly-built run going forward.
        n_rows, n_cols = grid.n_rows, grid.n_cols
        availability = np.full((len(bin_indices), n_rows, n_cols), uh_layers_available_value(init_time))
        augmented = augmented.assign(
            uh_layers_available=(
                ("sample", "row", "col"),
                availability,
                {"long_name": "1.0 if 0-2km/0-3km updraft helicity are real (not NaN) for this run, else 0.0"},
            )
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_suffix(".nc.tmp")
        augmented.to_netcdf(tmp_path)
        ds.close()
        tmp_path.replace(dst_path)  # atomic on the same filesystem

        cache_dir = hrrr_cache_root / "hrrr" / f"{init_time:%Y%m%d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        return True, str(dst_path)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
