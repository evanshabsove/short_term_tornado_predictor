"""
Hourly HRRR staging and whole-bin feature aggregation.

Every earlier dataset represented a 4-hour bin by ONE forecast hour (fxx 0
for bin 0, fxx 4 for bin 1 -- `dataset.representative_fxx`). This module
stages all 8 hourly snapshots (fxx 0-7) per run so a bin's features can be
built from all 4 of its hours.

Two separate steps, deliberately:

1. **Stage hourly snapshots** (network, slow, resumable): one file per run,
   each of the 14 base features as `(fxx=8, row, col)`. fxx 0 and 4 are
   copied from the run's existing v2 staged file (sample 0 / sample 1 are
   exactly those two snapshots) instead of re-pulled, so only 6 of 8 hours
   are downloaded. Written atomically and only when all 8 hours exist.
2. **Derive whole-bin staging** (local, fast): per field, time-mean of the
   `_mean` channel and time-max of the `_max` channel over the bin's hours
   (`aggregate_bin`), plus the labels, written in exactly the format
   `build_scaled` stages (per-run file, `sample` dim of 2 bins, 14 features +
   `label`). Anything that reads v2 staging (`combine_staged_runs`,
   `finalize`, `cross_validate_year.py --staging-dir`) works on it unchanged.
   Other aggregations (tendency, spread, 4 separate hours) can be derived
   from the same hourly store without any further download.

Pulling is parallelised across DATES, never within one: `process_one_hourly_run`
deletes its date's whole Herbie cache directory when it finishes, which would
race with another worker on the same date. The manifest is written only by
the parent process.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from tornado_predictor.build_scaled import load_manifest, run_staging_path, save_manifest
from tornado_predictor.dataset import label_grid_for_sample, representative_fxx
from tornado_predictor.features import EXTRA_FIELD_SPECS, extract_features, pool_field_to_grid, pull_hrrr_field
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.time_bins import MAX_LEAD_HOURS, N_BINS, fxx_values_for_bin

# Channel order of every existing 14-feature checkpoint (checked against
# models/unet/tornado_unet.pt) and of features.extract_features(include_uh=False).
BASE_FEATURE_NAMES: list[str] = [
    "cape_mean", "cape_max",
    "cin_mean", "cin_max",
    "srh_0_1km_mean", "srh_0_1km_max",
    "srh_0_3km_mean", "srh_0_3km_max",
    "t2m_mean", "t2m_max",
    "d2m_mean", "d2m_max",
    "shear_0_6km_mean", "shear_0_6km_max",
]

# Extra candidate fields (features.EXTRA_FIELD_SPECS), pooled the same way. The two
# 0-1 km shear components collapse into one magnitude, appended last like shear_0_6km.
EXTRA_PHYSICAL_NAMES: list[str] = [
    n for n, _ in EXTRA_FIELD_SPECS if n not in ("shear_u_0_1km", "shear_v_0_1km")
] + ["shear_0_1km"]
EXTRA_FEATURE_NAMES: list[str] = [f"{n}_{s}" for n in EXTRA_PHYSICAL_NAMES for s in ("mean", "max")]

FEATURE_SETS: dict[str, list[str]] = {"base": BASE_FEATURE_NAMES, "extra": EXTRA_FEATURE_NAMES}

# Derived from the hourly extras store (no extra download): surface-pressure tendency within a bin, the cell-mean
# pressure at the bin's last hour minus its first (Pa over 3 h). Appended after the extra channels.
TENDENCY_FEATURE_NAMES: list[str] = ["pres_sfc_tend"]

# Channels that experiment B z-scores (training-split statistics); the base 14 stay raw like every earlier model.
STANDARDIZED_FEATURE_NAMES: list[str] = EXTRA_FEATURE_NAMES + TENDENCY_FEATURE_NAMES

HOURLY_FXX: tuple[int, ...] = tuple(range(MAX_LEAD_HOURS))

# fxx already present in a v2 staged file -> its sample index there.
STAGED_FXX_TO_SAMPLE: dict[int, int] = {representative_fxx(b): b for b in range(N_BINS)}

Snapshot = dict[str, np.ndarray]  # feature name -> (row, col) float32


def hourly_run_path(staging_dir: Path, init_time: pd.Timestamp) -> Path:
    return run_staging_path(staging_dir, pd.Timestamp(init_time))


def pull_snapshot(init_time, fxx: int, grid: HrrrCoarseGrid) -> Snapshot:
    """One live (init_time, fxx) pull of the 14 base features."""
    ds = extract_features(init_time, fxx, grid, include_uh=False)
    return {name: ds[name].values.astype("float32") for name in BASE_FEATURE_NAMES}


# HRRR leaves the surface-based LCL undefined at a FIXED set of 5,712 native pixels (identical in
# every era and forecast hour checked, 2014-2024); they fall in 218 coarse cells, all over water
# (190) or Mexico (28), none US or Canada, and no cell is entirely undefined. LCL is pooled over
# each cell's valid pixels. Every other extra field must have no NaN at all.
NAN_TOLERANT_FIELDS: dict[str, float] = {"lcl_sb": 0.01}  # field -> max allowed NaN fraction of native pixels


def pool_field_nan_tolerant(field: np.ndarray, grid: HrrrCoarseGrid) -> tuple[np.ndarray, np.ndarray]:
    """features.pool_field_to_grid over each cell's non-NaN pixels. Raises if a
    cell has no valid pixel at all (nothing to pool)."""
    n_rows, n_cols, stride = grid.n_rows, grid.n_cols, grid.stride
    blocks = field[: n_rows * stride, : n_cols * stride].reshape(n_rows, stride, n_cols, stride)
    if np.isnan(blocks).all(axis=(1, 3)).any():
        raise ValueError("a coarse cell has no valid native pixel")
    return np.nanmean(blocks, axis=(1, 3)), np.nanmax(blocks, axis=(1, 3))


def pull_extra_snapshot(init_time, fxx: int, grid: HrrrCoarseGrid) -> Snapshot:
    """One live (init_time, fxx) pull of the extra candidate fields, pooled to
    mean/max per cell. Raises on any NaN except the known LCL undefined pixels
    (NAN_TOLERANT_FIELDS, within its stated fraction): a NaN anywhere else means a
    bad pull and must not be staged silently."""
    raw = {name: pull_hrrr_field(str(init_time), fxx, search) for name, search in EXTRA_FIELD_SPECS}
    raw["shear_0_1km"] = np.hypot(raw.pop("shear_u_0_1km"), raw.pop("shear_v_0_1km"))
    out: Snapshot = {}
    for name in EXTRA_PHYSICAL_NAMES:
        n_nan = int(np.isnan(raw[name]).sum())
        if name in NAN_TOLERANT_FIELDS:
            if n_nan / raw[name].size > NAN_TOLERANT_FIELDS[name]:
                raise ValueError(f"{n_nan} NaN pixels in {name!r} at {init_time} fxx={fxx}: more than expected")
            mean, max_ = pool_field_nan_tolerant(raw[name], grid)
        else:
            if n_nan:
                raise ValueError(f"NaN in extra field {name!r} at {init_time} fxx={fxx}")
            mean, max_ = pool_field_to_grid(raw[name], grid)
        out[f"{name}_mean"] = mean.astype("float32")
        out[f"{name}_max"] = max_.astype("float32")
    return out


def staged_snapshot(staged_path: Path, bin_index: int) -> Snapshot:
    """The 14 base features of one sample of an existing v2 staged file."""
    with xr.open_dataset(staged_path) as ds:
        return {name: ds[name].isel(sample=bin_index).values.astype("float32") for name in BASE_FEATURE_NAMES}


def build_hourly_run(
    init_time,
    grid: HrrrCoarseGrid,
    staged_dir: Path | None,
    pull: Callable[[object, int, HrrrCoarseGrid], Snapshot] = pull_snapshot,
    feature_names: list[str] = BASE_FEATURE_NAMES,
) -> xr.Dataset:
    """All 8 hourly snapshots of one run as a `(fxx, row, col)` dataset. fxx
    0 and 4 come from `staged_dir` when it is given (pass None to pull all
    8); every other hour is pulled. Raises if any hour cannot be obtained --
    a run is all-or-nothing."""
    init_time = pd.Timestamp(init_time)
    snapshots: dict[int, Snapshot] = {}
    reused: list[int] = []
    if staged_dir is not None:
        staged_path = run_staging_path(staged_dir, init_time)
        for fxx, sample in STAGED_FXX_TO_SAMPLE.items():
            snapshots[fxx] = staged_snapshot(staged_path, sample)
            reused.append(fxx)
    for fxx in HOURLY_FXX:
        if fxx not in snapshots:
            snapshots[fxx] = pull(init_time, fxx, grid)

    data_vars = {
        name: (("fxx", "row", "col"), np.stack([snapshots[f][name] for f in HOURLY_FXX]).astype("float32"))
        for name in feature_names
    }
    return xr.Dataset(
        data_vars,
        coords={"fxx": list(HOURLY_FXX), "row": np.arange(grid.n_rows), "col": np.arange(grid.n_cols)},
        attrs={
            "title": "Hourly HRRR snapshots (fxx 0-7) pooled onto the coarse grid, 14 base features",
            "init_time": str(init_time),
            "reused_fxx": ",".join(str(f) for f in reused),
        },
    )


def _cleanup_cache(cache_root: Path, init_time: pd.Timestamp) -> None:
    cache_dir = cache_root / "hrrr" / f"{init_time:%Y%m%d}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def process_one_hourly_run(
    init_time,
    grid: HrrrCoarseGrid,
    staged_dir: Path | None,
    out_dir: Path,
    cache_root: Path | None,
    pull: Callable[[object, int, HrrrCoarseGrid], Snapshot] | None = None,
    feature_set: str = "base",
) -> tuple[bool, str]:
    """Stages one run's hourly file (atomic write) and cleans up its raw
    HRRR cache. An already-staged run is left alone (cheap resume after a
    crash that lost the manifest entry). Returns (success, message)."""
    init_time = pd.Timestamp(init_time)
    out_path = hourly_run_path(out_dir, init_time)
    if out_path.exists():
        return True, str(out_path)
    if feature_set == "extra":
        # no staged copy of these fields exists, so all 8 hours are pulled
        staged_dir, pull = None, pull or pull_extra_snapshot
    pull = pull or pull_snapshot
    try:
        ds = build_hourly_run(init_time, grid, staged_dir, pull=pull, feature_names=FEATURE_SETS[feature_set])
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.stem + ".partial.nc")
    ds.to_netcdf(partial)
    partial.replace(out_path)
    if cache_root is not None:
        _cleanup_cache(cache_root, init_time)
    return True, str(out_path)


def group_runs_by_date(runs: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    """Runs grouped by UTC calendar date of init_time, each group sorted --
    the unit of parallel work (see module docstring)."""
    groups: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for r in sorted(pd.Timestamp(r) for r in runs):
        groups.setdefault(r.normalize(), []).append(r)
    return list(groups.values())


def _process_date_group(
    runs: list[pd.Timestamp],
    grid_path: Path,
    staged_dir: Path | None,
    out_dir: Path,
    cache_root: Path | None,
    feature_set: str = "base",
) -> list[tuple[pd.Timestamp, bool, str]]:
    grid = HrrrCoarseGrid.from_netcdf(grid_path)
    return [
        (r, *process_one_hourly_run(r, grid, staged_dir, out_dir, cache_root, feature_set=feature_set)) for r in runs
    ]


def run_hourly_pull(
    runs: list[pd.Timestamp],
    grid_path: Path,
    staged_dir: Path | None,
    out_dir: Path,
    manifest_path: Path,
    cache_root: Path | None,
    workers: int = 1,
    retry_failed: bool = False,
    log: Callable[[str], None] = print,
    feature_set: str = "base",
) -> dict:
    """Stages every run in `runs` not already `done` in the manifest (or
    failed, with retry_failed), `workers` dates at a time. Returns the
    manifest. Safe to interrupt and rerun."""
    manifest = load_manifest(manifest_path)
    todo = [
        r for r in runs
        if pd.Timestamp(r).isoformat() not in manifest
        or (manifest[pd.Timestamp(r).isoformat()]["status"] == "failed" and retry_failed)
    ]
    log(f"{len(todo)} runs to stage ({len(runs) - len(todo)} already in manifest)")
    groups = group_runs_by_date(todo)

    def record(results, done_count):
        for run, ok, message in results:
            manifest[run.isoformat()] = {"status": "done" if ok else "failed", "message": message}
            done_count += 1
            log(f"[{done_count}/{len(todo)}] {run}: {'OK' if ok else 'FAILED -- ' + message}")
        save_manifest(manifest_path, manifest)
        return done_count

    done = 0
    if workers <= 1:
        for group in groups:
            done = record(_process_date_group(group, grid_path, staged_dir, out_dir, cache_root, feature_set), done)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_date_group, g, grid_path, staged_dir, out_dir, cache_root, feature_set) for g in groups
            ]
            for fut in as_completed(futures):
                done = record(fut.result(), done)
    return manifest


# ---- step 2: derive whole-bin staging -------------------------------------------------


def aggregate_bin(hourly: xr.Dataset, bin_index: int, feature_names: list[str] = BASE_FEATURE_NAMES) -> xr.Dataset:
    """One bin's `(row, col)` features from all of its hours: time-mean of
    each `_mean` channel, time-max of each `_max` channel. Variable names and
    order are exactly `feature_names` (default BASE_FEATURE_NAMES)."""
    sub = hourly.sel(fxx=list(fxx_values_for_bin(bin_index)))
    out = {}
    for name in feature_names:
        if name.endswith("_mean"):
            reduced = sub[name].mean("fxx")
        elif name.endswith("_max"):
            reduced = sub[name].max("fxx")
        else:
            raise ValueError(f"cannot aggregate {name!r}: not a _mean/_max channel")
        out[name] = (("row", "col"), reduced.values.astype("float32"))
    return xr.Dataset(out)


def pressure_tendency(extra_hourly: xr.Dataset, bin_index: int) -> xr.DataArray:
    """(row, col) change in cell-mean surface pressure across a bin: last hour minus first."""
    hours = list(fxx_values_for_bin(bin_index))
    p = extra_hourly["pres_sfc_mean"]
    return xr.DataArray(p.sel(fxx=hours[-1]).values - p.sel(fxx=hours[0]).values, dims=("row", "col"))


def derive_wholebin_run(
    hourly_path: Path,
    labels_df: pd.DataFrame,
    grid: HrrrCoarseGrid,
    out_path: Path,
    extra_hourly_path: Path | None = None,
) -> None:
    """Writes one run's whole-bin staged file (v2 staging format) from its
    hourly file and the labels table. With extra_hourly_path, the extra
    candidate channels and the pressure-tendency channel are appended after the
    14 base channels (so a base-only experiment is just `--drop-columns` of them)."""
    with xr.open_dataset(hourly_path) as hourly:
        hourly = hourly.load()
    extra = None
    if extra_hourly_path is not None:
        with xr.open_dataset(extra_hourly_path) as e:
            extra = e.load()
    init_time = pd.Timestamp(hourly.attrs["init_time"])
    samples = []
    for bin_index in range(N_BINS):
        ds = aggregate_bin(hourly, bin_index)
        if extra is not None:
            ds = ds.assign(**aggregate_bin(extra, bin_index, EXTRA_FEATURE_NAMES).data_vars)
            ds["pres_sfc_tend"] = (("row", "col"), pressure_tendency(extra, bin_index).values.astype("float32"))
        label = label_grid_for_sample(labels_df, init_time, bin_index, grid)
        samples.append(ds.assign(label=(("row", "col"), label)))
    combined = xr.concat(samples, dim="sample").assign_coords(
        init_time=("sample", [init_time] * N_BINS),
        bin_index=("sample", list(range(N_BINS))),
        lat=(("row", "col"), grid.lat_center),
        lon=(("row", "col"), grid.lon_center),
    )
    combined.attrs = {
        "title": "HRRR severe-weather fields pooled onto the coarse dense-grid",
        "description": (
            "Whole-bin features: each channel is built from all 4 forecast hours of its bin "
            "(time-mean of the _mean channel, time-max of the _max channel)."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.stem + ".partial.nc")
    combined.to_netcdf(partial)
    partial.replace(out_path)


def derive_wholebin_staging(
    hourly_dir: Path,
    out_dir: Path,
    runs: list[pd.Timestamp],
    labels_df: pd.DataFrame,
    grid: HrrrCoarseGrid,
    log: Callable[[str], None] = print,
    extra_hourly_dir: Path | None = None,
) -> dict:
    """Derives whole-bin staging (+ a v2-format manifest) for every run in
    `runs` that has an hourly file (and, with extra_hourly_dir, an extras
    file too -- runs missing either are skipped). Returns the manifest
    written to out_dir/manifest.json."""
    manifest = {}
    for i, run in enumerate(sorted(pd.Timestamp(r) for r in runs)):
        src = hourly_run_path(hourly_dir, run)
        if not src.exists():
            continue
        extra_src = hourly_run_path(extra_hourly_dir, run) if extra_hourly_dir is not None else None
        if extra_src is not None and not extra_src.exists():
            continue
        dst = run_staging_path(out_dir, run)
        derive_wholebin_run(src, labels_df, grid, dst, extra_hourly_path=extra_src)
        manifest[run.isoformat()] = {"status": "done", "message": str(dst)}
        if (i + 1) % 500 == 0:
            log(f"derived {i + 1} runs")
    save_manifest(out_dir / "manifest.json", manifest)
    return manifest
