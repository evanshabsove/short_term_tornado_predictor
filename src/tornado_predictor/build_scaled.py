"""
Incremental, resumable, fault-tolerant orchestration for building a
large-scale consolidated dataset -- the full 2014-2025 active+quiet
scale-up runs ~40,000 HRRR pulls over an estimated day+ of wall time,
which the simple all-in-memory approach in dataset.build_dataset()
cannot safely support: a single network hiccup near the end would lose
all prior work, and there's no way to stop and resume.

Design: process one date at a time (its single run's 2 bin samples),
saving each date's small result to its own file in a staging directory
immediately after it succeeds, and recording progress in a JSON
manifest. A crash or interruption loses at most the one date in
progress (each date is ~16 HRRR pulls, well under a minute), not the
whole job. Re-running the build script re-reads the manifest and skips
every date already marked "done", so it's safe to stop and restart at
any point (laptop sleep, network drop, killed process).

Each date's raw HRRR GRIB cache (downloaded by Herbie under its own
cache directory, not by this module) is deleted after that date's
sample is safely staged -- at this job's scale (~40,000 individual
field pulls), leaving every raw GRIB file in place would consume on
the order of 100+ GB; the staged per-date outputs (already pooled onto
the coarse grid) are two to three orders of magnitude smaller.

Failures (after features.py's own internal retries are exhausted) are
recorded in the manifest as "failed" with the error message, not
retried automatically on the next run -- a genuinely bad date (e.g.
one that turns out to predate HRRR's archive, as happened once during
manual date selection) would otherwise retry forever. Use
--retry-failed to explicitly give failed dates another attempt.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie

from tornado_predictor.dataset import build_dataset
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.split import split_run_bins
from tornado_predictor.time_bins import N_BINS

ARCHIVE_START = pd.Timestamp("2014-09-01")  # verified live: CAPE/CIN/HLCY/shear are absent
# from the sfc product before this date (only a reduced ~57-field set, missing every
# severe-weather field this project needs, is available Aug 15-31 2014) -- confirmed
# by direct .idx inspection, not the earlier (wrong) "archive starts ~Aug 15" finding,
# which only checked raw file existence, not which fields it contained.


def hrrr_cache_root() -> Path:
    """Herbie's actual configured cache directory (~/.config/herbie/config.toml's
    save_dir), read from a live Herbie instance rather than hardcoded --
    this can differ per machine/environment."""
    return Herbie("2024-01-01 00:00", model="hrrr", product="sfc", fxx=0).save_dir


def select_active_dates(reports_df: pd.DataFrame) -> list[pd.Timestamp]:
    """Every unique UTC calendar date with >=1 SPC tornado report."""
    dates = pd.to_datetime(reports_df["timestamp_utc"]).dt.normalize().unique()
    return sorted(pd.Timestamp(d) for d in dates if pd.Timestamp(d) >= ARCHIVE_START)


def select_quiet_dates(reports_df: pd.DataFrame, n: int, end: pd.Timestamp, seed: int = 0) -> list[pd.Timestamp]:
    """n random UTC calendar dates with zero SPC tornado reports,
    verified within the HRRR archive (>= ARCHIVE_START). Sampled with
    replacement-free rejection from the full archive-to-end range, not
    hand-picked -- unbiased across years/seasons by construction."""
    active = set(pd.to_datetime(reports_df["timestamp_utc"]).dt.normalize().unique())
    rng = np.random.default_rng(seed)
    total_days = (end - ARCHIVE_START).days
    quiet: list[pd.Timestamp] = []
    seen = set()
    while len(quiet) < n:
        offset = int(rng.integers(0, total_days))
        candidate = ARCHIVE_START + pd.Timedelta(days=offset)
        if candidate in seen or candidate in active:
            continue
        seen.add(candidate)
        quiet.append(candidate)
    return sorted(quiet)


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def process_one_date(
    date: pd.Timestamp,
    run_hour: str,
    grid: HrrrCoarseGrid,
    labels_df: pd.DataFrame,
    staging_dir: Path,
    hrrr_cache_root: Path,
) -> tuple[bool, str]:
    """Builds the 2 (bin) samples for one date's single run, saves them
    to staging_dir, and cleans up that date's raw HRRR cache. Returns
    (success, message)."""
    init_time = pd.Timestamp(f"{date.date()} {run_hour}")
    run_bins = [(init_time, bin_index) for bin_index in range(N_BINS)]

    try:
        ds = build_dataset(run_bins, grid, labels_df)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = staging_dir / f"{date.date()}.nc"
    ds.to_netcdf(out_path)

    cache_dir = hrrr_cache_root / "hrrr" / f"{date:%Y%m%d}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    return True, str(out_path)


def finalize(
    staging_dir: Path,
    completed_dates: list[pd.Timestamp],
    run_hour: str,
    out_train: Path,
    out_val: Path,
    buffer_hours: float = 24.0,
) -> dict:
    """Combines every completed date's staged sample into leakage-safe
    train/val files, via the same split.split_run_bins used everywhere
    else in this project."""
    run_bins = [(pd.Timestamp(f"{d.date()} {run_hour}"), bin_index) for d in completed_dates for bin_index in range(N_BINS)]
    train_run_bins, val_run_bins = split_run_bins(run_bins, buffer_hours=buffer_hours)

    def combine(run_bins_subset, out_path):
        if not run_bins_subset:
            return 0
        per_date = {}
        datasets = []
        for init_time, bin_index in run_bins_subset:
            date = init_time.normalize()
            if date not in per_date:
                per_date[date] = xr.open_dataset(staging_dir / f"{date.date()}.nc")
            datasets.append(per_date[date].isel(sample=[bin_index]))
        combined = xr.concat(datasets, dim="sample")
        combined = combined.assign_coords(
            init_time=("sample", [i for i, _ in run_bins_subset]),
            bin_index=("sample", [b for _, b in run_bins_subset]),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_netcdf(out_path)
        for ds in per_date.values():
            ds.close()
        return combined.sizes["sample"]

    n_train = combine(train_run_bins, out_train)
    n_val = combine(val_run_bins, out_val)
    return {"n_train": n_train, "n_val": n_val, "n_dropped": len(run_bins) - len(train_run_bins) - len(val_run_bins)}
