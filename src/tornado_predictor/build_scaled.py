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

**Report-covering run selection (v2, replaces one-fixed-run-per-date).**
The original build pulled exactly one run per active date, fixed at
20:00 UTC. Auditing the completed dataset found this missed a real
tornado on 27.2% of active dates entirely (541/1989) -- the report's
timestamp simply fell outside that one run's [init, init+8h) window --
because most active dates have multiple reports spread across many
hours (median spread ~10.8h on multi-report days; only 46% of those
fit inside a single 8h window). `select_active_runs` fixes this with a
greedy interval-covering algorithm per date: anchor a run at the
earliest not-yet-covered report's hour, skip every report within the
next 8h (already covered by that run), repeat for whatever's left.
Checked against the real report data, this needs only 3,209 runs for
2,163 active dates (1.48x, not 8x) -- 1,305 dates need just 1 run, 670
need 2, 188 need 3. The unit of work is now a run (identified by its
full init_time, not just a date) since some dates need more than one;
quiet dates are unaffected (still exactly 1 fixed-hour run each, since
there's no report to anchor to).
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
from tornado_predictor.split import split_run_bins, split_run_bins_3way
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


def greedy_cover_run_inits(report_times: list[pd.Timestamp], window_hours: float = 8.0) -> list[pd.Timestamp]:
    """Minimum set of hourly-floored run init_times such that every
    report_time falls within some returned init's [init, init+window_hours)
    window. Greedy interval covering: anchor a run at the earliest
    not-yet-covered report's hour, skip everything that falls within the
    next window_hours (already covered), repeat. Optimal for this
    "cover a sorted 1-D line with fixed-length intervals starting at any
    covered point" case -- each greedy choice covers the maximum
    possible span before the next mandatory anchor."""
    times = sorted(report_times)
    runs: list[pd.Timestamp] = []
    i = 0
    while i < len(times):
        anchor = times[i].floor("h")
        runs.append(anchor)
        cutoff = anchor + pd.Timedelta(hours=window_hours)
        while i < len(times) and times[i] < cutoff:
            i += 1
    return runs


def select_active_runs(reports_df: pd.DataFrame) -> list[pd.Timestamp]:
    """For every UTC calendar date (>= ARCHIVE_START) with >=1 SPC
    report, the minimum set of HRRR run init_times needed so every
    report that date is captured by some run's 0-8h window (see
    greedy_cover_run_inits). Replaces the old "one fixed 20:00 UTC run
    per active date" scheme, which silently produced all-negative
    samples for 27.2% of active dates -- see this module's docstring."""
    ts = pd.to_datetime(reports_df["timestamp_utc"])
    df = reports_df.assign(_date=ts.dt.normalize(), _ts=ts)
    df = df[df["_date"] >= ARCHIVE_START]

    all_runs: list[pd.Timestamp] = []
    for _date, grp in df.groupby("_date"):
        all_runs.extend(greedy_cover_run_inits(list(grp["_ts"])))
    return sorted(all_runs)


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


def run_staging_path(staging_dir: Path, init_time: pd.Timestamp) -> Path:
    """Staging filename for one run -- keyed by full init_time (not just
    date), since v2 can pull more than one run per date."""
    return staging_dir / f"{init_time:%Y-%m-%dT%H}.nc"


def process_one_run(
    init_time: pd.Timestamp,
    grid: HrrrCoarseGrid,
    labels_df: pd.DataFrame,
    staging_dir: Path,
    hrrr_cache_root: Path,
) -> tuple[bool, str]:
    """Builds the 2 (bin) samples for one HRRR run, saves them to
    staging_dir, and cleans up that run's raw HRRR cache. Returns
    (success, message)."""
    run_bins = [(init_time, bin_index) for bin_index in range(N_BINS)]

    try:
        ds = build_dataset(run_bins, grid, labels_df)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    staging_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_staging_path(staging_dir, init_time)
    ds.to_netcdf(out_path)

    cache_dir = hrrr_cache_root / "hrrr" / f"{init_time:%Y%m%d}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    return True, str(out_path)


def finalize(
    staging_dir: Path,
    completed_runs: list[pd.Timestamp],
    out_train: Path,
    out_val: Path,
    buffer_hours: float = 24.0,
) -> dict:
    """Combines every completed run's staged sample into leakage-safe
    train/val files, via the same split.split_run_bins used everywhere
    else in this project. Multiple runs on the same calendar date land
    in the same split automatically -- split_run_bins groups by the
    UTC calendar date of init_time, independent of hour."""
    run_bins = [(init_time, bin_index) for init_time in completed_runs for bin_index in range(N_BINS)]
    train_run_bins, val_run_bins = split_run_bins(run_bins, buffer_hours=buffer_hours)

    def combine(run_bins_subset, out_path):
        if not run_bins_subset:
            return 0
        per_run = {}
        datasets = []
        for init_time, bin_index in run_bins_subset:
            if init_time not in per_run:
                per_run[init_time] = xr.open_dataset(run_staging_path(staging_dir, init_time))
            datasets.append(per_run[init_time].isel(sample=[bin_index]))
        combined = xr.concat(datasets, dim="sample")
        combined = combined.assign_coords(
            init_time=("sample", [i for i, _ in run_bins_subset]),
            bin_index=("sample", [b for _, b in run_bins_subset]),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_netcdf(out_path)
        for ds in per_run.values():
            ds.close()
        return combined.sizes["sample"]

    n_train = combine(train_run_bins, out_train)
    n_val = combine(val_run_bins, out_val)
    return {"n_train": n_train, "n_val": n_val, "n_dropped": len(run_bins) - len(train_run_bins) - len(val_run_bins)}


def finalize_3way(
    staging_dir: Path,
    completed_runs: list[pd.Timestamp],
    out_train: Path,
    out_test: Path,
    out_val: Path,
    buffer_hours: float = 24.0,
) -> dict:
    """Same combine-from-staging pattern as finalize(), but produces a
    genuine held-out train/test/val split via split.split_run_bins_3way
    instead of the 2-way split.split_run_bins -- added alongside
    finalize() (not replacing it) so every existing caller keeps
    working unchanged. staging_dir is not hardcoded to any particular
    build, so this works against data/interim/scaled_build_v2/ today
    and scaled_build_v3/ once available, with no code changes."""
    run_bins = [(init_time, bin_index) for init_time in completed_runs for bin_index in range(N_BINS)]
    train_run_bins, test_run_bins, val_run_bins = split_run_bins_3way(run_bins, buffer_hours=buffer_hours)

    def combine(run_bins_subset, out_path):
        if not run_bins_subset:
            return 0
        per_run = {}
        datasets = []
        for init_time, bin_index in run_bins_subset:
            if init_time not in per_run:
                per_run[init_time] = xr.open_dataset(run_staging_path(staging_dir, init_time))
            datasets.append(per_run[init_time].isel(sample=[bin_index]))
        combined = xr.concat(datasets, dim="sample")
        combined = combined.assign_coords(
            init_time=("sample", [i for i, _ in run_bins_subset]),
            bin_index=("sample", [b for _, b in run_bins_subset]),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_netcdf(out_path)
        for ds in per_run.values():
            ds.close()
        return combined.sizes["sample"]

    n_train = combine(train_run_bins, out_train)
    n_test = combine(test_run_bins, out_test)
    n_val = combine(val_run_bins, out_val)
    return {
        "n_train": n_train,
        "n_test": n_test,
        "n_val": n_val,
        "n_dropped": len(run_bins) - len(train_run_bins) - len(test_run_bins) - len(val_run_bins),
    }
