"""
Build a large-scale consolidated training dataset (v2): a
report-covering set of HRRR runs for every date with a confirmed SPC
tornado report (2014-2025, ~2,163 dates -> ~3,209 runs, see
src/tornado_predictor/build_scaled.py's docstring for why one run per
date isn't enough) plus one fixed-hour run per random quiet date, split
into leakage-safe train/val via split.py.

This is a LONG-RUNNING job (tens of thousands of HRRR pulls, likely a
day+ of wall time) -- see src/tornado_predictor/build_scaled.py for why
it's built to be interrupted and resumed safely: progress is
checkpointed per-run in a JSON manifest (--manifest), and re-running
this script skips every run already marked "done". Safe to stop
(Ctrl-C, laptop sleep, killed process) and restart at any time.

v2 (this version) replaced an earlier build that pulled exactly one
fixed 20:00 UTC run per active date -- auditing that dataset found it
missed a real tornado on 27.2% of active dates entirely, since most
active dates have multiple reports spread across many hours. v2 uses a
report-covering set of runs per date instead (see build_scaled.py's
docstring), validated as a real improvement (63-64% active maps vs.
46-48% before, at 2014-2025 real scale) and promoted to the canonical
`training_dataset_{train,val}_full.nc` filenames; the old build is
archived as `training_dataset_{train,val}_full_v1_superseded.nc`, kept
for reference rather than deleted. v2's own staging dir/manifest still
use a "_v2" suffix internally (unit of work changed from "date" to
"run", so its manifest keys and staged filenames are a different
format than the old build's -- not worth migrating).

Usage:
    python scripts/build_scaled_dataset.py                    # start or resume the full build
    python scripts/build_scaled_dataset.py --finalize-only     # just combine already-completed runs
    python scripts/build_scaled_dataset.py --retry-failed      # also retry runs previously marked failed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.build_scaled import (
    finalize,
    hrrr_cache_root as get_hrrr_cache_root,
    load_manifest,
    process_one_run,
    save_manifest,
    select_active_runs,
    select_quiet_dates,
)
from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2014_2025.csv"
DEFAULT_REPORTS_PATH = REPO_ROOT / "data" / "processed" / "spc_tornado_reports_2014_2025.csv"
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
DEFAULT_MANIFEST_PATH = DEFAULT_STAGING_DIR / "manifest.json"
DEFAULT_OUT_TRAIN = REPO_ROOT / "data" / "processed" / "training_dataset_train_full.nc"
DEFAULT_OUT_VAL = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"

N_QUIET_DATES = 300
QUIET_RUN_HOUR = "20:00"  # arbitrary but fixed -- no report to anchor to, so any hour is equally representative
SPC_LABEL_CEILING = pd.Timestamp("2025-09-09")  # verified live; see CLAUDE.md "Time binning"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_PATH)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--out-train", type=Path, default=DEFAULT_OUT_TRAIN)
    parser.add_argument("--out-val", type=Path, default=DEFAULT_OUT_VAL)
    parser.add_argument("--n-quiet", type=int, default=N_QUIET_DATES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true", help="skip pulling, just combine completed runs")
    parser.add_argument("--retry-failed", action="store_true", help="also retry runs previously marked failed")
    args = parser.parse_args()

    reports_df = pd.read_csv(args.reports, parse_dates=["timestamp_utc"])
    active_runs = select_active_runs(reports_df)
    quiet_dates = select_quiet_dates(reports_df, n=args.n_quiet, end=SPC_LABEL_CEILING, seed=args.seed)
    quiet_runs = [pd.Timestamp(f"{d.date()} {QUIET_RUN_HOUR}") for d in quiet_dates]
    all_runs = sorted(active_runs + quiet_runs)
    print(f"{len(active_runs)} active runs (report-covering) + {len(quiet_runs)} quiet runs = {len(all_runs)} total")

    manifest = load_manifest(args.manifest)

    def key(init_time: pd.Timestamp) -> str:
        return init_time.isoformat()

    if not args.finalize_only:
        grid = HrrrCoarseGrid.from_netcdf(args.grid)
        labels_df = pd.read_csv(args.labels, parse_dates=["init_time"])
        cache_root = get_hrrr_cache_root()

        todo = [
            r for r in all_runs
            if key(r) not in manifest
            or (manifest[key(r)]["status"] == "failed" and args.retry_failed)
        ]
        print(f"{len(todo)} runs to process ({len(all_runs) - len(todo)} already done)")

        for i, init_time in enumerate(todo):
            success, message = process_one_run(init_time, grid, labels_df, args.staging_dir, cache_root)
            manifest[key(init_time)] = {"status": "done" if success else "failed", "message": message}
            save_manifest(args.manifest, manifest)
            status = "OK" if success else "FAILED"
            print(f"[{i + 1}/{len(todo)}] {init_time}: {status}" + ("" if success else f" -- {message}"))

    n_done = sum(1 for v in manifest.values() if v["status"] == "done")
    n_failed = sum(1 for v in manifest.values() if v["status"] == "failed")
    print(f"\nManifest: {n_done} done, {n_failed} failed")

    completed_runs = [r for r in all_runs if manifest.get(key(r), {}).get("status") == "done"]
    result = finalize(args.staging_dir, completed_runs, args.out_train, args.out_val)
    print(f"Finalized: {result}")
    print(f"Saved train -> {args.out_train}")
    print(f"Saved val   -> {args.out_val}")


if __name__ == "__main__":
    main()
