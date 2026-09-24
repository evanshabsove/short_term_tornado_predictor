"""
Build a large-scale consolidated training dataset: every date with a
confirmed SPC tornado report (2014-2025, ~2,163 dates) plus a random
sample of quiet dates, one HRRR run per date, split into leakage-safe
train/val via split.py.

This is a LONG-RUNNING job (tens of thousands of HRRR pulls, likely
a day+ of wall time) -- see src/tornado_predictor/build_scaled.py for
why it's built to be interrupted and resumed safely: progress is
checkpointed per-date in a JSON manifest (--manifest), and re-running
this script skips every date already marked "done". Safe to stop
(Ctrl-C, laptop sleep, killed process) and restart at any time.

Usage:
    python scripts/build_scaled_dataset.py                    # start or resume the full build
    python scripts/build_scaled_dataset.py --finalize-only     # just combine already-completed dates
    python scripts/build_scaled_dataset.py --retry-failed      # also retry dates previously marked failed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.build_scaled import (
    finalize,
    hrrr_cache_root as get_hrrr_cache_root,
    load_manifest,
    process_one_date,
    save_manifest,
    select_active_dates,
    select_quiet_dates,
)
from tornado_predictor.grid import HrrrCoarseGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2014_2025.csv"
DEFAULT_REPORTS_PATH = REPO_ROOT / "data" / "processed" / "spc_tornado_reports_2014_2025.csv"
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build"
DEFAULT_MANIFEST_PATH = DEFAULT_STAGING_DIR / "manifest.json"
DEFAULT_OUT_TRAIN = REPO_ROOT / "data" / "processed" / "training_dataset_train_full.nc"
DEFAULT_OUT_VAL = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"

N_QUIET_DATES = 300
RUN_HOUR = "20:00"
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
    parser.add_argument("--finalize-only", action="store_true", help="skip pulling, just combine completed dates")
    parser.add_argument("--retry-failed", action="store_true", help="also retry dates previously marked failed")
    args = parser.parse_args()

    reports_df = pd.read_csv(args.reports, parse_dates=["timestamp_utc"])
    active_dates = select_active_dates(reports_df)
    quiet_dates = select_quiet_dates(reports_df, n=args.n_quiet, end=SPC_LABEL_CEILING, seed=args.seed)
    all_dates = sorted(active_dates + quiet_dates)
    print(f"{len(active_dates)} active dates + {len(quiet_dates)} quiet dates = {len(all_dates)} total")

    manifest = load_manifest(args.manifest)

    if not args.finalize_only:
        grid = HrrrCoarseGrid.from_netcdf(args.grid)
        labels_df = pd.read_csv(args.labels, parse_dates=["init_time"])
        cache_root = get_hrrr_cache_root()

        todo = [
            d for d in all_dates
            if str(d.date()) not in manifest
            or (manifest[str(d.date())]["status"] == "failed" and args.retry_failed)
        ]
        print(f"{len(todo)} dates to process ({len(all_dates) - len(todo)} already done)")

        for i, date in enumerate(todo):
            success, message = process_one_date(date, RUN_HOUR, grid, labels_df, args.staging_dir, cache_root)
            manifest[str(date.date())] = {"status": "done" if success else "failed", "message": message}
            save_manifest(args.manifest, manifest)
            status = "OK" if success else "FAILED"
            print(f"[{i + 1}/{len(todo)}] {date.date()}: {status}" + ("" if success else f" -- {message}"))

    n_done = sum(1 for v in manifest.values() if v["status"] == "done")
    n_failed = sum(1 for v in manifest.values() if v["status"] == "failed")
    print(f"\nManifest: {n_done} done, {n_failed} failed")

    completed_dates = [d for d in all_dates if manifest.get(str(d.date()), {}).get("status") == "done"]
    result = finalize(args.staging_dir, completed_dates, RUN_HOUR, args.out_train, args.out_val)
    print(f"Finalized: {result}")
    print(f"Saved train -> {args.out_train}")
    print(f"Saved val   -> {args.out_val}")


if __name__ == "__main__":
    main()
