"""
Builds a genuine held-out train/test/val split from an existing scaled
build's staged runs, with NO new HRRR pulls -- reuses the already-staged
per-run files (default: data/interim/scaled_build_v2/) via
build_scaled.finalize_3way().

val's composition here is identical to the existing 2-way split's val
(same day-of-year mod-20 values), so every AUC-PR already reported
against val describes the same data -- the new test set is carved from
what the 2-way split calls train, days never used for any architecture-
selection decision. See split.py's split_run_bins_3way docstring for
the exact rule and buffer-safeguard reasoning.

Output filenames (training_dataset_{train,test,val}_3way.nc) are
deliberately distinct from the canonical training_dataset_{train,val}_full.nc
files every prior ticket's scripts default to -- this does NOT replace
them. Whether to promote the new, smaller (70%) train split as canonical
for future training is a separate decision, not made here.

Usage:
    python scripts/build_3way_split.py
    python scripts/build_3way_split.py --staging-dir data/interim/scaled_build_v3 --manifest data/interim/scaled_build_v3/manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.build_scaled import finalize_3way, load_manifest
from tornado_predictor.split import report_split_balance_3way, split_run_bins_3way
from tornado_predictor.time_bins import N_BINS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING_DIR = REPO_ROOT / "data" / "interim" / "scaled_build_v2"
DEFAULT_MANIFEST = DEFAULT_STAGING_DIR / "manifest.json"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2014_2025.csv"
DEFAULT_OUT_TRAIN = REPO_ROOT / "data" / "processed" / "training_dataset_train_3way.nc"
DEFAULT_OUT_TEST = REPO_ROOT / "data" / "processed" / "training_dataset_test_3way.nc"
DEFAULT_OUT_VAL = REPO_ROOT / "data" / "processed" / "training_dataset_val_3way.nc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--out-train", type=Path, default=DEFAULT_OUT_TRAIN)
    parser.add_argument("--out-test", type=Path, default=DEFAULT_OUT_TEST)
    parser.add_argument("--out-val", type=Path, default=DEFAULT_OUT_VAL)
    parser.add_argument("--buffer-hours", type=float, default=24.0)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    completed_runs = sorted(pd.Timestamp(k) for k, v in manifest.items() if v["status"] == "done")
    print(f"{len(completed_runs)} completed runs in {args.manifest}")

    result = finalize_3way(args.staging_dir, completed_runs, args.out_train, args.out_test, args.out_val, buffer_hours=args.buffer_hours)
    print(f"Finalized: {result}")
    print(f"Saved train -> {args.out_train}")
    print(f"Saved test  -> {args.out_test}")
    print(f"Saved val   -> {args.out_val}")

    labels_df = pd.read_csv(args.labels, parse_dates=["init_time"])
    run_bins = [(init_time, bin_index) for init_time in completed_runs for bin_index in range(N_BINS)]
    train_run_bins, test_run_bins, val_run_bins = split_run_bins_3way(run_bins, buffer_hours=args.buffer_hours)
    balance = report_split_balance_3way(labels_df, train_run_bins, test_run_bins, val_run_bins)
    print(f"\nSplit balance: {balance}")


if __name__ == "__main__":
    main()
