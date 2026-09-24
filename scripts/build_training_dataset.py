"""
Combine feature extraction and label assignment across multiple HRRR
runs into train/val consolidated training datasets (see
src/tornado_predictor/dataset.py). Each run contributes 2 samples (its
2 time bins), each represented by a single fxx snapshot per bin (see
dataset.py's module docstring for why).

The full candidate run list is split into train/val via
src/tornado_predictor/split.py -- grouped by calendar date (every
sample from the same date goes to the same split), assigned
deterministically by day-of-year modulo (matching TorNet's own
methodology), with a buffer safeguard dropping training samples too
close in time to any validation sample. See split.py's module
docstring for the full rationale.

Usage:
    python scripts/build_training_dataset.py --init-times "2021-12-10 21:00" "2021-12-10 22:00" "2021-12-10 23:00"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.dataset import build_dataset, to_training_arrays
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.split import report_split_balance, split_run_bins
from tornado_predictor.time_bins import N_BINS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2014_2025.csv"
DEFAULT_OUT_TRAIN_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_train.nc"
DEFAULT_OUT_VAL_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val.nc"


def build_and_save(run_bins, grid, labels_df, out_path, split_name: str) -> None:
    if not run_bins:
        print(f"{split_name}: 0 samples -- skipping (nothing to build)")
        return
    ds = build_dataset(run_bins, grid, labels_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)

    n_positive_samples = int((ds["label"].sum(dim=("row", "col")) > 0).sum())
    print(f"{split_name}: {ds.sizes['sample']} samples, {n_positive_samples} with >=1 positive cell -> {out_path}")

    X, y, feature_names = to_training_arrays(ds)
    print(f"{split_name}: to_training_arrays X.shape={X.shape} y.shape={y.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-times", nargs="+", required=True, help='e.g. "2021-12-10 21:00" "2021-12-10 22:00"')
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--out-train", type=Path, default=DEFAULT_OUT_TRAIN_PATH)
    parser.add_argument("--out-val", type=Path, default=DEFAULT_OUT_VAL_PATH)
    parser.add_argument("--buffer-hours", type=float, default=24.0)
    args = parser.parse_args()

    grid = HrrrCoarseGrid.from_netcdf(args.grid)
    labels_df = pd.read_csv(args.labels, parse_dates=["init_time"])

    run_bins = [(init_time, bin_index) for init_time in args.init_times for bin_index in range(N_BINS)]
    print(f"{len(run_bins)} candidate samples from {len(args.init_times)} runs")

    train_run_bins, val_run_bins = split_run_bins(run_bins, buffer_hours=args.buffer_hours)
    n_dropped = len(run_bins) - len(train_run_bins) - len(val_run_bins)
    print(f"Split: {len(train_run_bins)} train, {len(val_run_bins)} val, {n_dropped} dropped by the buffer safeguard")

    balance = report_split_balance(labels_df, train_run_bins, val_run_bins)
    for split_name, stats in balance.items():
        print(f"  {split_name}: {stats}")

    build_and_save(train_run_bins, grid, labels_df, args.out_train, "train")
    build_and_save(val_run_bins, grid, labels_df, args.out_val, "val")


if __name__ == "__main__":
    main()
