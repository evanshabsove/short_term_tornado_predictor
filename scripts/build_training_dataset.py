"""
Combine feature extraction and label assignment across multiple HRRR
runs into one consolidated training dataset (see
src/tornado_predictor/dataset.py). Each run contributes 2 samples (its
2 time bins), each represented by a single fxx snapshot per bin (see
dataset.py's module docstring for why).

Usage:
    python scripts/build_training_dataset.py --init-times "2021-12-10 21:00" "2021-12-10 22:00" "2021-12-10 23:00"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.dataset import build_dataset, to_training_arrays
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.time_bins import N_BINS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2012_2022.csv"
DEFAULT_OUT_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_pilot.nc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-times", nargs="+", required=True, help='e.g. "2021-12-10 21:00" "2021-12-10 22:00"')
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    grid = HrrrCoarseGrid.from_netcdf(args.grid)
    labels_df = pd.read_csv(args.labels, parse_dates=["init_time"])

    run_bins = [(init_time, bin_index) for init_time in args.init_times for bin_index in range(N_BINS)]
    print(f"Building {len(run_bins)} samples from {len(args.init_times)} runs...")

    ds = build_dataset(run_bins, grid, labels_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)

    n_positive_cells = int((ds["label"] == 1).sum())
    n_positive_samples = int((ds["label"].sum(dim=("row", "col")) > 0).sum())
    print(f"Dataset shape: {dict(ds.sizes)}")
    print(f"Positive (sample, cell) labels: {n_positive_cells}")
    print(f"Samples with >=1 positive cell: {n_positive_samples} / {ds.sizes['sample']}")

    X, y, feature_names = to_training_arrays(ds)
    print(f"\nto_training_arrays: X.shape={X.shape} y.shape={y.shape}")
    print(f"Feature order: {feature_names}")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
