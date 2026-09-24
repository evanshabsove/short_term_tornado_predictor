"""
Build the sparse positive tornado-label table from SPC reports and the
coarse grid (see src/tornado_predictor/labels.py). One row per
(event_id, affected run init_time, affected bin_index, affected cell) --
every row is an implicit label of 1; anything absent is an implicit 0.

Usage:
    python scripts/build_labels.py
    python scripts/build_labels.py --reports data/processed/spc_tornado_reports_2014_2025.csv --grid data/processed/hrrr_coarse_grid_stride13.nc --out data/processed/tornado_labels_2014_2025.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.labels import build_positive_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_PATH = REPO_ROOT / "data" / "processed" / "spc_tornado_reports_2014_2025.csv"
DEFAULT_GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
DEFAULT_OUT_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2014_2025.csv"


def print_sanity_summary(reports_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    print(f"Total label rows: {len(labels_df)}")

    n_unique = labels_df.drop_duplicates(subset=["init_time", "bin_index", "row", "col"]).shape[0]
    print(f"Unique (init_time, bin_index, row, col) positives: {n_unique}")

    cells_per_event = labels_df.groupby("event_id").size() / 8
    non_integer = cells_per_event[cells_per_event != cells_per_event.round()]
    if len(non_integer):
        print(f"WARNING: {len(non_integer)} events have a non-integer cells-per-event ratio -- bug in run_bins_for_report")

    zero_cell_events = set(reports_df["event_id"]) - set(labels_df["event_id"])
    print(f"Reports with zero affected cells: {len(zero_cell_events)} / {len(reports_df)}")

    many_cell_events = cells_per_event[cells_per_event > 10]
    print(f"Reports with >10 affected cells: {len(many_cell_events)}")
    if len(many_cell_events):
        print(f"  examples: {list(many_cell_events.index[:5])}")

    print(f"Cells per event: min={cells_per_event.min():.0f} mean={cells_per_event.mean():.2f} max={cells_per_event.max():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_PATH)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    grid = HrrrCoarseGrid.from_netcdf(args.grid)
    reports_df = pd.read_csv(args.reports, parse_dates=["timestamp_utc"])

    labels_df = build_positive_labels(reports_df, grid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(args.out, index=False)

    print_sanity_summary(reports_df, labels_df)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
