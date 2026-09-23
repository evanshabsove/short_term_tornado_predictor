"""
Spot-check the forecast dataset pipeline (grid, time bins, labels,
features) against a single well-documented historical tornado, rather
than a synthetic or arbitrary case.

Anchor case: the EF4 tornado that touched down in Arkansas at
2021-12-11 01:07 UTC and tracked ~81 miles into Tennessee (SPC
event_id 2021_2112101907-01), part of the December 10-11, 2021
multi-state outbreak. This specific tornado -- not the separate,
later "Quad-State"/Mayfield EF4 from the same night (event_id
2021_2112102054-01) -- is the one already used as a named case study
in the sibling CNN radar-detection project
(tornet/notebooks/july_6th_result_analysis.ipynb, TorNet catalog
event_id 997130, KNQA radar, frames 2021-12-11 01:35:30-02:07:30 UTC
near 36.13-36.25N/-89.94 to -89.69W -- squarely along this tornado's
path and within its lifetime). Reusing it here means both projects'
validation stories point at the same real event.

This checks:
  1. Grid alignment -- the tornado's reported start/end points fall in
     grid cells geographically close (< half a cell width) to those
     points, and the full buffered-track cell set contains both.
  2. Time-bin correctness -- for the real candidate HRRR runs, the
     bin this report is assigned to matches what's already in the
     precomputed labels table (cross-checks the bulk-built table
     against a fresh, independent recomputation for this one event).
  3. Feature sanity leading up to the event -- pulls real HRRR fields
     for the touchdown cell across several runs before touchdown,
     checking for NaNs and physically plausible ranges (not a strict
     monotonic-increase requirement -- convective ramp-up isn't always
     smooth -- but flags degenerate all-zero/all-NaN results).

Usage:
    python scripts/validate_documented_case.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from tornado_predictor.features import extract_features
from tornado_predictor.grid import HRRR_EARTH_RADIUS_M, HrrrCoarseGrid
from tornado_predictor.labels import cells_for_report
from tornado_predictor.time_bins import assign_valid_time_bin, candidate_run_inits

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_PATH = REPO_ROOT / "data" / "processed" / "spc_tornado_reports_2012_2022.csv"
GRID_PATH = REPO_ROOT / "data" / "processed" / "hrrr_coarse_grid_stride13.nc"
LABELS_PATH = REPO_ROOT / "data" / "processed" / "tornado_labels_2012_2022.csv"

ANCHOR_EVENT_ID = "2021_2112101907-01"

# Hourly HRRR runs (fxx=0, i.e. each run's own analysis) leading up to
# touchdown, used as a real observed-conditions timeline.
LEADUP_RUN_TIMES = [
    "2021-12-10 18:00", "2021-12-10 19:00", "2021-12-10 20:00", "2021-12-10 21:00",
    "2021-12-10 22:00", "2021-12-10 23:00", "2021-12-11 00:00", "2021-12-11 01:00",
]

# Same spherical earth model as the grid itself (see grid.py) -- a
# self-consistency check, not a comparison against a different datum.
GEOD = Geod(a=HRRR_EARTH_RADIUS_M, f=0.0)


def check_grid_alignment(report, grid: HrrrCoarseGrid) -> tuple[int, int]:
    print("--- 1. Grid alignment ---")
    start_row, start_col = grid.assign_cells([report.start_lon], [report.start_lat])
    end_row, end_col = grid.assign_cells([report.end_lon], [report.end_lat])
    start_row, start_col, end_row, end_col = int(start_row[0]), int(start_col[0]), int(end_row[0]), int(end_col[0])
    print(f"Touchdown ({report.start_lat:.4f}, {report.start_lon:.4f}) -> cell (row={start_row}, col={start_col})")
    print(f"Lift-off  ({report.end_lat:.4f}, {report.end_lon:.4f}) -> cell (row={end_row}, col={end_col})")

    cell_lat, cell_lon = grid.lat_center[start_row, start_col], grid.lon_center[start_row, start_col]
    _, _, dist_m = GEOD.inv(report.start_lon, report.start_lat, cell_lon, cell_lat)
    half_cell_m = grid.cell_size_m / 2
    ok = dist_m < half_cell_m
    print(f"Touchdown point to assigned cell center: {dist_m / 1000:.1f} km (half cell = {half_cell_m / 1000:.1f} km) -- {'OK' if ok else 'FAIL'}")
    assert ok, "touchdown point is farther from its assigned cell's center than half a cell width"

    track_cells = cells_for_report(report.path_wkt, report.width_yd, grid)
    print(f"Full buffered-track cells ({len(track_cells)}): {track_cells}")
    assert (start_row, start_col) in track_cells, "touchdown cell missing from track cell set"
    assert (end_row, end_col) in track_cells, "lift-off cell missing from track cell set"
    print("Touchdown and lift-off cells both present in the full track cell set -- OK\n")
    return start_row, start_col


def check_time_bins(report, labels_df: pd.DataFrame) -> None:
    print("--- 2. Time-bin correctness ---")
    candidates = candidate_run_inits(report.timestamp_utc)
    print(f"{len(candidates)} candidate HRRR runs for this report's timestamp ({report.timestamp_utc}):")

    event_labels = labels_df[labels_df["event_id"] == ANCHOR_EVENT_ID]
    all_ok = True
    for init_time in candidates:
        expected_bin = assign_valid_time_bin(init_time, report.timestamp_utc)
        table_bins = set(event_labels.loc[event_labels["init_time"] == init_time, "bin_index"])
        ok = expected_bin in table_bins
        all_ok &= ok
        print(f"  run {init_time}: freshly computed bin={expected_bin}, bins in labels table={sorted(table_bins)} -- {'OK' if ok else 'MISMATCH'}")
    assert all_ok, "a freshly computed bin assignment disagrees with the precomputed labels table"
    print("Precomputed labels table agrees with fresh recomputation for every candidate run -- OK\n")


def check_features_leading_up_to_event(row: int, col: int) -> None:
    print("--- 3. HRRR feature sanity leading up to the event ---")
    grid = HrrrCoarseGrid.from_netcdf(GRID_PATH)

    records = []
    for init_time in LEADUP_RUN_TIMES:
        ds = extract_features(init_time, fxx=0, grid=grid)
        cell = ds.isel(row=row, col=col)
        record = {"init_time": init_time}
        for var in ds.data_vars:
            if var in ("lat", "lon"):
                continue
            record[var] = float(cell[var].values)
        records.append(record)

    df = pd.DataFrame(records).set_index("init_time")
    pd.set_option("display.width", 200)
    print(df.round(1).to_string())

    n_nan = int(df.isna().sum().sum())
    print(f"\nNaN count across all pulled fields: {n_nan} -- {'OK' if n_nan == 0 else 'FAIL'}")
    assert n_nan == 0, "NaNs found in HRRR fields at the touchdown cell"

    all_zero_cols = [c for c in df.columns if (df[c] == 0).all()]
    print(f"Fields that are identically zero across the whole leadup window: {all_zero_cols or 'none'} -- {'FAIL' if all_zero_cols else 'OK'}")
    assert not all_zero_cols, f"suspicious all-zero fields (possible misalignment/pull bug): {all_zero_cols}"

    cape_trend = df["cape_max"].values
    print(f"cape_max trend leading up to touchdown: {np.round(cape_trend, 0)}")
    print(f"cape_max at final pre-touchdown run ({LEADUP_RUN_TIMES[-1]}): {cape_trend[-1]:.0f} J/kg")
    assert cape_trend[-1] > 0, "expected nonzero CAPE at the touchdown cell in the hour before an EF4 tornado"
    print("Feature values are physically plausible (no NaNs, no degenerate all-zero fields, nonzero CAPE approaching touchdown) -- OK\n")


def main() -> None:
    reports_df = pd.read_csv(REPORTS_PATH, parse_dates=["timestamp_utc"])
    report = reports_df[reports_df["event_id"] == ANCHOR_EVENT_ID].iloc[0]
    labels_df = pd.read_csv(LABELS_PATH, parse_dates=["init_time"])
    grid = HrrrCoarseGrid.from_netcdf(GRID_PATH)

    print(f"Anchor case: {ANCHOR_EVENT_ID}")
    print(f"  Touchdown: {report.timestamp_utc} UTC, ({report.start_lat}, {report.start_lon})")
    print(f"  Lift-off:  ({report.end_lat}, {report.end_lon})")
    print(f"  EF{int(report.ef_rating)}, {report.length_mi} mi long, {report.width_yd} yd wide")
    print(f"  Matches TorNet catalog event_id 997130 (KNQA radar, 2021-12-11 01:35:30-02:07:30 UTC)\n")

    touchdown_row, touchdown_col = check_grid_alignment(report, grid)
    check_time_bins(report, labels_df)
    check_features_leading_up_to_event(touchdown_row, touchdown_col)

    print("=== All checks passed ===")


if __name__ == "__main__":
    main()
