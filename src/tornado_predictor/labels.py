"""
Assigns 1/0 tornado labels to (grid cell, run init_time, bin_index)
samples from SPC tornado reports.

Spatial matching uses the report's full track (path_wkt buffered by
half its damage-path width) so a long track is credited to every cell
it crosses, not just its touchdown cell -- per the guidance left when
path_wkt/width_yd were added to the SPC parser. Temporal matching
necessarily collapses to the report's single timestamp_utc -- SPC's
database has no track-relative timing or end-time field, so there is no
more granular alternative; this is a documented simplification, not an
oversight.

Buffering and cell-intersection both happen in projected LCC meters (via
grid.lonlat_to_xy), using each cell's exact square bounding box
(grid.cell_bounds_xy) rather than the curved-lat/lon approximation of
grid.cell_polygon.

Only positive labels are materialized here -- the full 0-label space
(every grid cell x bin x run not touched by a report) is intractable to
enumerate without a defined training run-list, and is out of scope.
"""

from __future__ import annotations

import math

import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, box

from tornado_predictor.grid import HrrrCoarseGrid, lonlat_to_xy
from tornado_predictor.time_bins import assign_valid_time_bin, candidate_run_inits

YARDS_TO_METERS = 0.9144


def buffer_radius_m(width_yd: float) -> float:
    """Converts a tornado's full damage-path width (yards) to a buffer
    radius in meters. Raises if width_yd <= 0: a zero-radius shapely
    buffer of a line is an EMPTY polygon, which would silently drop the
    report from the labels table with no error. The 2012-2022 SPC data
    has no zero-width reports, but this must not be assumed for other
    date ranges."""
    if width_yd <= 0:
        raise ValueError(
            f"width_yd must be > 0 to buffer a track (got {width_yd}); "
            "the 2012-2022 SPC data has no zero-width reports, but this "
            "must not be silently clamped for other date ranges"
        )
    return width_yd * YARDS_TO_METERS / 2.0


def cells_for_report(path_wkt: str, width_yd: float, grid: HrrrCoarseGrid) -> list[tuple[int, int]]:
    """The (row, col) grid cells whose exact projected bounding box
    intersects the report's track, buffered by half its width. Returns
    an empty list (never raises) if the track is entirely outside the
    grid domain."""
    line = wkt.loads(path_wkt)
    lons, lats = zip(*line.coords)
    xs, ys = lonlat_to_xy(lons, lats, grid.crs_proj4, grid.geodetic_crs_proj4)
    corridor = LineString(zip(xs, ys)).buffer(buffer_radius_m(width_yd))
    minx, miny, maxx, maxy = corridor.bounds

    # floor on both ends is conservative -- it never excludes a truly
    # intersecting cell; worst case includes one extra boundary
    # candidate that .intersects() filters out for free.
    col_lo = max(math.floor((minx - grid.x_min_m) / grid.cell_size_m), 0)
    col_hi = min(math.floor((maxx - grid.x_min_m) / grid.cell_size_m), grid.n_cols - 1)
    row_lo = max(math.floor((miny - grid.y_min_m) / grid.cell_size_m), 0)
    row_hi = min(math.floor((maxy - grid.y_min_m) / grid.cell_size_m), grid.n_rows - 1)

    if col_lo > col_hi or row_lo > row_hi:
        return []

    matched = []
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            if box(*grid.cell_bounds_xy(r, c)).intersects(corridor):
                matched.append((r, c))
    return matched


def run_bins_for_report(timestamp) -> list[tuple[pd.Timestamp, int]]:
    """The (init_time, bin_index) pairs this report is a positive label
    for -- always exactly 8, one per candidate hourly HRRR run."""
    result = []
    for init in candidate_run_inits(timestamp):
        bin_index = assign_valid_time_bin(init, timestamp)
        assert bin_index is not None, "candidate_run_inits guarantees all 8 inits are valid for this timestamp"
        result.append((init, bin_index))
    return result


def build_positive_labels(reports_df: pd.DataFrame, grid: HrrrCoarseGrid) -> pd.DataFrame:
    """Builds the sparse positive-label table: one row per (event,
    affected run, affected bin, affected cell). reports_df.timestamp_utc
    must already be parsed to pandas.Timestamp (e.g. via
    pd.read_csv(..., parse_dates=["timestamp_utc"]))."""
    records = []
    for report in reports_df.itertuples(index=False):
        cells = cells_for_report(report.path_wkt, report.width_yd, grid)
        run_bins = run_bins_for_report(report.timestamp_utc)
        for row, col in cells:
            for init_time, bin_index in run_bins:
                records.append(
                    {
                        "event_id": report.event_id,
                        "ef_rating": report.ef_rating,
                        "init_time": init_time,
                        "bin_index": bin_index,
                        "row": row,
                        "col": col,
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=["event_id", "ef_rating", "init_time", "bin_index", "row", "col"]
    )


def build_positive_label_index(labels_df: pd.DataFrame) -> frozenset:
    """A one-time index for repeated label_for_sample lookups -- build
    once per training run, not per call."""
    return frozenset(zip(labels_df["init_time"], labels_df["bin_index"], labels_df["row"], labels_df["col"]))


def label_for_sample(label_index: frozenset, init_time, bin_index: int, row: int, col: int) -> int:
    """1 if (init_time, bin_index, row, col) is a positive label, else 0."""
    return int((pd.Timestamp(init_time), bin_index, row, col) in label_index)
