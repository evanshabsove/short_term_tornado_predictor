from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from tornado_predictor.grid import build_coarse_grid, xy_to_lonlat
from tornado_predictor.labels import (
    build_positive_label_index,
    build_positive_labels,
    buffer_radius_m,
    cells_for_report,
    label_for_sample,
    run_bins_for_report,
)

GRID = build_coarse_grid()


def _cell_center_lonlat(row, col):
    x_min, y_min, x_max, y_max = GRID.cell_bounds_xy(row, col)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    lon, lat = xy_to_lonlat(cx, cy, GRID.crs_proj4, GRID.geodetic_crs_proj4)
    return float(lon), float(lat)


def _line_wkt(points_lonlat):
    return LineString(points_lonlat).wkt


def test_buffer_radius_m_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        buffer_radius_m(0)
    with pytest.raises(ValueError):
        buffer_radius_m(-5)


def test_single_cell_containment():
    row, col = 40, 69
    lon, lat = _cell_center_lonlat(row, col)
    # two points ~1.5km apart (well inside a 39km cell), centered on the cell
    line = _line_wkt([(lon - 0.01, lat), (lon + 0.01, lat)])
    assert cells_for_report(line, width_yd=50, grid=GRID) == [(row, col)]


def test_multi_cell_track_along_one_row():
    row = 40
    c0, c1, c2 = 69, 70, 71
    lon0, lat = _cell_center_lonlat(row, c0)
    lon2, _ = _cell_center_lonlat(row, c2)
    line = _line_wkt([(lon0, lat), (lon2, lat)])
    result = sorted(cells_for_report(line, width_yd=50, grid=GRID))
    assert result == [(row, c0), (row, c1), (row, c2)]


def test_degenerate_start_equals_end_track():
    row, col = 40, 69
    lon, lat = _cell_center_lonlat(row, col)
    line = _line_wkt([(lon, lat), (lon, lat)])
    assert cells_for_report(line, width_yd=75, grid=GRID) == [(row, col)]


def test_real_degenerate_track_from_spc_csv_matches_start_point_cell():
    csv_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "spc_tornado_reports_2012_2022.csv"
    if not csv_path.exists():
        pytest.skip("SPC reports CSV not present")
    df = pd.read_csv(csv_path)
    point_tracks = df[(df["start_lat"] == df["end_lat"]) & (df["start_lon"] == df["end_lon"])]
    if point_tracks.empty:
        pytest.skip("no degenerate start==end tracks in this CSV")
    report = point_tracks.iloc[0]

    row_idx, col_idx = GRID.assign_cells([report.start_lon], [report.start_lat])
    assert (row_idx[0], col_idx[0]) in cells_for_report(report.path_wkt, report.width_yd, GRID)


def test_track_entirely_outside_domain_returns_empty():
    line = _line_wkt([(0.0, 0.0), (0.0, 0.0)])
    assert cells_for_report(line, width_yd=75, grid=GRID) == []


def test_run_bins_for_report_returns_exactly_eight():
    ts = pd.Timestamp("2021-05-03 14:45:00")
    pairs = run_bins_for_report(ts)
    assert len(pairs) == 8
    inits = [p[0] for p in pairs]
    assert len(set(inits)) == 8
    assert min(inits) == pd.Timestamp("2021-05-03 07:00:00")
    assert max(inits) == pd.Timestamp("2021-05-03 14:00:00")
    assert all(bin_index in (0, 1) for _, bin_index in pairs)


def test_build_positive_labels_end_to_end():
    row = 40
    c0, c2 = 69, 71
    lon0, lat = _cell_center_lonlat(row, c0)
    lon2, _ = _cell_center_lonlat(row, c2)
    line = _line_wkt([(lon0, lat), (lon2, lat)])

    reports_df = pd.DataFrame(
        [{"event_id": "e1", "ef_rating": 1.0, "timestamp_utc": pd.Timestamp("2021-05-03 14:45:00"), "path_wkt": line, "width_yd": 50}]
    )
    labels = build_positive_labels(reports_df, GRID)

    assert set(labels.columns) == {"event_id", "ef_rating", "init_time", "bin_index", "row", "col"}
    assert len(labels) == 3 * 8
    assert labels.drop_duplicates(subset=["init_time", "bin_index", "row", "col"]).shape[0] == 3 * 8

    index = build_positive_label_index(labels)
    one_positive = labels.iloc[0]
    assert label_for_sample(index, one_positive.init_time, one_positive.bin_index, one_positive.row, one_positive.col) == 1
    assert label_for_sample(index, one_positive.init_time, one_positive.bin_index, row, 99) == 0
