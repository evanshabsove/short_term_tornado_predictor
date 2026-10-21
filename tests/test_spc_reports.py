import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tornado_predictor.replay_cases import REPLAY_CASES, case_by_key, case_for
from tornado_predictor.spc_reports import DEFAULT_REPORTS_PATH, ReportsCatalog

from conftest import REPORT_ROWS


@pytest.fixture
def catalog(reports_csv):
    return ReportsCatalog.load(reports_csv)


def test_window_is_half_open_on_the_touchdown_time(catalog):
    ids = lambda a, b: [f["properties"]["event_id"] for f in catalog.geojson(a, b)["features"] if f["properties"]["kind"] == "touchdown"]
    assert ids("2024-04-26", "2024-04-27") == ["2024_a-01", "2024_b-01", "2024_c-01"]  # 00:00 on the 27th belongs to the next window
    assert ids("2024-04-27", "2024-04-28") == ["2024_d-01"]
    assert ids("2024-04-26 18:30:00", "2024-04-26 18:30:01") == ["2024_a-01"]  # start inclusive
    assert ids("2024-04-26 18:30:01", "2024-04-26 19:10:00") == []  # end exclusive
    assert ids("2020-01-01", "2020-02-01") == []


def test_geojson_structure_and_properties(catalog):
    fc = catalog.geojson("2024-04-26", "2024-04-27")
    assert fc["type"] == "FeatureCollection" and fc["schema_version"] == 1 and fc["count"] == 3
    assert (fc["start"], fc["end"]) == ("2024-04-26T00:00:00Z", "2024-04-27T00:00:00Z")
    a_point = next(f for f in fc["features"] if f["properties"]["event_id"] == "2024_a-01" and f["properties"]["kind"] == "touchdown")
    assert a_point["type"] == "Feature" and a_point["geometry"] == {"type": "Point", "coordinates": [-96.0, 41.0]}  # lon, lat
    assert a_point["properties"] == {"kind": "touchdown", "event_id": "2024_a-01", "timestamp_utc": "2024-04-26T18:30:00Z",
                                     "ef_rating": 2, "state": "NE", "length_mi": 12.5, "width_yd": 400}
    a_track = next(f for f in fc["features"] if f["properties"]["event_id"] == "2024_a-01" and f["properties"]["kind"] == "track")
    assert a_track["geometry"]["type"] == "LineString" and [list(c) for c in a_track["geometry"]["coordinates"]] == [[-96.0, 41.0], [-95.8, 41.1]]
    json.dumps(fc)  # serializable as is


def test_zero_length_track_is_a_point_only_and_unknown_ef_is_null(catalog):
    fc = catalog.geojson("2024-04-26", "2024-04-27")
    b = [f for f in fc["features"] if f["properties"]["event_id"] == "2024_b-01"]
    assert [f["properties"]["kind"] for f in b] == ["touchdown"]
    assert b[0]["properties"]["ef_rating"] is None
    assert fc["count"] == 3 and len(fc["features"]) == 5  # 3 touchdowns + 2 real tracks


def test_every_report_has_one_touchdown_and_at_most_one_track(catalog):
    fc = catalog.geojson("2024-01-01", "2024-12-31")
    by_event = {}
    for f in fc["features"]:
        by_event.setdefault(f["properties"]["event_id"], []).append(f["properties"]["kind"])
    assert len(by_event) == fc["count"] == len(REPORT_ROWS)
    assert all(kinds.count("touchdown") == 1 and kinds.count("track") <= 1 for kinds in by_event.values())


def test_catalog_rejects_a_table_with_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ReportsCatalog(pd.DataFrame({"event_id": ["x"]}))


def test_catalog_sorts_by_time_and_exposes_its_range(tmp_path):
    rows = list(reversed(REPORT_ROWS))
    path = tmp_path / "r.csv"
    pd.DataFrame(rows, columns=["event_id", "timestamp_utc", "start_lat", "start_lon", "end_lat", "end_lon", "length_mi",
                                "width_yd", "path_wkt", "ef_rating", "state"]).to_csv(path, index=False)
    cat = ReportsCatalog.load(path)
    assert cat.first == pd.Timestamp("2024-04-26 18:30") and cat.last == pd.Timestamp("2024-05-30 12:00")


@pytest.mark.skipif(not DEFAULT_REPORTS_PATH.exists(), reason="the SPC reports table is not built locally")
def test_real_reports_for_the_mayfield_night():
    cat = ReportsCatalog.load()
    fc = cat.geojson("2021-12-10T18:00:00Z", "2021-12-11T12:00:00Z")
    assert fc["count"] == 68  # SPC reports in that window, counted directly from the CSV
    points = [f for f in fc["features"] if f["properties"]["kind"] == "touchdown"]
    assert len(points) == 68
    by_id = {f["properties"]["event_id"]: f for f in fc["features"] if f["properties"]["kind"] == "track"}
    for event_id in ("2021_2112102054-01", "2021_2112101907-01"):  # the two EF4s the project documents
        assert event_id in by_id and by_id[event_id]["properties"]["ef_rating"] == 4
    # every coordinate sits inside the continental US box: lon/lat order was not swapped
    for f in fc["features"]:
        coords = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Point" else f["geometry"]["coordinates"]
        for lon, lat in coords:
            assert -125 < lon < -66 and 24 < lat < 50


def test_replay_cases_are_well_formed():
    keys = [c["key"] for c in REPLAY_CASES]
    assert len(keys) == len(set(keys)) and keys
    for c in REPLAY_CASES:
        assert c["label"] and c["init_times"]
        for i in c["init_times"]:
            assert case_for(i)["key"] == c["key"]
            assert pd.Timestamp(i[:-1] + ":00:00").minute == 0
        assert case_by_key(c["key"]) is c
    assert case_for("2026-10-19T01Z") is None
    with pytest.raises(KeyError):
        case_by_key("nope")
