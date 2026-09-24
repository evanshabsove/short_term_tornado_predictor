import json

import pandas as pd

from tornado_predictor.build_scaled import ARCHIVE_START, load_manifest, save_manifest, select_active_dates, select_quiet_dates


def _synthetic_reports(dates):
    return pd.DataFrame({"timestamp_utc": [pd.Timestamp(f"{d} 18:30") for d in dates]})


def test_select_active_dates_deduplicates_and_normalizes_to_calendar_date():
    reports_df = _synthetic_reports(["2020-05-01", "2020-05-01", "2020-05-01", "2021-06-15"])
    active = select_active_dates(reports_df)
    assert active == [pd.Timestamp("2020-05-01"), pd.Timestamp("2021-06-15")]


def test_select_active_dates_excludes_dates_before_archive_start():
    reports_df = _synthetic_reports(["2014-08-20", "2014-09-01", "2014-09-02"])
    active = select_active_dates(reports_df)
    assert pd.Timestamp("2014-08-20") not in active
    assert active[0] >= ARCHIVE_START


def test_select_quiet_dates_never_overlaps_active_dates():
    reports_df = _synthetic_reports(["2020-05-01", "2020-05-02", "2020-05-03"])
    quiet = select_quiet_dates(reports_df, n=50, end=pd.Timestamp("2021-01-01"), seed=0)
    active = set(pd.to_datetime(reports_df["timestamp_utc"]).dt.normalize())
    assert len(quiet) == 50
    assert not (set(quiet) & active)


def test_select_quiet_dates_respects_archive_start_and_end():
    reports_df = _synthetic_reports(["2020-05-01"])
    end = pd.Timestamp("2020-12-31")
    quiet = select_quiet_dates(reports_df, n=100, end=end, seed=0)
    assert all(ARCHIVE_START <= d <= end for d in quiet)


def test_select_quiet_dates_is_deterministic_given_same_seed():
    reports_df = _synthetic_reports(["2020-05-01"])
    end = pd.Timestamp("2021-01-01")
    a = select_quiet_dates(reports_df, n=20, end=end, seed=7)
    b = select_quiet_dates(reports_df, n=20, end=end, seed=7)
    assert a == b


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"2020-05-01": {"status": "done", "message": "ok"}}
    save_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded == manifest


def test_load_manifest_returns_empty_dict_when_missing(tmp_path):
    assert load_manifest(tmp_path / "does_not_exist.json") == {}


def test_save_manifest_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "manifest.json"
    save_manifest(path, {"a": 1})
    assert path.exists()
    assert json.loads(path.read_text()) == {"a": 1}
