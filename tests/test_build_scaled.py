import json

import pandas as pd

from tornado_predictor.build_scaled import (
    ARCHIVE_START,
    greedy_cover_run_inits,
    load_manifest,
    save_manifest,
    select_active_dates,
    select_active_runs,
    select_quiet_dates,
)


def _synthetic_reports(dates):
    return pd.DataFrame({"timestamp_utc": [pd.Timestamp(f"{d} 18:30") for d in dates]})


def _synthetic_reports_at(timestamps):
    return pd.DataFrame({"timestamp_utc": [pd.Timestamp(t) for t in timestamps]})


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


def test_greedy_cover_single_report_anchors_at_its_floored_hour():
    runs = greedy_cover_run_inits([pd.Timestamp("2020-05-01 14:37")])
    assert runs == [pd.Timestamp("2020-05-01 14:00")]


def test_greedy_cover_two_close_reports_share_one_run():
    # 14:10 and 18:00 are 3h50m apart -- both fall within [14:00, 22:00)
    runs = greedy_cover_run_inits([pd.Timestamp("2020-05-01 14:10"), pd.Timestamp("2020-05-01 18:00")])
    assert runs == [pd.Timestamp("2020-05-01 14:00")]


def test_greedy_cover_two_far_reports_need_two_runs():
    # 06:00 and 20:00 are 14h apart -- 20:00 falls outside [06:00, 14:00)
    runs = greedy_cover_run_inits([pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 20:00")])
    assert runs == [pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 20:00")]


def test_greedy_cover_is_half_open_at_the_8h_boundary():
    # a report exactly 8h after the anchor is NOT covered (half-open window,
    # same convention as time_bins.BIN_EDGES_HOURS) -- must get its own run
    runs = greedy_cover_run_inits([pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 14:00")])
    assert runs == [pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 14:00")]


def test_greedy_cover_unsorted_input_and_empty_input():
    runs = greedy_cover_run_inits([pd.Timestamp("2020-05-01 20:00"), pd.Timestamp("2020-05-01 06:00")])
    assert runs == [pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 20:00")]
    assert greedy_cover_run_inits([]) == []


def test_select_active_runs_covers_multiple_dates_independently():
    reports_df = _synthetic_reports_at([
        "2020-05-01 14:10",  # date 1: single report -> 1 run
        "2020-05-02 06:00",  # date 2: two far-apart reports -> 2 runs
        "2020-05-02 20:00",
    ])
    runs = select_active_runs(reports_df)
    assert runs == [
        pd.Timestamp("2020-05-01 14:00"),
        pd.Timestamp("2020-05-02 06:00"),
        pd.Timestamp("2020-05-02 20:00"),
    ]


def test_select_active_runs_excludes_dates_before_archive_start():
    reports_df = _synthetic_reports_at(["2014-08-20 12:00", "2014-09-01 12:00"])
    runs = select_active_runs(reports_df)
    assert all(r >= ARCHIVE_START for r in runs)
    assert len(runs) == 1
