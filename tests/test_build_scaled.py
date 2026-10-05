import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tornado_predictor.build_scaled import (
    ARCHIVE_START,
    combine_staged_runs,
    finalize,
    finalize_3way,
    greedy_cover_run_inits,
    load_manifest,
    run_staging_path,
    save_manifest,
    select_active_dates,
    select_active_runs,
    select_all_quiet_dates,
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


# --- combine_staged_runs / finalize / finalize_3way ---


def _write_synthetic_staged_run(staging_dir, init_time, n_bins=2, feature_value=1.0, label_value=0.0, extra_fields=None):
    """Writes a tiny but structurally real staged per-run .nc file (2x2
    grid, one feature + label, "sample" dim of size n_bins) matching
    what dataset.build_dataset actually produces -- enough to exercise
    combine_staged_runs/finalize/finalize_3way end-to-end. extra_fields
    (dict of name -> fill value) simulates a run staged under a newer
    features.py schema with additional variables older runs don't have."""
    init_time = pd.Timestamp(init_time)
    data_vars = {
        "cape_mean": (("sample", "row", "col"), np.full((n_bins, 2, 2), feature_value)),
        "label": (("sample", "row", "col"), np.full((n_bins, 2, 2), label_value)),
    }
    for name, value in (extra_fields or {}).items():
        data_vars[name] = (("sample", "row", "col"), np.full((n_bins, 2, 2), value))
    ds = xr.Dataset(
        data_vars,
        coords={
            "init_time": ("sample", [init_time] * n_bins),
            "bin_index": ("sample", list(range(n_bins))),
        },
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(run_staging_path(staging_dir, init_time))
    return ds


def test_combine_staged_runs_handles_empty_subset():
    assert combine_staged_runs(None, []) is None


def test_combine_staged_runs_combines_multiple_runs_and_bins(tmp_path):
    run_a = pd.Timestamp("2021-12-10 21:00")
    run_b = pd.Timestamp("2021-12-11 05:00")
    _write_synthetic_staged_run(tmp_path, run_a, feature_value=1.0)
    _write_synthetic_staged_run(tmp_path, run_b, feature_value=2.0)

    run_bins_subset = [(run_a, 0), (run_a, 1), (run_b, 0)]
    combined = combine_staged_runs(tmp_path, run_bins_subset)

    assert combined.sizes["sample"] == 3
    assert list(combined["init_time"].values) == [run_a, run_a, run_b]
    assert list(combined["bin_index"].values) == [0, 1, 0]
    # last sample came from run_b's staged file (feature_value=2.0)
    assert combined["cape_mean"].isel(sample=2).values[0, 0] == 2.0


def test_combine_staged_runs_result_survives_source_file_deletion(tmp_path):
    """Regression guard for the .load() fix: the combined Dataset must
    be fully in-memory, not lazily backed by the source file handle --
    deleting the source file after combine_staged_runs returns must not
    break reading the result."""
    run_a = pd.Timestamp("2021-12-10 21:00")
    _write_synthetic_staged_run(tmp_path, run_a, feature_value=5.0)

    combined = combine_staged_runs(tmp_path, [(run_a, 0)])
    run_staging_path(tmp_path, run_a).unlink()  # delete the source file

    assert combined["cape_mean"].values[0, 0, 0] == 5.0  # still readable


def test_finalize_writes_leakage_safe_train_val_files(tmp_path):
    staging_dir = tmp_path / "staging"
    # 2021-12-02 (day 336, mod 16) -> train; 2021-12-03 (day 337, mod 17) -> val
    train_run = pd.Timestamp("2021-12-02 12:00")
    val_run = pd.Timestamp("2021-12-03 12:00")
    _write_synthetic_staged_run(staging_dir, train_run)
    _write_synthetic_staged_run(staging_dir, val_run)

    out_train, out_val = tmp_path / "train.nc", tmp_path / "val.nc"
    result = finalize(staging_dir, [train_run, val_run], out_train, out_val, buffer_hours=0)

    assert result == {"n_train": 2, "n_val": 2, "n_dropped": 0}
    assert xr.open_dataset(out_train).sizes["sample"] == 2
    assert xr.open_dataset(out_val).sizes["sample"] == 2


def test_finalize_3way_writes_train_test_val_files(tmp_path):
    staging_dir = tmp_path / "staging"
    # day-of-year mod 20: train < 14, 14 <= test < 17, val >= 17
    train_run = pd.Timestamp("2021-01-01 12:00")  # day 1, mod 1 -> train
    test_run = pd.Timestamp("2021-01-15 12:00")  # day 15, mod 15 -> test
    val_run = pd.Timestamp("2021-01-18 12:00")  # day 18, mod 18 -> val
    for r in (train_run, test_run, val_run):
        _write_synthetic_staged_run(staging_dir, r)

    out_train, out_test, out_val = tmp_path / "train.nc", tmp_path / "test.nc", tmp_path / "val.nc"
    result = finalize_3way(staging_dir, [train_run, test_run, val_run], out_train, out_test, out_val, buffer_hours=0)

    assert result == {"n_train": 2, "n_test": 2, "n_val": 2, "n_dropped": 0}
    assert xr.open_dataset(out_test).sizes["sample"] == 2


# --- select_all_quiet_dates ---


def test_select_all_quiet_dates_is_exhaustive_complement_of_active_dates():
    reports_df = _synthetic_reports(["2020-05-02", "2020-05-04"])
    end = pd.Timestamp("2020-05-06")  # ARCHIVE_START (2014-09-01) .. 2020-05-06, exclusive
    quiet = select_all_quiet_dates(reports_df, end)

    assert pd.Timestamp("2020-05-02") not in quiet
    assert pd.Timestamp("2020-05-04") not in quiet
    assert pd.Timestamp("2020-05-01") in quiet
    assert pd.Timestamp("2020-05-03") in quiet
    assert pd.Timestamp("2020-05-05") in quiet
    assert all(ARCHIVE_START <= d < end for d in quiet)


def test_select_all_quiet_dates_count_matches_total_minus_active():
    reports_df = _synthetic_reports(["2020-05-02", "2020-05-04", "2020-05-04"])  # dup date, still 2 unique active
    end = pd.Timestamp("2020-05-10")
    quiet = select_all_quiet_dates(reports_df, end)
    total_days = (end - ARCHIVE_START).days
    assert len(quiet) == total_days - 2


def test_select_all_quiet_dates_matches_select_quiet_dates_candidate_pool():
    """select_all_quiet_dates's output must be a superset of anything
    select_quiet_dates could ever sample -- both draw from the same
    "not active" pool, just exhaustively vs. randomly."""
    reports_df = _synthetic_reports(["2020-05-02", "2020-05-04"])
    end = pd.Timestamp("2020-06-01")
    all_quiet = set(select_all_quiet_dates(reports_df, end))
    sampled = select_quiet_dates(reports_df, n=10, end=end, seed=0)
    assert set(sampled) <= all_quiet


# --- finalize's drop_columns ---


def test_finalize_drop_columns_removes_variable_and_keeps_rest(tmp_path):
    staging_dir = tmp_path / "staging"
    train_run = pd.Timestamp("2021-12-02 12:00")  # train
    val_run = pd.Timestamp("2021-12-03 12:00")  # val
    _write_synthetic_staged_run(staging_dir, train_run, extra_fields={"uh_2_5km_mean": 9.0})
    _write_synthetic_staged_run(staging_dir, val_run, extra_fields={"uh_2_5km_mean": 9.0})

    out_train, out_val = tmp_path / "train.nc", tmp_path / "val.nc"
    finalize(staging_dir, [train_run, val_run], out_train, out_val, buffer_hours=0, drop_columns=["uh_2_5km_mean"])

    saved = xr.open_dataset(out_train)
    assert "uh_2_5km_mean" not in saved.data_vars
    assert "cape_mean" in saved.data_vars


def test_finalize_drop_columns_handles_mixed_schema_without_leaving_nan(tmp_path):
    """The real scenario this was built for: one run staged under an
    older schema (no uh_2_5km_mean at all), one under a newer schema
    (has it) -- xr.concat NaN-fills the old run's missing column, and
    drop_columns must remove it cleanly rather than leave that NaN in
    the final file."""
    staging_dir = tmp_path / "staging"
    old_run = pd.Timestamp("2021-12-02 12:00")  # train, no extra field
    new_run = pd.Timestamp("2021-12-09 12:00")  # train (day 343, mod 3), has extra field
    _write_synthetic_staged_run(staging_dir, old_run)
    _write_synthetic_staged_run(staging_dir, new_run, extra_fields={"uh_2_5km_mean": 9.0})

    out_train, out_val = tmp_path / "train.nc", tmp_path / "val.nc"
    finalize(staging_dir, [old_run, new_run], out_train, out_val, buffer_hours=0, drop_columns=["uh_2_5km_mean"])

    saved = xr.open_dataset(out_train)
    assert "uh_2_5km_mean" not in saved.data_vars
    assert not np.isnan(saved["cape_mean"].values).any()


def test_finalize_drop_columns_raises_on_unrelated_nan(tmp_path):
    """A genuine NaN in a column NOT being dropped must still raise --
    drop_columns only excuses NaN caused by the known old/new schema
    mismatch for the columns actually being dropped."""
    staging_dir = tmp_path / "staging"
    run = pd.Timestamp("2021-12-02 12:00")
    ds = xr.Dataset(
        {
            "cape_mean": (("sample", "row", "col"), np.full((2, 2, 2), np.nan)),
            "label": (("sample", "row", "col"), np.zeros((2, 2, 2))),
        },
        coords={"init_time": ("sample", [run, run]), "bin_index": ("sample", [0, 1])},
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(run_staging_path(staging_dir, run))

    out_train, out_val = tmp_path / "train.nc", tmp_path / "val.nc"
    with pytest.raises(AssertionError):
        finalize(staging_dir, [run], out_train, out_val, buffer_hours=0, drop_columns=["some_other_column"])
