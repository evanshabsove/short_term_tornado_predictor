import pandas as pd

from tornado_predictor.split import (
    assign_date_split,
    assign_date_split_3way,
    assign_year,
    available_years,
    report_split_balance,
    report_split_balance_3way,
    split_run_bins,
    split_run_bins_3way,
    split_run_bins_year_holdout,
)


def test_assign_date_split_matches_known_boundary_crossings():
    # real day-of-year / mod-20 values, verified by hand: 336->16 (train),
    # 337->17 (val), 339->19 (val), 340->0 (train)
    assert assign_date_split("2021-12-02") == "train"  # day 336, mod 16
    assert assign_date_split("2021-12-03") == "val"  # day 337, mod 17
    assert assign_date_split("2021-12-05") == "val"  # day 339, mod 19
    assert assign_date_split("2021-12-06") == "train"  # day 340, mod 0


def test_assign_date_split_is_deterministic_across_years():
    # same day-of-year, different years -> same split every time
    assert assign_date_split("2021-12-03") == assign_date_split("2019-12-03") == assign_date_split("2024-12-03")


def test_split_run_bins_keeps_whole_date_together():
    run_bins = [
        (pd.Timestamp("2021-12-10 21:00"), 0),
        (pd.Timestamp("2021-12-10 21:00"), 1),
        (pd.Timestamp("2021-12-10 22:00"), 0),
        (pd.Timestamp("2021-12-10 23:00"), 1),
    ]
    # 2021-12-10 is day 344, mod 4 -> train; every sample from that date
    # must land in the same split, none split off into val.
    train, val = split_run_bins(run_bins, buffer_hours=0)
    assert set(train) == set(run_bins)
    assert val == []


def test_split_run_bins_separates_different_dates():
    train_date_run = (pd.Timestamp("2021-12-02 12:00"), 0)  # train date (day 336, mod 16)
    # 2021-12-03 is a real val date (day 337, mod 17); use a far-apart hour
    # so the buffer safeguard (tested separately) doesn't remove it here.
    val_date_run = (pd.Timestamp("2021-12-03 12:00"), 0)

    train, val = split_run_bins([train_date_run, val_date_run], buffer_hours=1)

    assert train == [train_date_run]
    assert val == [val_date_run]


def test_buffer_safeguard_drops_nearby_training_samples():
    # 2021-12-02 -> train, 2021-12-03 -> val (real boundary crossing).
    # A train run at 23:00 on the train day is only 1h before a val run
    # at 00:00 the next (val) day -- within a 24h buffer, must be dropped.
    train_run = (pd.Timestamp("2021-12-02 23:00"), 0)
    val_run = (pd.Timestamp("2021-12-03 00:00"), 0)

    train, val = split_run_bins([train_run, val_run], buffer_hours=24)
    assert train == []  # dropped, not reassigned
    assert val == [val_run]


def test_buffer_safeguard_keeps_training_samples_far_from_val():
    train_run = (pd.Timestamp("2021-11-01 00:00"), 0)  # far from any val date below
    val_run = (pd.Timestamp("2021-12-03 00:00"), 0)

    train, val = split_run_bins([train_run, val_run], buffer_hours=24)
    assert train == [train_run]
    assert val == [val_run]


def test_report_split_balance_computes_positive_rate():
    labels_df = pd.DataFrame(
        [
            {"init_time": pd.Timestamp("2021-12-02 21:00"), "bin_index": 0},
            {"init_time": pd.Timestamp("2021-12-02 21:00"), "bin_index": 0},  # duplicate row, same key
            {"init_time": pd.Timestamp("2021-12-03 21:00"), "bin_index": 1},
        ]
    )
    train_run_bins = [(pd.Timestamp("2021-12-02 21:00"), 0), (pd.Timestamp("2021-11-01 00:00"), 0)]
    val_run_bins = [(pd.Timestamp("2021-12-03 21:00"), 1)]

    report = report_split_balance(labels_df, train_run_bins, val_run_bins)

    assert report["train"] == {"n_samples": 2, "n_positive_samples": 1, "positive_rate": 0.5}
    assert report["val"] == {"n_samples": 1, "n_positive_samples": 1, "positive_rate": 1.0}


def test_report_split_balance_handles_empty_split():
    labels_df = pd.DataFrame(columns=["init_time", "bin_index"])
    report = report_split_balance(labels_df, [], [])
    assert report["train"]["n_samples"] == 0
    assert report["val"]["n_samples"] == 0


# --- 3-way split ---
# real day-of-year / mod-20 values, verified by hand (same style as the
# 2-way tests above): 333->13 (train), 334->14 (test), 336->16 (test),
# 337->17 (val), 339->19 (val), 340->0 (train)


def test_assign_date_split_3way_matches_known_boundary_crossings():
    assert assign_date_split_3way("2021-11-29") == "train"  # day 333, mod 13
    assert assign_date_split_3way("2021-11-30") == "test"  # day 334, mod 14
    assert assign_date_split_3way("2021-12-02") == "test"  # day 336, mod 16
    assert assign_date_split_3way("2021-12-03") == "val"  # day 337, mod 17
    assert assign_date_split_3way("2021-12-05") == "val"  # day 339, mod 19
    assert assign_date_split_3way("2021-12-06") == "train"  # day 340, mod 0


def test_assign_date_split_3way_val_matches_2way_val_exactly():
    """The core design requirement: every date the 2-way split calls
    val must get the same "val" label under the 3-way split too, so
    every AUC-PR already reported against val still describes the same
    data. Checked across 2 full mod-20 cycles, not just the known
    boundary dates above."""
    for offset in range(40):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=offset)
        if assign_date_split(date) == "val":
            assert assign_date_split_3way(date) == "val"


def test_assign_date_split_3way_train_is_a_subset_of_2way_train():
    """The new, smaller train (mod 0-13) must be a strict subset of the
    old train (mod 0-16) -- nothing moves OUT of train into val, only
    some of train is carved into the new test."""
    for offset in range(40):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=offset)
        if assign_date_split_3way(date) == "train":
            assert assign_date_split(date) == "train"


def test_split_run_bins_3way_keeps_whole_date_together():
    run_bins = [
        (pd.Timestamp("2021-12-10 21:00"), 0),  # day 344, mod 4 -> train
        (pd.Timestamp("2021-12-10 22:00"), 0),
        (pd.Timestamp("2021-12-10 23:00"), 1),
    ]
    train, test, val = split_run_bins_3way(run_bins, buffer_hours=0)
    assert set(train) == set(run_bins)
    assert test == [] and val == []


def test_split_run_bins_3way_separates_all_three_groups():
    train_run = (pd.Timestamp("2021-11-29 12:00"), 0)  # train
    test_run = (pd.Timestamp("2021-11-30 12:00"), 0)  # test
    val_run = (pd.Timestamp("2021-12-03 12:00"), 0)  # val

    train, test, val = split_run_bins_3way([train_run, test_run, val_run], buffer_hours=1)

    assert train == [train_run]
    assert test == [test_run]
    assert val == [val_run]


def test_buffer_safeguard_drops_train_near_test():
    # 2021-11-29 (train) -> 2021-11-30 (test), real boundary crossing.
    train_run = (pd.Timestamp("2021-11-29 23:00"), 0)
    test_run = (pd.Timestamp("2021-11-30 00:00"), 0)

    train, test, val = split_run_bins_3way([train_run, test_run], buffer_hours=24)
    assert train == []  # dropped, not reassigned
    assert test == [test_run]  # test is never dropped


def test_buffer_safeguard_drops_val_near_test():
    # 2021-12-02 (test) -> 2021-12-03 (val), real boundary crossing.
    test_run = (pd.Timestamp("2021-12-02 23:00"), 0)
    val_run = (pd.Timestamp("2021-12-03 00:00"), 0)

    train, test, val = split_run_bins_3way([test_run, val_run], buffer_hours=24)
    assert val == []  # dropped
    assert test == [test_run]  # test is never dropped


def test_buffer_safeguard_drops_train_near_val_even_when_not_mod_adjacent():
    """train and val are no longer mod-adjacent under the 3-way rule
    (test always sits between them in the mod-20 cycle, a full 3-day
    block), so this scenario can't be demonstrated with two naturally
    calendar-adjacent dates the way the train-near-test/val-near-test
    tests above can -- the closest a real train date and a real val
    date can be is ~4 days apart (test_run_bins_3way_separates_all_three_groups'
    2021-11-29 train / 2021-12-03 val). The rule (drop train within
    buffer_hours of ANY val sample) must still hold at a large enough
    buffer to span that gap."""
    train_run = (pd.Timestamp("2021-11-29 12:00"), 0)  # mod 13 -> train
    val_run = (pd.Timestamp("2021-12-03 12:00"), 0)  # mod 17 -> val, 4 days later

    train, test, val = split_run_bins_3way([train_run, val_run], buffer_hours=96)
    assert train == []
    assert val == [val_run]


def test_buffer_safeguard_keeps_samples_far_from_test_and_val():
    train_run = (pd.Timestamp("2021-11-01 00:00"), 0)  # far from everything below
    test_run = (pd.Timestamp("2021-11-30 12:00"), 0)
    val_run = (pd.Timestamp("2021-12-03 12:00"), 0)

    train, test, val = split_run_bins_3way([train_run, test_run, val_run], buffer_hours=24)
    assert train == [train_run]
    assert test == [test_run]
    assert val == [val_run]


def test_report_split_balance_3way_computes_positive_rate():
    labels_df = pd.DataFrame(
        [
            {"init_time": pd.Timestamp("2021-11-29 21:00"), "bin_index": 0},
            {"init_time": pd.Timestamp("2021-11-30 21:00"), "bin_index": 0},
            {"init_time": pd.Timestamp("2021-12-03 21:00"), "bin_index": 1},
        ]
    )
    train_run_bins = [(pd.Timestamp("2021-11-29 21:00"), 0), (pd.Timestamp("2021-11-01 00:00"), 0)]
    test_run_bins = [(pd.Timestamp("2021-11-30 21:00"), 0)]
    val_run_bins = [(pd.Timestamp("2021-12-03 21:00"), 1)]

    report = report_split_balance_3way(labels_df, train_run_bins, test_run_bins, val_run_bins)

    assert report["train"] == {"n_samples": 2, "n_positive_samples": 1, "positive_rate": 0.5}
    assert report["test"] == {"n_samples": 1, "n_positive_samples": 1, "positive_rate": 1.0}
    assert report["val"] == {"n_samples": 1, "n_positive_samples": 1, "positive_rate": 1.0}


def test_report_split_balance_3way_handles_empty_splits():
    labels_df = pd.DataFrame(columns=["init_time", "bin_index"])
    report = report_split_balance_3way(labels_df, [], [], [])
    assert report["train"]["n_samples"] == 0
    assert report["test"]["n_samples"] == 0
    assert report["val"]["n_samples"] == 0


# --- year-holdout cross-validation split ---


def test_assign_year_extracts_calendar_year():
    assert assign_year("2021-12-10") == 2021
    assert assign_year(pd.Timestamp("2014-09-01 20:00")) == 2014


def test_available_years_returns_sorted_distinct_years():
    run_bins = [
        (pd.Timestamp("2021-12-10 21:00"), 0),
        (pd.Timestamp("2019-05-01 12:00"), 0),
        (pd.Timestamp("2021-01-01 00:00"), 1),
    ]
    assert available_years(run_bins) == [2019, 2021]


def test_split_run_bins_year_holdout_separates_held_out_year():
    run_2020 = (pd.Timestamp("2020-06-15 12:00"), 0)
    run_2021 = (pd.Timestamp("2021-06-15 12:00"), 0)
    run_2022 = (pd.Timestamp("2022-06-15 12:00"), 0)

    train, test = split_run_bins_year_holdout([run_2020, run_2021, run_2022], held_out_year=2021, buffer_hours=0)

    assert test == [run_2021]
    assert set(train) == {run_2020, run_2022}


def test_split_run_bins_year_holdout_buffer_drops_train_near_year_boundary():
    # Dec 31 of the prior year and Jan 1 of the held-out year are
    # consecutive calendar dates -- exactly the kind of abrupt split
    # flip the buffer safeguard exists to protect against, now at a
    # year boundary instead of a mod-20 one.
    train_candidate = (pd.Timestamp("2020-12-31 23:00"), 0)
    held_out_run = (pd.Timestamp("2021-01-01 00:00"), 0)

    train, test = split_run_bins_year_holdout([train_candidate, held_out_run], held_out_year=2021, buffer_hours=24)

    assert test == [held_out_run]
    assert train == []  # the Dec-31 run is within 1h of the held-out run -- dropped


def test_split_run_bins_year_holdout_keeps_train_far_from_year_boundary():
    far_train_run = (pd.Timestamp("2020-06-01 00:00"), 0)
    held_out_run = (pd.Timestamp("2021-06-01 00:00"), 0)

    train, test = split_run_bins_year_holdout([far_train_run, held_out_run], held_out_year=2021, buffer_hours=24)

    assert train == [far_train_run]
    assert test == [held_out_run]


def test_split_run_bins_year_holdout_handles_year_with_no_runs():
    run_2020 = (pd.Timestamp("2020-06-15 12:00"), 0)
    train, test = split_run_bins_year_holdout([run_2020], held_out_year=1999, buffer_hours=24)
    assert train == [run_2020]
    assert test == []
