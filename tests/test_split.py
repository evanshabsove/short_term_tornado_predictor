import pandas as pd

from tornado_predictor.split import assign_date_split, report_split_balance, split_run_bins


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
