"""
Train/validation split for the dense-grid dataset, avoiding leakage
from spatially/temporally correlated severe-weather outbreaks.

Adapts the splitting methodology from the TorNet benchmark paper
(Veillette et al. 2024, "Benchmark Dataset for Tornado Detection and
Prediction..."): group samples by a coarser leakage-safe unit (TorNet:
NOAA Storm Events Database "storm episode"; here: UTC calendar date of
the HRRR run's init_time -- our samples are full-domain maps, not
storm-centered crops, so every (run, bin) sample from the same date is
inherently correlated regardless of which grid cell is being looked
at), assign that unit to train/val via a deterministic Julian-day
modulo rule (TorNet: day-of-year mod 20 < 17 -> train, an 85/15 split;
ensures every split sees the full seasonal cycle every year, rather
than risking a lucky/unlucky contiguous year range), and apply a
buffer safeguard removing training samples too close in time to any
validation sample.

The buffer safeguard matters because the mod-20 rule flips abruptly at
two points every 20-day cycle (e.g. real dates: 2021-12-02 -> train,
2021-12-03 -> val -- consecutive calendar days, opposite splits). A
storm that straddles one of these boundaries (e.g. active in the
evening of a "train" day and past midnight into a "val" day) would
otherwise put a training sample and a validation sample an hour apart
looking at nearly the same storm. TorNet's own version of this is a
30-minute-and-0.25-degree buffer that removes training samples too
close to test samples; we only need the temporal half, since every one
of our samples already covers the whole domain (there's no
"far enough away spatially" case here).

Unlike TorNet, this module does NOT attempt to balance tornadic vs.
non-tornadic representation via the split rule itself -- TorNet
achieves that at corpus-construction time (which samples exist at
all), a decision this project makes separately when a training
run/date list is assembled (see dataset.py). This module only ensures
that whatever corpus already exists gets split without leakage, and
`report_split_balance` reports the resulting positive rate per split
so an accidental imbalance is caught, not silently shipped.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_MOD = 20
DEFAULT_TRAIN_THRESHOLD = 17  # day_of_year % 20 < 17 -> train; matches TorNet's 85/15 split
DEFAULT_BUFFER_HOURS = 24


def assign_date_split(date, mod: int = DEFAULT_MOD, train_threshold: int = DEFAULT_TRAIN_THRESHOLD) -> str:
    """"train" or "val" for a given date, via day-of-year mod `mod`
    compared to `train_threshold`. Pure and deterministic: the same
    date always gets the same split, and every `mod`-day cycle repeats
    across all years, so both splits see the full seasonal cycle."""
    day_of_year = pd.Timestamp(date).dayofyear
    return "train" if (day_of_year % mod) < train_threshold else "val"


def split_run_bins(
    run_bins: list[tuple],
    buffer_hours: float = DEFAULT_BUFFER_HOURS,
    mod: int = DEFAULT_MOD,
    train_threshold: int = DEFAULT_TRAIN_THRESHOLD,
) -> tuple[list[tuple], list[tuple]]:
    """Splits a list of (init_time, bin_index) run_bins into
    (train_run_bins, val_run_bins), grouped by calendar date of
    init_time (every run_bin from the same date goes to the same
    split), then drops any train run_bin whose init_time is within
    `buffer_hours` of any val run_bin's init_time (removed, not
    reassigned -- mirrors TorNet's own buffer-removal choice)."""
    run_bins = [(pd.Timestamp(init_time), bin_index) for init_time, bin_index in run_bins]

    train = [(init_time, bin_index) for init_time, bin_index in run_bins if assign_date_split(init_time) == "train"]
    val = [(init_time, bin_index) for init_time, bin_index in run_bins if assign_date_split(init_time) == "val"]

    if val and buffer_hours > 0:
        val_times = [init_time for init_time, _ in val]
        buffer = pd.Timedelta(hours=buffer_hours)
        train = [
            (init_time, bin_index)
            for init_time, bin_index in train
            if not any(abs(init_time - v) <= buffer for v in val_times)
        ]

    return train, val


DEFAULT_TRAIN_END = 14  # day_of_year % 20 < 14 -> train (70%)
DEFAULT_TEST_END = 17  # 14 <= ... < 17 -> test (15%); >= 17 -> val (15%, same as the 2-way split's val)


def assign_date_split_3way(
    date, mod: int = DEFAULT_MOD, train_end: int = DEFAULT_TRAIN_END, test_end: int = DEFAULT_TEST_END
) -> str:
    """"train", "test", or "val" for a given date. Added alongside
    (not replacing) assign_date_split -- val's day-of-year values here
    are identical to the 2-way split's val (mod >= 17, i.e.
    test_end == the 2-way split's train_threshold by construction), so
    every AUC-PR already reported against the 2-way val set describes
    the same data under this rule too. test is carved from what the
    2-way split calls train (mod 14-16) -- days never used for any
    architecture-selection decision, unlike the repeatedly-checked val."""
    day_of_year = pd.Timestamp(date).dayofyear
    m = day_of_year % mod
    if m < train_end:
        return "train"
    if m < test_end:
        return "test"
    return "val"


def split_run_bins_3way(
    run_bins: list[tuple],
    buffer_hours: float = DEFAULT_BUFFER_HOURS,
    mod: int = DEFAULT_MOD,
    train_end: int = DEFAULT_TRAIN_END,
    test_end: int = DEFAULT_TEST_END,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Splits run_bins into (train, test, val), grouped by calendar
    date via assign_date_split_3way, then applies the buffer safeguard
    -- generalized from the 2-way version's single "protect val, drop
    from train" rule to three groups: val and test are both protected
    relative to train (train must not leak into either); test is
    additionally protected relative to val, since val is checked
    repeatedly for tuning decisions that shape the final model, so val
    leaking into test would indirectly contaminate the "held-out"
    number too. Test samples are never dropped -- most protected,
    nothing is removed *from* it, the same directionality as the 2-way
    split's val just extended one level further."""
    run_bins = [(pd.Timestamp(init_time), bin_index) for init_time, bin_index in run_bins]

    def group(name):
        return [
            (init_time, bin_index)
            for init_time, bin_index in run_bins
            if assign_date_split_3way(init_time, mod, train_end, test_end) == name
        ]

    train, test, val = group("train"), group("test"), group("val")

    buffer = pd.Timedelta(hours=buffer_hours)

    def drop_near(subset, protect_times):
        if not protect_times or buffer_hours <= 0:
            return subset
        return [
            (init_time, bin_index)
            for init_time, bin_index in subset
            if not any(abs(init_time - p) <= buffer for p in protect_times)
        ]

    test_times = [init_time for init_time, _ in test]
    val_times = [init_time for init_time, _ in val]

    val = drop_near(val, test_times)
    train = drop_near(train, test_times + val_times)

    return train, test, val


def report_split_balance_3way(
    labels_df: pd.DataFrame, train_run_bins: list[tuple], test_run_bins: list[tuple], val_run_bins: list[tuple]
) -> dict:
    """Same per-split positive-rate computation as report_split_balance,
    extended to three groups."""
    label_keys = set(zip(pd.to_datetime(labels_df["init_time"]), labels_df["bin_index"]))

    def positive_rate(run_bins):
        if not run_bins:
            return {"n_samples": 0, "n_positive_samples": 0, "positive_rate": float("nan")}
        keys = {(pd.Timestamp(init_time), bin_index) for init_time, bin_index in run_bins}
        n_positive = len(keys & label_keys)
        return {
            "n_samples": len(run_bins),
            "n_positive_samples": n_positive,
            "positive_rate": n_positive / len(run_bins),
        }

    return {
        "train": positive_rate(train_run_bins),
        "test": positive_rate(test_run_bins),
        "val": positive_rate(val_run_bins),
    }


def report_split_balance(labels_df: pd.DataFrame, train_run_bins: list[tuple], val_run_bins: list[tuple]) -> dict:
    """Reports sample counts and the resulting tornado-positive rate
    for each split, so an accidental imbalance from the date-based
    split is caught rather than silently shipped -- TorNet balances
    tornadic/non-tornadic representation at corpus-construction time
    instead of via the split rule; this is the corresponding check
    for a project that doesn't yet do that."""

    label_keys = set(zip(pd.to_datetime(labels_df["init_time"]), labels_df["bin_index"]))

    def positive_rate(run_bins):
        if not run_bins:
            return {"n_samples": 0, "n_positive_samples": 0, "positive_rate": float("nan")}
        keys = {(pd.Timestamp(init_time), bin_index) for init_time, bin_index in run_bins}
        n_positive = len(keys & label_keys)
        return {
            "n_samples": len(run_bins),
            "n_positive_samples": n_positive,
            "positive_rate": n_positive / len(run_bins),
        }

    return {"train": positive_rate(train_run_bins), "val": positive_rate(val_run_bins)}
