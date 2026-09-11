import pandas as pd
import pytest

from tornado_predictor.time_bins import (
    assign_valid_time_bin,
    bin_valid_time_range,
    candidate_run_inits,
    fxx_to_bin_index,
    fxx_values_for_bin,
)

INIT = pd.Timestamp("2021-05-03 18:00:00")


def test_fxx_to_bin_index_bin0_boundaries():
    assert fxx_to_bin_index(0) == 0
    assert fxx_to_bin_index(3) == 0


def test_fxx_to_bin_index_bin1_boundaries():
    assert fxx_to_bin_index(4) == 1
    assert fxx_to_bin_index(7) == 1


def test_fxx_to_bin_index_excludes_upper_boundary():
    assert fxx_to_bin_index(8) is None


def test_fxx_to_bin_index_negative_is_none():
    assert fxx_to_bin_index(-1) is None


def test_fxx_values_for_bin():
    assert list(fxx_values_for_bin(0)) == [0, 1, 2, 3]
    assert list(fxx_values_for_bin(1)) == [4, 5, 6, 7]


def test_fxx_values_for_bin_invalid_index_raises():
    with pytest.raises(ValueError):
        fxx_values_for_bin(2)


def test_bin_valid_time_range():
    assert bin_valid_time_range(INIT, 0) == (INIT, INIT + pd.Timedelta(hours=4))
    assert bin_valid_time_range(INIT, 1) == (INIT + pd.Timedelta(hours=4), INIT + pd.Timedelta(hours=8))


def test_assign_valid_time_bin_report_before_init_is_none():
    assert assign_valid_time_bin(INIT, INIT - pd.Timedelta(minutes=1)) is None


def test_assign_valid_time_bin_at_bin_edge_goes_to_upper_bin():
    assert assign_valid_time_bin(INIT, INIT + pd.Timedelta(hours=4)) == 1


def test_assign_valid_time_bin_fractional_lead_time():
    assert assign_valid_time_bin(INIT, INIT + pd.Timedelta(hours=1, minutes=30)) == 0
    assert assign_valid_time_bin(INIT, INIT + pd.Timedelta(hours=4, minutes=45)) == 1


def test_assign_valid_time_bin_excludes_exact_8h():
    assert assign_valid_time_bin(INIT, INIT + pd.Timedelta(hours=8)) is None


def test_candidate_run_inits_count_and_range():
    ts = pd.Timestamp("2021-05-03 14:45:00")
    result = candidate_run_inits(ts)
    assert len(result) == 8
    assert min(result) == pd.Timestamp("2021-05-03 07:00:00")
    assert max(result) == pd.Timestamp("2021-05-03 14:00:00")


def test_candidate_run_inits_on_exact_hour():
    ts = pd.Timestamp("2021-05-03 14:00:00")
    result = candidate_run_inits(ts)
    assert min(result) == pd.Timestamp("2021-05-03 07:00:00")
    assert max(result) == pd.Timestamp("2021-05-03 14:00:00")


def test_candidate_run_inits_all_are_consistent_with_assign_valid_time_bin():
    ts = pd.Timestamp("2021-05-03 14:45:00")
    assert all(assign_valid_time_bin(init, ts) is not None for init in candidate_run_inits(ts))
