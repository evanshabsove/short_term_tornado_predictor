"""
Defines the forecast valid-time bin scheme for tornado forecasting (0-8h
lead time, two 4-hour bins) and the pure calendar-arithmetic functions
that map HRRR forecast hours (fxx) and absolute UTC timestamps (e.g. SPC
report times) onto those bins.

Every hourly HRRR run (00-23Z, all 24 runs/day) defines its own
independent 0-8h timeline and its own two bins, relative to *that run's*
init time -- not restricted to synoptic 00/06/12/18Z cycles. A
consequence: a single absolute valid time (e.g. one tornado report)
falls into a different bin depending on which run's timeline it is being
scored against, and it will appear as a positive label in the sample
sets of every run whose [init, init+8h) window contains it -- always
exactly 8 different runs (see candidate_run_inits). This multiplicity is
intentional (see CLAUDE.md "Time binning (locked in)"), not something to
deduplicate away.

All timestamps here are naive pandas.Timestamps treated as UTC by
convention (matching timestamp_utc in the SPC reports CSV) -- never
tz-aware; mixing naive and tz-aware timestamps will raise in pandas,
which is the desired fail-loud behavior rather than silently localizing.

This module is pure date/time arithmetic: it does not download or read
any data.
"""

from __future__ import annotations

import bisect

import pandas as pd

# Half-open bin i covers lead hours [BIN_EDGES_HOURS[i], BIN_EDGES_HOURS[i+1]).
# Locked in at two 4-hour bins over 0-8h per CLAUDE.md -- don't change
# without discussing with the user first, later labeling/feature
# aggregation depends on this.
BIN_EDGES_HOURS: tuple[int, ...] = (0, 4, 8)
N_BINS = len(BIN_EDGES_HOURS) - 1
MAX_LEAD_HOURS = BIN_EDGES_HOURS[-1]
HRRR_RUN_INTERVAL_HOURS = 1  # HRRR runs every hour, all 24 inits/day


def fxx_to_bin_index(fxx: int) -> int | None:
    """Map an integer HRRR forecast hour to a bin index. fxx=8 is
    deliberately excluded -- a valid HRRR pull, but not part of either
    bin under the half-open convention (bin 1 only covers fxx 4-7). For
    continuous/fractional lead times (e.g. SPC report timestamps) use
    assign_valid_time_bin instead. Returns None if fxx is outside
    [0, MAX_LEAD_HOURS)."""
    if fxx < BIN_EDGES_HOURS[0] or fxx >= MAX_LEAD_HOURS:
        return None
    return bisect.bisect_right(BIN_EDGES_HOURS, fxx) - 1


def fxx_values_for_bin(bin_index: int) -> range:
    """The integer HRRR forecast hours belonging to bin_index, for later
    pooling of multiple hourly HRRR fields into one bin's feature vector
    (the aggregation itself is out of scope here). Raises ValueError for
    an out-of-range bin_index -- a config-time/programmer error, not a
    per-row data path, so failing loudly is preferable to a sentinel."""
    if not (0 <= bin_index < N_BINS):
        raise ValueError(f"bin_index must be in [0, {N_BINS}), got {bin_index}")
    return range(BIN_EDGES_HOURS[bin_index], BIN_EDGES_HOURS[bin_index + 1])


def bin_valid_time_range(init_time, bin_index: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Absolute half-open [lo, hi) UTC valid-time bounds for bin_index
    relative to one run's init_time. Raises ValueError for an invalid
    bin_index (same rationale as fxx_values_for_bin)."""
    if not (0 <= bin_index < N_BINS):
        raise ValueError(f"bin_index must be in [0, {N_BINS}), got {bin_index}")
    init_time = pd.Timestamp(init_time)
    lo = init_time + pd.Timedelta(hours=BIN_EDGES_HOURS[bin_index])
    hi = init_time + pd.Timedelta(hours=BIN_EDGES_HOURS[bin_index + 1])
    return lo, hi


def assign_valid_time_bin(init_time, timestamp) -> int | None:
    """Given a run's init_time and an arbitrary absolute UTC timestamp
    (e.g. an SPC report's timestamp_utc, generally not on the hour),
    compute the continuous lead time in hours and return which bin it
    falls into relative to that run. Returns None if timestamp is
    strictly before init_time (a report before the run started cannot
    have been forecast by it) or >= init_time + MAX_LEAD_HOURS hours."""
    init_time = pd.Timestamp(init_time)
    ts = pd.Timestamp(timestamp)
    lead_hours = (ts - init_time) / pd.Timedelta(hours=1)
    if lead_hours < BIN_EDGES_HOURS[0] or lead_hours >= MAX_LEAD_HOURS:
        return None
    return bisect.bisect_right(BIN_EDGES_HOURS, lead_hours) - 1


def candidate_run_inits(timestamp) -> list[pd.Timestamp]:
    """The hourly HRRR run init times whose [init, init+8h) window
    contains timestamp -- every run that could have forecast this
    absolute valid time. Needed for labeling: given one tornado report,
    which runs' sample sets should get a positive label.

    Floors timestamp to the hour and walks back MAX_LEAD_HOURS hourly
    candidates. This always yields exactly MAX_LEAD_HOURS valid
    candidates, no filtering needed: for init = hour_floor - k*1h (k in
    [0, MAX_LEAD_HOURS-1]), hour_floor <= timestamp < hour_floor + 1h
    (definition of floor) gives init <= hour_floor <= timestamp (lower
    bound always holds), and init + MAX_LEAD_HOURS*1h = hour_floor +
    (MAX_LEAD_HOURS-k)*1h >= hour_floor + 1h > timestamp for every valid
    k (upper bound always holds). This relies on HRRR_RUN_INTERVAL_HOURS
    == 1 evenly dividing MAX_LEAD_HOURS; it is a pure calendar function
    and does not know about real HRRR data-availability gaps (e.g. the
    archive's start date) -- a later step must filter those separately.
    """
    hour_floor = pd.Timestamp(timestamp).floor("h")
    n_candidates = int(MAX_LEAD_HOURS / HRRR_RUN_INTERVAL_HOURS)
    return sorted(hour_floor - pd.Timedelta(hours=HRRR_RUN_INTERVAL_HOURS * k) for k in range(n_candidates))
