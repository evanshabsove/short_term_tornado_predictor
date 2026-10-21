"""
Curated historical cases for the forecast API's replay mode (ticket 8).

A replay is the production ensemble run on a past HRRR init time and stored in
the forecast store's "replay" namespace by scripts/09_serving/run_replay.py;
the API only serves what is stored. These cases make the Rails page
demonstrable when there is no live severe weather. Plain data, no heavy imports
(the API imports this module).

The served model trained on every year 2014-2025, including these dates, so a
replay of them is a retrospective illustration and never a held-out test.
"""

from __future__ import annotations

REPLAY_CASES = [
    {
        "key": "mayfield-2021",
        "label": "December 10-11, 2021: Quad-State / Mayfield tornado outbreak",
        "init_times": ["2021-12-10T21Z", "2021-12-10T22Z", "2021-12-10T23Z"],
    },
    {
        "key": "plains-2024-04-26",
        "label": "April 26-27, 2024: central Plains / Midwest outbreak",
        "init_times": ["2024-04-26T18Z", "2024-04-26T20Z", "2024-04-26T22Z"],
    },
]


def case_for(forecast_id: str) -> dict | None:
    """The curated case containing this forecast id (YYYY-MM-DDTHHZ), if any."""
    for case in REPLAY_CASES:
        if forecast_id in case["init_times"]:
            return case
    return None


def case_by_key(key: str) -> dict:
    for case in REPLAY_CASES:
        if case["key"] == key:
            return case
    raise KeyError(f"unknown replay case {key!r}; known: {[c['key'] for c in REPLAY_CASES]}")
