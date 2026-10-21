"""
Compute replays: the production ensemble run on past HRRR init times, stored in
the forecast store's "replay" namespace (data/live/replay/) for the API's
GET /replay/<id> (ticket 8). The API itself never computes anything; it serves
what is stored here.

The served model trained on every year 2014-2025, so a replay of any date in
that range is a retrospective illustration, not a held-out test (the API says so
in every replay payload).

Usage:
    python scripts/09_serving/run_replay.py --case mayfield-2021
    python scripts/09_serving/run_replay.py --case plains-2024-04-26
    python scripts/09_serving/run_replay.py --all-cases
    python scripts/09_serving/run_replay.py --init 2021-12-10T21Z --init 2021-12-10T22Z

Already-stored replays are skipped. Each run pulls ~16 small HRRR files (~30-40 s).
Exit code 1 if any run failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tornado_predictor.ensemble import DEFAULT_BUNDLE_MANIFEST_PATH, load_bundle
from tornado_predictor.forecast_job import FAILED, make_forecast_builder, run_replays
from tornado_predictor.forecast_store import DEFAULT_STORE_ROOT, ForecastStore
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.replay_cases import REPLAY_CASES, case_by_key

log = logging.getLogger("replay")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=[], help=f"curated case key: {[c['key'] for c in REPLAY_CASES]}")
    parser.add_argument("--all-cases", action="store_true", help="every curated case")
    parser.add_argument("--init", action="append", default=[], help="a forecast id such as 2021-12-10T21Z (on the hour, UTC)")
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_MANIFEST_PATH)
    args = parser.parse_args()

    inits = list(args.init)
    for key in ([c["key"] for c in REPLAY_CASES] if args.all_cases else args.case):
        inits += case_by_key(key)["init_times"]
    if not inits:
        parser.error("nothing to do: pass --case, --all-cases or --init")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bundle = load_bundle(args.bundle)
    build = make_forecast_builder(bundle, build_coarse_grid())
    store = ForecastStore(args.store_root, "replay")
    log.info("bundle %s (%d members), storing replays in %s", bundle.bundle_id, len(bundle.models), store.dir)

    outcomes = run_replays(store, build, list(dict.fromkeys(inits)))
    for o in outcomes:
        log.info("%s: %s%s", o.init_time, o.status, f" ({o.message})" if o.message else "")
    return 1 if any(o.status == FAILED for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
