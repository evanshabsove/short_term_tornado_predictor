"""
Hourly forecast job: stores a forecast for every recent HRRR run that has none.

Meant to be run by cron/launchd every ~10 minutes (runs land ~init+50-60 min,
and an invocation that finds nothing new is cheap and a no-op), or left running
with --loop. Safe to run overlapping: a lock makes a second invocation exit.

Each invocation checks the last --backfill-hours + 1 hourly runs, newest first:
stores those that are published and missing, leaves unpublished ones for a
later invocation, and records failures in <store-root>/status.json (the API's
/status reads it). See src/tornado_predictor/forecast_job.py.

Usage:
    python scripts/09_serving/run_forecast_job.py                       # one pass, then exit
    python scripts/09_serving/run_forecast_job.py --loop --interval-min 10
    python scripts/09_serving/run_forecast_job.py --backfill-hours 24   # fill gaps after downtime
    python scripts/09_serving/run_forecast_job.py --now "2026-10-19 03:00"   # pretend it is this UTC time

Exit code: 0 normally (including "nothing published yet" and "another job is running");
1 if work was attempted, failed, and nothing was stored.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tornado_predictor.ensemble import DEFAULT_BUNDLE_MANIFEST_PATH, load_bundle
from tornado_predictor.forecast_job import (
    DEFAULT_BACKFILL_HOURS,
    DEFAULT_MAX_ATTEMPTS,
    STATUS_FILENAME,
    JobStatus,
    make_forecast_builder,
    run_locked,
)
from tornado_predictor.forecast_store import DEFAULT_STORE_ROOT, ForecastStore
from tornado_predictor.grid import build_coarse_grid

log = logging.getLogger("forecast_job")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_MANIFEST_PATH, help="production bundle manifest.json")
    parser.add_argument("--backfill-hours", type=int, default=DEFAULT_BACKFILL_HOURS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="give up on a failing run after this many tries")
    parser.add_argument("--retry-failed", action="store_true", help="ignore --max-attempts and retry every failed run")
    parser.add_argument("--loop", action="store_true", help="keep running instead of exiting after one pass")
    parser.add_argument("--interval-min", type=float, default=10.0, help="minutes between passes with --loop")
    parser.add_argument("--now", default=None, help="override the current UTC time (testing/backfill)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bundle = load_bundle(args.bundle)  # verifies every member's sha256; once per process
    grid = build_coarse_grid()
    build = make_forecast_builder(bundle, grid)
    store = ForecastStore(args.store_root, "issued")
    status = JobStatus(args.store_root / STATUS_FILENAME)
    log.info("bundle %s (%d members), store %s", bundle.bundle_id, len(bundle.models), store.dir)

    exit_code = 0
    while True:
        try:
            result = run_locked(
                store,
                build,
                status,
                now=args.now,
                backfill_hours=args.backfill_hours,
                max_attempts=args.max_attempts,
                retry_failed=args.retry_failed,
            )
            log.info(
                "pass done: stored=%d failed=%d incomplete=%d skipped=%d%s",
                result.count("stored"), result.count("failed"), result.count("incomplete"),
                result.count("skipped_existing") + result.count("skipped_max_attempts"),
                " (locked)" if result.locked else "",
            )
            exit_code = 0 if result.ok else 1
        except Exception:  # noqa: BLE001 - in --loop mode a bad pass must not kill the process
            log.exception("pass crashed")
            exit_code = 1
            if not args.loop:
                raise
        if not args.loop:
            return exit_code
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    sys.exit(main())
