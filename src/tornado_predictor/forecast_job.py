"""
The hourly forecast job: finds recent HRRR runs that have no stored forecast
yet, runs the production ensemble on each, and writes the result to the
forecast store. A separate process from the API (the API only reads the
store), so a restart or a second API worker never loses or duplicates work.

One invocation (`run_once`) looks at the last `backfill_hours + 1` hourly
init times, newest first:
  - already stored             -> skipped (idempotent; re-running is a no-op)
  - files not all published    -> "incomplete", retry on a later invocation.
                                  NOT a failure: HRRR files land ~init+50 min,
                                  so the newest run is routinely not there yet
  - published                  -> extract features, run the ensemble, store
  - any error while doing that -> recorded as a failure and the remaining
                                  candidates still run (an HRRR pull fails
                                  ~3-5% of the time: missing .idx, EOF, ...)
Newest first means the most useful forecast appears before backfill finishes.

Failed runs are retried on later invocations, but only up to `max_attempts`
each: an old run that is permanently broken on AWS (missing .idx) must not be
re-pulled every hour until it ages out of the window. Attempt counts live in
the status file; `retry_failed=True` ignores them.

The status file (`<store root>/status.json`) is what GET /status exposes. Its
`consecutive_failures` counts consecutive *jobs* that attempted work, failed at
least once and stored nothing; a job that stores a forecast, or that has
nothing to do and nothing wrong, resets it to 0. A job that only found the
newest run unpublished is not a failure.

All network/model work is injected (`is_complete`, `build_forecast`) so the
logic is tested without either. `make_forecast_builder` wires the real path:
live_features.extract_live_inputs -> ensemble.predict_ensemble -> Forecast.

No tiers and no calibration here: the store keeps raw ensemble output
(see forecast_store.py).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from tornado_predictor.ensemble import EnsembleBundle, predict_ensemble
from tornado_predictor.forecast_store import (
    Forecast,
    ForecastStore,
    _iso_z,
    _naive_utc,
    format_forecast_id,
)
from tornado_predictor.grid import HrrrCoarseGrid
from tornado_predictor.live_features import PUBLISH_LATENCY_MINUTES, extract_live_inputs, run_is_complete

log = logging.getLogger(__name__)

DEFAULT_BACKFILL_HOURS = 6
DEFAULT_MAX_ATTEMPTS = 3
STATUS_FILENAME = "status.json"
LOCK_FILENAME = ".job.lock"
STATUS_SCHEMA_VERSION = 1
_MAX_RECENT_FAILURES = 20
_FAILED_RUN_RETENTION = pd.Timedelta(days=7)

STORED = "stored"
SKIPPED_EXISTING = "skipped_existing"
SKIPPED_MAX_ATTEMPTS = "skipped_max_attempts"
INCOMPLETE = "incomplete"
FAILED = "failed"


@dataclass(frozen=True)
class RunOutcome:
    init_time: pd.Timestamp
    status: str
    message: str | None = None


@dataclass
class JobResult:
    outcomes: list[RunOutcome] = field(default_factory=list)
    locked: bool = False  # another invocation held the lock; nothing was attempted

    def count(self, status: str) -> int:
        return sum(o.status == status for o in self.outcomes)

    @property
    def ok(self) -> bool:
        """False only if work failed and nothing was stored."""
        return not (self.count(FAILED) and not self.count(STORED))


# --- status file ---


class JobStatus:
    """Atomic JSON status file. `read()` returns the current state (defaults
    if absent); the job updates it once per invocation via `record_job`."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    @staticmethod
    def _default() -> dict:
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "last_job_at": None,
            "last_success_at": None,
            "last_success_init_time": None,
            "last_failure_at": None,
            "last_failure_init_time": None,
            "last_failure_message": None,
            "consecutive_failures": 0,
            "last_job": None,
            "recent_failures": [],
            "failed_runs": {},
        }

    def read(self) -> dict:
        if not self.path.is_file():
            return self._default()
        try:
            state = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            log.warning("status file %s is unreadable; starting from a fresh state", self.path)
            return self._default()
        return {**self._default(), **state}

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def failed_attempts(self, init_time) -> int:
        return int(self.read()["failed_runs"].get(format_forecast_id(init_time), {}).get("attempts", 0))

    def record_job(self, result: JobResult, now) -> None:
        now = _naive_utc(now)
        state = self.read()
        state["last_job_at"] = _iso_z(now)

        for o in result.outcomes:
            fid = format_forecast_id(o.init_time)
            if o.status == STORED:
                state["last_success_at"] = _iso_z(now)
                state["last_success_init_time"] = _iso_z(o.init_time)
                state["failed_runs"].pop(fid, None)
            elif o.status == FAILED:
                entry = state["failed_runs"].setdefault(fid, {"attempts": 0})
                entry.update(attempts=entry["attempts"] + 1, last_message=o.message, last_at=_iso_z(now))
                state["last_failure_at"] = _iso_z(now)
                state["last_failure_init_time"] = _iso_z(o.init_time)
                state["last_failure_message"] = o.message
                state["recent_failures"] = (
                    state["recent_failures"] + [{"init_time": _iso_z(o.init_time), "at": _iso_z(now), "message": o.message}]
                )[-_MAX_RECENT_FAILURES:]

        cutoff = now - _FAILED_RUN_RETENTION
        state["failed_runs"] = {k: v for k, v in state["failed_runs"].items() if _id_time(k) >= cutoff}

        attempted = result.count(STORED) + result.count(FAILED)
        if result.count(FAILED) and not result.count(STORED):
            state["consecutive_failures"] += 1
        elif attempted or not result.count(INCOMPLETE):
            # Stored something, or nothing to do and nothing wrong. A job that only
            # found its newest run unpublished leaves the counter alone.
            state["consecutive_failures"] = 0
        state["last_job"] = {
            "at": _iso_z(now),
            "stored": result.count(STORED),
            "failed": result.count(FAILED),
            "incomplete": result.count(INCOMPLETE),
            "skipped": result.count(SKIPPED_EXISTING) + result.count(SKIPPED_MAX_ATTEMPTS),
        }
        self._write(state)


def _id_time(forecast_id: str) -> pd.Timestamp:
    return pd.Timestamp(forecast_id[:-1] + ":00:00")


# --- lock ---


@contextmanager
def job_lock(path: Path | str) -> Iterator[bool]:
    """Non-blocking exclusive lock. Yields True if acquired, False if another
    process holds it (so overlapping cron/launchd invocations exit cleanly
    instead of double-pulling). The OS drops the lock if the process dies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- the job ---


def candidate_inits(now, backfill_hours: int) -> list[pd.Timestamp]:
    """Hourly init times, newest first: floor(now - publish latency), then
    `backfill_hours` earlier ones (so backfill_hours + 1 candidates)."""
    if backfill_hours < 0:
        raise ValueError(f"backfill_hours must be >= 0, got {backfill_hours}")
    newest = (_naive_utc(now) - pd.Timedelta(minutes=PUBLISH_LATENCY_MINUTES)).floor("h")
    return [newest - pd.Timedelta(hours=h) for h in range(backfill_hours + 1)]


def run_once(
    store: ForecastStore,
    build_forecast: Callable[[pd.Timestamp], Forecast],
    status: JobStatus | None = None,
    *,
    now=None,
    backfill_hours: int = DEFAULT_BACKFILL_HOURS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_failed: bool = False,
    is_complete: Callable[[pd.Timestamp], bool] = run_is_complete,
) -> JobResult:
    """One pass over the candidate runs (see module docstring). Never raises
    for a per-run problem; those become FAILED outcomes. `status`, if given,
    is updated once at the end."""
    clock_injected = now is not None
    now = _naive_utc(pd.Timestamp.now(tz="UTC") if now is None else now)
    result = JobResult()
    for init in candidate_inits(now, backfill_hours):
        if store.has(init):
            result.outcomes.append(RunOutcome(init, SKIPPED_EXISTING))
            continue
        if status is not None and not retry_failed and status.failed_attempts(init) >= max_attempts:
            result.outcomes.append(
                RunOutcome(init, SKIPPED_MAX_ATTEMPTS, f"failed {max_attempts} times already; use retry_failed to try again")
            )
            continue
        try:
            if not is_complete(init):
                log.info("%s: not fully published yet, will retry later", init)
                result.outcomes.append(RunOutcome(init, INCOMPLETE))
                continue
            forecast = build_forecast(init)
            store.put(forecast)
        except FileExistsError:
            # Another writer stored it between has() and put(): the goal is met.
            result.outcomes.append(RunOutcome(init, SKIPPED_EXISTING))
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the rest
            message = f"{type(e).__name__}: {e}"
            log.error("%s: failed: %s", init, message)
            result.outcomes.append(RunOutcome(init, FAILED, message))
        else:
            log.info("%s: stored", init)
            result.outcomes.append(RunOutcome(init, STORED))
    if status is not None:
        # Stamp the status with when the job finished (a pass can take minutes), unless
        # the caller injected a clock, in which case that clock is the only time there is.
        status.record_job(result, now if clock_injected else pd.Timestamp.now(tz="UTC"))
    return result


def run_locked(store: ForecastStore, *args, **kwargs) -> JobResult:
    """run_once under job_lock; if another invocation holds it, returns an
    empty JobResult with locked=True without touching anything."""
    with job_lock(store.root / LOCK_FILENAME) as acquired:
        if not acquired:
            log.info("another forecast job holds the lock; exiting")
            return JobResult(locked=True)
        return run_once(store, *args, **kwargs)


# --- real forecast builder ---


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def trained_years_from_manifest(manifest: dict) -> dict | None:
    years = manifest.get("training_data", {}).get("years")
    return {"first": int(min(years)), "last": int(max(years))} if years else None


def make_forecast_builder(
    bundle: EnsembleBundle, grid: HrrrCoarseGrid, *, commit: str | None = None
) -> Callable[[pd.Timestamp], Forecast]:
    """The real init_time -> Forecast path: pull and pool the live features,
    run the whole ensemble once on both bins, package the raw mean and spread."""
    commit = git_commit() if commit is None else commit
    years = trained_years_from_manifest(bundle.manifest)

    def build(init_time: pd.Timestamp) -> Forecast:
        live = extract_live_inputs(init_time, grid, bundle.feature_names)
        pred = predict_ensemble(bundle, live.X, feature_names=live.feature_names)
        return Forecast(
            init_time=live.init_time,
            mean=pred.mean,
            spread=pred.spread,
            bundle_id=bundle.bundle_id,
            n_members=len(bundle.models),
            feature_names=live.feature_names,
            bins=live.bins,
            grid_id=grid.grid_id,
            trained_years=years,
            pull_seconds=live.pull_seconds,
            generated_at=pd.Timestamp.now(tz="UTC").tz_localize(None),
            git_commit=commit,
        )

    return build
