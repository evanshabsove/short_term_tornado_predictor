import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tornado_predictor import forecast_job
from tornado_predictor.ensemble import DEFAULT_BUNDLE_MANIFEST_PATH, EnsemblePrediction, load_bundle
from tornado_predictor.forecast_job import (
    FAILED,
    INCOMPLETE,
    SKIPPED_EXISTING,
    SKIPPED_MAX_ATTEMPTS,
    STORED,
    JobResult,
    JobStatus,
    RunOutcome,
    candidate_inits,
    job_lock,
    make_forecast_builder,
    run_locked,
    run_once,
    trained_years_from_manifest,
)
from tornado_predictor.forecast_store import Forecast, ForecastStore
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.time_bins import bin_valid_time_range

SHAPE = (2, 4, 6)
NOW = "2026-10-19 02:40"  # newest candidate = floor(02:40 - 50 min) = 01:00


def ts(s):
    return pd.Timestamp(s)


def fake_forecast(init) -> Forecast:
    init = ts(init)
    return Forecast(
        init_time=init,
        mean=np.full(SHAPE, 0.01, "float32"),
        spread=np.zeros(SHAPE, "float32"),
        bundle_id="b" * 12,
        n_members=5,
        feature_names=["cape_mean"],
        bins=[{"bin_index": b, "fxx": 4 * b, "valid_start": lo, "valid_end": hi}
              for b, (lo, hi) in enumerate(bin_valid_time_range(init, b) for b in range(2))],
        grid_id="g" * 12,
    )


class Builder:
    """Records calls; `fail` maps init -> exception to raise."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def __call__(self, init):
        self.calls.append(init)
        if init in self.fail:
            raise self.fail[init]
        return fake_forecast(init)


@pytest.fixture
def store(tmp_path):
    return ForecastStore(tmp_path)


@pytest.fixture
def status(tmp_path):
    return JobStatus(tmp_path / "status.json")


def everything_published(init):
    return True


# --- candidate_inits ---


def test_candidate_inits_newest_first_with_latency_floor():
    got = candidate_inits(NOW, 2)
    assert got == [ts("2026-10-19 01:00"), ts("2026-10-19 00:00"), ts("2026-10-18 23:00")]


def test_candidate_inits_zero_backfill_is_one_candidate_and_negative_rejected():
    assert candidate_inits(NOW, 0) == [ts("2026-10-19 01:00")]
    with pytest.raises(ValueError):
        candidate_inits(NOW, -1)


def test_candidate_inits_accepts_tz_aware_now():
    assert candidate_inits(pd.Timestamp("2026-10-18 21:40", tz="US/Central"), 0) == [ts("2026-10-19 01:00")]


# --- run_once ---


def test_stores_published_runs_newest_first(store, status):
    build = Builder()
    result = run_once(store, build, status, now=NOW, backfill_hours=2, is_complete=everything_published)
    assert build.calls == [ts("2026-10-19 01:00"), ts("2026-10-19 00:00"), ts("2026-10-18 23:00")]
    assert [o.status for o in result.outcomes] == [STORED] * 3
    assert [m["id"] for m in store.list()] == ["2026-10-19T01Z", "2026-10-19T00Z", "2026-10-18T23Z"]


def test_second_run_is_a_noop(store, status):
    run_once(store, Builder(), status, now=NOW, backfill_hours=2, is_complete=everything_published)
    build = Builder()
    result = run_once(store, build, status, now=NOW, backfill_hours=2, is_complete=everything_published)
    assert build.calls == []
    assert [o.status for o in result.outcomes] == [SKIPPED_EXISTING] * 3
    assert len(store.list()) == 3


def test_skipping_stored_run_does_not_probe_the_network(store):
    store.put(fake_forecast("2026-10-19 01:00"))
    probed = []
    run_once(store, Builder(), now=NOW, backfill_hours=0, is_complete=lambda i: probed.append(i) or True)
    assert probed == []


def test_unpublished_run_is_incomplete_not_a_failure_and_older_ones_still_run(store, status):
    build = Builder()
    newest = ts("2026-10-19 01:00")
    result = run_once(store, build, status, now=NOW, backfill_hours=1, is_complete=lambda i: i != newest)
    assert [(o.init_time, o.status) for o in result.outcomes] == [(newest, INCOMPLETE), (ts("2026-10-19 00:00"), STORED)]
    assert build.calls == [ts("2026-10-19 00:00")]
    assert not store.has(newest)
    s = status.read()
    assert s["last_failure_at"] is None and s["recent_failures"] == [] and s["consecutive_failures"] == 0
    assert result.ok


def test_failure_is_recorded_and_remaining_candidates_still_run(store, status):
    boom = ts("2026-10-19 01:00")
    build = Builder(fail={boom: EOFError("Premature end of file")})
    result = run_once(store, build, status, now=NOW, backfill_hours=2, is_complete=everything_published)
    assert [o.status for o in result.outcomes] == [FAILED, STORED, STORED]
    assert "EOFError: Premature end of file" in result.outcomes[0].message
    assert not store.has(boom) and store.has("2026-10-19 00:00")
    s = status.read()
    assert s["last_failure_init_time"] == "2026-10-19T01:00:00Z"
    assert "EOFError" in s["last_failure_message"]
    assert s["failed_runs"]["2026-10-19T01Z"]["attempts"] == 1
    assert result.ok  # stored something


def test_error_in_completeness_check_is_a_failure_not_a_crash(store):
    def check(init):
        raise RuntimeError("network down")

    result = run_once(store, Builder(), now=NOW, backfill_hours=0, is_complete=check)
    assert result.outcomes[0].status == FAILED and "network down" in result.outcomes[0].message


def test_store_put_race_counts_as_already_stored(store):
    def build(init):
        store.put(fake_forecast(init))  # a concurrent writer wins before our put
        return fake_forecast(init)

    result = run_once(store, build, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert result.outcomes[0].status == SKIPPED_EXISTING


def test_nan_features_are_failures_never_imputed(store, status):
    def build(init):
        raise ValueError("3 NaN values in base features")

    result = run_once(store, build, status, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert result.outcomes[0].status == FAILED and store.list() == []
    assert not result.ok


# --- retry limits ---


def test_failed_run_is_retried_until_max_attempts_then_skipped(store, status):
    bad = ts("2026-10-19 01:00")
    build = Builder(fail={bad: OSError("no idx")})
    for _ in range(3):
        run_once(store, build, status, now=NOW, backfill_hours=0, max_attempts=3, is_complete=everything_published)
    assert build.calls == [bad] * 3
    result = run_once(store, build, status, now=NOW, backfill_hours=0, max_attempts=3, is_complete=everything_published)
    assert result.outcomes[0].status == SKIPPED_MAX_ATTEMPTS
    assert build.calls == [bad] * 3  # not attempted a 4th time
    result = run_once(store, build, status, now=NOW, backfill_hours=0, max_attempts=3, retry_failed=True,
                      is_complete=everything_published)
    assert result.outcomes[0].status == FAILED and len(build.calls) == 4


def test_success_clears_a_runs_failure_count(store, status):
    bad = ts("2026-10-19 01:00")
    run_once(store, Builder(fail={bad: OSError("x")}), status, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert status.failed_attempts(bad) == 1
    run_once(store, Builder(), status, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert status.failed_attempts(bad) == 0 and store.has(bad)


# --- status file ---


def test_status_defaults_when_absent(status):
    s = status.read()
    assert s["consecutive_failures"] == 0 and s["last_success_at"] is None and s["last_job"] is None


def test_consecutive_failures_counts_failed_jobs_and_resets_on_success(store, status):
    bad = ts("2026-10-19 01:00")
    for i in (1, 2):
        run_once(store, Builder(fail={bad: OSError("x")}), status, now=NOW, backfill_hours=0,
                 is_complete=everything_published, retry_failed=True)
        assert status.read()["consecutive_failures"] == i
    run_once(store, Builder(), status, now=NOW, backfill_hours=0, is_complete=everything_published)
    s = status.read()
    assert s["consecutive_failures"] == 0
    assert s["last_success_init_time"] == "2026-10-19T01:00:00Z"
    assert s["last_failure_message"]  # history is kept even after recovery
    assert s["last_job"]["stored"] == 1


def test_unpublished_newest_run_leaves_counter_alone(store, status):
    bad = ts("2026-10-19 01:00")
    run_once(store, Builder(fail={bad: OSError("x")}), status, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert status.read()["consecutive_failures"] == 1
    run_once(store, Builder(), status, now="2026-10-19 03:40", backfill_hours=0, is_complete=lambda i: False)
    assert status.read()["consecutive_failures"] == 1  # neither a failure nor a recovery


def test_idle_job_with_everything_stored_resets_counter(store, status):
    bad = ts("2026-10-19 01:00")
    run_once(store, Builder(fail={bad: OSError("x")}), status, now=NOW, backfill_hours=0, is_complete=everything_published)
    store.put(fake_forecast(bad))  # someone else filled the gap
    run_once(store, Builder(), status, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert status.read()["consecutive_failures"] == 0


def test_recent_failures_are_bounded(status):
    for i in range(30):
        init = ts("2026-10-01 00:00") + pd.Timedelta(hours=i)
        status.record_job(JobResult([RunOutcome(init, FAILED, f"m{i}")]), now="2026-10-02 12:00")
    s = status.read()
    assert len(s["recent_failures"]) == 20 and s["recent_failures"][-1]["message"] == "m29"


def test_failed_run_entries_expire_after_a_week(status):
    old, recent = ts("2026-10-01 00:00"), ts("2026-10-18 00:00")
    status.record_job(JobResult([RunOutcome(old, FAILED, "x"), RunOutcome(recent, FAILED, "y")]), now="2026-10-19 00:00")
    assert list(status.read()["failed_runs"]) == ["2026-10-18T00Z"]


def test_status_write_is_atomic_and_leaves_no_temp_files(store, status, tmp_path):
    run_once(store, Builder(), status, now=NOW, backfill_hours=1, is_complete=everything_published)
    assert json.loads(status.path.read_text())["schema_version"] == 1
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_corrupt_status_file_starts_fresh_instead_of_crashing(status):
    status.path.write_text("{not json")
    assert status.read()["consecutive_failures"] == 0
    status.record_job(JobResult([]), now=NOW)  # and can be overwritten
    assert json.loads(status.path.read_text())["last_job_at"]


def test_status_field_names_match_the_api_contract(status):
    """api/README.md's GET /status `scheduler` block."""
    s = status.read()
    for key in ("last_success_at", "last_failure_at", "last_failure_message", "consecutive_failures"):
        assert key in s


# --- lock ---


def test_second_job_cannot_take_the_lock(tmp_path):
    path = tmp_path / ".job.lock"
    with job_lock(path) as first:
        assert first is True
        with job_lock(path) as second:
            assert second is False
    with job_lock(path) as again:  # released afterwards
        assert again is True


def test_run_locked_does_nothing_when_locked(store, status):
    build = Builder()
    with job_lock(store.root / ".job.lock"):
        result = run_locked(store, build, status, now=NOW, backfill_hours=1, is_complete=everything_published)
    assert result.locked and result.outcomes == [] and build.calls == []
    assert status.read()["last_job_at"] is None  # a locked-out job leaves no trace
    result = run_locked(store, build, status, now=NOW, backfill_hours=0, is_complete=everything_published)
    assert not result.locked and result.count(STORED) == 1


# --- real builder wiring (extract + predict faked) ---


class FakeBundle:
    feature_names = ["cape_mean", "cape_max"]
    bundle_id = "abc123def456"
    models = [object()] * 5
    manifest = {"training_data": {"years": [2014, 2020, 2025]}}


class FakeLive:
    pull_seconds = 12.5

    def __init__(self, X, init_time, feature_names, bins):
        self.X, self.init_time, self.feature_names, self.bins = X, init_time, feature_names, bins


def test_builder_packages_extraction_and_ensemble_output(monkeypatch):
    grid = build_coarse_grid()
    seen = {}

    def fake_extract(init, g, names):
        seen["extract"] = (init, g is grid, names)
        X = np.zeros((2, 2, grid.n_rows, grid.n_cols), "float32")
        bins = [{"bin_index": b, "fxx": 4 * b, "valid_start": lo, "valid_end": hi}
                for b, (lo, hi) in enumerate(bin_valid_time_range(init, b) for b in range(2))]
        return FakeLive(X, init, names, bins)

    monkeypatch.setattr(forecast_job, "extract_live_inputs", fake_extract)

    def fake_predict(bundle, X, *, feature_names):
        seen["predict"] = (X.shape, feature_names)
        return EnsemblePrediction(mean=np.full((2, grid.n_rows, grid.n_cols), 0.02, "float32"),
                                  spread=np.full((2, grid.n_rows, grid.n_cols), 0.005, "float32"))

    monkeypatch.setattr(forecast_job, "predict_ensemble", fake_predict)

    f = make_forecast_builder(FakeBundle(), grid, commit="cafe123")(ts("2026-10-19 01:00"))
    assert seen["extract"] == (ts("2026-10-19 01:00"), True, ["cape_mean", "cape_max"])
    assert seen["predict"] == ((2, 2, grid.n_rows, grid.n_cols), ["cape_mean", "cape_max"])
    assert f.bundle_id == "abc123def456" and f.n_members == 5 and f.grid_id == grid.grid_id
    assert f.trained_years == {"first": 2014, "last": 2025}
    assert f.git_commit == "cafe123" and f.pull_seconds == 12.5
    assert f.mean.shape == (2, grid.n_rows, grid.n_cols)


def test_trained_years_from_manifest():
    assert trained_years_from_manifest({"training_data": {"years": [2016, 2014, 2025]}}) == {"first": 2014, "last": 2025}
    assert trained_years_from_manifest({}) is None


# --- end to end against the real archive and bundle (opt in) ---


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1" or not Path(DEFAULT_BUNDLE_MANIFEST_PATH).exists(),
    reason="set RUN_NETWORK_TESTS=1 and have the production bundle locally",
)
def test_end_to_end_on_real_run(tmp_path):
    bundle = load_bundle()
    grid = build_coarse_grid()
    store = ForecastStore(tmp_path)
    status = JobStatus(tmp_path / "status.json")
    build = make_forecast_builder(bundle, grid)
    result = run_once(store, build, status, backfill_hours=3)
    assert result.count(STORED) >= 1, [(o.init_time, o.status, o.message) for o in result.outcomes]
    got = store.latest()
    assert got.mean.shape == (2, grid.n_rows, grid.n_cols) and np.isfinite(got.mean).all()
    assert 0.0 <= got.mean.min() and got.mean.max() <= 1.0
    again = run_once(store, build, status, backfill_hours=3)
    assert again.count(STORED) == 0  # idempotent


# --- replays ---


def test_run_replays_stores_in_the_replay_namespace_and_skips_existing(tmp_path):
    from tornado_predictor.forecast_job import run_replays

    replay = ForecastStore(tmp_path, "replay")
    build = Builder()
    out = run_replays(replay, build, ["2021-12-10T21Z", ts("2021-12-10 22:00")])
    assert [o.status for o in out] == [STORED, STORED]
    assert [m["id"] for m in replay.list()] == ["2021-12-10T22Z", "2021-12-10T21Z"]
    assert replay.get("2021-12-10 21:00").meta["namespace"] == "replay"
    assert not ForecastStore(tmp_path, "issued").list()
    again = run_replays(replay, build, ["2021-12-10T21Z"])
    assert [o.status for o in again] == [SKIPPED_EXISTING] and len(build.calls) == 2  # nothing recomputed


def test_run_replays_records_a_failure_and_continues(tmp_path):
    from tornado_predictor.forecast_job import run_replays

    replay = ForecastStore(tmp_path, "replay")
    build = Builder(fail={ts("2021-12-10 21:00"): EOFError("Premature end of file")})
    out = run_replays(replay, build, ["2021-12-10T21Z", "2021-12-10T22Z"])
    assert [o.status for o in out] == [FAILED, STORED]
    assert "EOFError" in out[0].message and not replay.has("2021-12-10 21:00")


def test_run_replays_refuses_the_issued_namespace(tmp_path):
    from tornado_predictor.forecast_job import run_replays

    with pytest.raises(ValueError, match="replay"):
        run_replays(ForecastStore(tmp_path, "issued"), Builder(), ["2021-12-10T21Z"])
