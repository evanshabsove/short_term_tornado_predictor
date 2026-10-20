import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from tornado_predictor import forecast_store
from tornado_predictor.forecast_store import (
    Forecast,
    ForecastStore,
    ForecastStoreError,
    RetentionPolicy,
    format_forecast_id,
    parse_forecast_id,
)
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.time_bins import bin_valid_time_range

SHAPE = (2, 5, 7)


def make_forecast(init="2026-10-19 01:00", seed=0, **overrides) -> Forecast:
    rng = np.random.default_rng(seed)
    init = pd.Timestamp(init)
    kwargs = dict(
        init_time=init,
        mean=rng.random(SHAPE).astype("float32"),
        spread=rng.random(SHAPE).astype("float32") * 0.1,
        bundle_id="abc123def456",
        n_members=5,
        feature_names=["cape_mean", "cape_max"],
        bins=[
            {"bin_index": b, "fxx": 4 * b, "valid_start": lo, "valid_end": hi}
            for b, (lo, hi) in enumerate(bin_valid_time_range(init, b) for b in range(2))
        ],
        grid_id="0123456789ab",
        trained_years={"first": 2014, "last": 2025},
        pull_seconds=31.4159,
        generated_at=pd.Timestamp("2026-10-19 02:03:11"),
        git_commit="deadbeef",
    )
    kwargs.update(overrides)
    return Forecast(**kwargs)


@pytest.fixture
def store(tmp_path):
    return ForecastStore(tmp_path)


# --- ids ---


def test_id_round_trip():
    ts = pd.Timestamp("2026-10-19 01:00")
    assert format_forecast_id(ts) == "2026-10-19T01Z"
    assert parse_forecast_id("2026-10-19T01Z") == ts


def test_format_id_accepts_tz_aware_utc():
    assert format_forecast_id(pd.Timestamp("2026-10-19 01:00", tz="UTC")) == "2026-10-19T01Z"


def test_format_id_rejects_off_the_hour():
    with pytest.raises(ValueError, match="on the hour"):
        format_forecast_id(pd.Timestamp("2026-10-19 01:30"))


@pytest.mark.parametrize(
    "bad", ["2026-10-19T01", "2026-10-19 01Z", "../etc/passwd", "2026-13-19T01Z", "2026-10-19T25Z", "", None, 5]
)
def test_parse_id_is_strict(bad):
    with pytest.raises(ValueError):
        parse_forecast_id(bad)


# --- put / get ---


def test_round_trip(store):
    f = make_forecast()
    store.put(f)
    got = store.get("2026-10-19 01:00")
    np.testing.assert_array_equal(got.mean, f.mean)
    np.testing.assert_array_equal(got.spread, f.spread)
    assert got.mean.dtype == np.float32
    m = got.meta
    assert m["id"] == "2026-10-19T01Z" and m["namespace"] == "issued"
    assert m["init_time"] == "2026-10-19T01:00:00Z"
    assert m["generated_at"] == "2026-10-19T02:03:11Z"
    assert m["bundle_id"] == "abc123def456" and m["n_members"] == 5
    assert m["trained_years"] == {"first": 2014, "last": 2025}
    assert m["shape"] == [5, 7]
    assert m["bins"][0] == {
        "bin_index": 0, "fxx": 0, "valid_start": "2026-10-19T01:00:00Z", "valid_end": "2026-10-19T05:00:00Z",
    }
    assert m["bins"][1]["valid_start"] == "2026-10-19T05:00:00Z"
    assert m["pull_seconds"] == 31.42
    assert m["grid_id"] == "0123456789ab" and m["git_commit"] == "deadbeef"


def test_get_accepts_id_string_equivalent_timestamp(store):
    store.put(make_forecast())
    assert store.get(parse_forecast_id("2026-10-19T01Z")).meta["id"] == "2026-10-19T01Z"


def test_get_missing_raises_keyerror(store):
    with pytest.raises(KeyError):
        store.get("2026-10-19 01:00")
    with pytest.raises(KeyError):
        store.get_meta("2026-10-19 01:00")


def test_has(store):
    assert not store.has("2026-10-19 01:00")
    store.put(make_forecast())
    assert store.has("2026-10-19 01:00")
    assert not store.has("2026-10-19 02:00")


def test_put_never_overwrites(store):
    store.put(make_forecast(seed=0))
    with pytest.raises(FileExistsError):
        store.put(make_forecast(seed=1))
    # the original is untouched
    np.testing.assert_array_equal(store.get("2026-10-19 01:00").mean, make_forecast(seed=0).mean)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(mean=np.zeros((3, 5, 7), "float32")), "must be"),
        (dict(mean=np.zeros((2, 5), "float32")), "must be"),
        (dict(spread=np.zeros((2, 5, 8), "float32")), "shapes differ"),
        (dict(mean=np.full(SHAPE, np.nan, "float32")), "non-finite"),
        (dict(spread=np.full(SHAPE, np.inf, "float32")), "non-finite"),
    ],
)
def test_put_rejects_bad_arrays(store, overrides, match):
    with pytest.raises(ValueError, match=match):
        store.put(make_forecast(**overrides))
    assert store.list() == []


# --- atomicity ---


def test_failed_write_leaves_nothing_visible(store, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError, match="disk full"):
        store.put(make_forecast())
    assert store.list() == []
    assert not store.has("2026-10-19 01:00")
    assert list(store.dir.iterdir()) == []  # temp dir cleaned up too


def test_failed_write_does_not_block_a_retry(store, monkeypatch):
    real = np.savez_compressed
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        store.put(make_forecast())
    monkeypatch.setattr(np, "savez_compressed", real)
    store.put(make_forecast())
    assert store.has("2026-10-19 01:00")


def test_partial_directories_are_invisible(store):
    store.put(make_forecast("2026-10-19 01:00"))
    # a crashed writer's temp dir, and a bare directory with a valid-looking name but no meta
    (store.dir / ".tmp-2026-10-19T02Z-1-abcd1234").mkdir()
    (store.dir / ".tmp-2026-10-19T02Z-1-abcd1234" / "forecast.npz").write_bytes(b"junk")
    (store.dir / "2026-10-19T03Z").mkdir()
    assert [m["id"] for m in store.list()] == ["2026-10-19T01Z"]
    assert store.latest().meta["id"] == "2026-10-19T01Z"


def test_stale_temp_dirs_are_cleaned_on_next_put(store):
    store.put(make_forecast("2026-10-19 01:00"))
    old = store.dir / ".tmp-leftover"
    fresh = store.dir / ".tmp-in-flight"
    old.mkdir()
    fresh.mkdir()
    long_ago = time.time() - 2 * forecast_store._STALE_TMP_SECONDS
    os.utime(old, (long_ago, long_ago))
    store.put(make_forecast("2026-10-19 02:00"))
    assert not old.exists()
    assert fresh.exists()  # could belong to a live concurrent writer: left alone


# --- latest / list ---


def test_latest_and_list_ordering(store):
    assert store.latest() is None and store.latest_meta() is None and store.list() == []
    for h in (3, 1, 2):  # inserted out of order
        store.put(make_forecast(f"2026-10-19 0{h}:00", seed=h))
    assert store.latest().meta["id"] == "2026-10-19T03Z"
    assert store.latest_meta()["id"] == "2026-10-19T03Z"
    assert [m["id"] for m in store.list()] == ["2026-10-19T03Z", "2026-10-19T02Z", "2026-10-19T01Z"]


def test_list_limit_and_since(store):
    for h in range(1, 6):
        store.put(make_forecast(f"2026-10-19 0{h}:00", seed=h))
    assert [m["id"] for m in store.list(limit=2)] == ["2026-10-19T05Z", "2026-10-19T04Z"]
    assert [m["id"] for m in store.list(since="2026-10-19 04:00")] == ["2026-10-19T05Z", "2026-10-19T04Z"]
    assert [m["id"] for m in store.list(since="2026-10-19 04:00", limit=1)] == ["2026-10-19T05Z"]


def test_list_spans_days_in_time_order(store):
    store.put(make_forecast("2026-10-18 23:00"))
    store.put(make_forecast("2026-10-19 00:00"))
    assert [m["id"] for m in store.list()] == ["2026-10-19T00Z", "2026-10-18T23Z"]


# --- namespaces ---


def test_namespaces_are_isolated(tmp_path):
    issued = ForecastStore(tmp_path, "issued")
    replay = ForecastStore(tmp_path, "replay")
    issued.put(make_forecast("2026-10-19 01:00"))
    replay.put(make_forecast("2021-12-10 21:00"))
    assert [m["id"] for m in issued.list()] == ["2026-10-19T01Z"]
    assert [m["id"] for m in replay.list()] == ["2021-12-10T21Z"]
    assert replay.get("2021-12-10 21:00").meta["namespace"] == "replay"
    assert not issued.has("2021-12-10 21:00")
    # the same init time can exist in both without colliding
    replay.put(make_forecast("2026-10-19 01:00", seed=9))
    assert issued.get("2026-10-19 01:00").meta["namespace"] == "issued"


def test_unknown_namespace_rejected(tmp_path):
    with pytest.raises(ValueError, match="namespace"):
        ForecastStore(tmp_path, "scratch")


# --- corruption ---


def test_corrupt_npz_raises_clear_error(store):
    store.put(make_forecast())
    (store.dir / "2026-10-19T01Z" / "forecast.npz").write_bytes(b"not an npz")
    with pytest.raises(ForecastStoreError, match="unreadable forecast arrays"):
        store.get("2026-10-19 01:00")


def test_missing_npz_raises_clear_error(store):
    store.put(make_forecast())
    (store.dir / "2026-10-19T01Z" / "forecast.npz").unlink()
    with pytest.raises(ForecastStoreError, match="unreadable forecast arrays"):
        store.get("2026-10-19 01:00")


def test_corrupt_meta_raises_clear_error(store):
    store.put(make_forecast())
    (store.dir / "2026-10-19T01Z" / "meta.json").write_text("{not json")
    with pytest.raises(ForecastStoreError, match="unreadable metadata"):
        store.get_meta("2026-10-19 01:00")


def test_shape_disagreement_between_meta_and_arrays_raises(store):
    store.put(make_forecast())
    path = store.dir / "2026-10-19T01Z" / "meta.json"
    meta = json.loads(path.read_text())
    meta["shape"] = [9, 9]
    path.write_text(json.dumps(meta))
    with pytest.raises(ForecastStoreError, match="disagree"):
        store.get("2026-10-19 01:00")


# --- retention ---


def test_default_policy_never_deletes(store):
    store.put(make_forecast("2014-09-01 00:00"))
    assert store.prune(RetentionPolicy()) == []
    assert store.has("2014-09-01 00:00")


def test_prune_deletes_only_past_max_age(store):
    for init in ("2026-10-01 00:00", "2026-10-10 00:00", "2026-10-18 00:00"):
        store.put(make_forecast(init))
    removed = store.prune(RetentionPolicy(max_age_days=7), now="2026-10-19 00:00")
    assert removed == ["2026-10-01T00Z", "2026-10-10T00Z"]
    assert [m["id"] for m in store.list()] == ["2026-10-18T00Z"]
    assert list(p.name for p in store.dir.iterdir()) == ["2026-10-18T00Z"]  # no temp dirs left behind


def test_prune_is_scoped_to_its_namespace(tmp_path):
    issued, replay = ForecastStore(tmp_path, "issued"), ForecastStore(tmp_path, "replay")
    issued.put(make_forecast("2020-01-01 00:00"))
    replay.put(make_forecast("2020-01-01 00:00"))
    replay.prune(RetentionPolicy(max_age_days=1), now="2026-10-19 00:00")
    assert issued.has("2020-01-01 00:00") and not replay.has("2020-01-01 00:00")


# --- grid_id ---


def test_grid_id_is_stable_and_geometry_dependent():
    grid = build_coarse_grid()
    assert grid.grid_id == build_coarse_grid().grid_id
    assert len(grid.grid_id) == 12
    assert build_coarse_grid(stride=14).grid_id != grid.grid_id
