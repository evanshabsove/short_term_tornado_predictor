"""Payload builders (no Flask): shapes, tiering, status, tier-set integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tornado_predictor import forecast_api as api
from tornado_predictor import tiers as T
from tornado_predictor.cell_mask import us_trusted_mask
from tornado_predictor.forecast_api import (
    BadRequest,
    NoForecast,
    NotFound,
    ServerFault,
    ServingContext,
    TierConfig,
    build_forecast,
    build_forecasts,
    build_grid,
    build_status,
    build_tiers,
    parse_limit,
    parse_since,
    read_scheduler_status,
)
from tornado_predictor.forecast_store import ForecastStore

NOW = pd.Timestamp("2026-10-19 02:30")  # 90 min after the 01Z run's init


@pytest.fixture
def ctx(store_root, country, grid, tier_config):
    return ServingContext.create(store_root, country=country, grid=grid, tier_config=tier_config, now=lambda: NOW)


# --- argument parsing ---


@pytest.mark.parametrize("value", ["2026-10-19T01Z", "2026-10-19T01:00:00Z", "2026-10-19T01:00:00", "2026-10-19T01:00",
                                   "2026-10-19T01:00:00+00:00", "2026-10-18T20:00:00-05:00"])
def test_parse_since_accepts_ids_and_iso_datetimes(value):
    assert parse_since(value) == pd.Timestamp("2026-10-19 01:00")


@pytest.mark.parametrize("value", ["", "yesterday", "2026-10-19", "2026", "2026-13-01T00:00:00Z", "2026-10-19 01:00", "../x"])
def test_parse_since_rejects_everything_else(value):
    with pytest.raises(BadRequest):
        parse_since(value)


def test_parse_limit():
    assert parse_limit(None) == 24 and parse_limit("1") == 1 and parse_limit("168") == 168
    for bad in ("0", "169", "-1", "abc", "1.5", "", " 5"):
        with pytest.raises(BadRequest):
            parse_limit(bad)


# --- tier set integrity ---


def test_tier_config_loads_the_real_study_and_reproduces_its_id(tier_config):
    assert tier_config.tier_set_id == "ts-e2c497b6"
    assert (tier_config.operator, tier_config.radius) == ("max", 1)
    assert tier_config.cutoffs["elevated"] < tier_config.cutoffs["high"]
    assert T.tier_set_id(tier_config.operator, tier_config.radius, tier_config.cutoffs) == tier_config.tier_set_id


def test_tier_config_fails_fast_on_an_inconsistent_id(tmp_path):
    study = json.loads(api.DEFAULT_TIER_CONFIG_PATH.read_text())
    study["serving"]["cutoffs"]["high"] = 0.3  # edited cutoff, stale id
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="tier_set_id"):
        TierConfig.load(path)


def test_tier_config_missing_file_names_the_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="tier_study.py"):
        TierConfig.load(tmp_path / "nope.json")


def test_context_rejects_a_country_mask_of_the_wrong_shape(store_root, grid, tier_config):
    with pytest.raises(ValueError, match="shape"):
        ServingContext.create(store_root, country=np.zeros((5, 5), np.int8), grid=grid, tier_config=tier_config)


# --- status ---


def test_status_no_data(ctx):
    payload, etag = build_status(ctx)
    assert etag is None
    assert payload["status"] == "no_data" and payload["latest_init_time"] is None
    assert payload["latest_age_seconds"] is None and payload["stale"] is None
    assert payload["now"] == "2026-10-19T02:30:00Z"
    assert payload["scheduler"] == {"last_success_at": None, "last_failure_at": None, "last_failure_message": None, "consecutive_failures": 0}


def test_status_ok_then_stale_with_the_injected_clock(ctx, store, make_forecast):
    store.put(make_forecast("2026-10-19 01:00"))
    ok, _ = build_status(ctx)
    assert (ok["status"], ok["stale"], ok["latest_age_seconds"], ok["latest_init_time"]) == ("ok", False, 5400, "2026-10-19T01:00:00Z")
    ctx.now = lambda: pd.Timestamp("2026-10-19 04:00:01")  # init + 3h00m01s: just past 10,800 s
    stale, _ = build_status(ctx)
    assert (stale["status"], stale["stale"], stale["latest_age_seconds"]) == ("stale", True, 10801)
    ctx.now = lambda: pd.Timestamp("2026-10-19 04:00:00")  # exactly 3 h is not yet stale
    assert build_status(ctx)[0]["stale"] is False


def test_status_reads_the_scheduler_file_and_survives_a_bad_one(ctx, store_root):
    store_root.mkdir(parents=True)
    (store_root / "status.json").write_text(json.dumps({
        "last_success_at": "2026-10-19T02:03:11Z", "last_failure_at": "2026-10-19T01:00:00Z",
        "last_failure_message": "EOFError: x", "consecutive_failures": 2, "failed_runs": {"a": 1}, "recent_failures": []}))
    sched = build_status(ctx)[0]["scheduler"]
    assert sched == {"last_success_at": "2026-10-19T02:03:11Z", "last_failure_at": "2026-10-19T01:00:00Z",
                     "last_failure_message": "EOFError: x", "consecutive_failures": 2}  # internal fields are not exposed
    (store_root / "status.json").write_text("{not json")
    assert read_scheduler_status(store_root / "status.json")["consecutive_failures"] == 0


def test_status_filename_matches_the_scheduler():
    from tornado_predictor import forecast_job  # heavy import, so only here

    assert api.STATUS_FILENAME == forecast_job.STATUS_FILENAME


# --- grid ---


def test_grid_payload_shape_and_content(ctx, grid, country):
    payload, etag = build_grid(ctx)
    assert etag == grid.grid_id == payload["grid_id"]
    assert payload["shape"] == [81, 138] and payload["vertex_shape"] == [82, 139] and payload["cell_size_km"] == 39
    assert len(payload["vertices"]["lon"]) == len(payload["vertices"]["lat"]) == 82 * 139 == 11398
    assert payload["country"] == country.ravel().tolist() and len(payload["country"]) == 11178
    assert payload["country_codes"] == {"0": "water", "1": "US", "2": "Canada", "3": "Mexico", "4": "other land"}
    assert build_grid(ctx)[0] is payload  # built once


def test_grid_vertices_follow_the_documented_corner_indexing(ctx, grid):
    payload, _ = build_grid(ctx)
    lon, lat = np.array(payload["vertices"]["lon"]), np.array(payload["vertices"]["lat"])
    for row, col in [(0, 0), (40, 60), (80, 137), (12, 3)]:
        v = row * 139 + col
        idx = [v, v + 1, v + 139 + 1, v + 139]  # SW, SE, NE, NW
        poly = np.array(grid.cell_polygon(row, col).exterior.coords)[:4]
        got = np.stack([lon[idx], lat[idx]], axis=1)
        metres = np.hypot((got[:, 1] - poly[:, 1]) * 111_195, (got[:, 0] - poly[:, 0]) * 111_195 * np.cos(np.radians(poly[:, 1])))
        assert metres.max() < 7.5


# --- forecasts list ---


def test_forecasts_empty_store(ctx):
    payload, etag = build_forecasts(ctx)
    assert etag is None
    assert payload == {"schema_version": 1, "count": 0, "latest_id": None, "forecasts": []}


def test_forecasts_listing_order_limit_since_and_fields(ctx, store, make_forecast):
    for h in (1, 2, 3):
        store.put(make_forecast(f"2026-10-19 0{h}:00", bundle_id=f"b{h}"))
    payload, _ = build_forecasts(ctx)
    assert payload["count"] == 3 and payload["latest_id"] == "2026-10-19T03Z"
    assert [f["id"] for f in payload["forecasts"]] == ["2026-10-19T03Z", "2026-10-19T02Z", "2026-10-19T01Z"]
    first = payload["forecasts"][0]
    assert set(first) == {"id", "init_time", "generated_at", "age_seconds", "stale", "bundle_id", "tier_set_id"}
    assert first["tier_set_id"] == "ts-e2c497b6" and first["bundle_id"] == "b3"
    assert first["age_seconds"] == 0  # init 03:00 is after the fixed clock (02:30): age is clamped, never negative
    assert [f["id"] for f in build_forecasts(ctx, limit=2)[0]["forecasts"]] == ["2026-10-19T03Z", "2026-10-19T02Z"]
    since = build_forecasts(ctx, since=pd.Timestamp("2026-10-19 02:00"))[0]
    assert [f["id"] for f in since["forecasts"]] == ["2026-10-19T03Z", "2026-10-19T02Z"]
    # a `since` after everything leaves the list empty but still names the newest stored forecast
    late = build_forecasts(ctx, since=pd.Timestamp("2026-10-20 00:00"))[0]
    assert late["count"] == 0 and late["latest_id"] == "2026-10-19T03Z"


def test_forecast_age_in_the_listing_is_measured_from_init_time(ctx, store, make_forecast):
    store.put(make_forecast("2026-10-19 01:00"))
    item = build_forecasts(ctx)[0]["forecasts"][0]
    assert item["age_seconds"] == 5400 and item["stale"] is False  # now 02:30, init 01:00


# --- one forecast ---


def test_forecast_payload_matches_the_contract(ctx, store, make_forecast, grid):
    store.put(make_forecast("2026-10-19 01:00"))
    payload, etag = build_forecast(ctx, "2026-10-19T01Z")
    assert etag == "2026-10-19T01Z-3c02381f1611-ts-e2c497b6"
    assert list(payload)[:3] == ["schema_version", "id", "mode"]
    assert payload["mode"] == "live" and payload["id"] == "2026-10-19T01Z"
    assert payload["init_time"] == "2026-10-19T01:00:00Z" and payload["age_seconds"] == 5400 and payload["stale"] is False
    assert payload["grid_id"] == grid.grid_id and payload["shape"] == [81, 138]
    assert payload["model"] == {"bundle_id": "3c02381f1611", "n_members": 5, "trained_years": {"first": 2014, "last": 2025}}
    assert payload["tiering"] == {"tier_set_id": "ts-e2c497b6", "smoothing": {"operator": "max", "radius_cells": 1}}
    assert payload["calibrated_percent"] is None
    assert [b["bin_index"] for b in payload["bins"]] == [0, 1]
    b0, b1 = payload["bins"]
    assert (b0["valid_start"], b0["valid_end"]) == ("2026-10-19T01:00:00Z", "2026-10-19T05:00:00Z")
    assert (b1["valid_start"], b1["valid_end"]) == ("2026-10-19T05:00:00Z", "2026-10-19T09:00:00Z")
    for b in payload["bins"]:
        assert len(b["tiers"]) == len(b["smoothed_score"]) == 11178
    # raw probabilities and spread are not part of the payload
    assert "mean" not in payload and "spread" not in payload and "mean" not in b0 and "spread" not in b0


def test_forecast_tiers_use_row_major_indexing_and_the_real_cutoffs(ctx, store, make_forecast, country):
    store.put(make_forecast())
    payload, _ = build_forecast(ctx, "latest")
    t0 = np.array(payload["bins"][0]["tiers"]).reshape(81, 138)
    t1 = np.array(payload["bins"][1]["tiers"]).reshape(81, 138)
    assert payload["bins"][0]["tiers"][40 * 138 + 60] == 2  # i = row * 138 + col, row 0 southernmost
    assert (t0 == 2).sum() == 9 and (t0[39:42, 59:62] == 2).all()  # 0.25 peak, max-smoothed over 3x3
    assert (t0 == 1).sum() == 9 and (t0[49:52, 99:102] == 1).all()  # 0.15 peak -> Elevated
    assert (t1 == 2).sum() == 0 and (t1 == 1).sum() == 9 and (t1[44:47, 29:32] == 1).all()  # 0.13 -> Elevated only
    # no coverage exactly where the country code is Canada / Mexico / other land, whatever the score
    expected_cov = us_trusted_mask(country)
    for t in (t0, t1):
        assert ((t == -1) == ~expected_cov).all()
    assert t0[4, 60] == -1 and t0[3:6, 59:62].max() == -1  # the 0.30 peak inside the Canadian band stays grey
    assert ((t0 == 0) | (t0 == 1) | (t0 == 2) | (t0 == -1)).all()


def test_forecast_tiers_equal_the_study_path(ctx, store, make_forecast, tier_config, country):
    rng = np.random.default_rng(5)
    mean = (rng.random((2, 81, 138)) ** 5 * 0.4).astype(np.float32)
    store.put(make_forecast(mean=mean))
    payload, _ = build_forecast(ctx, "2026-10-19T01Z")
    cov = us_trusted_mask(country)
    for i, b in enumerate(payload["bins"]):
        smoothed = T.quantize(T.smooth(mean[i], "max", 1)).astype("float32")
        direct = T.assign_tiers(smoothed, tier_config.cutoffs, cov)
        np.testing.assert_array_equal(np.array(b["tiers"]).reshape(81, 138), direct)
        np.testing.assert_allclose(np.array(b["smoothed_score"]).reshape(81, 138), smoothed, atol=5e-5)


def test_score_exactly_at_a_cutoff_goes_up_a_tier(ctx, store, make_forecast, tier_config):
    cut = tier_config.cutoffs
    mean = np.full((2, 81, 138), 0.001, np.float32)
    mean[0, 40, 20] = cut["elevated"]  # exactly at the Elevated cutoff
    mean[0, 40, 60] = cut["high"]  # exactly at the High cutoff
    mean[0, 60, 60] = cut["elevated"] - 0.0001  # a full float16 step below: stays Low
    store.put(make_forecast(mean=mean))
    t0 = np.array(build_forecast(ctx, "latest")[0]["bins"][0]["tiers"]).reshape(81, 138)
    assert t0[40, 20] == 1 and t0[40, 60] == 2 and t0[60, 60] == 0


def test_smoothed_score_is_rounded_and_has_no_float32_noise(ctx, store, make_forecast):
    store.put(make_forecast())
    scores = build_forecast(ctx, "latest")[0]["bins"][0]["smoothed_score"]
    assert all(isinstance(s, float) for s in scores[:50])
    assert all(len(repr(s).split(".")[-1]) <= 4 for s in scores[:2000] if s)  # at most 4 decimals


def test_forecast_latest_picks_the_newest_and_empty_store_is_503(ctx, store, make_forecast):
    with pytest.raises(NoForecast):
        build_forecast(ctx, "latest")
    store.put(make_forecast("2026-10-19 01:00"))
    store.put(make_forecast("2026-10-19 02:00"))
    assert build_forecast(ctx, "latest")[0]["id"] == "2026-10-19T02Z"


def test_forecast_not_found_and_malformed_ids(ctx, store, make_forecast):
    store.put(make_forecast())
    with pytest.raises(NotFound):
        build_forecast(ctx, "2026-10-19T09Z")
    for bad in ("2026-10-19", "yesterday", "2026-10-19T01", "..%2f"):
        with pytest.raises(BadRequest):
            build_forecast(ctx, bad)


def test_grid_mismatch_is_a_server_fault_never_silently_tiered(ctx, store, make_forecast):
    store.put(make_forecast(grid_id="ffffffffffff"))
    with pytest.raises(ServerFault, match="grid_id"):
        build_forecast(ctx, "latest")


def test_shape_mismatch_is_a_server_fault(ctx, store, make_forecast):
    store.put(make_forecast(mean=np.full((2, 10, 10), 0.001, np.float32)))
    with pytest.raises(ServerFault, match="shape"):
        build_forecast(ctx, "latest")


def test_changing_the_tier_set_changes_the_id_and_etag(store_root, country, grid, tier_config, store, make_forecast):
    store.put(make_forecast())
    cut = {"elevated": tier_config.cutoffs["elevated"], "high": 0.30}
    other = TierConfig(T.tier_set_id("max", 1, cut), "max", 1, cut, tier_config.study)
    ctx2 = ServingContext.create(store_root, country=country, grid=grid, tier_config=other, now=lambda: NOW)
    p1, e1 = build_forecast(ServingContext.create(store_root, country=country, grid=grid, tier_config=tier_config, now=lambda: NOW), "latest")
    p2, e2 = build_forecast(ctx2, "latest")
    assert e1 != e2 and p1["tiering"]["tier_set_id"] != p2["tiering"]["tier_set_id"]
    assert (np.array(p1["bins"][0]["tiers"]) == 2).sum() == 9 and (np.array(p2["bins"][0]["tiers"]) == 2).sum() == 0  # 0.25 < 0.30


# --- /tiers ---


def test_tiers_payload_comes_from_the_study_file(ctx, tier_config):
    payload, etag = build_tiers(ctx)
    study = tier_config.study
    assert etag == payload["tier_set_id"] == "ts-e2c497b6"
    rows = {r["name"]: r for r in payload["tiers"]}
    assert list(rows) == ["Low", "Elevated", "High"] and [r["code"] for r in payload["tiers"]] == [0, 1, 2]
    point = study["tier_table_A"]["primary_20Z"]["point"]
    for name, key in (("Low", "low"), ("Elevated", "elevated"), ("High", "high")):
        assert rows[name]["hit_rate"] == point[key]["hit_rate"]
        assert rows[name]["area_share"] == point[key]["area_share"] and rows[name]["capture_share"] == point[key]["capture_share"]
        assert len(rows[name]["hit_rate_ci95"]) == 2 and rows[name]["hit_rate_ci95"][0] <= rows[name]["hit_rate"] <= rows[name]["hit_rate_ci95"][1]
    assert rows["High"]["hit_rate"] > rows["Elevated"]["hit_rate"] > rows["Low"]["hit_rate"]
    assert payload["no_coverage"]["code"] == -1
    era = payload["basis"]["high_hit_rate_by_era"]
    assert 0.09 < era["2015-2020"] < 0.12 and 0.18 < era["2021-2025"] < 0.22  # the measured doubling
    assert payload["basis"]["sample"].startswith("init 20Z runs") and "4,410" in payload["basis"]["sample"]
    assert 0.05 < payload["basis"]["cutoffs_checked_on"]["high_hit_rate"] < 0.25
    assert "cutoffs" not in payload and "calibrated_percent" not in payload  # cutoffs stay server-side; no percentage exists
