"""Flask layer: routes, headers, conditional requests, error shapes, read-only behavior."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tornado_predictor import forecast_api as api
from tornado_predictor import tiers as T
from tornado_predictor.forecast_app import create_app

NOW = pd.Timestamp("2026-10-19 02:30")


@pytest.fixture
def app(store_root, country, tier_config):
    return create_app(store_root, country=country, tier_config=tier_config, now=lambda: NOW)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def filled(store, make_forecast):
    store.put(make_forecast("2026-10-19 00:00", bundle_id="b0"))
    store.put(make_forecast("2026-10-19 01:00"))
    return store


# --- every route answers with the contract's shape ---


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}


def test_status_route(client, filled):
    r = client.get("/status")
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok" and body["latest_init_time"] == "2026-10-19T01:00:00Z"
    assert r.headers["Cache-Control"] == "no-cache" and "ETag" not in r.headers
    assert list(body)[:3] == ["schema_version", "status", "now"]  # field order follows the contract


def test_status_is_always_200_even_when_empty(client):
    r = client.get("/status")
    assert r.status_code == 200 and r.get_json()["status"] == "no_data"


def test_grid_route_headers_and_etag(client, grid):
    r = client.get("/grid")
    body = r.get_json()
    assert r.status_code == 200 and body["grid_id"] == grid.grid_id
    assert r.headers["Cache-Control"] == "public, max-age=86400" and r.headers["ETag"] == f'"{grid.grid_id}"'
    assert len(body["vertices"]["lon"]) == 11398 and len(body["country"]) == 11178
    assert client.get("/grid", headers={"If-None-Match": r.headers["ETag"]}).status_code == 304


def test_forecasts_route(client, filled):
    r = client.get("/forecasts")
    body = r.get_json()
    assert r.status_code == 200 and body["count"] == 2 and body["latest_id"] == "2026-10-19T01Z"
    assert [f["id"] for f in body["forecasts"]] == ["2026-10-19T01Z", "2026-10-19T00Z"]
    assert r.headers["Cache-Control"] == "no-cache" and "ETag" not in r.headers
    assert client.get("/forecasts?limit=1").get_json()["count"] == 1
    assert client.get("/forecasts?since=2026-10-19T01Z").get_json()["count"] == 1
    assert client.get("/forecasts?since=2026-10-19T00:30:00Z&limit=5").get_json()["count"] == 1


def test_forecasts_on_an_empty_store_is_200_with_an_empty_list(client):
    r = client.get("/forecasts")
    assert r.status_code == 200 and r.get_json() == {"schema_version": 1, "count": 0, "latest_id": None, "forecasts": []}


def test_forecast_latest_and_by_id_share_a_payload_but_not_caching(client, filled):
    latest, by_id = client.get("/forecast/latest"), client.get("/forecast/2026-10-19T01Z")
    assert latest.status_code == by_id.status_code == 200
    assert latest.get_json() == by_id.get_json()
    assert latest.headers["Cache-Control"] == "no-cache"
    assert by_id.headers["Cache-Control"] == "public, max-age=300"
    etag = '"2026-10-19T01Z-3c02381f1611-ts-e2c497b6"'
    assert latest.headers["ETag"] == by_id.headers["ETag"] == etag
    assert client.get("/forecast/2026-10-19T00Z").get_json()["model"]["bundle_id"] == "b0"


def test_conditional_get_returns_304_with_no_body_and_a_stale_etag_returns_200(client, filled):
    etag = client.get("/forecast/latest").headers["ETag"]
    r = client.get("/forecast/latest", headers={"If-None-Match": etag})
    assert r.status_code == 304 and r.data == b""
    assert client.get("/forecast/2026-10-19T01Z", headers={"If-None-Match": etag}).status_code == 304
    assert client.get("/forecast/latest", headers={"If-None-Match": '"something-else"'}).status_code == 200


def test_tiers_route(client):
    r = client.get("/tiers")
    body = r.get_json()
    assert r.status_code == 200 and body["tier_set_id"] == "ts-e2c497b6"
    assert r.headers["Cache-Control"] == "public, max-age=300" and r.headers["ETag"] == '"ts-e2c497b6"'
    assert [t["name"] for t in body["tiers"]] == ["Low", "Elevated", "High"]
    assert client.get("/tiers", headers={"If-None-Match": r.headers["ETag"]}).status_code == 304


def test_a_new_tier_set_changes_the_forecast_etag(store_root, country, tier_config, store, make_forecast):
    store.put(make_forecast())
    cut = {"elevated": tier_config.cutoffs["elevated"], "high": 0.30}
    other = api.TierConfig(T.tier_set_id("max", 1, cut), "max", 1, cut, tier_config.study)
    c1 = create_app(store_root, country=country, tier_config=tier_config, now=lambda: NOW).test_client()
    c2 = create_app(store_root, country=country, tier_config=other, now=lambda: NOW).test_client()
    e1, e2 = c1.get("/forecast/latest").headers["ETag"], c2.get("/forecast/latest").headers["ETag"]
    assert e1 != e2
    # a client holding the old tier set's ETag gets a fresh 200, not a 304
    assert c2.get("/forecast/latest", headers={"If-None-Match": e1}).status_code == 200


# --- errors ---


def test_error_shapes(client, filled):
    cases = [
        ("/forecast/not-an-id", 400), ("/forecast/2026-10-19", 400), ("/forecasts?limit=0", 400), ("/forecasts?limit=169", 400),
        ("/forecasts?limit=abc", 400), ("/forecasts?since=yesterday", 400), ("/forecasts?since=", 400),
        ("/forecast/2026-10-20T09Z", 404), ("/nope", 404),
    ]
    for path, status in cases:
        r = client.get(path)
        assert r.status_code == status, path
        body = r.get_json()
        assert list(body) == ["error"] and isinstance(body["error"], str) and body["error"], path


def test_empty_store_latest_is_503_with_retry_after(client):
    r = client.get("/forecast/latest")
    assert r.status_code == 503 and r.headers["Retry-After"] == "300"
    assert "No forecast" in r.get_json()["error"]


def test_method_not_allowed_is_json(client):
    r = client.post("/forecast/latest")
    assert r.status_code == 405 and list(r.get_json()) == ["error"]


def test_unexpected_exceptions_are_500_without_details(app, client, monkeypatch):
    def boom(ctx):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api, "build_tiers", boom)
    r = client.get("/tiers")
    assert r.status_code == 500 and r.get_json() == {"error": "Internal server error"}
    assert b"secret" not in r.data


def test_grid_mismatch_is_a_generic_500_and_is_logged(client, store, make_forecast, caplog):
    store.put(make_forecast(grid_id="ffffffffffff"))
    with caplog.at_level("ERROR"):
        r = client.get("/forecast/latest")
    assert r.status_code == 500 and r.get_json() == {"error": "Internal server error"}
    assert b"ffffffffffff" not in r.data and "ffffffffffff" in caplog.text


def test_corrupt_stored_forecast_is_a_generic_500(client, store, make_forecast):
    store.put(make_forecast())
    (store.dir / "2026-10-19T01Z" / "forecast.npz").write_bytes(b"not an npz")
    r = client.get("/forecast/latest")
    assert r.status_code == 500 and r.get_json() == {"error": "Internal server error"}


# --- CORS ---


def test_cors_allows_any_origin_by_default(client):
    # flask-cors answers a wildcard policy with "*" or by echoing the caller's origin; either way every origin is allowed
    for origin in ("http://localhost:3000", "https://rails.example"):
        r = client.get("/health", headers={"Origin": origin})
        assert r.headers["Access-Control-Allow-Origin"] in ("*", origin)


def test_cors_can_be_restricted_to_listed_origins(store_root, country, tier_config):
    c = create_app(store_root, country=country, tier_config=tier_config, cors_origins="https://rails.example, https://b.example").test_client()
    ok = c.get("/health", headers={"Origin": "https://rails.example"})
    assert ok.headers["Access-Control-Allow-Origin"] == "https://rails.example"
    assert "Access-Control-Allow-Origin" not in c.get("/health", headers={"Origin": "https://evil.example"}).headers


def test_cors_env_var_is_read(store_root, country, tier_config, monkeypatch):
    monkeypatch.setenv("FORECAST_CORS_ORIGINS", "https://only.example")
    c = create_app(store_root, country=country, tier_config=tier_config).test_client()
    assert c.get("/health", headers={"Origin": "https://only.example"}).headers["Access-Control-Allow-Origin"] == "https://only.example"
    assert "Access-Control-Allow-Origin" not in c.get("/health", headers={"Origin": "https://other.example"}).headers


# --- configuration ---


def test_stale_threshold_comes_from_the_environment(store_root, country, tier_config, store, make_forecast, monkeypatch):
    store.put(make_forecast("2026-10-19 01:00"))
    monkeypatch.setenv("FORECAST_STALE_AFTER_SECONDS", "3600")  # now is 90 minutes after init
    c = create_app(store_root, country=country, tier_config=tier_config, now=lambda: NOW).test_client()
    assert c.get("/status").get_json()["stale"] is True
    assert c.get("/forecast/latest").get_json()["stale"] is True
    default = create_app(store_root, country=country, tier_config=tier_config, now=lambda: NOW, stale_after_seconds=10800).test_client()
    assert default.get("/status").get_json()["stale"] is False


def test_store_root_comes_from_the_environment(tmp_path, country, tier_config, make_forecast, monkeypatch):
    from tornado_predictor.forecast_store import ForecastStore

    root = tmp_path / "elsewhere"
    ForecastStore(root, "issued").put(make_forecast())
    monkeypatch.setenv("FORECAST_STORE_ROOT", str(root))
    c = create_app(country=country, tier_config=tier_config, now=lambda: NOW).test_client()
    assert c.get("/forecasts").get_json()["count"] == 1


# --- size and speed budget ---


def test_forecast_payload_stays_under_the_size_budget(client, store, make_forecast):
    rng = np.random.default_rng(1)
    store.put(make_forecast(mean=(rng.random((2, 81, 138)) ** 4 * 0.05).astype(np.float32)))
    r = client.get("/forecast/latest")
    assert len(r.data) < 300_000  # README budget


# --- read-only ---


def _tree(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.rglob("*")) if p.is_file()}


def test_requests_never_modify_the_store(client, filled, store_root):
    before = _tree(store_root)
    for path in ["/health", "/status", "/grid", "/forecasts", "/forecast/latest", "/forecast/2026-10-19T00Z", "/tiers",
                 "/forecast/2026-10-20T09Z", "/forecasts?limit=0"]:
        client.get(path)
    assert _tree(store_root) == before


def test_post_cannot_write_anything(client, filled, store_root):
    before = _tree(store_root)
    for path in ["/forecast/latest", "/forecasts", "/status"]:
        assert client.post(path, json={"x": 1}).status_code == 405
    assert _tree(store_root) == before


def test_importing_the_api_never_loads_a_model_or_the_hrrr_stack():
    code = (
        "import sys, json\n"
        "import tornado_predictor.forecast_app\n"
        "banned = ['torch', 'herbie', 'tornado_predictor.forecast_job', 'tornado_predictor.ensemble',\n"
        "          'tornado_predictor.live_features', 'tornado_predictor.model', 'tornado_predictor.unet']\n"
        "print(json.dumps([m for m in banned if m in sys.modules]))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_creating_the_app_starts_no_background_threads(store_root, country, tier_config):
    import threading

    before = threading.active_count()
    create_app(store_root, country=country, tier_config=tier_config)
    assert threading.active_count() == before
