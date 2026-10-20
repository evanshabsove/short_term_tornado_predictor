"""
Flask layer of the forecast API (contract: api/README.md). Thin on purpose:
routing, headers and error mapping only; every payload comes from
forecast_api's builders. Read-only: no request handler loads a model, pulls
HRRR data or writes to the forecast store (the scheduler, ticket 5, is the only
writer), and importing this module never starts a background thread.

Run: `python api/app.py` (serves on :5060, override with PORT). Environment:
  PORT                          port (default 5060; read by api/app.py)
  FORECAST_STORE_ROOT           directory holding issued/ and status.json (default data/live)
  FORECAST_STALE_AFTER_SECONDS  a forecast whose run is older than this is flagged stale (default 10800)
  FORECAST_CORS_ORIGINS         "*" (default) or a comma-separated list of allowed origins
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from tornado_predictor import forecast_api as api
from tornado_predictor.forecast_store import DEFAULT_STORE_ROOT

logger = logging.getLogger(__name__)

# Cache-Control per route, from the "Caching" table in api/README.md.
CACHE_GRID = "public, max-age=86400"
CACHE_FORECAST_BY_ID = "public, max-age=300"
CACHE_TIERS = "public, max-age=300"
NO_CACHE = "no-cache"


def _cors_origins(value: str | None):
    value = (value if value is not None else os.environ.get("FORECAST_CORS_ORIGINS", "*")).strip()
    return "*" if value in ("", "*") else [o.strip() for o in value.split(",") if o.strip()]


def create_app(
    store_root: Path | str | None = None,
    *,
    country=None,
    tier_config: api.TierConfig | None = None,
    now: Callable[[], pd.Timestamp] = api.utc_now,
    stale_after_seconds: float | None = None,
    cors_origins: str | None = None,
) -> Flask:
    """Application factory. Everything has an environment/default fallback so
    `create_app()` is the production configuration; tests pass explicit values
    (a synthetic country mask, a fixture store, a fixed clock)."""
    if store_root is None:
        store_root = os.environ.get("FORECAST_STORE_ROOT", DEFAULT_STORE_ROOT)
    if stale_after_seconds is None:
        stale_after_seconds = float(os.environ.get("FORECAST_STALE_AFTER_SECONDS", api.DEFAULT_STALE_AFTER_SECONDS))

    ctx = api.ServingContext.create(
        store_root, country=country, tier_config=tier_config, stale_after_seconds=stale_after_seconds, now=now
    )
    api.build_grid(ctx)  # the static geometry is built once, here, not on the first request

    app = Flask(__name__)
    app.json.sort_keys = False  # keep the field order the contract documents
    app.config["FORECAST_CONTEXT"] = ctx
    CORS(app, origins=_cors_origins(cors_origins))  # consumed cross-origin by the Rails frontend

    def respond(payload: dict, etag: str | None, cache_control: str):
        response = jsonify(payload)
        response.headers["Cache-Control"] = cache_control
        if etag is not None:
            response.set_etag(etag)
            return response.make_conditional(request)  # 304 when If-None-Match matches
        return response

    @app.errorhandler(api.ApiError)
    def handle_api_error(e: api.ApiError):
        if isinstance(e, api.ServerFault):
            app.logger.error("Server fault: %s", e.message)
            response = jsonify({"error": "Internal server error"})
        else:
            response = jsonify({"error": e.message})
        response.status_code = e.status
        if e.status == 503:
            response.headers["Retry-After"] = str(api.RETRY_AFTER_SECONDS)
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        response = jsonify({"error": e.description})
        response.status_code = e.code
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_exception(e: Exception):
        app.logger.exception("Unhandled error in request")
        response = jsonify({"error": "Internal server error"})
        response.status_code = 500
        return response

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/status")
    def status():
        payload, _ = api.build_status(ctx)
        return respond(payload, None, NO_CACHE)

    @app.route("/grid")
    def grid():
        payload, etag = api.build_grid(ctx)
        return respond(payload, etag, CACHE_GRID)

    @app.route("/forecasts")
    def forecasts():
        limit = api.parse_limit(request.args.get("limit"))
        since = api.parse_since(request.args["since"]) if "since" in request.args else None
        payload, _ = api.build_forecasts(ctx, limit, since)
        return respond(payload, None, NO_CACHE)

    @app.route("/forecast/<forecast_id>")
    def forecast(forecast_id: str):
        payload, etag = api.build_forecast(ctx, forecast_id)
        return respond(payload, etag, NO_CACHE if forecast_id == "latest" else CACHE_FORECAST_BY_ID)

    @app.route("/tiers")
    def tiers():
        payload, etag = api.build_tiers(ctx)
        return respond(payload, etag, CACHE_TIERS)

    return app
