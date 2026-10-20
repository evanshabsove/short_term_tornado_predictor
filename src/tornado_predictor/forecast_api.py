"""
Payload builders for the forecast API (contract: api/README.md). No Flask
here: each builder turns the forecast store, the tier set and the grid into a
JSON-ready dict (plus an ETag where the contract has one), and the Flask layer
(forecast_app.py) only routes, sets headers and maps errors to status codes.

Read-only by construction: nothing here loads a model, pulls HRRR data or
imports the scheduler (forecast_job pulls in torch and herbie). The scheduler's
status file is read as plain JSON, and a test pins its filename to
forecast_job's constant.

Tiers are never stored. They are computed at read time from the stored raw
ensemble mean by tiers.tier_forecast -- the same smooth -> float16 -> cutoff
path the tier study measured -- using the tier set in models/tiers/
tier_study.json. Raw probabilities and the member spread are not part of any
payload: they are uncalibrated and the spread's meaning is untested.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from tornado_predictor import tiers as T
from tornado_predictor.cell_mask import load_country, us_trusted_mask
from tornado_predictor.forecast_store import ForecastStore, parse_forecast_id
from tornado_predictor.grid import HrrrCoarseGrid, build_coarse_grid

SCHEMA_VERSION = 1
STATUS_FILENAME = "status.json"
DEFAULT_TIER_CONFIG_PATH = Path(__file__).resolve().parents[2] / "models" / "tiers" / "tier_study.json"
DEFAULT_STALE_AFTER_SECONDS = 10800.0
DEFAULT_LIMIT, MAX_LIMIT = 24, 168
SCORE_DECIMALS = 4
VERTEX_DECIMALS = 4
COUNTRY_CODES = {"0": "water", "1": "US", "2": "Canada", "3": "Mexico", "4": "other land"}
RETRY_AFTER_SECONDS = 300

_SINCE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}(:\d{2}(:\d{2})?)?(Z|[+-]\d{2}:?\d{2})?$")

TIER_DESCRIPTIONS = {
    "low": "Lower relative risk. Not a safe cell.",
    "elevated": "Elevated relative risk.",
    "high": "Highest relative risk.",
}


# --- errors (the Flask layer maps these to status codes) ---


class ApiError(Exception):
    status = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequest(ApiError):
    status = 400


class NotFound(ApiError):
    status = 404


class NoForecast(ApiError):
    """Nothing has been issued yet: 503 with Retry-After."""

    status = 503


class ServerFault(ApiError):
    """Stored data is inconsistent with the running API. The detail is logged;
    the client only sees a generic 500."""

    status = 500


# --- time helpers ---


def iso_z(ts) -> str:
    ts = pd.Timestamp(ts)
    ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
    return f"{ts:%Y-%m-%dT%H:%M:%S}Z"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def parse_since(value: str) -> pd.Timestamp:
    """A forecast id (`2026-10-19T01Z`) or a full ISO-8601 datetime, as naive UTC."""
    try:
        return parse_forecast_id(value)
    except ValueError:
        pass
    if not _SINCE_RE.match(value or ""):
        raise BadRequest(f"since must be a forecast id (YYYY-MM-DDTHHZ) or an ISO-8601 datetime, got {value!r}")
    try:
        ts = pd.Timestamp(value)
    except ValueError as e:
        raise BadRequest(f"since is not a valid datetime: {value!r}") from e
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def parse_limit(value: str | None) -> int:
    if value is None:
        return DEFAULT_LIMIT
    if not re.fullmatch(r"\d+", value) or not (1 <= int(value) <= MAX_LIMIT):
        raise BadRequest(f"limit must be an integer from 1 to {MAX_LIMIT}, got {value!r}")
    return int(value)


# --- tier set ---


@dataclass(frozen=True)
class TierConfig:
    """The served tier set: operator, radius and cutoffs from the tier study's
    `serving` block, plus the study itself for the /tiers evidence."""

    tier_set_id: str
    operator: str
    radius: int
    cutoffs: dict
    study: dict = field(repr=False, compare=False)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_TIER_CONFIG_PATH) -> "TierConfig":
        path = Path(path)
        try:
            study = json.loads(path.read_text())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"tier study results not found at {path}; run scripts/09_serving/tier_study.py") from e
        s = study["serving"]
        cutoffs = {"elevated": float(s["cutoffs"]["elevated"]), "high": float(s["cutoffs"]["high"])}
        recomputed = T.tier_set_id(s["operator"], int(s["radius_cells"]), cutoffs)
        if recomputed != s["tier_set_id"]:
            raise ValueError(
                f"{path}: serving tier_set_id {s['tier_set_id']} does not match the one recomputed from its "
                f"operator/radius/cutoffs ({recomputed}); the file was edited or is inconsistent"
            )
        if not cutoffs["elevated"] < cutoffs["high"]:
            raise ValueError(f"{path}: elevated cutoff must be below the high cutoff")
        return cls(tier_set_id=s["tier_set_id"], operator=s["operator"], radius=int(s["radius_cells"]), cutoffs=cutoffs, study=study)


# --- serving context ---


@dataclass
class ServingContext:
    store: ForecastStore
    grid: HrrrCoarseGrid
    country: np.ndarray
    tiers: TierConfig
    status_path: Path
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS
    now: Callable[[], pd.Timestamp] = utc_now
    _grid_payload: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.country.shape != (self.grid.n_rows, self.grid.n_cols):
            raise ValueError(f"country mask shape {self.country.shape} does not match the grid")
        self.coverage = us_trusted_mask(self.country)

    @classmethod
    def create(
        cls,
        store_root: Path | str,
        *,
        country: np.ndarray | None = None,
        grid: HrrrCoarseGrid | None = None,
        tier_config: TierConfig | None = None,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        now: Callable[[], pd.Timestamp] = utc_now,
    ) -> "ServingContext":
        store_root = Path(store_root)
        if country is None:
            try:
                country = load_country()
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{e}; build the country mask with scripts/08_canada_transfer/build_country_mask.py"
                ) from e
        return cls(
            store=ForecastStore(store_root, "issued"),
            grid=grid if grid is not None else build_coarse_grid(),
            country=np.asarray(country),
            tiers=tier_config if tier_config is not None else TierConfig.load(),
            status_path=store_root / STATUS_FILENAME,
            stale_after_seconds=stale_after_seconds,
            now=now,
        )

    def age_seconds(self, init_time) -> int:
        """Seconds from the run's init time (stored as an ISO string with a
        trailing Z) to now; the project convention is naive timestamps in UTC."""
        init = pd.Timestamp(init_time)
        init = init.tz_convert("UTC").tz_localize(None) if init.tzinfo is not None else init
        return max(0, int((self.now() - init).total_seconds()))

    def is_stale(self, age_seconds: int) -> bool:
        return age_seconds > self.stale_after_seconds


# --- builders: each returns (payload, etag-or-None) ---


def read_scheduler_status(path: Path) -> dict:
    """The four scheduler fields GET /status exposes, from status.json. A
    missing or unreadable file gives the defaults (the scheduler has not run)."""
    state = {}
    try:
        state = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {
        "last_success_at": state.get("last_success_at"),
        "last_failure_at": state.get("last_failure_at"),
        "last_failure_message": state.get("last_failure_message"),
        "consecutive_failures": int(state.get("consecutive_failures", 0)),
    }


def build_status(ctx: ServingContext) -> tuple[dict, None]:
    meta = ctx.store.latest_meta()
    scheduler = read_scheduler_status(ctx.status_path)
    if meta is None:
        return {
            "schema_version": SCHEMA_VERSION, "status": "no_data", "now": iso_z(ctx.now()),
            "latest_init_time": None, "latest_age_seconds": None, "stale": None, "scheduler": scheduler,
        }, None
    age = ctx.age_seconds(meta["init_time"])
    stale = ctx.is_stale(age)
    return {
        "schema_version": SCHEMA_VERSION, "status": "stale" if stale else "ok", "now": iso_z(ctx.now()),
        "latest_init_time": meta["init_time"], "latest_age_seconds": age, "stale": stale, "scheduler": scheduler,
    }, None


def build_grid(ctx: ServingContext) -> tuple[dict, str]:
    """Static geometry; built once and reused (it only changes with the grid)."""
    if ctx._grid_payload is None:
        g = ctx.grid
        lon, lat = g.vertex_lonlat()
        ctx._grid_payload = {
            "schema_version": SCHEMA_VERSION,
            "grid_id": g.grid_id,
            "shape": [g.n_rows, g.n_cols],
            "vertex_shape": [g.n_rows + 1, g.n_cols + 1],
            "cell_size_km": int(g.cell_size_m / 1000.0) if float(g.cell_size_m / 1000.0).is_integer() else g.cell_size_m / 1000.0,
            "vertices": {
                "lon": np.round(lon.astype(np.float64), VERTEX_DECIMALS).ravel().tolist(),
                "lat": np.round(lat.astype(np.float64), VERTEX_DECIMALS).ravel().tolist(),
            },
            "country": ctx.country.astype(int).ravel().tolist(),
            "country_codes": COUNTRY_CODES,
        }
    return ctx._grid_payload, ctx.grid.grid_id


def build_forecasts(ctx: ServingContext, limit: int = DEFAULT_LIMIT, since: pd.Timestamp | None = None) -> tuple[dict, None]:
    metas = ctx.store.list(limit=limit, since=since)
    newest = ctx.store.latest_meta()
    items = []
    for m in metas:
        age = ctx.age_seconds(m["init_time"])
        items.append({
            "id": m["id"], "init_time": m["init_time"], "generated_at": m["generated_at"],
            "age_seconds": age, "stale": ctx.is_stale(age),
            "bundle_id": m["bundle_id"], "tier_set_id": ctx.tiers.tier_set_id,
        })
    return {
        "schema_version": SCHEMA_VERSION, "count": len(items),
        "latest_id": newest["id"] if newest else None, "forecasts": items,
    }, None


def forecast_etag(meta: dict, tiers: TierConfig) -> str:
    return f"{meta['id']}-{meta['bundle_id']}-{tiers.tier_set_id}"


def build_forecast(ctx: ServingContext, forecast_id: str) -> tuple[dict, str]:
    """`forecast_id` is a forecast id such as 2026-10-19T01Z, or "latest"."""
    if forecast_id == "latest":
        stored = ctx.store.latest()
        if stored is None:
            raise NoForecast("No forecast has been issued yet")
    else:
        try:
            init = parse_forecast_id(forecast_id)
        except ValueError as e:
            raise BadRequest(str(e)) from e
        try:
            stored = ctx.store.get(init)
        except KeyError:
            raise NotFound(f"No forecast stored for {forecast_id}") from None

    meta, grid = stored.meta, ctx.grid
    if meta["grid_id"] != grid.grid_id or tuple(stored.mean.shape[1:]) != (grid.n_rows, grid.n_cols):
        raise ServerFault(
            f"stored forecast {meta['id']} has grid_id {meta['grid_id']} / shape {stored.mean.shape[1:]}, "
            f"but the API runs grid {grid.grid_id} {(grid.n_rows, grid.n_cols)}"
        )

    tier_arr, smoothed = T.tier_forecast(stored.mean, ctx.tiers.operator, ctx.tiers.radius, ctx.tiers.cutoffs, ctx.coverage)
    bins = []
    for b in meta["bins"]:
        i = int(b["bin_index"])
        bins.append({
            "bin_index": i,
            "valid_start": b["valid_start"],
            "valid_end": b["valid_end"],
            "tiers": tier_arr[i].ravel().tolist(),
            "smoothed_score": np.round(smoothed[i].astype(np.float64), SCORE_DECIMALS).ravel().tolist(),
        })
    age = ctx.age_seconds(meta["init_time"])
    return {
        "schema_version": SCHEMA_VERSION,
        "id": meta["id"],
        "mode": "live",
        "init_time": meta["init_time"],
        "generated_at": meta["generated_at"],
        "age_seconds": age,
        "stale": ctx.is_stale(age),
        "grid_id": meta["grid_id"],
        "shape": [grid.n_rows, grid.n_cols],
        "model": {"bundle_id": meta["bundle_id"], "n_members": meta["n_members"], "trained_years": meta.get("trained_years")},
        "tiering": {
            "tier_set_id": ctx.tiers.tier_set_id,
            "smoothing": {"operator": ctx.tiers.operator, "radius_cells": ctx.tiers.radius},
        },
        "calibrated_percent": None,
        "bins": bins,
    }, forecast_etag(meta, ctx.tiers)


def _pooled_hit_rate(per_year: dict, years: range) -> float | None:
    cells = sum(per_year[str(y)]["high"]["n_cells"] for y in years)
    hits = sum(per_year[str(y)]["high"]["n_cells"] * per_year[str(y)]["high"]["hit_rate"]
               for y in years if per_year[str(y)]["high"]["n_cells"])
    return hits / cells if cells else None


def build_tiers(ctx: ServingContext) -> tuple[dict, str]:
    """What each tier means, from the tier study's own results file (not
    hard-coded), so a new tier set only needs a new study JSON."""
    study = ctx.tiers.study
    table = study["tier_table_A"]["primary_20Z"]
    rows = []
    for code, name in ((0, "low"), (1, "elevated"), (2, "high")):
        point, ci = table["point"][name], table["ci95"][name]
        rows.append({
            "code": code, "name": name.capitalize(), "description": TIER_DESCRIPTIONS[name],
            "area_share": point["area_share"], "capture_share": point["capture_share"],
            "hit_rate": point["hit_rate"], "hit_rate_ci95": ci["hit_rate"],
        })
    b_high = study["transfer_check"]["B_table_at_serving_cutoffs"]
    per_year = study["tier_table_A"]["primary_per_year"]
    return {
        "schema_version": SCHEMA_VERSION,
        "tier_set_id": ctx.tiers.tier_set_id,
        "smoothing": {"operator": ctx.tiers.operator, "radius_cells": ctx.tiers.radius},
        "basis": {
            "ensemble": "proxy: 2-model leave-one-year-out ensemble (not the served 5-member ensemble)",
            "sample": f"init 20Z runs, 2014-2025 ({study['sample']['primary_samples']:,} samples), held-out years",
            "hit_definition": "a tornado touched the cell or one of its 8 neighbours (~40 km) in that 4-hour bin",
            "cutoffs_checked_on": {
                "description": "5-member EMA ensembles, 2016/2020/2024",
                "high_hit_rate": b_high["point_median_over_subsets"]["high"]["hit_rate"],
                "high_hit_rate_ci95": b_high["ci95"]["high"]["hit_rate"],
            },
            "high_hit_rate_by_era": {
                "2015-2020": _pooled_hit_rate(per_year, range(2015, 2021)),
                "2021-2025": _pooled_hit_rate(per_year, range(2021, 2026)),
            },
        },
        "tiers": rows,
        "no_coverage": {
            "code": -1, "name": "No model coverage",
            "description": "Canada, Mexico and other land: no tornado labels exist there, so no skill is claimed.",
        },
    }, ctx.tiers.tier_set_id
