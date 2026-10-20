"""
Disk-backed archive of issued forecasts: the raw ensemble output for one
hourly HRRR run, written once and read back by the API.

Replaces tornet's in-memory cache (lost on restart) with files, so a restart
loses nothing and every issued forecast is kept for later verification.

What is stored, and what deliberately is not: the per-cell ensemble *mean*
probability and member *spread* for both time bins, as float32 arrays of shape
(N_BINS, n_rows, n_cols). Never tiers, smoothed scores or percentages -- those
are derived from the raw arrays at read time (ticket 4's module), so a change
to the smoothing or cutoffs never requires regenerating forecasts. The spread
is a disagreement signal of untested meaning, not a confidence interval.

On-disk layout, one directory per forecast:

    <root>/<namespace>/<YYYY-MM-DDTHHZ>/forecast.npz   (mean, spread; compressed)
    <root>/<namespace>/<YYYY-MM-DDTHHZ>/meta.json

`namespace` is a constructor argument: "issued" for forecasts the scheduler
produced live, "replay" for retrospective runs (ticket 8). They are separate
directories, so a replay can never be mistaken for an issued forecast.

Writes are atomic: everything goes into a temp directory next to the target
and is renamed into place, so a reader sees either a whole forecast or none.
`put` never overwrites -- an existing id raises. Nothing is ever deleted unless
`prune` is called explicitly with an age limit; the default policy keeps
everything (the archive has research value).

Forecast ids are `YYYY-MM-DDTHHZ` (UTC, on the hour), as in api/README.md.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tornado_predictor.time_bins import N_BINS

SCHEMA_VERSION = 1
NAMESPACES = ("issued", "replay")
DEFAULT_STORE_ROOT = Path(__file__).resolve().parents[2] / "data" / "live"

_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}Z$")
_TMP_PREFIX = ".tmp-"
# A temp dir this old cannot belong to a live writer (a put takes well under a
# second), so it is leftover from a crashed one and safe to remove.
_STALE_TMP_SECONDS = 3600


class ForecastStoreError(Exception):
    """A stored forecast is missing, malformed or inconsistent."""


def format_forecast_id(init_time) -> str:
    """`YYYY-MM-DDTHHZ` for an on-the-hour UTC init time (naive, or tz-aware
    in UTC). Raises ValueError for a time that is not on the hour."""
    ts = _naive_utc(init_time)
    if ts != ts.floor("h"):
        raise ValueError(f"init_time {ts} is not on the hour")
    return f"{ts:%Y-%m-%dT%H}Z"


def parse_forecast_id(forecast_id: str) -> pd.Timestamp:
    """Strict inverse of format_forecast_id; returns a naive-UTC Timestamp.
    Raises ValueError for anything else (the id comes from URLs and
    directory names, so it is validated before it touches the filesystem)."""
    if not isinstance(forecast_id, str) or not _ID_RE.match(forecast_id):
        raise ValueError(f"malformed forecast id {forecast_id!r}; expected YYYY-MM-DDTHHZ")
    try:
        return pd.Timestamp(forecast_id[:-1] + ":00:00")
    except ValueError as e:
        raise ValueError(f"malformed forecast id {forecast_id!r}: {e}") from e


def _naive_utc(ts) -> pd.Timestamp:
    """Project convention: naive timestamps, treated as UTC (see time_bins.py)."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _iso_z(ts) -> str:
    return f"{_naive_utc(ts):%Y-%m-%dT%H:%M:%S}Z"


@dataclass(frozen=True, eq=False)
class Forecast:
    """One run's raw ensemble output plus the provenance the API needs.

    mean, spread: (N_BINS, n_rows, n_cols) float arrays (stored as float32).
    bins: per-bin dicts with bin_index, fxx, valid_start, valid_end (the shape
    live_features.LiveInputs.bins already has).
    trained_years: {"first": int, "last": int} from the bundle manifest, so a
    replay response can say which years the served model trained on."""

    init_time: pd.Timestamp
    mean: np.ndarray
    spread: np.ndarray
    bundle_id: str
    n_members: int
    feature_names: list[str]
    bins: list[dict]
    grid_id: str
    trained_years: dict | None = None
    pull_seconds: float | None = None
    generated_at: pd.Timestamp | None = None
    git_commit: str | None = None


@dataclass(frozen=True, eq=False)
class StoredForecast:
    meta: dict
    mean: np.ndarray
    spread: np.ndarray


@dataclass(frozen=True)
class RetentionPolicy:
    """max_age_days=None means never delete (the default)."""

    max_age_days: float | None = None


class ForecastStore:
    def __init__(self, root: Path | str = DEFAULT_STORE_ROOT, namespace: str = "issued"):
        if namespace not in NAMESPACES:
            raise ValueError(f"namespace must be one of {NAMESPACES}, got {namespace!r}")
        self.root = Path(root)
        self.namespace = namespace
        self.dir = self.root / namespace

    # --- paths ---

    def _path(self, init_time) -> Path:
        return self.dir / format_forecast_id(init_time)

    def has(self, init_time) -> bool:
        return (self._path(init_time) / "meta.json").is_file()

    # --- write ---

    def put(self, forecast: Forecast) -> Path:
        """Atomically writes one forecast. Raises FileExistsError if its id is
        already stored (never overwrites) and ValueError for malformed arrays."""
        meta = self._build_meta(forecast)
        mean = np.asarray(forecast.mean, dtype=np.float32)
        spread = np.asarray(forecast.spread, dtype=np.float32)
        for name, arr in (("mean", mean), ("spread", spread)):
            if arr.ndim != 3 or arr.shape[0] != N_BINS:
                raise ValueError(f"{name} must be ({N_BINS}, n_rows, n_cols), got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains non-finite values; refusing to store it")
        if mean.shape != spread.shape:
            raise ValueError(f"mean {mean.shape} and spread {spread.shape} shapes differ")
        meta["shape"] = list(mean.shape[1:])

        final = self._path(forecast.init_time)
        if final.exists():
            raise FileExistsError(f"forecast {final.name} already stored in {self.namespace!r}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._remove_stale_tmp()

        tmp = self.dir / f"{_TMP_PREFIX}{final.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            with open(tmp / "forecast.npz", "wb") as f:
                np.savez_compressed(f, mean=mean, spread=spread)
                f.flush()
                os.fsync(f.fileno())
            with open(tmp / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.rename(tmp, final)
            except OSError as e:
                # A concurrent writer won the race (rename onto a non-empty dir fails).
                if final.exists():
                    raise FileExistsError(f"forecast {final.name} already stored in {self.namespace!r}") from e
                raise
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return final

    def _build_meta(self, forecast: Forecast) -> dict:
        init = _naive_utc(forecast.init_time)
        generated = _naive_utc(forecast.generated_at) if forecast.generated_at is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
        return {
            "schema_version": SCHEMA_VERSION,
            "id": format_forecast_id(init),
            "namespace": self.namespace,
            "init_time": _iso_z(init),
            "generated_at": _iso_z(generated),
            "bundle_id": forecast.bundle_id,
            "n_members": int(forecast.n_members),
            "trained_years": forecast.trained_years,
            "feature_names": list(forecast.feature_names),
            "bins": [
                {
                    "bin_index": int(b["bin_index"]),
                    "fxx": int(b["fxx"]),
                    "valid_start": _iso_z(b["valid_start"]),
                    "valid_end": _iso_z(b["valid_end"]),
                }
                for b in forecast.bins
            ],
            "grid_id": forecast.grid_id,
            "pull_seconds": None if forecast.pull_seconds is None else round(float(forecast.pull_seconds), 2),
            "git_commit": forecast.git_commit,
        }

    def _remove_stale_tmp(self) -> None:
        now = time.time()
        for p in self.dir.glob(f"{_TMP_PREFIX}*"):
            try:
                if now - p.stat().st_mtime > _STALE_TMP_SECONDS:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass

    # --- read ---

    def _ids(self) -> list[str]:
        """Ids of complete forecasts, oldest first. A directory only counts if
        its name is a valid id and meta.json exists (a rename is atomic, so
        this is belt-and-braces against hand-made or partial directories)."""
        if not self.dir.is_dir():
            return []
        return sorted(
            p.name for p in self.dir.iterdir() if _ID_RE.match(p.name) and (p / "meta.json").is_file()
        )

    def get_meta(self, init_time) -> dict:
        """Metadata only (cheap; no arrays). Raises KeyError if not stored."""
        path = self._path(init_time) / "meta.json"
        if not path.is_file():
            raise KeyError(format_forecast_id(init_time))
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ForecastStoreError(f"{path}: unreadable metadata ({e})") from e

    def get(self, init_time) -> StoredForecast:
        """Raises KeyError if not stored, ForecastStoreError if the files are
        corrupt or disagree with each other."""
        meta = self.get_meta(init_time)
        npz_path = self._path(init_time) / "forecast.npz"
        try:
            with np.load(npz_path) as z:
                mean, spread = z["mean"], z["spread"]
        except (OSError, ValueError, KeyError) as e:
            raise ForecastStoreError(f"{npz_path}: unreadable forecast arrays ({e})") from e
        if list(mean.shape[1:]) != meta.get("shape") or mean.shape != spread.shape:
            raise ForecastStoreError(
                f"{npz_path}: array shapes {mean.shape}/{spread.shape} disagree with metadata shape {meta.get('shape')}"
            )
        return StoredForecast(meta=meta, mean=mean, spread=spread)

    def latest(self) -> StoredForecast | None:
        ids = self._ids()
        return self.get(parse_forecast_id(ids[-1])) if ids else None

    def latest_meta(self) -> dict | None:
        ids = self._ids()
        return self.get_meta(parse_forecast_id(ids[-1])) if ids else None

    def list(self, limit: int | None = None, since=None) -> list[dict]:
        """Metadata of stored forecasts, newest first. `since` keeps only
        init times at or after it."""
        ids = self._ids()[::-1]
        if since is not None:
            cutoff = _naive_utc(since)
            ids = [i for i in ids if parse_forecast_id(i) >= cutoff]
        if limit is not None:
            ids = ids[:limit]
        return [self.get_meta(parse_forecast_id(i)) for i in ids]

    # --- retention ---

    def prune(self, policy: RetentionPolicy, now=None) -> list[str]:
        """Deletes forecasts whose init_time is older than policy.max_age_days
        and returns their ids. With max_age_days=None (the default policy)
        nothing is deleted. Never called implicitly."""
        if policy.max_age_days is None:
            return []
        now = _naive_utc(pd.Timestamp.now(tz="UTC") if now is None else now)
        cutoff = now - pd.Timedelta(days=policy.max_age_days)
        removed = []
        for fid in self._ids():
            if parse_forecast_id(fid) < cutoff:
                # Rename first so a concurrent reader never sees a half-deleted dir.
                doomed = self.dir / f"{_TMP_PREFIX}deleting-{fid}-{uuid.uuid4().hex[:8]}"
                os.rename(self.dir / fid, doomed)
                shutil.rmtree(doomed, ignore_errors=True)
                removed.append(fid)
        return removed
