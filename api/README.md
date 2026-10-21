# Tornado Forecast API

> **Status: implemented (ticket 7).** Served by `api/app.py` (code: `src/tornado_predictor/forecast_app.py` and `forecast_api.py`).
> This document is what the Flask API and the Rails frontend (tickets 9-10) both code against. Changing a response shape is a
> coordinated change with the Rails app, as with the tornet API's `/predict`.

Flask API serving the short-range tornado **forecast** produced by this repo's HRRR-based U-Net ensemble
(0-8 h, two 4-hour bins per hourly HRRR run, ~39 km grid cells). It is the forecasting counterpart of the
`tornet` radar-detection API (`../tornet/api/`) and follows its conventions: JSON everywhere, `CORS(app)`,
a `/health` liveness route, and `{"error": "..."}` error bodies.

It is a **read-only** layer over forecasts that a separate scheduler process writes to disk (ticket 5).
No request handler loads a model or pulls HRRR data.

## Differences from the tornet API

| | tornet radar API | this forecast API |
|---|---|---|
| Unit of output | one probability per storm / radar tile | a probability **map** (81 x 138 cells) per time bin |
| Port | 5050 | **5060** (so both can run together) |
| Freshness | `staleness_seconds` = age of the cache entry | `age_seconds` = age of the **model run** (now minus init time) |
| Cache | in-memory, lost on restart | on disk; every issued forecast is kept |
| Display unit | percentage (0-100, UI multiplies) | **three tiers**, not a percentage (see "Presentation rules") |

## Setup

```bash
python api/app.py          # serves on :5060 (override with PORT)
```

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5060` | Port the Flask server binds to |
| `FORECAST_STORE_ROOT` | `data/live` | Directory holding `issued/` (written by the scheduler) and `status.json` |
| `FORECAST_STALE_AFTER_SECONDS` | `10800` (3 h) | A forecast whose run is older than this is flagged `stale: true` |
| `FORECAST_CORS_ORIGINS` | `*` | `*`, or a comma-separated list of allowed origins (ticket 11 sets this for production) |
| `FORECAST_REPORTS_CSV` | `data/processed/spc_tornado_reports_2014_2025.csv` | SPC reports for `/reports`; optional (the route is `503` without it) |

The API needs the tracked tier results (`models/tiers/tier_study.json`) and the gitignored country mask
(`data/processed/hrrr_coarse_grid_stride13_countries.nc`), and refuses to start without them, or if the tier file's
`tier_set_id` does not match the one recomputed from its own cutoffs. It never loads a model, pulls HRRR data, writes to the
store or starts a background thread: fill the store with `scripts/09_serving/run_forecast_job.py`.
The Rails app reads the base URL from `FORECAST_API_BASE_URL` (default `http://localhost:5060`).

## Conventions

- **Timestamps** are UTC ISO-8601 with a trailing `Z`: `2026-10-19T01:00:00Z`.
- **Forecast ids** (used in URLs) are `YYYY-MM-DDTHHZ`, e.g. `2026-10-19T01Z`. Parsing is strict.
- **`schema_version`** (integer, currently `1`) is on every JSON payload. Adding fields does not bump it;
  removing or re-typing a field does.
- **Grid arrays are flat and row-major**, with the shape stated in the payload (`[81, 138]`):
  index `i = row * 138 + col`; **row 0 is the southernmost row, col 0 the westernmost column**. The same
  indexing applies to every per-cell array (`tiers`, `smoothed_score`, `country`).
- **Time bins** are relative to the run's init time: bin 0 = `[init, init+4h)`, bin 1 = `[init+4h, init+8h)`.
  A run is published about one hour after its init time (HRRR files land ~init+50 min), so **bin 0 is already
  about an hour old when a forecast is issued**. Always show the valid window, never "the next four hours".
- **Tier codes**: `-1` no model coverage, `0` Low, `1` Elevated, `2` High.
- **Errors** use `{"error": "<message>"}` with the codes in the table at the end.

---

## Routes

### `GET /health`

Liveness check, identical to the tornet API.

```json
{ "status": "ok" }
```

### `GET /status`

Pipeline health, for monitoring and for the UI's staleness badge. Always `200`.

```json
{
  "schema_version": 1,
  "status": "ok",
  "now": "2026-10-19T02:03:40Z",
  "latest_init_time": "2026-10-19T01:00:00Z",
  "latest_age_seconds": 3820,
  "stale": false,
  "scheduler": {
    "last_success_at": "2026-10-19T02:03:11Z",
    "last_failure_at": null,
    "last_failure_message": null,
    "consecutive_failures": 0
  }
}
```

`status` is `ok` (latest forecast not stale), `stale` (a forecast exists but is older than
`FORECAST_STALE_AFTER_SECONDS`), or `no_data` (nothing issued yet; `latest_*` are `null`).

### `GET /grid`

The static grid geometry. Fetch once and cache; it only changes if the grid itself changes (`grid_id`).

```json
{
  "schema_version": 1,
  "grid_id": "5d0c1f9a2b7e",
  "shape": [81, 138],
  "vertex_shape": [82, 139],
  "cell_size_km": 39,
  "vertices": { "lon": [-134.1, "... 11,398 values"], "lat": [21.1, "... 11,398 values"] },
  "country": [0, 0, 1, "... 11,178 values"],
  "country_codes": { "0": "water", "1": "US", "2": "Canada", "3": "Mexico", "4": "other land" }
}
```

`vertices` is the shared lattice of cell corners (flat, row-major, 4 decimals), so neighbouring cells share
vertices instead of repeating them. For cell `(row, col)`, with `V = row * 139 + col` the vertex index:

| corner | vertex index |
|---|---|
| south-west | `V` |
| south-east | `V + 1` |
| north-east | `V + 139 + 1` |
| north-west | `V + 139` |

A cell has model coverage when its `country` code is `0` (water) or `1` (US); in a forecast, `tiers` is
`-1` for exactly the cells with code `2`, `3` or `4`. Canadian and Mexican cells are always unlabeled in the
training data (SPC reports are US-only), so no skill is claimed there.

### `GET /forecasts`

Stored forecasts, newest first. An empty store is **not** an error here: `200` with `count: 0`, `latest_id: null` and an
empty list (only `GET /forecast/latest` is `503`). `latest_id` is the newest stored forecast even when `since` filters the list to nothing.

| Query param | Default | Description |
|---|---|---|
| `limit` | `24` | Max items (1-168) |
| `since` | none | Only forecasts with `init_time` at or after this (ISO-8601 or forecast id) |

```json
{
  "schema_version": 1,
  "count": 2,
  "latest_id": "2026-10-19T01Z",
  "forecasts": [
    {
      "id": "2026-10-19T01Z",
      "init_time": "2026-10-19T01:00:00Z",
      "generated_at": "2026-10-19T02:03:11Z",
      "age_seconds": 3820,
      "stale": false,
      "bundle_id": "e0dc23a53c86",
      "tier_set_id": "ts-e2c497b6"
    }
  ]
}
```

### `GET /forecast/latest` and `GET /forecast/<id>`

One forecast: both time bins. `latest` is the newest stored run; `<id>` is a forecast id such as
`2026-10-19T01Z`.

```json
{
  "schema_version": 1,
  "id": "2026-10-19T01Z",
  "mode": "live",
  "init_time": "2026-10-19T01:00:00Z",
  "generated_at": "2026-10-19T02:03:11Z",
  "age_seconds": 3820,
  "stale": false,
  "grid_id": "5d0c1f9a2b7e",
  "shape": [81, 138],
  "model": {
    "bundle_id": "e0dc23a53c86",
    "n_members": 5,
    "trained_years": { "first": 2014, "last": 2025 }
  },
  "tiering": {
    "tier_set_id": "ts-e2c497b6",
    "smoothing": { "operator": "max", "radius_cells": 1 }
  },
  "calibrated_percent": null,
  "bins": [
    {
      "bin_index": 0,
      "valid_start": "2026-10-19T01:00:00Z",
      "valid_end": "2026-10-19T05:00:00Z",
      "tiers": [0, 0, 1, "... 11,178 values"],
      "smoothed_score": [0.0021, 0.0034, "... 11,178 values"]
    },
    { "bin_index": 1, "valid_start": "2026-10-19T05:00:00Z", "valid_end": "2026-10-19T09:00:00Z", "tiers": ["..."], "smoothed_score": ["..."] }
  ]
}
```

| Field | Meaning |
|---|---|
| `age_seconds` | Now minus **`init_time`**, as of this response (never negative). The age of the model run, which is what matters to a forecaster. The `ETag` does not change with time, so a revalidated `304` leaves the client with its older body: **compute age from `init_time` client-side** rather than trusting a cached `age_seconds`. |
| `stale` | `age_seconds` exceeds `FORECAST_STALE_AFTER_SECONDS`. A stale forecast is still returned with `200`. |
| `grid_id` | Must equal `/grid`'s `grid_id`; if it differs, refetch `/grid`. |
| `model` | Which production bundle produced it. `trained_years` matters for historical replays, which are retrospective, not held-out tests (ticket 8). |
| `tiering` | Which tier set and smoothing produced `tiers`. Tiers are derived from the stored raw ensemble output, so they can be re-derived if the cutoffs change (the id changes with them). |
| `calibrated_percent` | Reserved. `null` unless ticket 4 concluded a calibrated percentage is defensible, in which case a per-bin percent array is added. |
| `tiers` | Per cell: `-1` no coverage, `0` Low, `1` Elevated, `2` High. Computed server-side; clients never compute tiers. |
| `smoothed_score` | The neighbourhood-smoothed ensemble score the tiers are cut from. **Uncalibrated relative index; do not display it as a percentage.** For hover text and ordering only. Quantized to float16 (the precision the tiers were measured at) and rounded to 4 decimals. |

Raw per-member probabilities and the member spread are stored by the scheduler but are **not** part of this
contract by default: they are uncalibrated, and the spread's meaning is untested (it is a disagreement signal,
not a confidence interval).

### `GET /tiers`

What each tier means, with the measured evidence behind it (ticket 4; full results in `docs/tier_study_results.md`).
Values below are the real ones for tier set `ts-e2c497b6`.

```json
{
  "schema_version": 1,
  "tier_set_id": "ts-e2c497b6",
  "smoothing": { "operator": "max", "radius_cells": 1 },
  "basis": {
    "ensemble": "proxy: 2-model leave-one-year-out ensemble (not the served 5-member ensemble)",
    "sample": "init 20Z runs, 2014-2025 (4,410 samples), held-out years",
    "hit_definition": "a tornado touched the cell or one of its 8 neighbours (~40 km) in that 4-hour bin",
    "cutoffs_checked_on": {
      "description": "5-member EMA ensembles, 2016/2020/2024",
      "high_hit_rate": 0.139, "high_hit_rate_ci95": [0.103, 0.176]
    },
    "high_hit_rate_by_era": { "2015-2020": 0.103, "2021-2025": 0.198 }
  },
  "tiers": [
    { "code": 0, "name": "Low", "description": "Lower relative risk. Not a safe cell.",
      "area_share": 0.9977, "capture_share": 0.399, "hit_rate": 0.00012, "hit_rate_ci95": [0.00010, 0.00014] },
    { "code": 1, "name": "Elevated", "description": "Elevated relative risk.",
      "area_share": 0.0020, "capture_share": 0.337, "hit_rate": 0.039, "hit_rate_ci95": [0.033, 0.044] },
    { "code": 2, "name": "High", "description": "Highest relative risk.",
      "area_share": 0.00035, "capture_share": 0.263, "hit_rate": 0.139, "hit_rate_ci95": [0.117, 0.160] }
  ],
  "no_coverage": { "code": -1, "name": "No model coverage", "description": "Canada, Mexico and other land: no tornado labels exist there, so no skill is claimed." }
}
```

`area_share` is the fraction of covered cells (US land and water, over all maps, so it is tiny) in the tier, `capture_share` the
fraction of tornado cells inside it (**Low holds 40%**), and `hit_rate` the fraction of its cells that verified. Tiers are exclusive.
`hit_rate_ci95` is a day-block bootstrap interval. The cutoffs themselves stay server-side (they are scores on an uncalibrated
scale), in `models/tiers/tier_study.json` under `serving`, keyed by `tier_set_id`; the API reads them at startup.
Every number in this response is read from that file at runtime, so a new tier set only needs a new study file.
`GET /forecast/*` carries the same `tier_set_id`. A calibrated percentage was **not** found defensible (see the results doc), so
no percentage field exists.

### `GET /replay/<id>`

A past case: the production ensemble run on a historical HRRR init time, tiered exactly like a live forecast. `<id>` is a forecast id
(`2021-12-10T21Z`; there is no `latest`). Same envelope as `GET /forecast/<id>` with these differences:

```json
{ "schema_version": 1, "id": "2021-12-10T21Z", "mode": "replay",
  "retrospective": true, "trained_on_years": { "first": 2014, "last": 2025 },
  "note": "Retrospective: the served model was trained on 2014-2025, which includes this date, so this replay illustrates the model's output on a past case. It is not a held-out test.",
  "init_time": "2021-12-10T21:00:00Z", "age_seconds": null, "stale": null, "...": "the rest as in /forecast/<id>" }
```

- **A replay is retrospective, never a test.** The served model trained on every year, so any replay of 2014-2025 is in-sample; a client must label it as such (show `note`).
- `age_seconds` and `stale` are `null`: the run is years old, so neither means anything.
- **The API never computes a replay.** It serves what `scripts/09_serving/run_replay.py` has stored in the `replay` namespace (`data/live/replay/`); an id that was not computed is `404` and the message says how to compute it. (The ticket text proposed computing on request; that would put model loading and ~16 HRRR pulls, 30+ s, inside a request handler, which the read-only design rules out.)
- Replays and live forecasts are separate: a replay id is `404` under `/forecast/<id>` and the reverse, and a replay never counts as the latest forecast.

### `GET /replays`

Stored replays (newest first; same `limit` and `since` as `/forecasts`) and the curated showcase cases with which of their runs are available, so a client can offer a case picker without guessing ids.

```json
{
  "schema_version": 1, "count": 1,
  "cases": [
    { "key": "mayfield-2021", "label": "December 10-11, 2021: Quad-State / Mayfield tornado outbreak",
      "runs": [ { "id": "2021-12-10T21Z", "init_time": "2021-12-10T21:00:00Z", "available": true },
                { "id": "2021-12-10T22Z", "init_time": "2021-12-10T22:00:00Z", "available": false } ] }
  ],
  "replays": [ { "id": "2021-12-10T21Z", "init_time": "2021-12-10T21:00:00Z", "generated_at": "2026-10-20T06:48:50Z",
                 "bundle_id": "3c02381f1611", "tier_set_id": "ts-e2c497b6", "case": "mayfield-2021" } ]
}
```

The curated cases live in `src/tornado_predictor/replay_cases.py` (`mayfield-2021` and `plains-2024-04-26`, three runs each).

### `GET /reports?start=&end=`

SPC tornado reports in a half-open window `[start, end)` on the touchdown time, as GeoJSON, for the verification overlay.
`start` and `end` are both required and accept a date (`2024-04-26`, midnight UTC), a forecast id, or an ISO-8601 datetime; the window may span at most 31 days.

```json
{ "type": "FeatureCollection", "schema_version": 1, "start": "2024-04-26T00:00:00Z", "end": "2024-04-27T00:00:00Z", "count": 3,
  "features": [
    { "type": "Feature", "geometry": { "type": "Point", "coordinates": [-96.0, 41.0] },
      "properties": { "kind": "touchdown", "event_id": "2024_2404261830-01", "timestamp_utc": "2024-04-26T18:30:00Z",
                      "ef_rating": 2, "state": "NE", "length_mi": 12.5, "width_yd": 400 } },
    { "type": "Feature", "geometry": { "type": "LineString", "coordinates": [[-96.0, 41.0], [-95.8, 41.1]] },
      "properties": { "kind": "track", "event_id": "2024_2404261830-01", "...": "same properties" } }
  ] }
```

- Each report is a `touchdown` Point and, when it has a real track, a `track` LineString sharing its `event_id`; `count` is the number of **reports**, not features. A zero-length track (about 8% of reports) is a Point only.
- Coordinates are `[lon, lat]`. `ef_rating` is `null` when SPC gave none (about 10% of reports).
- The track is SPC's straight line from touchdown to lift-off, not a surveyed path, and the single timestamp is the touchdown time: SPC has no per-point timing, so the track does not say where the tornado was at any other moment.
- Source: `data/processed/spc_tornado_reports_2014_2025.csv` (2014-01-11 to 2025-12-29). If that file is not installed the API still starts and this route answers `503` (no `Retry-After`).

---

## Presentation rules

These come from the tier design (ticket 4) and apply to any client:

- Show **Low / Elevated / High** and a separate grey **no model coverage** state. Never render a no-coverage cell as Low.
- Do **not** show percentages. The raw probabilities are not calibrated, and the old tornet-detector cutoffs
  (0.1 / 0.4) must not be reused: this model's scores sit on a different scale.
- "High" means *highest relative risk*, not "likely": about 1 in 7 High cells verified, so most do not. A Low
  cell is not a safe cell: 40% of tornado cells fell in Low. Say "lower relative risk", never "low chance" or "safe".
- If hit rates are shown, use the measured sentence and its caveats, e.g. "In past forecasts, about 14% of High cells had a
  tornado within ~40 km in that 4-hour window (about 10% in 2015-2020, 20% in 2021-2025)". Never turn it into a
  per-forecast probability, and say it was measured on a proxy ensemble.
- Expect most runs to show **no High cells** (about 11% of afternoon runs have any); an empty High tier is normal, not a failure.
- Show the **valid window** and the **age**, and flag `stale` forecasts.

## Caching

| Route | `Cache-Control` | `ETag` |
|---|---|---|
| `/grid` | `public, max-age=86400` | `grid_id` |
| `/forecast/latest` | `no-cache` | `<id>-<bundle_id>-<tier_set_id>` |
| `/forecast/<id>` | `public, max-age=300` | `<id>-<bundle_id>-<tier_set_id>` |
| `/forecasts`, `/status` | `no-cache` | none |
| `/tiers` | `public, max-age=300` | `tier_set_id` |
| `/replay/<id>` | `public, max-age=300` | `<id>-<bundle_id>-<tier_set_id>-replay` |
| `/replays` | `no-cache` | none |
| `/reports` | `public, max-age=3600` | none |

Conditional requests (`If-None-Match`) return `304`. Compression (gzip) is a deployment decision (ticket 11).

## Payload size (measured against the real grid)

| Payload | Raw JSON | gzip |
|---|---|---|
| `/grid` (vertex mesh + country) | 219 KB | 76 KB |
| `/forecast/<id>` (2 bins x tiers + score) | 200 KB | 17 KB (a real, quiet forecast; an active map compresses worse) |
| (rejected) full cell polygons | 883 KB | 162 KB |

Budget: a forecast stays under 300 KB raw. The vertex mesh reproduces every one of the 11,178 cells' corners
(from `HrrrCoarseGrid.cell_polygon`) exactly before rounding; the 4-decimal rounding moves a vertex by at most 5.6 m
in either axis, 7.4 m as a distance (measured over all 11,398 vertices; an earlier version of this note said 5.5 m, which was the per-axis figure).

## Errors

| Code | When |
|---|---|
| `400` | Malformed forecast id, bad `limit`/`since`, missing or bad `/reports` `start`/`end` (or a window over 31 days) |
| `404` | Forecast or replay id not stored; unknown route |
| `503` | `GET /forecast/latest`: no forecast has been issued yet (empty store); sends `Retry-After: 300`. `GET /reports`: the SPC reports file is not installed (no `Retry-After`) |
| `500` | Unexpected error, or a stored forecast that does not match the running API (different `grid_id` or shape, unreadable files). Body is `{"error": "Internal server error"}`; details only in the server log |

A **stale** forecast is never an error: it is returned with `stale: true`.

## Quick test

```bash
curl http://localhost:5060/health
curl http://localhost:5060/status
curl http://localhost:5060/grid
curl 'http://localhost:5060/forecasts?limit=3'
curl http://localhost:5060/forecast/latest
curl http://localhost:5060/forecast/2026-10-19T01Z
curl http://localhost:5060/tiers
curl http://localhost:5060/replays
curl http://localhost:5060/replay/2021-12-10T21Z
curl 'http://localhost:5060/reports?start=2021-12-10&end=2021-12-12'
```
