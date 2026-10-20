# Tornado Forecast API (contract)

> **Status: contract only.** Nothing here is served yet (the `/tiers` numbers are filled in from ticket 4). This document is what the Flask API (ticket 7)
> and the Rails frontend (tickets 9-10) both code against. Changing a response shape is a coordinated change
> with the Rails app, as with the tornet API's `/predict`.

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

## Setup (planned)

```bash
python api/app.py          # serves on :5060 (override with PORT)
```

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5060` | Port the Flask server binds to |
| `FORECAST_STALE_AFTER_SECONDS` | `10800` (3 h) | A forecast whose run is older than this is flagged `stale: true` |

Further variables (forecast store location, allowed CORS origins) are decided in tickets 3, 7 and 11.
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

Stored forecasts, newest first.

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
| `age_seconds` | Now minus **`init_time`**. The age of the model run, which is what matters to a forecaster. |
| `stale` | `age_seconds` exceeds `FORECAST_STALE_AFTER_SECONDS`. A stale forecast is still returned with `200`. |
| `grid_id` | Must equal `/grid`'s `grid_id`; if it differs, refetch `/grid`. |
| `model` | Which production bundle produced it. `trained_years` matters for historical replays, which are retrospective, not held-out tests (ticket 8). |
| `tiering` | Which tier set and smoothing produced `tiers`. Tiers are derived from the stored raw ensemble output, so they can be re-derived if the cutoffs change (the id changes with them). |
| `calibrated_percent` | Reserved. `null` unless ticket 4 concluded a calibrated percentage is defensible, in which case a per-bin percent array is added. |
| `tiers` | Per cell: `-1` no coverage, `0` Low, `1` Elevated, `2` High. Computed server-side; clients never compute tiers. |
| `smoothed_score` | The neighbourhood-smoothed ensemble score the tiers are cut from. **Uncalibrated relative index; do not display it as a percentage.** For hover text and ordering only. |

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
    "cutoffs_checked_on": "5-member EMA ensembles, 2016/2020/2024: High hit rate 13.9% [10.3, 17.6]",
    "era_note": "High hit rate was 10.3% in 2015-2020 and 19.8% in 2021-2025"
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
scale), in `models/tiers/tier_study.json` under `serving`, keyed by `tier_set_id`; the Flask layer (ticket 7) reads them from there.
`GET /forecast/*` carries the same `tier_set_id`. A calibrated percentage was **not** found defensible (see the results doc), so
no percentage field exists.

### Reserved for later

`GET /replay/<id>` and `GET /reports` (ticket 8). A replay returns the same envelope as `/forecast/<id>` with
`"mode": "replay"`, `"retrospective": true` and `trained_on_years`; `/reports` returns SPC tornado reports as GeoJSON.
Their shapes are specified in that ticket.

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

Conditional requests (`If-None-Match`) return `304`. Compression (gzip) is a deployment decision (ticket 11).

## Payload size (measured against the real grid)

| Payload | Raw JSON | gzip |
|---|---|---|
| `/grid` (vertex mesh + country) | 219 KB | 76 KB |
| `/forecast/<id>` (2 bins x tiers + score) | 200 KB | 36 KB |
| (rejected) full cell polygons | 883 KB | 162 KB |

Budget: a forecast stays under 300 KB raw. The vertex mesh reproduces every one of the 11,178 cells' corners
(from `HrrrCoarseGrid.cell_polygon`) to within 5.5 m, which is just the 4-decimal rounding.

## Errors

| Code | When |
|---|---|
| `400` | Malformed forecast id, bad `limit`/`since` |
| `404` | Forecast id not stored; unknown route |
| `503` | No forecast has been issued yet (empty store); sends `Retry-After: 300` |
| `500` | Unexpected error (body is `{"error": "Internal server error"}`; details only in the server log) |

A **stale** forecast is never an error: it is returned with `stale: true`.

## Quick test (once implemented)

```bash
curl http://localhost:5060/health
curl http://localhost:5060/status
curl http://localhost:5060/grid
curl http://localhost:5060/forecasts?limit=3
curl http://localhost:5060/forecast/latest
curl http://localhost:5060/forecast/2026-10-19T01Z
curl http://localhost:5060/tiers
```
