# Forecast API + Rails integration: ticket set

Drafted 2026-10-18. Goal: serve this project's HRRR U-Net ensemble to the sibling Rails app
(`../tornado_prediction_radar`) through a Flask API modeled on `../tornet/api/app.py`.
Each ticket is a Trello card: title, description, starter prompt (paste back to start it).

## Design findings behind the set

- tornet pattern: Rails has no DB; Stimulus controllers `fetch()` the Flask API directly
  (`RADAR_API_BASE_URL`); Flask has CORS, `/health`, JSON error handlers.
- Ours is a map (81x138 grid, two 4h bins per HRRR run), not a storm point, so send grid
  geometry once and compact arrays per forecast.
- HRRR files land on AWS about init+50 min (fxx=0) and init+59 min (fxx=4) (checked
  2026-10-18), so a run is usable about 1h after init and bin 0 is already about 1h old at issue.
- Scheduler is a separate process writing to disk (tornet's in-thread, in-memory design wipes
  on restart and duplicates under multiple workers).
- No input normalization in training: `DenseGridDataset` feeds raw pooled values, NaN->0.
  Serving must reproduce that exactly (parity test in ticket 2).
- Raw probabilities are uncalibrated; the old Rails tiers (>=0.1, >=0.4) must not be reused.
- Canadian/Mexican cells are always labeled 0, so the UI greys them out.
- Port 5060 (tornet uses 5050) so both APIs can run together.

## Decisions made

- Serve a 5-member all-years EMA(0.9999) U-Net ensemble, 14 features, alpha=0.25, plain mean.
- UI shows three tiers (Low / Elevated / High) plus grey "no model coverage", computed on a
  neighborhood-smoothed score (post-hoc, no retraining). No "% chance" unless ticket 4's
  calibration check passes.
- Store raw ensemble probabilities, never only tiers.

## Order and dependencies

Tickets 1, 2 and 6 in parallel; then 3, 5 and 4 (4 alongside); 7 needs 3 and 6; 8 needs 1-3;
9 needs 4, 6, 7; 10 needs 8 and 9; 11 last (or earlier for a partial deploy).

---

### 1. Production model bundle: 5-member EMA ensemble, trained on all years

**Status: done 2026-10-19** (bundle `3c02381f1611`; results and caveats in CLAUDE.md, "Forecast API for the Rails app").

**Description:** Train a 5-member ensemble of U-Nets on all available years, the model the
forecast API will serve. Add a tested `predict_ensemble` that averages member probabilities and
checks feature order, plus a manifest recording what was trained and how.

> Start ticket: production model bundle. **Decided (don't re-litigate):** serve a 5-member
> ensemble of `TornadoUNet(base_channels=16)`, all years (2014-2025, no held-out year), 14 base
> features (drop the 7 UH columns from the v4 files), `alpha=0.25`, `gamma=2.0`, 50 epochs,
> `quiet_keep_fraction=0.2`, EMA decay 0.9999 for every member (use the EMA weights, not the raw
> ones), fresh seeds (e.g. 20-24; seeds 10-15 are already used by the EMA study). Combine by plain
> mean of per-cell probabilities. Weighted or calibrated ensembling is untested, so leave it out.
> Also record the per-cell spread across members, but don't present it as a confidence interval,
> since its meaning is untested.
>
> **Why this recipe:** ensembles of >=3 EMA-averaged members meet the project's stability
> criterion, though narrowly (disjoint 3-vs-3 gap 6.9%, with 20.0% of splits beyond noise,
> exactly at the line). Gains over a single seed are about +19% at 2 members and +29% at 4, and
> flatten after 4-5. Ensembling reduces model variance only: the ceiling is unchanged and the
> deployment-like AUC-PR is still about 0.02-0.03. It does not fix calibration (separate ticket).
>
> **Facts:** the training data is `data/processed/training_dataset_{train,val}_full_v4.nc`, built
> from staged runs in `data/interim/scaled_build_v2/`. The UH columns are mostly NaN there, so
> drop them as the CV scripts do with `--drop-columns` (see
> `scripts/05_year_cv_diagnostics/cross_validate_year.py`). Checkpoints are self-describing
> (`feature_names`, `model_class`, `model_hyperparameters`), and `inference.load_checkpoint` plus
> `assert_feature_order_matches` already exist. Training uses no input normalization. Each
> training takes roughly 45 min on CPU, so about 4 hours in total. There is no held-out year for
> these exact weights, so the expected skill we quote must come from the existing 12-fold CV for
> the same recipe (ensemble AUC-PR about 0.053 for 2 members, single seed 0.044-0.049 +/- 25%).
> Don't claim a measured score for the production weights.
>
> **Open for planning:** how to split the train/val files for an all-years run (train on
> train+val combined, or keep val for monitoring only), where to write the checkpoints (e.g.
> `models/production/`), what goes in the manifest (checkpoint paths and hashes, `feature_names`,
> EMA decay, seeds, training years, git commit, expected-skill reference), and the
> `predict_ensemble` interface so the forecast store can record the member spread. Add tests for
> feature-order mismatch and ensemble averaging. Please plan before writing code.

### 2. Live HRRR feature extraction with a train/serve parity test

**Status: done 2026-10-19** (`src/tornado_predictor/live_features.py`; parity test passes on two historical runs).

**Description:** Add a function that finds the latest complete HRRR run, pulls fxx=0 (bin 0) and
fxx=4 (bin 1), and builds the exact 14-channel input the model trained on. A parity test replays a
historical run through this path and compares it with the stored training features.

> Start ticket: live HRRR feature extraction. Context: `features.extract_features(init_time, fxx,
> grid)` already pools to the 81x138 grid, and `dataset.representative_fxx` maps bin 0 to fxx=0 and
> bin 1 to fxx=4. Verified 2026-10-18: on AWS `noaa-hrrr-bdp-pds`, fxx=0 lands about init+50 min
> and fxx=4 about init+59 min, so a run is usable about 1h after init. Training applies no
> normalization: `DenseGridDataset` feeds raw pooled values with NaN->0, with channel order taken
> from the checkpoint's `feature_names`. Only the 14 base features are needed (don't pull the UH
> fields). Scope: latest-complete-run discovery with retry/backoff, a function returning a
> `(2, 14, 81, 138)` array in checkpoint order, and a parity test that runs one historical
> `(init_time, bin)` from a staged training file through the live path and asserts it matches the
> stored features. Known wrinkle: pull failures (missing `.idx`, EOF errors) happened in about
> 3-5% of archive pulls. Plan before coding.

### 3. Forecast store (disk-backed archive of raw ensemble output)

**Status: done 2026-10-19** (`src/tornado_predictor/forecast_store.py`, 38 tests in `tests/test_forecast_store.py`; `HrrrCoarseGrid.grid_id` added in `grid.py`; `data/live/` gitignored).
Layout: `data/live/<issued|replay>/<YYYY-MM-DDTHHZ>/{forecast.npz,meta.json}`; `put` is atomic (temp dir + rename) and never overwrites; `prune` only runs when called with an age limit.

**Description:** A small module that atomically writes each issued forecast to disk and reads it
back, with a retention policy. It replaces tornet's in-memory cache so restarts lose nothing and
every issued forecast is kept for later verification. It stores the raw ensemble probabilities,
not tiers, so cutoffs can be revised and the archive re-tiered later without regenerating
forecasts.

> Start ticket: forecast store. Context: tornet's `LiveCache` is in-memory and wiped on restart.
> We issue one forecast per hourly HRRR run, each holding a `(2 bins, 81, 138)` probability array,
> so disk storage is cheap and the archive has research value (prospective verification later).
> **Decided:** the store keeps the raw ensemble mean probabilities and the per-cell member spread,
> never only derived tiers. Tiers and the neighborhood-smoothed score are computed from the raw
> arrays by ticket 4's module, at read time or in a derived layer, so a later cutoff change doesn't
> require regenerating forecasts. The spread is stored but not presented as a confidence interval,
> since its meaning is untested.
>
> Design the on-disk format (e.g. one compressed `.npz` + JSON metadata per `init_time`, under a
> gitignored `data/live/`). Metadata should cover init time, valid windows per bin, the production
> model-bundle manifest id (ticket 1), generation time and the HRRR fxx values used. Needs atomic
> writes (temp file + rename), `latest()`, `get(init_time)`, `list()`, and a retention policy with
> an explicit "never delete" option. Also keep replay results (ticket 8) in a separate namespace so
> they can't be mistaken for issued forecasts. Depends on ticket 1's manifest id. Plan first.

### 4. Risk tiers on a neighborhood-smoothed score (and a calibration decision)

**Status: done 2026-10-20** (`src/tornado_predictor/tiers.py`, 27 tests; `scripts/09_serving/tier_study.py`; plan `docs/tier_study_preregistration.md`; results `docs/tier_study_results.md` and `models/tiers/tier_study.json`).
Outcome: max 3x3 smoothing; Elevated >= 0.122, High >= 0.190 (`ts-e2c497b6`); High hit rate 13.9% per cell, Low holds 40% of tornado cells; calibrated percentage **not** defensible by the pre-registered rule (C1 narrowly failed); `/tiers` filled in `api/README.md`. Real hour-to-hour tier flicker is deferred to the live archive.

**Description:** Turn the ensemble's raw per-cell scores into three honest tiers (Low / Elevated /
High) on a neighborhood-smoothed score, with each tier's meaning backed by measured numbers from
the year-holdout CV. It also decides whether a calibrated "% chance" is defensible. If calibration
fails, tiers are the only display, which is an acceptable outcome. This gates the Rails UI.

> Start ticket: risk tiers. **Decided (don't re-litigate):** the UI shows three tiers, *Low /
> Elevated / High*, plus a separate grey "no model coverage" state for Canadian and Mexican cells
> (use the country mask `data/processed/hrrr_coarse_grid_stride13_countries.nc`). The tiers are
> computed on a neighborhood-smoothed score, applied after the model, with no retraining. Don't
> label tiers as "% chance" unless the calibration check below passes.
>
> **Why:** raw probabilities are not calibrated (focal loss alpha=0.25, plus a training mix with a
> higher positive rate than an all-hourly-runs stream; the deployment-like AUC-PR is about
> 0.02-0.03 versus 0.044 on the CV mix). The best-F1 threshold is about 0.21 per model, recall at
> 0.5 is exactly 0, and the old Rails cutoffs (>=0.1 moderate, >=0.4 high) were set for the tornet
> detector and must not be reused. Neighborhood-dilated verification (scores unchanged) lifted
> precision at best-F1 from about 8% per cell to about 20% within one cell (about 39 km), so a
> smoothed score matches what a forecaster can verify and avoids a speckled map. Seed noise is
> about 19% for a single model, so tier cutoffs must be set on the ensemble output.
>
> **Scope:**
> 1. Choose the smoothing operator (e.g. max versus mean over a 3x3 window, or k=2) by comparing
>    candidates on the year-CV scores, with the choice pre-registered before looking.
> 2. Define tier cutoffs from measured behavior. For each tier, report its share of the domain
>    area, the share of tornado cells it captures, and its hit rate, using dilated labels at the
>    same radius as the smoothing. Use a deployment-like sample, not the CV mix (apply a
>    prior-shift correction or restrict to the 20Z stratum, whichever planning justifies). Cached
>    held-out scores are under `data/interim/diag_scores/`.
> 3. Measure tier stability: how often a cell changes tier between consecutive hourly runs, and
>    how much of that is seed noise versus real change, using the ensemble.
> 4. Calibration check: test whether the smoothed score can be mapped to a defensible probability
>    with a reliability analysis (SPC's tornado probabilities are defined within about 25 miles of
>    a point; confirm from SPC docs, not memory). Pre-register the pass/fail criterion. If it
>    fails, record that and ship tiers only.
> 5. Write the wording rules: High means "highest relative risk", not "likely" (even at best-F1,
>    about 9% of flagged cells per cell are hits, roughly 10 flagged cells per tornado), and avoid
>    "low chance" phrasing.
>
> **Out of scope:** changing the training labels to a neighborhood definition would alter the
> locked labeling method and needs a separate discussion. Output: a small module (smoothing, tier
> assignment, tier stats) with tests, a results JSON, and a short written conclusion. Report
> numbers exactly as measured, including bad news. Please plan before writing code.

### 5. Hourly forecast scheduler (separate process)

**Status: done 2026-10-20** (`src/tornado_predictor/forecast_job.py`, `scripts/09_serving/run_forecast_job.py`, 26 tests in `tests/test_forecast_job.py`; verified on the real latest HRRR runs, second pass a no-op).
Retry cap per failing run (default 3) and a job lockfile were added beyond the original brief; `status.json` lives at `<store root>/status.json`.

**Description:** A CLI job that runs once per hour, finds the newest complete HRRR run, runs the
ensemble and writes to the forecast store. It skips runs already done and can backfill gaps after
downtime.

> Start ticket: hourly forecast scheduler. Context: tornet runs its scheduler as a thread inside
> Flask, which breaks under multiple workers and loses state on restart. We want a separate script
> (`scripts/09_serving/...`, run via cron/launchd/a loop) writing to the forecast store from ticket
> 3, using tickets 1-2. Requirements: idempotent (skip an `init_time` already stored), backfill the
> last N missed runs, treat a missing or incomplete HRRR run as "retry later" rather than an error,
> and log failures into a small status file the API can expose. Known fact: files land about
> init+50-60 min. Don't do any calibration here (ticket 4). Plan first.

### 6. Flask API contract and README

**Status: done 2026-10-19** (`api/README.md`; payload sizes and mesh fidelity measured against the real grid).

**Description:** Specify the response shapes before building, written the way tornet's
`api/README.md` is: route tables, field meanings and error codes. This is the contract the Rails
app and the Flask app both code against. It carries the tier per cell and the metadata the legend
needs.

> Start ticket: API contract. Context: tornet's Flask API (`../tornet/api/app.py`,
> `api/README.md`) is the reference. Ours serves a grid, not storm points, so design a compact
> contract. Proposed routes: `GET /health`; `GET /grid` (static cell polygons or bounds + country
> mask, cacheable); `GET /forecasts` (available init_times); `GET /forecast/latest` and
> `GET /forecast/<init_time>`; and `GET /tiers` (the current tier definitions and per-tier
> historical hit rate and area share, from ticket 4).
>
> **Decided in ticket 4:** each `/forecast` response carries, per bin: the valid window, the tier
> per cell (Low / Elevated / High, computed server-side), the neighborhood-smoothed score, and a
> coverage flag (US or water versus no model coverage for Canada, Mexico and other land), so Rails
> never computes tiers itself. Tiers are computed from the raw stored probabilities (ticket 3), so
> the response must say which smoothing operator and cutoff set produced them, and that identifier
> must be part of (or traceable from) the manifest id. Raw probabilities are not a public field by
> default, because they are uncalibrated and the UI mustn't present them as percentages; expose
> them only if planning decides a forecaster view needs them. A calibrated percentage field exists
> only if ticket 4 concluded calibration is defensible. Also include issue time, generation time,
> staleness and the model-manifest id.
>
> Other decisions to make: array encoding for 11,178 cells (e.g. tier as a small-int array,
> rounded score), how non-US cells are represented, error shapes (503 with a clear message when no
> forecast exists yet), and caching headers for `/grid` and `/tiers`. Port 5060 (tornet uses
> 5050), `CORS(app)`, `PORT` env var. A change here is a coordinated change with Rails, as with
> tornet's `/predict` gotcha. Write the contract and a README only; no implementation. Plan first.

### 7. Flask API implementation

**Status: done 2026-10-20** (`src/tornado_predictor/forecast_api.py` builders, `forecast_app.py` Flask factory, `api/app.py` entrypoint; 63 new tests; verified over real HTTP against the real `data/live/` archive).
Decisions made in implementation: `/forecasts` on an empty store is 200 with an empty list (only `/forecast/latest` is 503); clients compute age from `init_time` because a 304 keeps an old `age_seconds`; a stored forecast whose grid does not match is a logged 500, never silently tiered; tiers use the same smooth -> float16 -> cutoff path the tier study measured (`tiers.tier_forecast`).

**Description:** Implement the routes from ticket 6 as a thin read-only layer over the forecast
store, with the same error handling, CORS and `/health` conventions as tornet.

> Start ticket: Flask API implementation. Context: the contract is in ticket 6's README, the store
> is ticket 3's module, and tornet's `api/app.py` is the style reference (JSON error handlers for
> `HTTPException` and generic exceptions, `CORS(app)`). Keep Flask read-only: no model loading and
> no HRRR pulls in request handlers, since the scheduler (ticket 5) writes to disk. Serve `/grid`
> with caching headers. Return 503 with a clear message when no forecast exists yet, and expose
> staleness. Add pytest tests using a fixture forecast on disk. Plan before coding.

### 8. Historical replay and SPC reports endpoint

**Description:** The equivalent of tornet's `/predict/<storm_id>`: let the UI pull the forecast
for a past case plus the SPC reports that verified it. It makes the Rails page demonstrable in
October with no live severe weather.

> Start ticket: historical replay. Context: the live route only shows now, and there's little
> severe weather to look at in the off-season. tornet's `/predict/<storm_id>` lets the UI pull any
> historical storm. Add `GET /replay/<init_time>` (compute via the same feature + ensemble path as
> live, cache the result in the store under a separate namespace) and `GET /reports?start=&end=`
> returning SPC tornado reports from `data/processed/spc_tornado_reports_2014_2025.csv` as GeoJSON
> (touchdown point plus track). Suggested showcase cases: the 2021-12-10/11 Mayfield outbreak (used
> in the validation script) and a 2024 outbreak. Caveat: a replay run is genuinely held-out only if
> the ensemble wasn't trained on that date, so the response must say which years the served model
> trained on (the production model trains on all years, so replays are retrospective, not tests).
> Depends on tickets 1-3. Plan first.

### 9. Rails forecast page (three tiers plus coverage state)

**Description:** Add a `/forecast` page to the Rails app: a Leaflet map of the grid showing Low /
Elevated / High tiers, with controls for run time and 4-hour bin. It shows a legend with each
tier's historical hit rate, and greys out cells with no model coverage. It follows the app's
existing conventions (Stimulus, custom DOM events, API base URL from a Rails initializer).

> Start ticket: Rails forecast page in `../tornado_prediction_radar`. Context: that app is Rails 8
> + Stimulus + importmap + Tailwind, with no database. Controllers just pass `@radar_api_base_url`
> to views, and Stimulus controllers `fetch()` the API directly and talk to each other via custom
> DOM events (see its CLAUDE.md). Add a `FORECAST_API_BASE_URL` initializer (default
> `http://localhost:5060`), a `ForecastsController#index` + route `/forecast`, and Stimulus
> controllers for run/bin selection and the map.
>
> **Display decisions (from ticket 4):** three tiers, *Low / Elevated / High*, computed
> server-side on the neighborhood-smoothed score, and the API sends the tier per cell (the client
> does not compute tiers). Canadian and Mexican cells render as a separate grey "no model
> coverage" state, not as Low. Don't show percentages unless ticket 4 concluded calibration is
> defensible. The old tier cutoffs in `radar_map_controller.js` (>=0.1 moderate, >=0.4 high) must
> not be reused. Wording: "highest relative risk", never "likely".
>
> **Scope:** render with Leaflet canvas polygons from `/grid` + the per-cell tier array, not 11k
> SVG paths. Show the valid window ("valid HH-HH+4Z"), since a run lands about 1h after init, so
> bin 0 is already about an hour old at issue time. Show issue time and a staleness badge. The
> legend should show each tier's historical hit rate and area share from ticket 4's table. Add a
> Rails controller test for the page. Depends on tickets 4, 6 and 7. Plan before coding.

### 10. Rails verification overlay

**Description:** Add SPC tornado reports on top of the forecast map in Rails, driven by the
replay endpoint, so forecasters can compare forecast and outcome.

> Start ticket: Rails SPC report overlay. Context: ticket 8 adds `/replay/<init_time>` and
> `/reports`; ticket 9 adds the `/forecast` page. Add a replay mode (pick a historical case or init
> time) and a report layer (touchdown points plus tracks, colored by EF rating). Keep to the app's
> conventions: a new Stimulus controller communicating with the map via custom events. Label
> replayed cases clearly as retrospective. Plan first.

### 11. Packaging and deployment

**Description:** Decide where the API and scheduler run and make them reproducible: dependencies,
system packages, Herbie cache, process model. The Rails app deploys with Kamal to one server.

> Start ticket: deploy the forecast API. Context: need a pinned `requirements/api.txt`-style file
> (torch CPU, herbie-data, cfgrib/eccodes system dependency, flask, flask-cors, gunicorn), a
> Dockerfile, and a plan for running two processes (Flask API read-only; scheduler writing to a
> shared volume). Inference is cheap (about 450k-parameter U-Net on an 81x138 grid, CPU is fine).
> Unknowns to settle first: where tornet's Flask API runs today and whether this one shares that
> host, how CORS origins are restricted in production, and where the Herbie download cache and
> forecast archive live. Pitfall from tornet: don't start a scheduler thread at import time under a
> multi-worker server. Plan first.
