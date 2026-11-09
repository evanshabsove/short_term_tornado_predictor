# Forecast API for the Rails app (tickets 1-8)

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Forecast API for the Rails app (side track; tickets 1-8 of 11 done)

Goal: serve the HRRR U-Net ensemble to the sibling Rails app (`../tornado_prediction_radar`) through a Flask API modeled on `../tornet/api/`.
The full ticket set (title/description/starter prompt per ticket, plus the decisions behind it) is `docs/forecast_api_tickets.md`; the contract is
`api/README.md`. Decided: a 5-member all-years EMA ensemble; the UI shows three tiers (Low / Elevated / High) on a neighborhood-smoothed score plus a grey
"no model coverage" state for Canada/Mexico, and no "% chance" unless ticket 4's calibration check passes; raw ensemble probabilities are stored, never only tiers.

- **Ticket 1, production bundle (`models/production/`, `src/tornado_predictor/ensemble.py`, `scripts/09_serving/train_production_ensemble.py`).** 5 members,
  `TornadoUNet(base_channels=16)`, seeds 20-24, every `done` run (4,919 runs / 9,838 samples, 43.2% active, years 2014-2025, no validation set), 14 base features
  (UH columns dropped), alpha=0.25, gamma=2.0, 50 epochs, EMA 0.9999 weights saved. ~47-78 min per member on CPU (~4.9 h total). Bundle id `3c02381f1611`.
  `ensemble.load_bundle` verifies each checkpoint's sha256, the bundle id, and one shared feature order; `predict_ensemble` returns the plain mean plus the
  population-std spread. `models/production/manifest.json` is tracked (`.gitignore` exception); the checkpoints are not.
  **There is no held-out score for these exact weights** (every year is in training). The expected skill is the existing 12-fold CV of the same recipe, with
  the usual +/-19-25% single-seed noise; the deployment-like expectation stays ~0.02-0.03 AUC-PR. Mechanics checks only
  (`scripts/09_serving/check_production_bundle.py`, `models/production/mechanics_check.json`):
  final train loss 0.000129-0.000136 vs. 0.000132 mean for the 18 CV runs (range 0.000120-0.000146), so training behaved like the CV runs;
  members agree closely (pairwise map correlation 0.902-0.945); mean predicted p over US cells 0.0068-0.0083 per member vs. 0.0064-0.0084 for the nine CV EMA models
  scored on the same 200-run sample (an earlier ~0.0128 reference came from a different sample and is not comparable; I first read the gap as a problem, then
  scored both on the same sample). **In-sample AUC-PR is 0.0333-0.0600 per member and 0.0500 for the ensemble: training-set fit, not skill, and not above the
  held-out CV range (0.044-0.049 single seed).** Two things it shows: the models are not memorizing, and member-to-member AUC-PR spread (0.033-0.060) is as large as the
  seed noise documented above, with this ensemble's in-sample AUC-PR below its best member's (a plain mean need not beat the best member).
- **Ticket 2, live feature extraction (`src/tornado_predictor/live_features.py`, `scripts/09_serving/live_features_smoke.py`).** `latest_complete_run` walks back
  hourly from `floor(now - 50 min)` and returns the newest run whose fxx=0 and fxx=4 `.idx` files are on AWS with all 8 live search strings (None means retry later);
  `extract_live_inputs` returns `(2, 14, 81, 138)` float32 in checkpoint order, wrapping `features.extract_features` (now `include_uh=False` for the live path: 8 pulls per fxx,
  default behavior unchanged). Raises on NaN instead of imputing 0 (the base fields have none in training). Verified HRRR timing on AWS: fxx=0 lands ~init+50 min and
  fxx=4 ~init+59 min, so a run is usable ~1 h after init and bin 0 is already ~1 h old at issue. **Parity test** (gated by `RUN_NETWORK_TESTS=1`): the live path reproduces the
  staged training features for a 2019-era and a 2023-era run (rtol 1e-6) and a CV checkpoint gives identical predictions on both. Smoke run on the real latest
  run (2026-10-19 01:00Z): 16 pulls in 31 s, every channel inside the training sample's range. Herbie returns `grib=None, idx=None` (does not raise) for an unpublished run
  and refuses future dates.
- **Ticket 6, API contract (`api/README.md`).** Port 5060 (tornet uses 5050); flat row-major arrays (`i = row*138 + col`, row 0 southernmost); `/grid` is a shared 82x139
  vertex mesh (219 KB raw vs. 883 KB for full polygons; reproduces all 11,178 `cell_polygon` corners to within 5.5 m); a forecast payload is 200 KB raw (budget 300 KB);
  `age_seconds` is measured from the run's init time, unlike tornet's cache-age `staleness_seconds`; `stale` after 3 h (env-configurable).
- **Data note (does not change any result):** the SPC CSV now runs to 2025-12-29, not 2025-09-09 as the "Time binning" section says (that figure was true when checked).
  37 staged runs fall after the old 2025-09-09 label ceiling (active-date runs only, since quiet dates were selected up to the ceiling); they are in the production training set
  and recorded in its manifest.
- **Ticket 3, forecast store (`src/tornado_predictor/forecast_store.py`, `HrrrCoarseGrid.grid_id` in `grid.py`).** `data/live/<issued|replay>/<YYYY-MM-DDTHHZ>/{forecast.npz,meta.json}`
  (gitignored): raw ensemble `mean` and `spread`, float32 `(2, 81, 138)`, never tiers; metadata carries bundle id, trained years, valid windows, `grid_id`, git commit. `put` writes a temp
  dir and renames it (atomic), never overwrites, and refuses non-finite arrays; retention is never-delete unless `prune` is called with an age limit. `grid_id` hashes the grid's defining
  constants (not the cell-center arrays), so it is platform-stable.
- **Ticket 5, hourly scheduler (`src/tornado_predictor/forecast_job.py`, `scripts/09_serving/run_forecast_job.py`).** One pass checks the last `--backfill-hours + 1` (default 7) hourly runs,
  newest first: already stored -> skipped (no network); not fully published -> "incomplete", retried by a later pass and **not** a failure (the newest run is routinely unpublished); else
  extract -> `predict_ensemble` -> store. A per-run error is recorded and the rest still run. A failing run is retried at most `--max-attempts` (3) times so an AWS-side gap (missing `.idx`)
  is not re-pulled hourly; `--retry-failed` overrides. `status.json` (next to the store) feeds `GET /status`; `consecutive_failures` counts consecutive *jobs* that failed at least once and
  stored nothing. An `flock` lockfile makes overlapping cron/launchd invocations exit cleanly. Intended to run every ~10 min (or `--loop`); cron/launchd packaging is ticket 11.
  **Real run (2026-10-20 ~03:54 UTC):** stored the 01Z and 02Z runs (~32 s of pulls each), left 03Z as incomplete (published later, ~init+55 min); a second pass stored nothing and exited 0.
  Stored values are finite in [0, 0.072]; mean over US/water cells ~0.0029-0.0030 per bin. That is lower than the 0.0068-0.0083 per-member figure in the ticket 1 mechanics check, but
  that check used an active-heavy training sample (43% active maps), so the two are not comparable and this is a plausibility check, not a validation. Known nuisance: Herbie's per-pull
  console output is very noisy for a cron log (not suppressed yet).
- **Ticket 4, risk tiers (`src/tornado_predictor/tiers.py`, `scripts/09_serving/tier_study.py`; pre-registration `docs/tier_study_preregistration.md`, results `docs/tier_study_results.md`, numbers `models/tiers/tier_study.json`).**
  Evaluation-only, on **proxy** scores (A = old+v4 2-model leave-one-year-out ensemble, 12 years; B = 5-member EMA ensembles on 2016/2020/2024); the served ensemble has no held-out scores.
  Rules were written before any score was loaded (the JSON records the pre-registration's sha256; one erratum, 7 bins not 8, was fixed before running). Primary sample = init-20Z stratum.
  - **Serving tiers:** max-smoothing over 3x3 (mean was +3.8% AUC-PR but the paired interval [-0.0008, +0.0070] includes 0, so the default applies); **Elevated >= 0.122, High >= 0.190**
    (High = best-F1 point, Elevated = 60% capture), tier set `ts-e2c497b6`. Cutoffs measured on A transfer to 5-member EMA ensembles (area ratio 1.14 Elevated / 1.24 High, limit 1.33), so A's fixed cutoffs are served.
  - **Per tier (per cell, 4-hour bin, hit = tornado within the 3x3 neighborhood):** Low 0.012% hit rate, holds **39.9% of tornado cells**; Elevated 3.9% [3.3, 4.4], 33.7% of tornado cells; High **13.9% [11.7, 16.0]**,
    26.3% [21.4, 31.2]. Areas are tiny (Elevated+High 0.235% of cell-samples; ~11% of 20Z maps contain any High cell, and 45% of those had a tornado somewhere vs ~11% base). The weighted full mix
    and B's three folds give the same picture (High hit 15.1% and 13.9%).
  - **The pooled hit rate hides an era shift:** High-tier hit rate 10.3% in 2015-2020 vs 19.8% in 2021-2025 (same calendar trend as the CV; partly tornado-population composition).
  - **Calibration: not defensible by the pre-registered rule, though close.** Reliability ratios are 0.98-1.03 in the six populated bins, but C1 required the bootstrap interval to contain 1 and it did not in two bins (by <0.01);
    C2-C4 passed. The High-tier ratio drifts with year (1.1-1.9 early, 0.58-0.80 from 2021; Spearman -0.73, p=0.011), so a single map would be off 30-50% in either era. `calibrated_percent` stays null; do not
    show percentages. The criterion was not relaxed after seeing the result; revising it needs a new pre-registration.
  - **Stability:** disjoint 3-member ensembles agree on 55% of High cells (Jaccard 0.554); hour-to-hour flicker could not be measured (only 20 consecutive-hour run pairs in the staged data) -- deferred to the live archive.
  - **Predictions scored:** P1, P6 met; P2 (High area 0.3-3%) missed by ~10x (observed 0.035%), P3 (Elevated cap binds) and P4 (cutoffs fail to transfer) wrong, P5 right that it fails but wrong on which criterion.
- **Ticket 7, Flask API (`src/tornado_predictor/forecast_api.py` builders without Flask, `forecast_app.py` factory, `api/app.py`; contract `api/README.md`).**
  Routes `/health`, `/status`, `/grid`, `/forecasts`, `/forecast/latest`, `/forecast/<id>`, `/tiers` on :5060, read-only over the store: importing it never loads torch/herbie or the scheduler (tested in a fresh interpreter), no
  background thread, no writes (tested). Tiers are computed per request from the stored raw ensemble mean by `tiers.tier_forecast` -- smooth, quantize to float16, compare with the cutoffs, the exact order the tier study
  measured -- using `models/tiers/tier_study.json`; the app refuses to start if that file's `tier_set_id` does not match its own recomputation. `/tiers` numbers are read from the study file at runtime. Added
  `HrrrCoarseGrid.vertex_lonlat()` for the `/grid` mesh and pinned `flask`, `flask-cors` and their dependencies in `requirements.txt`.
  **Verified over real HTTP** against the two real forecasts (01Z, 02Z 2026-10-20): every route answers in 2-28 ms, process RSS 157 MB, `/forecast/latest` is 200,528 B raw / 17,123 B gzip, served tiers equal a direct
  computation, 304 and CORS preflight work, an empty store gives `/forecasts` 200 (empty) and `/forecast/latest` 503 + `Retry-After: 300`. Both real forecasts are quiet-night maps with **no Elevated or High cells**, so the
  tier boundaries are covered by synthetic-peak tests (a score exactly at a cutoff goes up a tier; a score a full float16 step below stays down; the 3x3 spread; Canada/Mexico cells stay -1 whatever their score), not by a real severe case.
  **Two corrections to ticket 6's contract found while building:** the README's "vertex mesh within 5.5 m" was a per-axis figure -- over all 11,398 vertices the 4-decimal rounding moves a corner up to 5.6 m per axis and 7.4 m as a
  distance (now stated); and `age_seconds` cannot be trusted after a 304 because the ETag does not change with time (clients compute age from `init_time`). flask-cors 6.x answers a wildcard policy by echoing the caller's origin.
  Needs the gitignored country mask at startup (a deployment item for ticket 11). Nothing is scheduled yet, so `/status` reports `stale` until `run_forecast_job.py` is run on a timer.
- **Ticket 8, replay and SPC reports (`GET /replay/<id>`, `GET /replays`, `GET /reports?start=&end=`; `replay_cases.py`, `spc_reports.py`, `forecast_job.run_replays`, `scripts/09_serving/run_replay.py`).**
  A replay is the production ensemble run on a past init time, stored in the store's separate `replay` namespace and tiered by the same `tiers.tier_forecast` path as live. **The API only serves stored replays; it never computes
  one** (model loading plus ~16 HRRR pulls would not belong in a request handler -- a departure from the ticket text, which proposed compute-on-request). `run_replay.py --all-cases` computed six (Mayfield 2021-12-10 21/22/23Z;
  2024-04-26 18/20/22Z), ~32 s each. Replay payloads carry `mode: "replay"`, `retrospective: true`, `trained_on_years` and a note that the date is inside the training years, and null `age_seconds`/`stale`.
  `/reports` returns SPC reports for a half-open window (max 31 days) as GeoJSON: a `touchdown` Point plus a `track` LineString (SPC's straight touchdown-to-lift-off line; zero-length tracks, 7.9% of reports, are points only; unknown EF, 10%,
  is null). If the reports CSV is missing the API still starts and `/reports` is 503 without `Retry-After`. `/replays` lists stored replays and curated cases with availability (added for the Rails case picker).
  **Real-server check (all in-sample dates, so plumbing only, not skill):** the six replays serve in ~13 ms at ~200 KB; unlike the quiet live night they have 180-328 High and 385-511 Elevated cells each; of the SPC touchdowns inside each
  bin window, 83-100% lie within one cell of an Elevated-or-High cell (e.g. 5 of 6 and 11 of 12 in the two earliest bins, all of the rest). That figure is **not** evidence of skill: the model trained on these dates (and on these reports as labels), and the 60% Elevated-or-High capture measured out of sample
  is the honest number. One bug found by a real-data test while building: `ReportsCatalog` failed on timezone-aware timestamps (fixed; the API always normalized before calling it).
- **Not done:** tickets 9-11 (Rails forecast page, Rails verification overlay, deployment). No percentage field exists; the API serves tiers only.


