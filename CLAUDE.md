# CLAUDE.md

Context for AI-assisted sessions in this repo. Read this before making
methodology or architecture decisions.

**This file is the compact, current view: rules, locked decisions, results
ladder, open items.** Full evidence (tables, numbers, predictions scored,
superseded findings) lives in `docs/history/` (verbatim archive of the old
2,170-line version of this file) and in the per-study pre-registration/results
files in `docs/`. See "Where to look" below; when a question needs a number or
a derivation, open the pointed file instead of guessing from the summary here.

## What this project is

Short-range (0-8h lead time) tornado forecasting using HRRR forecast fields as
input features and SPC storm reports as labels.

A **separate but complementary** project to the user's CNN radar tornado
*detection* model (`../tornet`, near-real-time). Do not conflate them or assume
shared code: independent codebases, independent goals (forecasting ahead vs.
detecting now). Long-term goal (see memory): forecaster-aid tool, then
transfer to Canada. A side track serves the model to the sibling Rails app
(`../tornado_prediction_radar`) through a Flask API.

## Working rules (apply to every new experiment)

- **Methodology is dense-grid classification** (Sobash et al. 2020), not
  storm-centered. Every `(grid cell, run, bin)` is a sample. Don't switch
  without discussing.
- **Judge by AUC-PR** (plus best-threshold F1/precision/recall). Never cite
  accuracy (>99.8% for anything), and treat AUC-ROC (0.92-0.99 for everything) and Cohen's d
  as secondary: they have disagreed with AUC-PR repeatedly. Loss is not a quality
  metric (flat/rescaled across configs). 0.5 is not a usable threshold for this
  focal-loss family; use each config's best-F1 threshold.
- **Seed noise is large (~19% relative SD of single-seed AUC-PR).** Single-seed
  rankings within ~+/-25-30% are not findings. Compare **ensembles of >=3 seeds**
  (minimum detectable effect ~12% with 3 seeds, ~21% with 1, 12-fold paired).
  Report 12-fold pooled results with bootstrap intervals, not single years or
  single splits (one split swung AUC-PR 2x: val 0.0398 vs test 0.0836).
- **Pre-register** (write the decision rule and predictions to `docs/*_preregistration.md`
  and commit before launching), then score predictions honestly, including misses.
  Never spin negative results; keep superseded findings with a "superseded" note
  rather than deleting them.
- **Log per-epoch held-out metrics** (AUC-PR/ROC, precision, recall), not only loss.
  **Early stopping must use a separate validation split of the training years**, never
  the test year. Studies that train 50 fixed epochs can compare an overfit model with a
  non-overfit one (this is exactly what hid the ConvLSTM result).
- **Never change without discussing first:** grid stride/projection (13x13 block-aligned),
  `BIN_EDGES_HOURS`, the every-hourly-run decision, the full-track spatial/single-timestamp
  labeling method (don't fabricate track timing), neighborhood-style training labels,
  a synthetic 0-8 km shear layer, interpolated grids, or storm-centered sampling.
- Ticket/future-work notes are written as title + description + starter prompt, not full
  technical plans (see memory). Plan first, per ticket.
- Commit only when asked.

## Locked decisions (condensed; full text in `docs/history/00_locked_decisions_detail.md`)

**Grid** (`src/tornado_predictor/grid.py`, built by `scripts/01_data_pipeline/build_grid.py`
-> `data/processed/hrrr_coarse_grid_stride13.nc`). Block-aligned stride coarsening of the
native HRRR LCC grid (std parallels 38.5N, central meridian -97.5, spherical R=6371229 m,
3 km native). Coarse cells = exact contiguous **13x13 native-pixel blocks (39 km)**, shape
**(81, 138)** rows x cols (11,178 cells); leftover partial blocks dropped. Full native domain,
no land clipping (includes ocean/Mexico/Canada; NW corner reaches ~134W because of LCC
curvature, not a bug). Row 0 is southernmost. Pure projection math, no downloads.
Constants verified against live GRIB (`verify_hrrr_grid_params.py`).

**Time bins** (`time_bins.py`, pure date arithmetic). Lead 0-8h, two half-open 4h bins:
bin 0 = [0,4h), bin 1 = [4h,8h). **Every hourly HRRR run (all 24/day) gets its own bins**
relative to its own init, deliberately, for sample density. **Label multiplicity is
intentional:** one report is positive in exactly 8 different hourly runs' sample sets
(`candidate_run_inits`); do not dedupe to one run per event. `fxx_to_bin_index` (integer
fxx; fxx=8 excluded) vs `assign_valid_time_bin` (continuous lead time for off-the-hour
reports). **Training period ~Sept 2014 - Sept 2025**: lower bound is the AWS HRRR archive start
(~Aug 15 2014), grid/fields/forecast length identical back to Sept 2014; upper bound is
SPC label availability (SPC CSV now runs to 2025-12-29; 37 staged runs fall after the old
2025-09-09 ceiling; earlier archive years, esp. 2014-2018, have more missing pulls).

**Features** (`features.py`, `scripts/01_data_pipeline/extract_features.py`). 7 physical fields
from HRRR `sfc` via one Herbie call each, exact search strings (a loose `"CAPE:"` matches other
CAPE variants): `cape` `CAPE:surface`, `cin` `CIN:surface`, `srh_0_1km` `HLCY:1000-0 m above ground`,
`srh_0_3km` `HLCY:3000-0 m above ground`, `shear_0_6km` = hypot(`VUCSH`,`VVCSH` `:0-6000 m above ground`),
`t2m`, `d2m` (`:2 m above ground`). **0-6 km shear, not 0-8** (HRRR has no 0-8 km field). Pooled onto
the coarse grid by reshape + mean/max per 13x13 block (no interpolation): 14 variables (`_mean`, `_max`
per field). **Canonical model input = these 14 base features.** The training dataset uses one
representative fxx per bin (fxx 0 and 4) unless the hourly/whole-bin store is used (below).

**Labels** (`labels.py`, `scripts/01_data_pipeline/build_labels.py`
-> `data/processed/tornado_labels_2014_2025.csv`). SPC `path_wkt` buffered by half `width_yd`
(must be >0), intersected with each cell's exact LCC bounding box (`grid.cell_bounds_xy`),
credits every cell a track crosses. Temporal matching uses the single touchdown timestamp
against the 8 candidate runs (permanent simplification). Only positives are materialized (one row
per event, run, cell); collapse with `.drop_duplicates()` only when building a sample table.
12 of 15,294 reports fall outside the grid. SPC has only US reports: **Canadian/Mexican cells are
always labeled 0** (label noise on ~17% of cells; see Canada section).

**Split** (`split.py`). Unit = UTC calendar date of `init_time`; `day-of-year mod 20 < 17` -> train (TorNet's rule),
24h temporal buffer drops (not reassigns) train samples near val. 3-way variant: train mod 0-13, test 14-16, val 17-19.
Year-holdout variant (`split_run_bins_year_holdout`) is the headline evaluation (12 leave-one-year-out folds).

**Class imbalance** (`training.py`): `FocalLoss` (alpha=0.25, gamma=2.0 stays the default; the alpha sweep
"prefer 0.9" conclusion was reversed by AUC-PR), `DenseGridDataset`, `QuietMapDownsampler` (map-level, keeps all
active maps, `quiet_keep_fraction=0.2`, redrawn per epoch; per-cell downsampling is impossible for a spatial CNN).
Models output raw logits; apply sigmoid at inference. Checkpoints are self-describing (`feature_names`,
`model_class`, hyperparameters); `inference.load_checkpoint` dispatches via `MODEL_REGISTRY`;
`assert_feature_order_matches` guards channel order.

## Canonical data (what exists and which to use)

- **Labels / SPC:** `data/processed/spc_tornado_reports_2014_2025.csv` (15,294 reports),
  `tornado_labels_2014_2025.csv` (147,440 rows, 111,580 unique positives).
- **Run store:** 4,919 of 5,050 targeted HRRR runs done (131 failures, 107 are permanently missing `.idx` files),
  staged in `data/interim/scaled_build_v2/` (14 base features only). Active runs are chosen by greedy interval covering
  (every report is inside some pulled run's 0-8h window; ~1.48 runs per active date); quiet dates are one fixed 20Z run
  each (all remaining quiet dates in v4). **Active-map share 43% (v4), per-cell positive rate ~1.4e-4.**
- **Dataset files** in `data/processed/`: `training_dataset_{train,val}_full_v4.nc` (current, 7,652 / 1,478 samples;
  **UH columns are mostly NaN there: drop the 7 UH columns before training**), `_full.nc` (v2, UH-enriched, older),
  `_3way.nc` (train/test/val), `_v1_superseded`/`_v2_superseded` (archived). For UH work use
  `data/interim/scaled_build_uh_merged/`. Hourly (fxx 0-7, plus 9 extra fields) store:
  `data/interim/scaled_build_hourly{,_extra}/` (4,916 runs), whole-bin derived: `scaled_build_wholebin{,_extra}/`.
- **Country mask:** `data/processed/hrrr_coarse_grid_stride13_countries.nc` (0 water, 1 US, 2 CA, 3 MX, 4 other;
  US-trusted = water+US, 9,269 cells).
- All data, models, outputs are gitignored and regenerable except `models/production/manifest.json` and tracked docs.

## Results ladder (where we are)

Headline numbers are AUC-PR on held-out data. Caveats above apply (seed noise ~19%).

| step | result | verdict / pointer |
|---|---|---|
| Baseline CNN (32ch, 3 layers, 7x7 field) | 0.0283 (single split) | `history/02` |
| Receptive field matters, not width | 5-layer 0.0319, dilated [1,2,4,8] 0.0322; wider channels hurt | `history/02` (single-seed, inside noise band) |
| **U-Net `TornadoUNet(base_channels=16)`**, 14 features | 0.0387 val; **12-fold year CV 0.0486 +/- 0.018 (old), 0.0442 +/- 0.0142 (v4)**, 2-model ensemble 0.0532 | `history/02`, `history/03` |
| vs. alternatives | GBT 0.029 vs U-Net 0.084 on same test (U-Net 2-3x); STP-like physics baseline ~0.016; climatology ~0.0013 | `history/03` |
| Extra quiet dates (v4) | no detectable change (-4%, p=0.54) | `history/01`, `history/03` |
| Updraft helicity features | **no detectable effect** after multi-seed re-test (original "hurts" was seed noise); keep 14 features | `history/03` (Parts 7-8 in sample_mix notebook) |
| Ensembling | +20-36% over a single seed (largest gain found); use EMA(0.9999) members, >=3-5 | `history/03` |
| **A: whole-bin hourly features** (all 4 fxx per bin, time-mean/max) | **+22.5%** vs baseline ensembles, 3/3 folds, confirmed | `docs/hourly_features_results.md`, `history/04` |
| B: A + MLCAPE/MUCAPE/LCL/0-1km shear/REFC/pressure (+tendency) | +9.7% vs A, weak, inside noise; no ablation, can't attribute to pressure | `docs/hourly_features_results.md` |
| ConvLSTM-U-Net on hourly store | first attempt -19.7% (saturated gates); repaired C3 -1.5%; +dropout +1.9% (no effect) | `docs/convlstm_*_results.md` |
| **Early-stopped C3 vs early-stopped A** | **+29.4% pooled, helps, confirmed** (2016/2020/2024 ensemble AUC-PR 0.0706/0.0847/0.1150) | `docs/convlstm_early_stopping_results.md` |
| Tuned arm-A control (lr 3e-4, clip 1) vs C3 | C3 still **+25.8%**, helps, confirmed (~88% survives) | `docs/unet_control_results.md` |
| US-only masked training | no detectable effect on US skill (valid clean baseline) | `history/06` |

**Best research model:** the early-stopped repaired ConvLSTM-U-Net (C3: GroupNorm before LSTM, pooled
skips, grad clip 1.0, lr 1e-4, EMA 0.9999) on hourly features. **Not shown:** that recurrence is the cause (C3 also
has GroupNorm and pooled skips; no no-recurrence ablation under early stopping). **Honest expectation:** AUC-PR is a
~190-260x improvement over no-skill but remains low in absolute terms; the deployment-like (all hourly runs, 20Z stratum)
estimate for the single-snapshot U-Net is ~0.02-0.03, not 0.044. Map-level discrimination (is *this* map active?) is the
weak link; false alarms concentrate on quiet bins of active runs (timing).

**Known confounds:** AUC-PR rises with calendar year (r~0.8) but is largely confounded with tornado-population composition
(track length, EF0 share), not cleanly attributable to HRRR improvements; do not cite it as HRRR skill gains.
Raw AUC-PR is prevalence-dependent; compare lift only within a stratum.

## Forecast API side track (tickets 1-8 of 11 done; detail in `history/07_forecast_api.md`, `docs/forecast_api_tickets.md`, `api/README.md`)

- **Served model:** production bundle `models/production/` (id `3c02381f1611`), 5-member EMA ensemble of 14-feature
  `TornadoUNet`, seeds 20-24, trained on **all** years (no held-out score for these exact weights; expected skill = the 12-fold CV).
  Still **single-snapshot features** (fxx 0 and 4); the whole-bin/ConvLSTM models are not in serving (would need fxx 0-7 live pulls and a retrain).
- Live extraction (`live_features.py`), forecast store (`forecast_store.py`, `data/live/`), hourly job (`forecast_job.py`,
  `scripts/09_serving/run_forecast_job.py`), tiers (`tiers.py`), Flask API on port 5060 (`forecast_api.py`, `forecast_app.py`, `api/app.py`),
  stored replays (6 in-sample cases) and SPC reports endpoint.
- **Tiers only, no percentages:** Elevated >= 0.122, High >= 0.190 on 3x3-max-smoothed ensemble mean (tier set `ts-e2c497b6`; High hit
  rate ~13.9%, hit rate era-dependent 10% early vs 20% since 2021); calibration failed the pre-registered criterion by a hair, so
  `calibrated_percent` stays null. Canada/Mexico cells are a grey "no model coverage" state.
- **Not done:** tickets 9-11 (Rails forecast page, verification overlay, deployment/cron packaging). Nothing is scheduled; `/status` reports stale.

## Canada transfer groundwork (detail in `history/06_canada_and_us_only.md`, `docs/canada_data_availability.md`)

HRRR domain stops ~52.5N (central Alberta at/over the edge). 1,243 Canadian cells, ~1 of them ever labeled positive. Masked-loss
(US-only) training: no detectable effect. **Stage C** (add Northern Tornadoes Project / ECCC labels, compare vs the US-only baseline)
is unplanned; an NTP data-access inquiry had no recorded reply.

## Open items / not done

- No-recurrence ablation (3 seeds, same early-stopping rule); hybrid of C3 with A's pooled fields; 12-fold confirmation of A and C3.
- Deployment-like (20Z) stratum evaluation; drop-one-group ablation of B; standardizing the base 14 features.
- U-Net `base_channels`/depth sweep and per-architecture hyperparameter tuning (everything shared one recipe).
- Serving bundle on hourly features; API tickets 9-11.
- Canada Stage C labels; engineered neighborhood features for GBT; precision-at-fixed-recall comparison of old vs v4.
- Nothing above is scheduled; ask the user which to take next.

## Pitfalls that have actually bitten

- **Activate `.venv` before running anything** (bare `python` hits a broken pyenv shim; `| tee` hides the exit code). Studies launched
  without the venv failed instantly.
- Mac sleeps on lid close during long runs (`caffeinate -i` doesn't stop that); wall clock can be 2-3x estimates.
- v4 `_full_v4.nc` UH columns are NaN for old runs: drop them. `xr.concat` silently NaN-fills mismatched variables.
- `DenseGridDataset` takes channel order from the first staged run; staged sources order variables differently (UH after shear in `_v3` copies).
  `inference.assert_feature_order_matches` and the CV script's test alignment guard this; always assert.
- Cohen's d can rise while AUC-PR falls; AUC-PR is the arbiter.
- numpy 2.0.2 on this Mac prints spurious matmul warnings in bootstrap code (values verified equal to einsum).
- Select dates/runs only inside the verified HRRR archive window (>= ~Aug 15 2014) *and* labeled range.
- Receptive-field gradient tests must use random (not zero) input: all-zero input lets ReLUs die uniformly.
- Herbie returns `grib=None` (doesn't raise) for unpublished runs; its console output is noisy in cron logs.

## Where to look

| question | file (search the heading in it) |
|---|---|
| Why grid/bins/labels/features/split are the way they are; training-period verification | `docs/history/00_locked_decisions_detail.md` |
| How each dataset build (pilot, 44-sample, v1, v2, v4) was made, failures, class balance | `docs/history/01_dataset_builds.md` |
| First full-scale training, alpha sweep, model-comparison metrics, architecture tickets 2-5 | `docs/history/02_architecture_search.md` |
| 3-way test set, GBT, UH (original), year CV, baselines, sample-mix diagnostics, seed variance, ensemble size, EMA, UH re-tests | `docs/history/03_evaluation_and_variance.md` |
| Hourly features A/B, ConvLSTM series, per-epoch, early stopping, tuned control (summaries) | `docs/history/04_hourly_and_convlstm.md` |
| Validation case, class imbalance, baseline CNN, training loop, inference, prediction-vs-detection notebook | `docs/history/05_pipeline_mechanics.md` |
| Country mask, Canadian coverage, US-only study | `docs/history/06_canada_and_us_only.md` |
| Forecast API tickets 1-8 details | `docs/history/07_forecast_api.md`, `docs/forecast_api_tickets.md`, `api/README.md` |
| The pre-compaction "Current status" paragraph | `docs/history/08_old_current_status.md` |
| Narrative of how it all got built | `docs/project_walkthrough.md` |
| Pre-registrations, results per study | `docs/hourly_features_*`, `docs/convlstm_*` (spec, tickets, repair plan, diagnostic ladder, c3, dropout, per-epoch, early stopping), `docs/unet_control_*`, `docs/tier_study_*` |
| Feature ideas and research notes | `docs/feature_candidates_research.md`, `docs/canada_data_availability.md` |
| Which script does what | `scripts/README.md` (numbered phase folders `01_data_pipeline` ... `11_convlstm`) |
| Result tables as notebooks | `notebooks/` (architecture_comparison, cv_year_holdout_results, sample_mix_diagnostics, convlstm_study, model_comparison_summary, cv_v4_vs_v2_comparison, data_exploration, alpha_sweep_results) |
| Raw numbers | `models/<study>/results.json` (gitignored; regenerable) |

Whole-archive search: `grep -rn "<term>" docs/ CLAUDE.md`. The old single-file version is also in git
history (`git show 6d2eaed:CLAUDE.md`).

**Old citations still resolve.** Code comments, docstrings, notebooks and docs say things like
`see CLAUDE.md "Time binning (locked in)"` or `CLAUDE.md "Seed-variance study and ensembles"`. Those were
section headings of the old file; each is now a heading in `docs/history/0*.md` with the same title.
Find it with `grep -rn "^### <title>" docs/history/` or the heading list in `docs/history/README.md`.

## Repo layout

- `src/tornado_predictor/`: package code (editable install via `pyproject.toml`). Key modules: `grid`, `time_bins`, `features`, `labels`,
  `dataset`, `split`, `training`, `evaluate`, `model`/`unet`/`convlstm_unet`, `inference`, `ensemble`, `build_scaled`/`build_hourly`,
  `cell_mask`, `tiers`, `live_features`, `forecast_*`, `replay_cases`, `spc_reports`, `gbt`.
- `scripts/`: CLI entry points by phase (`01_data_pipeline` ... `11_convlstm`), see `scripts/README.md`
- `data/{raw,interim,processed,live}/`, `models/`, `outputs/`: gitignored contents, regenerable
- `docs/`: pre-registrations, results, tickets, walkthrough; `docs/history/`: archive of the old CLAUDE.md
- `notebooks/`, `tests/`, `api/`

## Environment

Python virtualenv at `.venv/`:
```
pip install -r requirements.txt
pip install -e .
```
The editable install is required for scripts to `import tornado_predictor`. Run tests with the venv active (`pytest`).
