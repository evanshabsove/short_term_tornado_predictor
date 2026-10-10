# CLAUDE.md

Context for AI-assisted sessions in this repo. Read this before making
methodology or architecture decisions.

## What this project is

Short-range (0–8h lead time) tornado forecasting model using HRRR forecast
fields as input features and SPC storm reports as labels.

This is a **separate but complementary** project to the user's existing
CNN-based tornado detection model, which operates on radar data for
near-real-time detection. Do not conflate the two or assume shared code —
they are independent codebases with independent goals (forecasting ahead
of time vs. detecting now).

## Methodology (decided, follow unless told otherwise)

**Dense-grid classification**, following Sobash et al. (2020) — NOT a
storm-centered approach (e.g. TorNet-style, where samples are built
around known storm objects).

Key properties:
- **Sample unit**: every `(grid cell, time bin)` pair is one sample.
  Sampling is exhaustive over the domain and forecast period, not
  restricted to cells near observed storms.
- **Grid resolution**: ~40 km cells.
- **Time binning**: two 4-hour bins across the 0–8h HRRR forecast window,
  defined independently per hourly HRRR run (see "Time binning (locked
  in)" below).
- **Features**: HRRR forecast fields (severe-weather-relevant fields —
  CAPE, shear, helicity, etc.), retrieved via Herbie.
- **Labels**: derived from SPC storm reports — a cell/time bin is
  positive if a tornado report falls within it (exact spatial/temporal
  matching rules are still to be finalized as the labeling pipeline is
  built).

When implementing data loading, feature engineering, or model training,
preserve this dense-grid structure: don't silently switch to a
storm-centered sampling scheme without discussing it with the user first,
since that would change the entire methodology and break comparability
with the Sobash et al. baseline this project follows.

### Grid definition (locked in)

The ~40 km grid cells are built by **block-aligned stride coarsening of
the native HRRR grid**, not an independently reprojected/interpolated
grid:

- Reuses HRRR's own native Lambert Conformal Conic projection (standard
  parallels 38.5°N, central meridian -97.5°, spherical earth
  R=6371229m, 3km native spacing) — see `src/tornado_predictor/grid.py`.
  These constants are verified against a live HRRR grib file's `GRIB_*`
  keys by `scripts/verify_hrrr_grid_params.py`.
- Coarse cells are exact contiguous 13x13 blocks of native HRRR pixels
  (13 * 3km = 39km, the closest integer stride to "~40km"). Coarse grid
  shape is (81, 138) rows x cols; any leftover partial block at the
  north/east domain edge is dropped rather than kept as a ragged cell.
- Full native HRRR domain is used — no clipping to a CONUS land
  boundary, so the grid extends over ocean/Mexico/Canada at the domain
  edges. Note the domain is *not* a simple lat/lon bounding box: LCC
  projection curvature means the NW corner reaches as far as ~134°W
  despite the SW corner only being at ~123°W — this is genuine grid
  geometry, not a bug, confirmed by computing all four native corners
  directly.
- Grid geometry (cell centers, plus everything needed to rebuild cell
  boundary polygons) is derived purely from projection math — it does
  not require downloading HRRR data — and is saved to
  `data/processed/hrrr_coarse_grid_stride13.nc` by
  `scripts/build_grid.py`.
- Do not switch to an interpolated/reprojected grid or change the
  stride without discussing with the user first — later feature
  aggregation depends on the exact block-alignment between coarse
  cells and native HRRR pixels.

### Time binning (locked in)

Forecast lead time scope is **0–8h** (extended from an earlier 0–6h —
0–6h doesn't divide evenly into 4-hour bins), split into two equal,
half-open 4-hour bins: bin 0 = [0h, 4h), bin 1 = [4h, 8h). Defined
purely as date/time arithmetic in `src/tornado_predictor/time_bins.py`
— no data download required, same spirit as `grid.py` being pure
projection math.

- **Every hourly HRRR run gets its own bins.** All 24 hourly runs/day
  (not just synoptic 00/06/12/18Z) each define an independent 0–8h
  timeline and pair of bins relative to *their own* init time — this
  is deliberate, to maximize sample density, not an oversight.
- **Label multiplicity is intentional.** A consequence of per-run
  binning: one absolute valid time (e.g. one tornado report) falls in
  a different bin depending on which run's timeline it's scored
  against, and it becomes a positive label in the sample sets of
  *every* run whose [init, init+8h) window contains it — always
  exactly 8 different hourly runs, split across their two bins each
  (see `candidate_run_inits`). Do not "fix" this by deduplicating to
  one run per event; it is the correct behavior for this per-run
  sampling scheme.
- **Integer fxx vs. continuous lead time**: `fxx_to_bin_index` maps
  integer HRRR forecast hours (fxx=8 itself is excluded from both
  bins under the half-open convention, even though a fxx=8 HRRR pull
  is otherwise valid); `assign_valid_time_bin` maps an arbitrary
  absolute UTC timestamp (e.g. an SPC report, which is not on the
  hour) to a bin using continuous lead-time hours.
- **Training period (locked in, resolved from an earlier "known future
  constraint"): ~Sept 2014 – Sept 2025.** Verified live, not assumed:
  - Lower bound is HRRR's actual public AWS archive start
    (`noaa-hrrr-bdp-pds`) — confirmed via HTTP HEAD checks: 404 before
    ~Aug 15 2014, 200 after.
  - **HRRR's native grid, all needed fields, and forecast length are
    identical all the way back to Sept 2014** — confirmed by pulling
    live GRIB metadata (Nx/Ny, LCC projection params, Dx/Dy) and `.idx`
    field listings at 2014, 2016, 2017, and 2018 test dates, all
    matching `grid.py`'s constants and `features.py`'s field table
    exactly. Also confirmed fxx=7 (needed for bin 1) is available even
    on off-synoptic hourly runs back to Sept 2014. **There is no
    structural reason tied to HRRR versions (v1/v2/v3/v4) to start
    later than 2014** — resolution and grid did not change; only
    model physics/skill improved over versions, which is a data-quality
    consideration, not an availability one.
  - Upper bound is **SPC's label data, not HRRR** — HRRR itself is
    produced continuously to the present. SPC's published bulk file
    (`1950-2025_actual_tornadoes.csv`) was confirmed live to currently
    end at 2025-09-09 — SPC's ground truth lags real time by roughly a
    year (likely survey/QC completion), so labels, not HRRR, are the
    binding constraint on the upper end.
  - SPC reports are now downloaded/parsed for this range:
    `data/processed/spc_tornado_reports_2014_2025.csv` (15,294 reports,
    up from 12,649 for the original, arbitrarily-chosen 2012–2022
    range) via `scripts/download_spc_tornado_reports.py` (defaults
    updated to `--start-year 2014 --end-year 2025`). The labels table
    was rebuilt to match: `data/processed/tornado_labels_2014_2025.csv`
    (147,440 rows; 111,580 unique positives) via `scripts/build_labels.py`
    (defaults updated to match). All downstream default paths
    (`scripts/build_training_dataset.py`, `scripts/validate_documented_case.py`,
    `tests/test_labels.py`) were repointed to the new filenames; the
    old `_2012_2022` files were deleted (gitignored/regenerable, not a
    data-loss concern). The existing pilot dataset/checkpoint/notebook
    (all built from within-range 2021-12-10/11 runs) needed no rebuild
    — their content was already materialized and unaffected by the
    rename.
  - This does not by itself scale the training dataset — it only
    defines the target range for that future work.
- Do not change `BIN_EDGES_HOURS` or the every-hourly-run decision
  without discussing with the user first — later labeling/feature
  aggregation depends on both.

### Feature extraction (locked in)

`src/tornado_predictor/features.py` pulls HRRR forecast fields for one
`(init_time, fxx)` and pools each onto the coarse grid, via
`scripts/extract_features.py`.

- **Fields** (verified against a live HRRR `sfc` `.idx` file, exact
  Herbie search strings — a loose `"CAPE:"` would also match HRRR's
  other CAPE variants, so these must stay exact):

  | field | search string |
  |---|---|
  | `cape` (SBCAPE) | `CAPE:surface` |
  | `cin` (SBCIN) | `CIN:surface` |
  | `srh_0_1km` | `HLCY:1000-0 m above ground` |
  | `srh_0_3km` | `HLCY:3000-0 m above ground` |
  | `shear_0_6km` | computed as `hypot(u, v)` from `VUCSH:0-6000 m above ground` + `VVCSH:0-6000 m above ground` |
  | `t2m` | `TMP:2 m above ground` |
  | `d2m` | `DPT:2 m above ground` |

- **0–6km shear, not 0–8km.** HRRR has no native 0–8km shear field at
  all (confirmed by grepping a live `.idx` file — zero hits for
  "shear"/"8000"/"8 km"); 0–6km is what HRRR natively provides and is
  the standard severe-weather bulk-shear layer in this literature
  anyway. Don't try to synthesize a 0–8km layer from isobaric winds
  without discussing with the user first — meaningfully more complex
  (needs geopotential height for level selection) for a nonstandard
  layer.
- **Pooling, not interpolation.** Reuses the grid's exact 13x13
  native-pixel block alignment (reshape + `mean`/`max` per block) —
  zero interpolation error, consistent with the grid's own "no
  interpolation" design. Every field gets both `_mean` and `_max`
  variables (14 output variables total for 7 physical fields) to
  capture peak vs. typical conditions per cell.
- **One Herbie call per field**, not combined regex calls — sidesteps
  uncertainty about cfgrib's assigned variable names and Herbie's
  multi-hypercube merge behavior when combining different level-types
  in one call. Slightly more HTTP requests, negligible cost (each pull
  is already a small byte-range GRIB subset).
- This extracts a single `(init_time, fxx)` snapshot only. Aggregating
  multiple `fxx` snapshots within one `time_bins` bin into one
  bin-level feature set is a separate future step, not built yet.

### Labeling (locked in)

`src/tornado_predictor/labels.py` assigns positive tornado labels to
`(grid cell, run init_time, bin_index)` samples from SPC reports, via
`scripts/build_labels.py`.

- **Full-track spatial matching, not touchdown-point-only.** Each
  report's `path_wkt` is buffered by half its `width_yd` (converted to
  meters) to build a corridor polygon, then intersected against the
  *exact* projected bounding box of each candidate grid cell
  (`grid.cell_bounds_xy`, not the lat/lon-curved `cell_polygon`) —
  reprojection and buffering both happen in LCC meters via
  `grid.lonlat_to_xy`. This credits every cell a tornado crossed, not
  just where it touched down, per the intent flagged when
  `path_wkt`/`width_yd` were added to the SPC parser.
- **Single-touchdown-timestamp temporal matching is a real, permanent
  simplification.** SPC has no per-point track timing or end-time
  field, so the *entire* buffered track is matched against the report's
  one `timestamp_utc` via `time_bins.candidate_run_inits` +
  `assign_valid_time_bin` — spatial matching uses the full track,
  temporal matching collapses to one instant. Not fixable without
  fabricating a synthetic track-speed/duration; don't attempt that
  without discussing with the user first.
- **Only positive labels are materialized.** The output table
  (`data/processed/tornado_labels_2014_2025.csv`) contains one row per
  `(event_id, init_time, bin_index, row, col)` — every row is an
  implicit label of 1; anything not present is an implicit 0. The full
  0-label space is still not materialized — even with the training
  period now defined (~Sept 2014 – Sept 2025, see "Time binning"
  above), that's ~2.19 billion `(run, bin, cell)` combinations, not yet
  built as an actual training-run list (a future scaling step).
- **Row granularity is per-(event, run, cell), not deduplicated.** One
  report produces `(number of cells it touches) x 8` rows (always
  exactly 8 candidate runs per `time_bins`), preserving which report(s)
  caused which label. Collapse to unique `(init_time, bin_index, row,
  col)` positives later via `.drop_duplicates()` when building an
  actual training sample table — don't do that collapse in this module,
  since it would throw away traceability.
- `width_yd` must be strictly positive for the buffer to be a real
  polygon (a zero-radius buffer of a straight line is empty in shapely,
  silently dropping the report) — `labels.buffer_radius_m` raises
  rather than allowing this to pass silently. Not observed in the
  2014–2025 CSV, but not guaranteed for other date ranges.
- Validated against the real dataset: 12 of 15,294 reports fall outside
  the grid domain (proportionally consistent with the original
  2012–2022 pass's 10/12,649), and the single outlier with >10 affected
  cells (13) is still the December 10, 2021 Quad-State/Mayfield EF4 —
  the longest track in the dataset (~165 miles) — not a bug.

### Consolidated training dataset (pilot + first scale-up)

`src/tornado_predictor/dataset.py` combines feature extraction and
labeling across multiple HRRR runs into one dataset, via
`scripts/build_training_dataset.py`.

- **One representative fxx per bin, not full-bin aggregation.** A bin
  spans 4 forecast hours (`time_bins.fxx_values_for_bin`), but each
  sample here uses only the bin's first hour (fxx=0 for bin 0, fxx=4
  for bin 1) as a stand-in for the whole bin — keeps Herbie network
  cost proportional to sample count rather than 4x it. Aggregating all
  4 fxx per bin (spatial pooling already done, plus a further temporal
  reduction) is a possible future refinement, not built here — this was
  an explicit scope tradeoff, not an oversight.
- **Storage keeps (sample, row, col) structure**, one variable per
  field/statistic plus `label` — self-describing, consistent with
  every other saved artifact in this repo. `to_training_arrays(ds)` is
  a separate, explicit function that flattens this into literal
  `(sample, grid, feature)` + `(sample, grid)` numpy arrays for direct
  ML consumption — that flattening is lossy (loses which axis was row
  vs. col) and is deliberately not baked into the storage format.
  "Sample" is the accurate name for the leading axis (each is one
  independent (init_time, bin_index) pair, not a continuous timeline,
  per the every-hourly-run time-binning design) even though a
  consuming ML pipeline will likely treat it as "time".
- **This is a pilot, not a production build.** Demonstrated on 3 hourly
  HRRR runs (2021-12-10 21:00/22:00/23:00 UTC, bracketing the
  2021-12-11 02:54 UTC Quad-State/Mayfield EF4 and the broader outbreak
  that night) — chosen specifically because it's within the labeled
  SPC range *and* produces genuine positive labels (167 positive
  (sample, cell) pairs across all 6 samples), not an all-zero sanity
  check.

### First scale-up (44 train / 6 val samples)

Built via `scripts/build_training_dataset.py`, applying `split.py`'s
leakage-safe split (see above) to a **stratified sample of 25 dates**
(2014–2025) rather than the full training period — deliberately
sized down, per an explicit scope decision, to something that
completes in one interactive session (~200 HRRR pulls) while
meaningfully exercising the pipeline at real scale, not the full
multi-year archive (that's further-out future work).

- **Date selection**: 12 real high-activity dates (one busiest day per
  year 2014–2025, by SPC report count that day — e.g. 2014-09-01 IA
  outbreak, 2021-12-15, 2024-12-29) + 13 genuine quiet dates (verified
  zero SPC reports that day, spread across years and seasons — the
  first time this project has included real negative-class diversity;
  every previous dataset was 100% positive). One run per date (20:00
  UTC, a fixed representative hour for every date — a deliberate
  simplification, not a per-event-tuned choice, same spirit as the
  single-representative-fxx-per-bin approximation above).
- **A real date-selection bug was caught before it silently corrupted
  the dataset**: `2014-02-10` was initially picked as a quiet date
  without cross-checking it against the verified HRRR archive start
  (~Aug 15, 2014) — it's *before* the archive begins, so the pull
  correctly failed loudly (`"Did not find"` from Herbie) rather than
  silently returning wrong data. Fixed by swapping in `2014-11-10`
  (verified quiet AND within the archive) before re-running. Any
  future date selection must check both conditions.
- **Result**: 44 train samples (13 active / 31 quiet — `QuietMapDownsampler`
  finally does real, non-trivial work for the first time) / 6 val
  samples (3 active / 3 quiet), split 44/6 by the deterministic
  mod-20 rule with 0 dropped by the buffer safeguard (the 25 dates are
  naturally well-separated).
  `data/processed/training_dataset_train_scaled.nc` /
  `training_dataset_val_scaled.nc`.
- **Training result** (`models/tornado_cnn_scaled.pt`, 50 epochs,
  `quiet_keep_fraction=0.5`): loss and val loss both converge smoothly
  to ~0.0002–0.0003 — much lower in absolute terms than the 10-sample
  pilot's ~0.0007/0.0017, but this is **not** a sign of better fit; a
  mostly-quiet dataset naturally has lower mean loss regardless of
  discriminative quality (most cells are easy true-negatives).
- **The honest, important finding**: probability separation between
  positive and negative cells on the **held-out val set** is much
  weaker than the pilot's training-set-only numbers ever showed —
  ranging from ~2.5x (one val sample) to ~50x (another) rather than
  the pilot's consistent ~40–90x, and one positive cell in a 14-cell
  val sample got a probability of ~9e-6 (essentially zero — a clear
  miss). This is the first genuine generalization signal this project
  has produced, and it says plainly: **44 training samples is still
  not enough.** Not addressed here — surfacing it accurately is the
  point of having a real val set; don't paper over it with a rosier
  training-loss-only story.
- Scaling further toward the full ~Sept 2014 – Sept 2025 training
  period (see "Time binning" above) is the natural next step, now that
  the archive-availability question is resolved and this smaller
  scale-up has validated the mechanics.

### Full-scale build v1 (superseded — see v2 below)

Built via `scripts/build_scaled_dataset.py` + `src/tornado_predictor/build_scaled.py`
(see that module's docstring for the manifest-based resumable design).
Per explicit user direction ("everything all active and quiet days...
happy to have this running in the background"), this covers **every**
UTC calendar date with >=1 SPC tornado report in the ~Sept 2014–Sept
2025 archive window (2,044 active dates), plus 300 randomly-sampled
quiet dates (`seed=0`) — 2,344 dates total, one HRRR run per date
(20:00 UTC, same fixed-representative-hour simplification as the first
scale-up). Ran unattended over ~30 hours wall time.

- **2,279 of 2,344 dates succeeded (97.2%); 65 failed** after
  `features.py`'s internal retries were exhausted. **Failures cluster
  entirely in 2014–2018** (13 in Sept 2014 near the archive-start edge,
  the rest scattered through 2015–2018) — **zero failures from 2019
  onward**. Consistent with the earlier-discovered Sept 2014
  reduced-field-set issue (see "Time binning" above): the earliest
  years of AWS's HRRR archive are less complete/consistent than later
  years, not a bug in this pipeline.
- **Split result**: 3,578 train / 682 val / 298 dropped by the buffer.
  1,660/3,578 train maps (46%) and 325/682 val maps (48%) were
  "active" (>=1 positive cell), per-cell positive rate ~1.5e-04.
- **Superseded by v2 below.** A data-exploration pass
  (`notebooks/data_exploration.ipynb`) surfaced that only 46-48% of
  maps built from dates specifically chosen for having a confirmed
  report were actually "active" — lower than expected. Auditing found
  the cause: this build's one-fixed-20:00-UTC-run-per-date design
  missed a real tornado on **27.2% of active dates entirely** (541 of
  1,989), because most active dates have multiple reports spread
  across many hours (median spread ~10.8h on multi-report days; only
  46% fit inside one 8h window) — the single fixed run's window simply
  didn't bracket most of them. Not a labeling bug (`labels.py` was
  already correct); the bug was in which of a report's 8 valid
  candidate runs this build chose to actually pull.
- Archived, not deleted, for reference:
  `data/processed/training_dataset_{train,val}_full_v1_superseded.nc`.

### Full-scale build v2 (report-covering runs, canonical)

Fixes v1's run-selection gap. `build_scaled.select_active_runs` uses a
greedy interval-covering algorithm per active date: anchor a run at
the earliest not-yet-covered report's hour, skip every report within
the next 8h (already covered), repeat for whatever's left — the
minimum number of runs needed so *every* report that date is captured
by some run's 0-8h window (see `build_scaled.py`'s docstring for the
full derivation). Checked against the real report data before
launching: only 3,209 runs needed for 2,163 active dates (1.48x, not
8x) — 1,305 dates need just 1 run, 670 need 2, 188 need 3. Quiet dates
are unchanged (still exactly 1 fixed-hour run each — no report to
anchor to, so any hour is equally representative). The unit of work
changed from "date" to "run" (`process_one_run`, keyed by full
init_time, since a date can now need multiple runs); `finalize` still
groups by calendar date for the train/val split, so multi-run dates
split as one unit, unchanged from before.

- Validated on real data before the full run: pulled 5 real runs
  across 2 known multi-report dates and confirmed 9 of 10 resulting
  samples came out active (vs. what v1's single-run design would have
  caught).
- **3,335 total runs targeted** (3,035 active + 300 quiet) vs. v1's
  2,344 — 1.4x the volume, as predicted. **3,183 succeeded (95.5%)
  after one retry pass** (`--retry-failed` recovered 40 of the first
  attempt's 192 failures — most were transient `EOFError`/
  `ConnectionError`/`PrematureEndOfFileError`, not archive gaps like
  v1's failures were; 152 still failed, scattered across 2014–2021 and
  2025 with no strong pattern, not investigated further).
- **Split result**: 5,078 train / 980 val / 308 dropped by the buffer
  — meaningfully more samples than v1 (6,058 vs. 4,260 total) from
  fewer targeted active dates, because runs are no longer wasted on
  windows with nothing in them.
  `data/processed/training_dataset_train_full.nc` (3.0GB) /
  `training_dataset_val_full.nc` (596MB) — promoted to the canonical
  filenames (v1's are archived, see above).
- **Class balance improved substantially, as intended**: 3,208/5,078
  train maps (**63.2%**, up from v1's 46%) and 631/980 val maps
  (**64.4%**, up from 48%) are active. Per-cell positive rate ~2.13e-04
  in both splits (train 12,090/56,761,884; val 2,327/10,954,440),
  consistent between splits. Zero NaNs, zero degenerate all-zero
  fields.
- **Not yet done**: retraining `TornadoCNN` on this dataset and
  re-running the generalization evaluation that flagged v1's smaller
  25-date scale-up as insufficient (see "First scale-up" above) — this
  build only produces the dataset; `scripts/train_model.py --val-dataset ...`
  against it is the natural next step, not performed automatically
  here.

### Full-scale training run (first statistically real generalization result)

`scripts/train_model.py --dataset training_dataset_train_full.nc
--val-dataset training_dataset_val_full.nc --epochs 50` against the
full-scale v2 dataset, default hyperparameters (`alpha=0.25`,
`gamma=2.0`, `quiet_keep_fraction=0.2`, `lr=1e-3`, `batch_size=2`,
`seed=0`). Took ~13 minutes on a MacBook (CPU only — verified this is
plenty; see chat history, no GPU/cloud needed at this model/data
scale). Checkpoint: `models/tornado_cnn_full.pt`.

- **Loss**: 0.0011 -> 0.0002 (train), 0.0002 -> 0.0001 (val), both
  converged by ~epoch 5 and flat afterward. Low absolute loss is
  expected from the per-cell class imbalance alone (per "Class
  imbalance strategy" below) and is **not** by itself evidence of good
  discrimination — same caveat as every earlier training run in this
  project.
- **The real result is the held-out separation, measured properly for
  the first time.** The first scale-up's val set had only 6 samples (3
  active) — too few for a real distribution. This dataset's val set
  has **631 active samples**, giving an actual statistical picture:
  pooled across all 980 val samples (2,327 positive cells vs.
  10,952,113 negative cells), mean positive-cell probability 0.1195 vs.
  mean negative-cell probability 0.0078 — a **15.3x pooled separation**,
  and a probability-space Cohen's d of **2.49**. That's *stronger* than
  any single raw feature's Cohen's d from the data-exploration notebook
  (0.9-1.4) — the model has learned a nonlinear combination of the 14
  input fields that discriminates better than any one field alone, a
  genuine sign it learned something real, not noise.
- **Per-sample view**: among the 631 active val samples, separation
  ratio (mean positive proba / mean negative proba) has median 12.5x
  (p5=2.2x, p95=32.5x) — weaker than the training-set-only pilot's
  claimed ~40-90x (expected: that was memorization, this is genuine
  held-out data), but a real, usable signal. Only **16/631 (2.5%)**
  samples are outright misses (weakest true positive cell scored below
  the sample's mean negative-cell probability) — most active val
  samples show clear separation.
- **Absolute confidence is still modest** (mean positive-cell proba
  ~0.12, nowhere near 1.0) — same `FocalLoss` `alpha=0.25` explanation
  flagged throughout this project, now with much stronger evidence
  behind it: the underlying features and the model's own learned
  probabilities both show large effect sizes, so the low absolute
  confidence is a loss-function calibration choice, not a sign the
  model failed to learn. Retuning `alpha` was the concrete next lever
  — see "Alpha sweep" below (which also corrects the direction this
  note originally guessed).
- **Conclusion**: the full-scale dataset resolved the earlier "44
  samples isn't enough" finding — this is now a real, reproducible,
  statistically meaningful generalization result, not a mechanics
  check. The architecture itself (3-layer, ~270km receptive field, 32
  channels, 22,593 parameters — tiny next to TorNet's ~8M-parameter
  VGG-style baseline) remains the acknowledged next limitation, not
  yet addressed.

### Alpha sweep (resolved — corrects earlier "lower alpha" guidance)

`scripts/sweep_alpha.py` trains 5 identical `TornadoCNN` runs on the
full-scale dataset (same data, architecture, epochs=50, seed=0),
varying only `FocalLoss`'s `alpha`: 0.1, 0.25 (prior default), 0.5,
0.75, 0.9. Analyzed in `notebooks/alpha_sweep_results.ipynb`.

- **The prior guidance was backwards, now corrected.** Earlier text in
  this file and in `training.FocalLoss`'s own docstring claimed modest
  positive-cell confidence would need "a lower alpha" to fix. Rereading
  the actual formula (`alpha_t = alpha` for the positive class, `1 -
  alpha` for negative) suggested the opposite — a *higher* alpha gives
  the rare positive class *more* weight, not less — and the sweep
  confirms it empirically: `mean_proba_positive` rose monotonically
  with alpha (0.082 at 0.1 -> 0.308 at 0.9) across every value tested.
  `training.py`'s docstring is corrected; this note replaces the old
  (wrong) one.
- **Higher alpha measurably improves held-out separation, with a real
  tradeoff.** `mean_proba_negative` also rises with alpha (0.005 ->
  0.028) — pushing alpha up isn't free. But on the two more
  distributionally-robust metrics, higher alpha still wins: Cohen's d
  climbs from 2.32 (alpha=0.1) to a peak of **3.08 at alpha=0.9**, and
  miss rate (fraction of active val samples where the weakest true
  positive cell scores below that sample's own mean negative-cell
  probability) drops from 3.3% to **1.7%** — both best at alpha=0.9,
  the highest value tested. Simple mean-ratio metrics (pooled ratio,
  per-sample median ratio) are noisier and non-monotonic across this
  sweep — a reminder that they're a weaker signal here than Cohen's d.
- **alpha=0.9 is the best performer among the 5 values tested, by
  Cohen's d and miss rate specifically.** Nothing above 0.9 was tried.
- **Does not touch the architecture** — `TornadoCNN` is identical
  (22,593 parameters) across all 5 runs; this is a pure loss-function
  tuning result, independent of the architecture-widening work planned
  next.
- Checkpoints: `models/alpha_sweep/tornado_cnn_alpha_<alpha>.pt`.
  Results: `models/alpha_sweep/results.json`.
- **This recommendation was superseded almost immediately — see "Model
  comparison summary" below.** Cohen's d and miss rate are both
  mean-separation metrics; once threshold-integrated metrics (AUC-PR,
  best-threshold F1) were computed, they showed the *opposite* ranking
  — alpha=0.25 (the original default) wins, alpha=0.9 is the *worst*
  of the 5 by AUC-PR. **The project default `alpha=0.25` was NOT
  changed, and this sweep's "prefer higher alpha" conclusion should
  not be acted on** — kept here, uncollapsed, as a real example of two
  legitimate metrics disagreeing, not deleted or quietly fixed.

### Model comparison summary (standardized metrics, all runs)

`notebooks/model_comparison_summary.ipynb` computes a single
consistent metric set — accuracy, AUC-ROC, AUC-PR (average precision),
precision/recall/F1 (at threshold=0.5 and at each model's own
best-F1 threshold), Cohen's d — across every checkpoint trained so
far: pilot, first scale-up, split-demo, full-scale v2, and all 5
alpha-sweep runs. `scikit-learn` (+ `scipy`, its dependency) added to
`requirements.txt` for this — the first use of standard
precision/recall/ROC/PR tooling in the project; everything before this
used hand-rolled mean-probability comparisons.

- **Accuracy is confirmed useless at this imbalance, with real
  numbers**: every model scores >99.8% (up to 99.98% at full scale) —
  including, implicitly, an always-predict-negative model, which would
  score `1 - pos_rate`. Never cite accuracy as evidence of quality in
  this project.
- **AUC-ROC is misleadingly high (0.92–0.98) for every model** — a
  known trap under extreme imbalance (ROC integrates over the huge,
  easy true-negative population). **AUC-PR (average precision) is the
  honest metric: only ~0.02–0.03** for every model — still ~100-150x
  better than the no-skill baseline (the positive rate itself), a real
  signal, but a much more sober picture than ROC-AUC alone suggests.
  **Report AUC-PR alongside AUC-ROC from here on.**
- **`recall_at_0.5` is exactly 0.0000 for every model without
  exception** — not one prediction across any val set crosses the
  conventional 0.5 threshold for a real positive cell, even at
  alpha=0.9 (mean positive-cell probability 0.31). For the full-scale
  and alpha-sweep models, `precision_at_0.5` is `NaN` — literally zero
  cells out of ~11 million cross 0.5 at all, positive or negative.
  Confirms, more starkly than any earlier check, that `FocalLoss`
  calibration keeps every prediction well under 0.5 regardless of
  alpha in the tested range — 0.5 is not a usable operating threshold
  for this model family; each config's own best-F1 threshold (found to
  range 0.08–0.40) would be required for any real deployment.
- **The alpha-sweep's "higher alpha wins" conclusion is reversed by
  AUC-PR/F1** — see "Alpha sweep" above, now corrected there too.
  Mean-separation metrics (Cohen's d, miss rate) and
  threshold-integrated metrics (AUC-PR, F1) legitimately disagree here;
  prefer AUC-PR/F1 going forward, since they better reflect real
  operating-point usefulness.
- **Val loss does not predict AUC-PR** — no clean relationship across
  the 8 models with a val set (loss is also rescaled by
  `alpha`/`gamma`, so it isn't even comparable across the sweep). Loss
  curves remain useful for confirming a run trained without
  instability, but shouldn't be used to judge or compare model
  quality.
- **Carries forward into architecture widening**: use AUC-PR (plus a
  tuned-threshold F1/recall/precision) as the primary comparison
  metric for any new architecture — not loss, not ROC-AUC alone, not
  raw probability separation alone. The number a wider architecture
  needs to beat: **AUC-PR = 0.0283** (full-scale v2, alpha=0.25 — still
  the best config found to date).
- **The metric logic above is now a tested library module**, not just
  notebook code: `src/tornado_predictor/evaluate.py`
  (`get_proba_labels`, `compute_metrics`, `evaluate_model`,
  `evaluate_checkpoint`) + `scripts/evaluate_model.py` (CLI, prints a
  summary and can save full metrics as JSON via `--out`). Architecture
  ticket 1 of the widening plan — every future architecture variant
  (tickets 2-5) should call this instead of re-deriving the metric
  computation. Verified to reproduce the comparison notebook's exact
  numbers bit-for-bit (`python scripts/evaluate_model.py` against the
  default full-scale/alpha=0.25 config prints AUC-ROC=0.9819,
  AUC-PR=0.0283, matching above precisely). `scikit-learn`'s
  `roc_auc_score`/`average_precision_score`/`precision_recall_curve`/
  `roc_curve` do the underlying computation; a zero-positives or
  zero-negatives `y_true` returns `nan` for the AUC fields instead of
  raising (`sklearn` itself raises `ValueError` on a single-class
  input) — same convention `inference.sanity_check_against_labels`
  already used. **`notebooks/model_comparison_summary.ipynb` and
  `scripts/sweep_alpha.py` were NOT retrofitted to call this module** —
  they keep their own inline copies of the same logic; only future work
  uses the extracted version.

### Architecture widening, ticket 2: width vs. depth ablation

`TornadoCNN` (`model.py`) generalized to accept `n_hidden_layers`
(default 3, exactly reproducing the original hardcoded architecture —
verified bit-identical against `tornado_cnn_full.pt`'s real predictions
before/after the change, so every existing checkpoint still loads
correctly with no migration). Added `receptive_field_cells(n)` =
`1 + 2*n`. `scripts/sweep_architecture.py` trained 4 new
(`hidden_channels`, `n_hidden_layers`) configs on the full-scale v2
dataset, all other hyperparameters fixed at the established best
config (`alpha=0.25`, 50 epochs, seed=0), and re-evaluated the existing
baseline via `evaluate.evaluate_checkpoint` (not retrained) as the
comparison point. Results in `models/architecture_sweep/results.json`.

| config | params | receptive field | AUC-PR | AUC-ROC | Cohen's d | train/val loss |
|---|---|---|---|---|---|---|
| baseline (32ch, 3L) | 22,593 | 7x7 | 0.0283 | 0.9819 | 2.49 | 0.00016 / 0.00012 |
| 64ch, 3L | 82,049 | 7x7 | 0.0265 | 0.9828 | 2.34 | 0.00016 / 0.00013 |
| 128ch, 3L | 311,553 | 7x7 | 0.0257 | 0.9831 | 2.67 | 0.00017 / 0.00012 |
| **32ch, 5L** | 41,089 | 11x11 | **0.0319** | 0.9766 | 2.45 | 0.00016 / 0.00013 |
| 128ch, 5L | 606,721 | 11x11 | 0.0256 | 0.9653 | 1.28 | 0.00016 / **0.00029** |

- **A real, if modest, improvement — and it confirms the receptive-field
  hypothesis, not the "more capacity" one.** The *only* config that beat
  the baseline's AUC-PR is **32 channels / 5 layers (0.0319, +12.7% over
  0.0283)** — the cheapest of the 4 new variants (41K params, ~1.8x
  baseline) and the one that isolates *more receptive field* (11x11
  cells, ~430km) without adding channel width. This is exactly the
  mechanism argued for when this ticket was planned: tornadic
  environments are organized at meso/synoptic scales the original 7x7
  window couldn't see.
- **Widening channels alone did not help** — both 64ch/3L and 128ch/3L
  scored *below* the baseline on AUC-PR (0.0265, 0.0257) despite 3.6x
  and 13.8x more parameters. More per-cell representational capacity,
  without more spatial context, isn't the bottleneck.
- **Combining both axes (128ch, 5L) was the worst of all 5 configs**,
  and shows real overfitting: it's the only config where val loss
  (0.00029) is meaningfully *higher* than train loss (0.00016) — every
  other config has them roughly equal. 607K parameters (~27x baseline)
  on a 5,078-sample training set is a lot of capacity, and Cohen's d
  (1.28, vs. 2.3-2.7 everywhere else) and AUC-ROC (0.9653, notably
  below the 0.976-0.983 range elsewhere) confirm it's the clear outlier
  in the wrong direction.
- **Caveat, stated when this ticket was planned and still true**: this
  is a 5-point targeted ablation at one fixed hyperparameter recipe
  (the baseline's `alpha`/`lr`/`epochs`), not an exhaustive search, and
  no per-architecture hyperparameter retuning was attempted. The 32ch/5L
  win is real evidence for the receptive-field direction, not proof
  it's the ceiling — a next step (not done here) could retune
  hyperparameters specifically for a depth-only widening direction, or
  test more depth values (6, 7 layers) before moving to dilated convs.
- Observed wall-clock scaled sub-linearly with parameter count (e.g.
  128ch/3L's 311K params took ~51min, far less than the ~3hr a naive
  linear extrapolation from the 128ch/5L timing test predicted) —
  widening channels parallelizes well on CPU (larger matrix
  multiplies), while adding depth is more sequential; worth remembering
  when estimating cost for tickets 3-4.
- Checkpoints: `models/architecture_sweep/tornado_cnn_h<hidden>_l<layers>.pt`.

### Architecture widening, ticket 3: dilated convolutions (new best result)

`TornadoCNN` generalized again (same in-place precedent as ticket 2) to
accept `dilations: list[int] | None` — each hidden layer becomes
`Conv2d(..., padding=dilation, dilation=dilation)`, preserving spatial
size at any dilation rate. `None` (default) -> `[1] * n_hidden_layers`,
exactly reproducing prior behavior — verified bit-identical against
both the original baseline and ticket 2's winner's real predictions
before/after the change. `receptive_field_cells` generalized to
`1 + 2*sum(dilations)`. Channel width fixed at 32 throughout (ticket
2's best/cheapest width) to isolate the dilation effect.
`scripts/sweep_dilation.py` mirrors ticket 2's sweep script pattern.

**A real bug was caught and fixed while writing this ticket's tests,
worth recording**: the standard way to empirically verify a CNN's
receptive field is backprop-from-one-output-pixel-through-a-zero-input
and check which input pixels got nonzero gradient. This is a trap —
with an all-zero input, every conv's pre-activation is spatially
*constant* (bias only, since input contributes nothing anywhere), so a
ReLU can go uniformly dead across the *entire* feature map purely by
chance on the bias's sign, silently collapsing the measured field to
nothing. Fixed by using random input instead (verified reliable across
5 seeds, `hidden_channels=32` so the chance of every channel being
simultaneously dead at one exact position is astronomically small).
Documented in the test itself
(`test_receptive_field_formula_matches_gradient_based_measurement`) so
it isn't silently rediscovered later.

**Cost estimate was wrong, corrected empirically**: the plan predicted
~32 min total for both new configs (reasoning: dilation doesn't change
FLOP count vs. a plain conv of the same kernel/channel size). Real
measurement: ~76 min (31min + 45min). Dilated convs have real CPU
overhead beyond raw FLOP count on this hardware, likely from less
cache-friendly strided/dilated memory access patterns — still cheap
enough to just run, but the "same FLOPs = same wall-clock" assumption
doesn't hold on CPU.

| config | params | receptive field | AUC-PR | AUC-ROC | Cohen's d | train/val loss |
|---|---|---|---|---|---|---|
| baseline (32ch, 3L, no dilation) | 22,593 | 7x7/273km | 0.0283 | 0.9819 | 2.49 | 0.00016 / 0.00012 |
| ticket 2 winner (32ch, 5L, no dilation) | 41,089 | 11x11/429km | 0.0319 | 0.9766 | 2.45 | 0.00016 / 0.00013 |
| 32ch, dilations=[1,2,4] | 22,593 | 15x15/585km | 0.0289 | 0.9823 | 2.32 | 0.00016 / 0.00013 |
| **32ch, dilations=[1,2,4,8]** | 31,841 | 31x31/1209km | **0.0322** | **0.9861** | 2.33 | 0.00016 / 0.00012 |

- **New best result across both architecture tickets: `dilations=[1,2,4,8]` (AUC-PR 0.0322, AUC-ROC 0.9861, best-F1 0.0883)**
  — beats ticket 2's winner on every metric, with **fewer parameters
  (31,841 vs. 41,089, ~22% fewer) and fewer layers (4 vs. 5)**. This is
  the clearest evidence yet that receptive field, not raw parameter
  count, is what this problem needs — dilation grows it far more
  cheaply than stacking more full layers did.
- **`[1,2,4]` (same 3 layers as the original baseline, zero extra
  parameters) already beats the baseline** (0.0289 vs. 0.0283) — a
  free improvement from dilation alone at identical parameter count.
  It does *not* reach ticket 2's winner, though — matching ticket 2's
  ~430km depth-widening benefit needed the bigger `[1,2,4,8]` schedule
  (~1209km), not just any dilation.
- **No overfitting signature at the biggest receptive field this
  time** — unlike ticket 2's 128ch/5L (which combined more parameters
  *and* more depth and overfit badly, val loss almost 2x train loss),
  `[1,2,4,8]`'s val loss (0.00012) is actually slightly *below* train
  loss (0.00016), the same healthy pattern as every other config here.
  Supports the emerging picture: parameter growth (ticket 2's channel
  widening, and combining width+depth) is what caused overfitting risk
  before, not receptive-field growth by itself — dilation decouples the
  two, growing spatial context without growing capacity nearly as much.
- **Caveat carried over from planning**: the known "gridding artifact"
  risk of naive stacked dilation schedules (`[1,2,4,8]`) wasn't
  engineered around (e.g. hybrid dilation scheduling) — the results
  here don't show obvious signs of it, but a more careful schedule is a
  candidate future refinement if this direction is pursued further, not
  ruled out or confirmed necessary by this ticket alone.
- Checkpoints: `models/dilation_sweep/tornado_cnn_d<dilations-joined-by-dash>.pt`.

### Architecture widening, ticket 4: a real U-Net (new best result)

`src/tornado_predictor/unet.py` — `TornadoUNet`, a standard 3-downsample-
stage U-Net (4 resolution levels, matching TorNet's own 4-block
backbone shape): `ConvBlock` = two `(Conv3x3, ReLU)` pairs per level,
`MaxPool2d(2)` downsampling, bilinear `Upsample` + `Conv1x1`
channel-reduce + skip-concat for the decoder (avoids the
checkerboard-artifact failure mode of `ConvTranspose2d`). Channels:
14→16→32→64→128 (bottleneck) →64→32→16→1 — deliberately conservative
width (`base_channels=16`, not TorNet's 64) given ticket 2's
overfitting lesson at high parameter counts.

**Non-power-of-2 grid, resolved with pad-once/crop-once**: the 81x138
grid needs H,W divisible by 8 for 3 clean downsample stages. Pads to
88x144 (`F.pad(x, (0, 6, 0, 7))`, right/bottom only, so the real region
sits in the padded canvas's top-left corner) and crops the output back
to `[:81, :138]` — no per-block offset bookkeeping needed, unlike the
asymmetric-padding alternative the ticket considered and rejected.
Downsample sequence is exact with no further rounding: 88x144 → 44x72
→ 22x36 → 11x18 (bottleneck).

**`inference.load_checkpoint` now dispatches via a small model
registry** (`MODEL_REGISTRY = {"TornadoCNN": ..., "TornadoUNet": ...}`)
keyed by each checkpoint's own `model_class` field — self-describing
checkpoints, consistent with this project's stated design principle
(see "Training loop" above). Every checkpoint saved before this ticket
has no `model_class` key and defaults to `"TornadoCNN"`, reconstructing
exactly as before — verified bit-identical against a real checkpoint's
predictions. **No changes were needed to `training.py` or
`evaluate.py`** — both already operated generically on any `nn.Module`,
the direct payoff of ticket 1 being built architecture-agnostic from
the start.

**A real bug was caught writing the U-Net's tests**: none needed —
worth noting the *lack* of one, since U-Net skip-connection wiring has
many more ways to accidentally leave a branch disconnected than a flat
conv stack; the gradient-flow-through-every-parameter test (same
pattern as `test_model.py`) passed cleanly on the first real run.

Single training run (not a sweep — this ticket built and trained one
config, not an ablation, matching its starter prompt): `alpha=0.25`,
50 epochs, seed=0, same recipe as every prior ticket. 451,361
parameters (measured, not estimated).

| config | params | AUC-PR | AUC-ROC | Cohen's d | train/val loss |
|---|---|---|---|---|---|
| original baseline | 22,593 | 0.0283 | 0.9819 | 2.49 | — |
| ticket 2 winner (32ch, 5L) | 41,089 | 0.0319 | 0.9766 | 2.45 | — |
| ticket 3 winner (dilated [1,2,4,8]) | 31,841 | 0.0322 | 0.9861 | 2.33 | 0.00016 / 0.00012 |
| **U-Net (base_channels=16)** | 451,361 | **0.0387** | **0.9882** | 2.30 | 0.00015 / **0.00012** |

- **New best result overall: AUC-PR 0.0387** — a +20.2% relative
  improvement over ticket 3's dilated winner (0.0322) and +36.7% over
  the original baseline (0.0283). Best-F1 (0.0960) and AUC-ROC (0.9882)
  are also both the best of any config across all three architecture
  tickets.
- **No overfitting signature, despite ~14x more parameters than the
  previous best** — val loss (0.00012) is essentially equal to (very
  slightly below) train loss (0.00015), the same healthy pattern every
  non-overfit config has shown. This is a genuinely interesting result
  on its own: ticket 2 showed raw parameter count *can* cause
  overfitting (607K params, flat architecture, val loss ~2x train
  loss), but this 451K-parameter U-Net shows no such symptom. Parameter
  count alone doesn't determine overfitting risk — the U-Net's
  multi-scale downsample/upsample structure (a strong architectural
  prior for spatial data, forcing the network to represent information
  at multiple scales rather than memorizing at full resolution
  everywhere) appears to have real regularizing value beyond just
  "more capacity, more overfitting risk."
- Checkpoint: `models/unet/tornado_unet.pt`. Results:
  `models/unet/results.json`.
- Not attempted in this ticket (out of scope, per plan): sweeping
  `base_channels` or downsample depth — this ticket built and trained
  exactly one deliberately-conservative config; a wider or deeper
  U-Net might do even better, or might start overfitting the way
  ticket 2's widening did. Open question for a future ticket.

### Architecture widening, ticket 5: full comparison (concludes this round)

`notebooks/architecture_comparison.ipynb` — consolidates all 3 sweep
results files (`architecture_sweep`, `dilation_sweep`, `unet`) into one
comparison, deduplicating reference checkpoints re-evaluated more than
once across files (by `checkpoint` path — verified identical metrics
across duplicates, as expected from re-evaluating the same file). 8
unique real models total.

- **Confirms the U-Net as the outright winner** across every metric
  shown (AUC-PR 0.0387, AUC-ROC 0.9882, best-F1 0.0960) — see ticket 4
  above for the full table.
- **Quantifies the overfitting finding precisely**: val-loss/train-loss
  ratio is 0.72–0.83 for every config *except* ticket 2's `128ch/5L`,
  which sits at **1.80** — a stark, isolated outlier, not a fuzzy
  judgment call.
- **No clean "bigger is better" relationship between parameter count
  and AUC-PR** — ticket 2's widest configs (`128ch/3L`, `128ch/5L`)
  score *below* several far smaller models; the U-Net is both the
  largest model tried and the best, but the very next-best (`dilated
  [1,2,4,8]`) is one of the smallest. What mattered was growing
  receptive field, not raw capacity.
- **Honest caveats stated directly in the notebook, not buried**: all
  8 models share one hyperparameter recipe never tuned per-architecture
  (the U-Net's real ceiling is unknown); the U-Net itself was one
  config, not swept; and absolute AUC-PR (0.0387) is still low in
  absolute terms — a real ~190-260x improvement over the no-skill
  baseline, but this remains a hard, unsolved problem, not a finished
  one.
- This concludes the architecture-widening round (tickets 1-5). The
  U-Net (`models/unet/tornado_unet.pt`) is now this project's
  best-performing checkpoint and the natural default choice for any
  future work building on it.

### Genuine held-out test set (Tier 1 roadmap, ticket 2)

Every AUC-PR reported through ticket 5 (including the U-Net's 0.0387)
came from `training_dataset_val_full.nc` — a set checked repeatedly
across 5 architecture-selection rounds. `split.py` gained
`assign_date_split_3way`/`split_run_bins_3way`/`report_split_balance_3way`
(added alongside the existing 2-way functions, not replacing them —
nothing in tickets 1-5 needed to change), and `build_scaled.py` gained
`finalize_3way`, to build a genuine third bucket: **train (day-of-year
mod 20: 0-13, 70%) / test (14-16, 15%, new) / val (17-19, 15%)**. val's
*date-assignment rule* is unchanged from the 2-way split (mod 17-19,
same as always); test is carved from what the 2-way split called
train — days never touched by any past architecture decision, since
train was only ever used for gradient updates. No new HRRR pulls
needed — built entirely from the existing v2 staged runs via
`scripts/build_3way_split.py`, output to clearly-distinct
`training_dataset_{train,test,val}_3way.nc` files (the canonical
`_full.nc` files tickets 1-5 depend on are untouched).

- **A real inconsistency in the plan was caught by the actual numbers,
  not assumed away.** The plan claimed val's composition would stay
  "unchanged" — true for the *date-assignment rule* (verified exactly:
  with `buffer_hours=0`, the 3-way val is byte-identical to the 2-way
  val, both 980 samples) but **false once the buffer safeguard is
  applied**: protecting test from indirect val-leakage (dropping any
  val sample within 24h of a test sample) removes 182 of those 980
  val samples (18.6%), leaving **798**. User chose the more rigorous
  option: keep the buffer drop, and re-evaluate the U-Net on the new,
  smaller val too, rather than silently weakening test's protection to
  preserve exact historical comparability.
- **Real split sizes** (3,183 completed v2 runs, buffer_hours=24):
  train 4,174 / test 922 / val 798 / dropped 472 (7.4%, vs. the 2-way
  split's 6.5% — expected, there are now two boundaries needing buffer
  protection instead of one). Positive rate is consistent across all
  three (train 63.2%, test 62.6%, val 62.9% active maps) — no
  accidental imbalance.
- **U-Net evaluated on all three, via the identical `evaluate.py` code
  path**:

  | split | samples | AUC-PR | AUC-ROC | Cohen's d |
  |---|---|---|---|---|
  | val (new, 798) | 798 | 0.0398 | 0.9880 | 2.27 |
  | val (old, 980, for reference) | 980 | 0.0387 | — | — |
  | **test (new, 922, never touched before)** | 922 | **0.0836** | 0.9924 | 2.59 |

- **The new val's AUC-PR (0.0398) is reassuringly close to the old
  val's (0.0387, +2.8%)** — removing the 182 test-adjacent samples
  didn't meaningfully change the number, a good sign the buffer-drop
  isn't introducing some weird bias.
- **The genuinely new, never-touched test set scores notably
  *higher*** (0.0836, more than double val's ~0.04) — not a leakage
  red flag (leakage would show up as test scoring *worse* than a val
  set the architecture search implicitly fit to; scoring *better* on
  fresh data is the reassuring direction). Checked whether this is a
  trivial composition artifact before taking the number at face value:
  active-map rate (62.6% test vs. 62.9% val) and per-cell positive
  rate (2.01e-04 vs. 2.06e-04) are both nearly identical between the
  two splits — so the gap isn't explained by one split being an
  obviously easier class-balance mix. **The honest conclusion: at this
  project's current scale (~2,344 total dates, split 3 ways by a
  deterministic day-of-year rule), which specific storms land in which
  partition carries real variance** — a single train/val/test split
  can swing AUC-PR by 2x depending on date luck, not just model
  quality. This is itself a genuine, useful finding, not noise to
  explain away: it's direct evidence that year-based cross-validation
  (already flagged as a Tier 3 roadmap item) would give a much more
  trustworthy performance estimate than any single split, including
  this new one.
- **Not done in this ticket** (explicitly out of scope, per the plan):
  whether to promote the new, smaller (70%) train as canonical for
  future training — a separate decision, not folded in silently. No
  model has been retrained on the new train split; only the
  already-trained U-Net was evaluated against the new val/test.

### Gradient-boosted trees (Tier 1 roadmap, ticket 3) — CNN family wins, decisively

Literature found an HGBT beat a U-Net on a closely analogous CAM-grid
task — this ticket tested whether that holds here too.
`src/tornado_predictor/gbt.py` (`flatten_for_gbt`,
`downsample_negative_cells`) + `scripts/train_gbt.py` trained
`sklearn.ensemble.HistGradientBoostingClassifier` on
`training_dataset_train_3way.nc` flattened to one row per grid cell,
keeping every positive cell and downsampling negatives to 20:1 (9,992
positive + 199,840 negative = 209,832 rows, vs. 46.66M total — per-cell
downsampling a spatial CNN structurally can't do, see "Class imbalance
strategy" above). Evaluated on val and test, full/non-downsampled, via
the same `evaluate.compute_metrics` code path as every CNN variant.
Fit took **1.9 seconds** (HGBT's whole design point is scaling past
this dataset's size easily).

| model | val AUC-PR | test AUC-PR | val Cohen's d | test Cohen's d |
|---|---|---|---|---|
| U-Net (ticket 4/5) | 0.0398 | **0.0836** | 2.27 | 2.59 |
| **GBT (this ticket)** | 0.0186 | **0.0290** | **3.86** | **4.04** |

- **The literature's finding did not replicate here — the U-Net beats
  GBT by 2.1x (val) to 2.9x (test), decisively, not a close call.**
  Reported exactly as found, not spun: this is a real, useful answer to
  "should this project keep chasing bigger CNNs" — yes, for now.
- **Best-supported explanation, consistent with this project's own
  throughline**: every architecture-widening ticket (2-4) found that
  *receptive field* — seeing neighboring cells, not just more
  per-cell capacity — is what drove real improvement. A per-cell GBT
  has **zero** receptive field by construction: each grid cell is an
  i.i.d. tabular row with no access to its neighbors at all. The
  literature's own HGBT success likely came from richer, already
  spatially-aware engineered predictors (Sobash et al. 2020 used 174
  engineered predictors, not 14 raw pooled fields) and/or
  neighborhood-radius label matching — this project's GBT had neither.
- **A second, independent confirmation that Cohen's d and AUC-PR can
  sharply disagree** (first found comparing alpha-sweep configs, see
  "Model comparison summary" above) — this time *across model
  families*: GBT's Cohen's d (3.86-4.04) is actually *larger* than the
  U-Net's (2.27-2.59), and its mean positive-cell probability is far
  higher (~0.88 vs. the U-Net's much more modest, better-calibrated
  confidence) — yet its AUC-PR is far worse. GBT is confident and
  well-separated *on average*, but ranks enough of the ~9-10 million
  negative cells above true positives somewhere in the distribution's
  tail to hurt precision badly at any real operating point — exactly
  the failure mode a per-cell model with no spatial disambiguation
  would be expected to have (an isolated favorable-looking cell
  surrounded by unfavorable neighbors scores high but is often a false
  alarm; the U-Net can tell the difference, GBT structurally cannot).
- **Feature importance (permutation, on val) validates the literature's
  CAPE+shear+helicity emphasis**: `cape_mean`, `srh_0_3km_max`,
  `cape_max`, and `shear_0_6km_max` are the top 4 of 14, `cin_mean` is
  dead last — consistent with the STP-style severe-weather literature
  and with this project's own earlier Cohen's d findings (data
  exploration notebook: CIN was the weakest single-feature
  discriminator there too).
- **Two concrete, well-motivated follow-ups, not done here**: (1)
  engineer neighborhood-aware features for GBT (e.g. a 3x3-cell local
  average/spread per field) to give it *some* spatial context without
  building a full CNN — would directly test whether the gap is really
  about receptive field or something else; (2) re-run once ticket 1's
  UH-enriched dataset is ready — updraft helicity was the literature's
  single most-cited tornado surrogate and wasn't available for this
  run.
- Model: `models/gbt/tornado_gbt.joblib`. Results:
  `models/gbt/results.json`.

### Updraft helicity features (Tier 1 roadmap, ticket 1) — did not help

The data pull (section above, `scripts/augment_dataset_fields.py`)
finished successfully: 3,182 of 3,183 runs augmented (one sustained
network outage mid-job cost 851 runs, recovered via `--retry-failed`
once connectivity returned — down to 1 genuine remaining failure).
Verified before use: 22 variables, `uh_0_2km`/`uh_0_3km` NaN fraction
(~31%) matches the known 2018-07-13 archive boundary exactly,
`uh_2_5km` has zero NaN as expected. Promoted to canonical filenames
(`training_dataset_{train,val}_full.nc` and the 3-way equivalents),
archiving the prior 14-feature v2 files as `_v2_superseded` — same
precedent as the v1→v2 promotion.

The U-Net (21 features now: the original 14 + 6 UH mean/max + the
`uh_layers_available` indicator) was retrained on the enriched 3-way
train set and evaluated on both val and test, identical recipe
(`alpha=0.25`, 50 epochs, seed=0) to every prior run:

| model | val AUC-PR | test AUC-PR | val Cohen's d | test Cohen's d |
|---|---|---|---|---|
| U-Net, 14 features (no UH) | 0.0398 | **0.0836** | 2.27 | 2.59 |
| U-Net, 16 features (UH25 only, zero NaN) | 0.0377 | 0.0528 | 2.97 | 3.01 |
| U-Net, 21 features (+UH, NaN-imputed) | 0.0359 | 0.0491 | 2.77 | 2.86 |

- **Adding the literature's single most-cited tornado surrogate made
  AUC-PR *worse*** — val -9.8%, test -41.3% — despite Cohen's d
  *improving* on both splits. Reported exactly as found: this is the
  literature's actual answer not replicating, same honest treatment as
  the GBT result above, not spun positive because it was expected to
  help.
- **A third, independent instance of the same Cohen's d / AUC-PR
  divergence** (alpha sweep; GBT vs. U-Net; now UH features) — a
  genuinely recurring pattern in this project at this point, not a
  one-off. Mean separation between positive and negative cells keeps
  improving across these three cases while rank-based precision against
  the full negative population gets worse or stays flat — a strong
  argument that any future hyperparameter/feature decision in this
  project should be judged by AUC-PR specifically, never by Cohen's d
  alone, no exceptions.
- **No overfitting signature** (final val loss 0.000114 vs. train loss
  0.000146, the same healthy pattern as every non-overfit run in this
  project) — the regression isn't explained by the model fitting worse
  to begin with.
- **Resolved by a targeted ablation (user-proposed): UH itself is the
  dominant cause, not the NaN-imputation handling.** Retrained with
  *only* `uh_2_5km` (the field available for the whole archive, zero
  NaN anywhere — eliminates the missing-data/`uh_layers_available`
  explanation entirely): test AUC-PR = 0.0528, still 36.8% below the
  no-UH baseline (0.0836) and barely above the full 21-feature version
  (0.0491, +7.5% relative). If missing-data confusion had been the main
  driver, this clean-data version should have landed close to 0.0836 —
  it didn't. The small residual gap (0.0491 → 0.0528) shows the NaN
  handling was a real but secondary contributor, not the main story.
  **This is also the cleanest instance yet of the Cohen's d / AUC-PR
  divergence**: Cohen's d rises monotonically as UH is added (2.59 →
  2.86 → 3.01, UH25-only alone reaching the *highest* separation of
  all three configs) while AUC-PR falls at every step. Likely
  explanation: UH25 is a general supercell/rotation surrogate — it's
  elevated on many strong non-tornadic storms too, so it widens the
  average positive/negative gap (truly tornadic cells have even higher
  UH) while also creating more confidently-wrong false positives among
  rotating-but-non-tornadic cells, which is precisely what hurts
  precision-sensitive AUC-PR without hurting mean separation.
- Checkpoints: `models/unet_v3features/tornado_unet_v3features.pt` (21
  features), `models/unet_uh25only/tornado_unet_uh25only.pt` (16
  features, UH25-only ablation). Results:
  `models/unet_v3features/results.json`,
  `models/unet_uh25only/results.json`. **The project's best checkpoint
  remains the original 14-feature U-Net** (`models/unet/tornado_unet.pt`,
  test AUC-PR 0.0836) — neither UH variant beats it; both are
  documented negative results, and UH (at least these layers, at this
  training budget) is not recommended as a feature for this task.

### Year-holdout cross-validation (resolves the val/test instability)

Every AUC-PR reported above came from one fixed train/val(/test) split —
and the 3-way val (0.0398) vs. test (0.0836) numbers for the identical
frozen U-Net checkpoint differed by >2x, with no way to tell whether
either was "the truth." `src/tornado_predictor/split.py` gained
`assign_year`, `available_years`, `split_run_bins_year_holdout` (additive,
same pattern as the 2-way/3-way functions); `build_scaled.py`'s duplicated
`combine()` closure was extracted into a shared, reusable
`combine_staged_runs` helper (with a real `.load()` correctness fix —
the original inline version only worked because its callers happened to
call `to_netcdf` before closing source file handles). `scripts/cross_validate_year.py`
runs 12 leave-one-year-out folds (2014-2025) on the identical established
recipe (`TornadoUNet(base_channels=16)`, alpha=0.25, 50 epochs, seed=0),
reusing the already-staged v2 runs — no new HRRR pulls. Full results and
plots: `notebooks/cv_year_holdout_results.ipynb`.

| held-out year | test samples | AUC-PR | AUC-ROC | Cohen's d |
|---|---|---|---|---|
| 2014 | 88 | 0.0314 | 0.9881 | 2.80 |
| 2015 | 528 | 0.0276 | 0.9773 | 1.95 |
| 2016 | 484 | 0.0309 | 0.9765 | 2.37 |
| 2017 | 582 | 0.0466 | 0.9896 | 2.55 |
| 2018 | 524 | 0.0427 | 0.9887 | 2.86 |
| 2019 | 644 | 0.0362 | 0.9868 | 2.98 |
| 2020 | 524 | 0.0554 | 0.9879 | 2.43 |
| 2021 | 572 | 0.0447 | 0.9895 | 3.05 |
| 2022 | 564 | 0.0740 | 0.9902 | 2.80 |
| 2023 | 616 | 0.0526 | 0.9879 | 2.85 |
| 2024 | 660 | 0.0486 | 0.9810 | 2.66 |
| 2025 | 580 | 0.0925 | 0.9912 | 3.06 |

- **The trustworthy performance number for `models/unet/tornado_unet.pt`
  is AUC-PR = 0.0486 ± 0.0180** (12 folds, coefficient of variation
  ~37%, min 0.0276, max 0.0925) — this, not any single split, is what
  should be cited as "how good is this model" going forward.
- **The old test number (0.0836) was an unusually favorable draw, not a
  stable baseline**: z = +1.94 relative to the CV distribution, and 11 of
  12 independent year-folds score *below* it. The old val number (0.0398)
  was, by contrast, fairly representative (z = -0.49, 4/12 folds below
  it). This directly explains the >2x val/test swing that motivated this
  ticket — both numbers were just different draws from a genuinely wide
  underlying distribution; test happened to land around the 92nd
  percentile of it, not because of leakage or a stronger model, just split
  luck.
- **An unexpected, statistically significant finding**: AUC-PR correlates
  strongly with calendar year (Pearson r = 0.79, p = 0.002, still r = 0.77,
  p = 0.005 excluding the small-sample 2014 fold) — 2020-2025 averages
  0.0613 vs. 2015-2019's 0.0359, a 71% relative increase. Not something
  this project changed (identical model/features/recipe every fold) —
  leading, unconfirmed hypothesis is that it reflects HRRR's own forecast
  skill improving over its operational history (notably the HRRRv3→v4
  upgrade, operational since Dec 2020, right at this trend's inflection
  point) rather than anything about this pipeline. SPC labeling-practice
  changes over time are an uneliminated alternative explanation. Worth
  investigating before the next architecture/feature round, since it may
  mean recent years are systematically easier to predict on for reasons
  outside this project's control.
- Checkpoints: `models/cv_year_holdout/tornado_unet_heldout_<year>.pt`.
  Results: `models/cv_year_holdout/results.json`.
- **Not yet done**: re-running this CV on the quiet-date-scaled dataset or
  with pressure features once those tickets land — a natural follow-up,
  not required immediately. This CV describes the dataset as it stood
  before both.

### Full-scale build v4 (expanded quiet dates + one retry pass)

Same `scripts/build_scaled_dataset.py` / manifest (`data/interim/scaled_build_v2/`),
re-run with `--quiet-mode all` (every remaining quiet date, not 300 sampled) and
then `--retry-failed`. Outputs: `data/processed/training_dataset_{train,val}_full_v4.nc`
(the canonical `_full.nc` files are untouched).

- **Pulls**: first pass 4,822 done / 228 failed; after one retry pass
  **4,919 done / 131 failed** (97 recovered). Of the original 228 failures,
  107 were `ValueError: No index file was found` (the `.idx` is absent from
  the AWS archive -- will not recover on retry), 49 `EOFError`, 38
  `PrematureEndOfFileError`, 24 `FileNotFoundError`, 10 `ConnectionError`
  (the last four are transient/cache). 195/228 failures are 2014-2018, same
  early-archive pattern as v1/v2. Remaining 131 not pursued further.
- **Split** (2-way, mod-20 rule, 24h buffer): 7,652 train / 1,478 val /
  708 dropped.
- **Sanity check (verified)**: the 14 core features have zero NaNs, no
  all-constant samples, plausible ranges; train/val means nearly identical.
  Active maps **43.4% train / 44.5% val** (down from ~63% -- expected, quiet
  expansion); per-cell positive rate **1.44e-04** in both (was 2.1e-04).
- **UH columns are mostly NaN in the v4 `_full_v4.nc` files, but NOT because the new runs lack UH (CORRECTED --
  an earlier version of this note had it backwards):** the 3,182 original runs staged in
  `data/interim/scaled_build_v2/` carry only the 14 base features (their UH-augmented copies live in
  `data/interim/scaled_build_v3/`), while the 1,737 newer runs carry all 21 features natively. The v4 train/val
  files were assembled from `scaled_build_v2/`, so the old runs' UH is NaN there (77% for `uh_0_2km`/`uh_0_3km`,
  65% for `uh_2_5km`/`uh_layers_available`; the 0-2/0-3 km layers are additionally missing before 2018-07-13).
  `uh_layers_available` is NaN (not 0/1) for those. **Do not train on the 21-feature `_full_v4.nc` files as-is
  (drop the 7 UH columns, as the CV runs do).** For any UH work use `data/interim/scaled_build_uh_merged/`
  (symlinks: the `_v3` augmented copy for the old runs, the native file for the new ones): all 4,918 of 4,919 runs
  carry UH, `uh_2_5km` has zero NaN, `uh_0_2km` is all-NaN exactly for pre-2018-07-13 runs, and the 14 base
  features and labels are identical to the `_v2` copies (checked on 40 runs); the one exception is 2022-10-14.
- `scripts/cross_validate_year.py` gained `--drop-columns` (needed because
  staged runs mix with/without UH, and xr.concat silently NaN-fills).
- **v4 year-holdout CV result (12 folds, `models/cv_year_holdout_v4/`, 14
  features, identical recipe): v4 is statistically indistinguishable from the
  original -- not better, not worse.** As reported (different test sets) AUC-PR
  is 0.0442 +/- 0.0142 vs. the original 0.0486 +/- 0.0180 (-9%, v4 wins 3/12
  folds), but those test sets differ. Scoring both model sets on the *same*
  samples (`scripts/compare_cv_old_vs_v4.py`, inference only; results in
  `models/cv_year_holdout_v4/like_for_like.json`; analysis in
  `notebooks/cv_v4_vs_v2_comparison.ipynb`): AUC-PR **-4.0%** on the full v4
  test years (5/12 wins, paired-t p=0.54) and **-3.7%** on the old-run subset
  (5/12, p=0.55); best-F1 flat (+0.6%/+1.1%), AUC-ROC flat (+0.1%). Per-fold
  swings are +/-25-35% in both directions (v4 +33% in 2016, +36% in 2024; -26%
  in 2020, -23% in 2025), so 12 single-seed folds can't resolve a ~4% mean
  difference. Old models on the old-run subset reproduce the original CV's
  AUC-PR to within 5e-7, validating the like-for-like setup. **Cited number
  for the 14-feature U-Net stays ~0.045-0.049 AUC-PR**; the ~1,700 added quiet
  runs gave no measurable ranking gain (and no measurable harm; dataset is
  closer to natural class balance). Correction to an earlier guess: the lower
  base rate (test positive rate -34%) only explains ~-6% of AUC-PR with the
  model held fixed, not a large mechanical artifact. Calibration unchanged
  (mean positive-cell p 0.1365 vs 0.1364). The calendar-year skill trend
  persists on identical test samples (r=0.79 old, 0.87 v4). Untested:
  `quiet_keep_fraction=0.2` means only ~20% of the new quiet maps are seen per
  epoch. (Per-seed variance was subsequently measured and is large -- see "Seed-variance study and ensembles" below.)
  **Precision/recall at each model's own best-F1 threshold** (now reported
  alongside AUC-PR; already computed by `evaluate.compute_metrics`): at that
  operating point both model sets flag ~1 in 5 positive cells (recall ~0.19-0.20)
  at ~9% precision (~10 flagged cells per hit); best-F1 thresholds ~0.21 for
  both. On the old-run subset precision/recall differ by 1-2% (p>0.7). On the
  full v4 test years v4 has +20% recall (0.162 -> 0.195, 8/12 wins, p=0.061,
  suggestive only) with -7% precision (p=0.25); the old models' recall falls
  when scored on the quieter v4 test mix while v4 models' doesn't -- a
  threshold-dependent operating-point effect with no matching AUC-PR gain. Not
  a matched-recall/precision comparison (each model uses its own threshold,
  tuned on the data it is scored on); a precision-at-fixed-recall comparison
  from the full PR curves is the cleaner follow-up, not done.

### Post-v4 diagnostics: baselines, trend confound, and sample-mix artifacts

Cheap, evaluation-only analyses run after the v4 CV (scripts:
`scripts/physics_baselines.py`, `scripts/climatology_baseline.py`; results
`models/cv_year_holdout_v4/{physics_baselines,climatology_baseline}.json`).
Baselines use the v4 train+val files, i.e. ~7% fewer samples than the CV test
sets (buffer-dropped samples) -- comparison is approximate, not sample-identical.

- **U-Net vs. untrained baselines (per-year AUC-PR, 12 folds):** best simple
  physics baseline (STP-like composite, no LCL term, 5x5-smoothed max fields)
  averages 0.016; unsmoothed STP 0.012-0.013; CAPE x SRH 0.0075; single fields
  <=0.002; leave-one-year-out climatology (per cell, per month) ~0.0013. The v4
  U-Net (0.044) beats the best physics baseline in 12/12 years by ~3-4x, and
  beats climatology by ~30x. The U-Net/baseline ratio shows no trend with year
  (r=0.13), i.e. the model's value-add is stable.
- **The calendar-year skill trend is largely confounded with tornado-population
  composition, not cleanly attributable to HRRR:** mean SPC track length rose
  2.8 -> 4.9 mi and the EF0 share fell 0.54 -> 0.30 over 2016-2025 (year vs EF0
  share r=-0.91). Per-year AUC-PR correlates with mean track length (rho
  0.85-0.87) and EF0 share (rho -0.80 to -0.90). Controlling for track length,
  the year effect's partial r falls from 0.79 to 0.40 (p=0.19) for the old
  models and 0.87 to 0.61 (p=0.04) for v4. The untrained smoothed-STP baseline
  also drifts upward with year (r~0.55). A linear trend fits slightly better
  than a step at an HRRR upgrade boundary. 12 points cannot separate the
  explanations; do not cite the trend as evidence of HRRR forecast-skill gains.
- **Run-selection design shapes the sample mix (a sample-selection artifact,
  not a labeling bug):** 70% of quiet samples are init 20Z (the fixed quiet-date
  hour) vs. 11% of active samples, so P(active | init hour) is ~11% at 20Z but
  60-70% at 21Z/00Z. Active runs are anchored at the earliest report's hour, so
  positives skew to short lead: bin 0 has 2x the positive cells of bin 1 (9,778
  vs 4,924; active frac 60.5% vs 26.6%), and per-cell positive rate is 1.4e-4
  vs ~5e-5 over all hourly runs. Quantified below
  (`scripts/diagnose_sample_mix.py` -> `models/cv_year_holdout_v4/
  sample_mix_diagnostics.json`; per-fold cached scores in
  `data/interim/diag_scores/`; 12 folds, both model sets, full v4 test years).

### Sample-mix, map-level, and neighborhood diagnostics (evaluation-only)

- **Raw AUC-PR is strongly prevalence-dependent; compare lift (AUC-PR / positive
  rate) within a stratum, and never across strata.** Mean per-year (v4 models):
  all samples AUC-PR 0.0442 (pos rate 1.39e-4, lift 321x); init 20Z only 0.0217
  (3.1e-5, 689x); init not-20Z 0.0481 (2.3e-4, 206x); bin 0 0.0578 (1.85e-4,
  328x); bin 1 0.0341 (9.3e-5, 368x). Lift is *not* prevalence-invariant (it falls
  as prevalence rises: dilation k=0/1/2 -> 321x/182x/111x), so it is a
  within-stratum yardstick only.
- **Time-of-day shortcut: not supported as the source of skill.** Restricting to
  init 20Z (where active and quiet maps are both ordinary) does not reduce skill
  relative to base rate -- lift is higher (689x vs 321x), not lower. Old and v4
  models behave the same. (Not a proof the shortcut is unused, only that skill
  does not collapse where it is unavailable.)
- **Short-lead skew: not supported as inflating skill either.** Bin 1 lift (368x)
  is >= bin 0's (328x); the raw AUC-PR gap (0.058 vs 0.034) is prevalence (2x
  fewer positives in bin 1), not worse ranking. The model's skill is apparently
  environment-scale, not nowcast-lead-dependent.
- **Deployment-like estimate:** the 20Z stratum (11% active maps, positive rate
  3.1e-5) is the closest available proxy to an all-hourly-runs stream (17%
  active, ~5e-5): **raw AUC-PR ~0.02 there, vs. 0.044 on the CV mix.** Quote
  ~0.02-0.03 as the deployment-like expectation, not 0.044. Proxy is imperfect.
- **Map-level discrimination is the weak link (mean over 12 years):** max
  predicted probability separates active from quiet maps with AUC-ROC only
  ~0.79-0.81 and map AUC-PR 0.71-0.72 vs. an active fraction of 0.42 (lift 1.7x).
  At 50% recall of active maps, 13.5% (old) / 14.9% (v4) of quiet maps
  false-alarm; at 80% recall, 34% / 36%. The false alarms concentrate on the
  quiet bins of *active* runs (31-33% at 50% recall, 66-67% at 80%) rather than
  on quiet-date runs (5.5-6.2% at 50% recall, 19-20% at 80%): the model mostly
  rejects ordinary quiet days but struggles with *when* during a severe day
  (timing within an active run), consistent with using a single start-of-bin
  snapshot (fxx 0/4) per 4-hour bin. Within active maps, spatial AUC-PR is 0.054
  (lift ~162-168x).
- **The extra quiet training data did not improve quiet-day false alarms**, even
  on the metric designed to see it: quiet-date-run false-alarm rate at 50%
  recall is 5.5% (old) vs 6.2% (v4); map AUC-ROC 0.805 vs 0.794 (v4 wins 4/12,
  p=0.27). No direction is significant. Quiet-date negatives are not where the
  model's errors are.
- **Neighborhood-dilated verification (scores unchanged; positive if a tornado
  touched the cell or any cell within k cells, k=1 ~ 39 km, k=2 ~ 78 km):** v4
  AUC-PR 0.044 -> 0.148 (k=1) -> 0.210 (k=2); best-F1 0.115 -> 0.237 -> 0.295;
  precision at best-F1 0.084 -> 0.199 -> 0.259; recall 0.194 -> 0.298 -> 0.349.
  Positive rate rises 6x (k=1) and 14x (k=2), so most of the AUC-PR rise is
  prevalence (lift falls 321x -> 182x -> 111x); the informative part is that
  precision goes 8% -> 20%, i.e. roughly 12% of flagged cells are adjacent
  misses within one cell. v4 vs. old stays flat (-2.0% at k=1, p=0.67; -1.4% at
  k=2, p=0.74). This is verification only -- changing the *training* labels to a
  neighborhood definition would alter the locked labeling method and needs a
  discussion first.
- **Documented in `notebooks/sample_mix_diagnostics.ipynb`** (Part 1: the diagnostics
  above; Part 2: the seed-variance study and ensembles below, both executed).

### Seed-variance study and ensembles (changes how every earlier comparison should be read)

v4 14-feature U-Net CV recipe retrained with seeds 1 and 2 on folds 2016/2020/2024
(seed 0 = the existing v4 CV model); scripts `scripts/run_seed_study.sh`,
`scripts/seed_study_eval.py`, `scripts/ensemble_followup.py` (`cross_validate_year.py`
gained `--seed`); results in `models/seed_study/`. Predictions were pre-registered in the
notebook before results.

- **Seed noise is about as large as the signal in every single-seed comparison made so
  far.** Within a fold, AUC-PR of the *identical* recipe varies by a mean CV of **24%**
  across seeds (16-34%); max-min range is 32-69% of the fold mean (2024: 0.0638 / 0.0307 /
  0.0501). Pooled seed SD 0.0110 (27.9% of mean); excluding seed 0 (the member affected
  by fold selection) gives 26.5%. Estimated from 3 folds x 3 seeds (6 d.f.; 95% interval
  on the SD ~0.64x-2.2x), and the folds were selected for large v4-vs-old gaps.
- **Detectability:** a single-seed, 12-fold paired comparison can only detect an effect of
  ~**32%** (80% power); 3 seeds per fold ~18%. About half of the observed fold-to-fold
  variance of the single-seed 12-fold CV (SD 0.0149) may be seed noise (implied true
  between-fold SD ~0.010).
- **v4 vs. old, re-read:** seed 0's apparent +33% / -26% / +36% vs. the old model on the
  three folds becomes +7% / -23% / +3% (mean -4.4%) with the 3-seed v4 mean -- the per-fold
  swings were largely seed luck; the 12-fold null stands.
- **Ensembling (mean predicted probability) is the largest gain found in this project.**
  3-seed ensemble vs. the mean single seed: AUC-PR **+36%** (+33/+45/+30% per fold), best-F1
  +23%, map AUC-ROC +3.5%; it matches or beats the best single seed in 2 of 3 folds
  (+6/+23/-2%). **Across all 12 folds, the old+v4-seed-0 two-model ensemble scores AUC-PR
  0.0532 vs. 0.0461 (old) / 0.0442 (v4)** -- +18% over the mean of its members, beats both
  members in 12/12 folds, paired p=0.0002 / 0.0016; best-F1 0.124 vs 0.114/0.115.
- **Old-recipe and v4-recipe models are exchangeable:** pair-ensemble gain is +26.0%
  (old+seed) vs. +26.5% (seed+seed) (9 pairs each), reinforcing that the quiet-data
  expansion changed nothing detectable.
- **What this does to earlier conclusions (unresolved, not wrong):** single-seed rankings
  within about +/-30% -- the alpha sweep, the architecture-widening ladder (+12% to +37%
  steps), UH ablations (single split + single seed), v4 vs. old, and Part 1's map-level and
  false-alarm comparisons (quiet-date false-alarm rate has a seed CV of ~53%) -- are inside
  the noise band and cannot be called findings without repeats. Large effects survive:
  U-Net vs. GBT (2.1-2.9x), U-Net vs. physics baselines (~3-4x) and climatology (~30x).
  The calendar-year skill trend should be re-checked on ensembles (single-seed trend r=0.79;
  2-model ensemble r=0.86).
- **Instability vs. metric noise (day-block bootstrap, `scripts/bootstrap_stability.py`,
  `scripts/ensemble_stability.py`; notebook Part 3; decision rule written before looking):
  the models are genuinely unstable -- not "stable".** Resampling the test *dates* with
  one fixed model moves a year's AUC-PR by ~16% (relative SD; ~+/-30% 95% intervals; 2014's
  95 days far noisier) -- this metric noise is real and is NOT reduced by ensembling. But the
  seed spread (CV 16-34%) is larger and survives resampling (mean seed CV under resampling
  23/18/35%), and **7 of 9 seed pairs differ beyond day-resampling noise** (most with the
  same sign in all 2,000 resamples); old vs. v4-seed-0 differs significantly in 7/12 folds
  with mixed signs (3 +, 4 -), i.e. instability not a data effect. Exceptions: 2016 seed 1 vs 2
  and 2020 seed 0 vs 2 are within noise (2020's per-model sampling noise is ~30%).
- **Ensembles halve the instability but 2 members are not "stable":** disjoint 2-member
  ensembles disagree by ~13% vs. ~27% for single-model pairs (differ beyond noise in 4/9 vs
  12/18; one 2020 split still differs by 33%). The 2-model old+v4 ensemble beats each member
  beyond resampling noise in 8-9 of 12 folds and is never significantly worse. Caveats: days
  resampled as independent (consecutive-day correlation makes intervals somewhat too narrow);
  float16-quantized scores; ensemble stability measured only at 2 members.
- **Plan implied:** keep ensembling as the baseline (only thing tested that reduces model
  variance) but use more members (>=5 is a reasonable target; the number needed is untested);
  report 12-fold pooled means with bootstrap intervals, not single years; consider reducing
  instability at the source (EMA/SWA weight averaging, LR decay, earlier stopping -- loss
  flattened by ~epoch 5 in earlier runs -- gentler loss); verify with two disjoint 3-member
  ensembles (3 more seeds on the 3 seed folds, ~7h).
- **Ensemble-size study (notebook Part 4; seeds 3/4/5 added on folds 2016/2020/2024 ->
  six seeds per fold; `scripts/ensemble_size.py`, results `models/seed_study/ensemble_size.json`).**
  Disjoint-ensemble disagreement (mean relative AUC-PR gap) falls **25.5% (single) -> 13.0% (2
  members) -> 9.5% (3 members)**; the share of splits differing beyond day-resampling noise falls
  56% -> 33% -> 20% (2024 alone: 50% at k=3). **Pre-registered criterion (gap <=8% AND <=20% of
  splits beyond noise) is narrowly NOT met at 3 members** (gap 9.5%). Gap x size is roughly constant
  (25.5/26.0/28.5), so by extrapolation (NOT a direct test -- disjoint splits stop at 3+3 with six
  seeds) 4 members ~6-7%, 5-6 members ~4-5%.
- **Accuracy vs. size:** mean gain over a single seed +22% (2), +30% (3), +35% (4), +37% (5), +39%
  (6); marginal gain of the k-th member +22%, +6.8%, +3.4%, +2.0%, +1.3% -- 5 members capture ~95% of
  the gain in this pool. Test-sampling noise is identical at every ensemble size (~19% bootstrap rel.
  SD): ensembling reduces model variance only. 6-seed ensembles: AUC-PR 0.042 / 0.062 / 0.070
  (2016/2020/2024), +6-12% over the 3-seed ensemble and +19% to +54% over the original single-seed CV
  models; each still has wide day-bootstrap intervals (+/-25-45% relative). All four pre-registered
  predictions held (6-member gain 39% vs. predicted 40-45%, marginally under).
- **Decision (per the rule written beforehand): use a >=5-member ensemble as the project baseline;
  variance reduction at the source (EMA/SWA, LR decay, earlier stopping) is the next experiment.**
  A 12-fold 5-member baseline would cost ~22-25h CPU (the old and v4-seed-0 models already give 2
  members per fold and were found exchangeable).
- **Weight-averaging (EMA) study (notebook Part 5; `training.train_model(ema_decays=...)`, default off, tests
  verify tracking the average does not change training; `cross_validate_year.py --ema-decays`;
  `scripts/run_ema_study.sh`, `scripts/ema_eval.py`, results `models/ema_study/ema_eval.json`).** Seeds 10/11/12
  on folds 2016/2020/2024; plain and averaged weights saved from the same run (paired). **Pre-registered
  outcome "works" for EMA decay 0.9999 (~4-epoch horizon):** C1 seed CV 10.4% -> 3.7% (ratio 0.36) PASS;
  C2 pooled AUC-PR +2.9% PASS; C4 disjoint-pair gap 12.7% -> 4.7% (ratio 0.37; pairs beyond test noise
  4/9 -> 0/9) PASS; **C3 FAIL (EMA >= raw in only 5/9 runs, threshold 6)** -- averaging moves models toward a common
  value (per-run changes -15% to +19%) rather than improving each one. EMA 0.999 (~0.4-epoch horizon) mostly
  does not help (CV 10.4% -> 7.7%). Prediction check: I predicted "partial"; the paired result was stronger.
- **RECALIBRATION (supersedes the ~24% / ~32% figures above and in Part 2-3 text):** pooled over **nine plain
  seeds per fold** (0-5, 10-12), plain-model seed spread of AUC-PR is **~19%** (18.8%; 19.5% excluding the
  selection-affected seed 0), not ~24-28%: the first estimate used 3 seeds on folds selected for large v4-vs-old
  gaps, and 3-seed CVs range 9%-34% across seed groups. Minimum detectable effect for a 12-fold paired
  comparison: ~21% with one seed per fold, ~12% with three. About a third (not half) of the single-seed 12-fold CV
  variance is seed noise.
- **Direct 4-member plain-ensemble test (8 unselected seeds 1-5,10-12; `scripts/ensemble_size.py --extra-raw-seeds
  10 11 12 --exclude-s0 --max-k 4`, `models/seed_study/ensemble_size_8seeds.json`): near miss, and my extrapolation
  was too optimistic.** Disjoint plain ensembles disagree by 22.5% / 11.3% / 9.2% / 8.2% at 1/2/3/4 members, with
  50% / 30% / 26% / 25% of splits beyond test noise (2024 ~49% at k=4) -- the gap flattens, not 1/k. Gain over a
  single seed +19% / +25% / +29% at 2/3/4 members and +33-36% for all 8 (the earlier 6-seed figures
  +22/+30/+35/+37/+39% and the "5 members capture ~95%" statement were relative to that small pool and slightly
  high). A single EMA(0.9999) model's paired stability (ratio 0.37 -> ~8% when scaled to the pool's 22.5%) is
  roughly that of a 4-member plain ensemble, from one model (a scaled comparison, not a direct test).
- **Decision:** adopt EMA(0.9999) for every ensemble member (negligible cost). Untested: whether 3 EMA members
  meet the stability criterion (needs six EMA models per fold for a disjoint 3-vs-3 test: seeds 13-15 with
  EMA, ~7h). Plain ensembles need >=5 and still sit near the criterion at 4.
- **Averaged-ensemble stability test (notebook Part 6; seeds 13/14/15 with EMA 0.9999 added on folds
  2016/2020/2024 -> six averaged and six plain models per fold from the same runs, seeds 10-15;
  `scripts/ema_eval.py --seeds 10..15`, `scripts/ensemble_size.py --emastudy-variant {raw,ema0.9999}`; outputs
  `models/ema_study/ema_eval_6seeds.json`, `ens_size_6seeds_{raw,ema0.9999}.json`).** Disjoint-ensemble
  criterion (gap <=8% AND <=20% of splits beyond day-resampling noise), pooled over 3 folds:
  **averaged 3-vs-3 MEETS it (gap 6.9%, 6/30 = 20.0% beyond noise -- exactly at the line); plain 3-vs-3 does not
  (8.5%, 8/30 = 26.7%).** Averaged 1v1 12.7% / 2v2 7.4% (24% beyond noise, fails); plain 1v1 19.5% / 2v2 9.7%.
  Accuracy ceiling NOT raised: 6-model ensemble AUC-PR averaged 0.042/0.060/0.076 vs plain 0.045/0.065/0.073
  (pooled ratio 0.973); averaged ensembles gain less from ensembling (3-member +21/+16/+20% vs plain +27/+26/+23%).
  Predictions: 1v1 gap 5-8% NOT confirmed (12.7%); 3v3 result confirmed (narrowly); ceiling within +/-5% only pooled;
  smaller ensembling gain confirmed in direction.
- **REVISED Part 5 verdict ("works" -> "partial"), from re-evaluating EMA 0.9999 vs plain on 18 runs (six seeds
  per fold):** seed CV 16.3% -> 10.8% (ratio 0.66; C1 FAIL), pair gap 19.5% -> 12.7% (ratio 0.65; C4 FAIL), pooled
  AUC-PR +5.5% (C2 pass), EMA >= plain in 13/18 runs (C3 pass), single-model pairs beyond test noise 17/45 plain
  vs 16/45 averaged (no difference). The 9-run ratios (0.36/0.37; 4/9 -> 0/9) were optimistic -- seeds 13-15 added
  low outliers; a 3-seed spread estimate is unreliable (flagged at the time, now borne out). I originally
  predicted "partial".
- **Decision (updated):** use averaged (EMA 0.9999) members; 3 averaged members is the minimum that meets the bar
  (thin margin, thresholds are my choice), 4-5 for a safety margin; expect no higher accuracy ceiling than a plain
  ensemble of the same size. Cost of a 12-fold baseline: the 9 folds without averaged models need fresh training
  (plain old/v4 models cannot be converted): 3 members ~27 trainings (~21h), 5 members ~45 (~36h).
- **Pitfall found during the UH re-test setup (feature channel order):** `DenseGridDataset` takes channel order from the
  variable order of the *first staged run* in the combined set, and staged files from different sources order their
  variables differently (UH columns come after shear in the `scaled_build_v3` augmented copies but before it in
  natively pulled files). So a train set and a test set built from the same merged staging can disagree on channel order
  (the 14-feature studies are unaffected: with UH dropped the order is identical). `inference.assert_feature_order_matches`
  caught this in `uh_eval.py` (which sorts runs; the CV script builds its run list in manifest order). Checked directly
  for seed 10 / fold 2016: the CV script's own in-script AUC-PR (0.0326 plain / 0.0366 averaged) equals the correctly
  aligned value, whereas swapped channels would give 0.0293 / 0.0305 -- so the UH runs' in-script numbers were valid, but only
  by luck of ordering. `cross_validate_year.py` now aligns the test set to the training order (a no-op when they match) and
  `uh_eval.py` aligns each model's scoring set to its checkpoint's order, then asserts. Training itself was never affected.
- **UH re-test with averaged ensembles (in progress, notebook Part 7):** UH 2-5 km only (14 base + `uh_2_5km`
  mean/max = 16 features), EMA 0.9999, seeds 10-12, folds 2016/2020/2024 (`scripts/run_uh_study.sh`, ~8h;
  evaluation `scripts/uh_eval.py` -> `models/uh_study/uh_eval.json`), trained on the same 4,919 runs/splits as the
  baseline averaged models (`models/ema_study/`, seeds 10-15), so only the UH features differ. Compares the UH
  3-member ensemble to the 20 possible 3-member baseline ensembles. Pre-registered: hurts if E<=-10% and below the
  baseline-triple p10 in >=2/3 folds; helps if E>=+10% and above p90 in >=2/3; else "no detectable effect" (keep 14
  features). Predicted: no detectable effect; Cohen's d higher for UH in >=2/3 folds.
- **Working rules going forward:** report ensembles (>=3 seeds) as the headline model and
  cite single-seed AUC-PR only with a +/-25% caveat; require >=3 seeds per configuration
  (or ensemble-vs-ensemble) before treating any comparison as a finding. Cited performance
  for the 14-feature U-Net: single seed ~0.044-0.049 (+/-25%); 2-model ensemble 0.053 (12
  folds). Not measured: ensemble-vs-ensemble variance, weighted/calibrated ensembling,
  seed sensitivity of other architectures.
- **Open caveats:** these are each model's own best-F1 thresholds, 12 single-seed
  folds, v4 test years only; baselines were not re-scored under dilation or per
  stratum.

### Validation against a documented case

`scripts/validate_documented_case.py` spot-checks the grid/time-bin/
label/feature pipeline against one specific, real, well-documented
tornado: the EF4 that touched down in Arkansas at 2021-12-11 01:07 UTC
and tracked ~81 miles into Tennessee (SPC `event_id 2021_2112101907-01`)
— chosen because it's the exact tornado the sibling CNN radar-detection
project (`tornet/notebooks/july_6th_result_analysis.ipynb`, TorNet
catalog `event_id 997130`, KNQA radar) already uses as a named case
study, so both projects' validation stories anchor on the same real
event. Note this is a **different** tornado than the one used in the
pilot training dataset above (`2021_2112102054-01`, the separate,
later Quad-State/Mayfield EF4 from the same outbreak night) — don't
conflate the two.

All checks passed: touchdown point lands 9.5km from its assigned grid
cell's center (well inside the ~19.5km half-cell tolerance); the
6-cell buffered track includes both the touchdown and lift-off cells;
the precomputed labels table agrees with fresh recomputation for all 8
candidate HRRR runs; and pulled HRRR features show no NaNs, no
degenerate all-zero fields, and a physically realistic signature
leading up to touchdown — 0-1km and 0-3km storm-relative helicity more
than double in the final hour before the tornado (textbook tornadic
supercell behavior), while CAPE stays substantial (1200-2000 J/kg)
throughout.

### Class imbalance strategy (active)

`src/tornado_predictor/training.py` implements `FocalLoss`,
`DenseGridDataset`, and `QuietMapDownsampler` — real, tested code as of
this ticket (`torch==2.8.0` now a dependency). This is how the dense
grid's class imbalance is handled once modeling starts:

- Grounded in this project's real numbers (`tornado_labels_2014_2025.csv`):
  training on every hourly HRRR run in this period would give a
  per-cell positive rate of ~5.1e-05 (~1 in 19,600), but ~17% of
  `(run, bin)` grid-maps contain at least one positive cell — most of
  the imbalance is "positives are rare within an active map," not
  "active maps are rare." (These are updated figures after the SPC
  range was extended from 2012–2022 to 2014–2025 — same order of
  magnitude, not a materially different picture.)
- **Focal loss over per-cell class weighting**, and **quiet-map
  downsampling over per-cell downsampling** — per-cell downsampling
  isn't really executable for a spatial CNN (can't drop arbitrary
  pixels from a grid map without breaking the convolutional receptive
  field). Downsampling instead happens at the map level: keep every
  active map, keep only a fraction of quiet (all-negative) maps,
  re-drawn each epoch.
- `alpha=0.25, gamma=2.0` (RetinaNet defaults) and
  `quiet_keep_fraction=0.2` are starting points, not tuned to this
  dataset — expect to adjust after real validation-set experimentation.
- **`QuietMapDownsampler` is currently a no-op.** The pilot dataset
  (`training_dataset_pilot.nc`) has only 6 samples, and all 6 are
  "active" (it was deliberately built around a known outbreak — see
  "Consolidated training dataset" above). There are no quiet maps yet
  to downsample; this only starts doing real work once the training
  set is scaled beyond a single outbreak. Verified directly in
  `tests/test_training.py::test_downsampler_noop_on_real_pilot_dataset`.
  Don't mistake this for a bug in the sampler.
- Verified against the real pilot dataset: `DenseGridDataset` correctly
  reports 14 features, `(81, 138)` grid shape, and zero NaNs across all
  6 samples (`tests/test_training.py::test_dense_grid_dataset_against_real_pilot_dataset`).
- The CNN architecture itself is still a separate, open decision, not
  covered by this module (see next ticket in the modeling roadmap).

### Baseline CNN architecture (implemented, first pass)

`src/tornado_predictor/model.py` defines `TornadoCNN` — a minimal first
baseline, explicitly not a final architecture:

- **Full-resolution convolutions only, no pooling/upsampling.** The
  grid's shape (81, 138) isn't a clean power of 2, so a U-Net's
  downsample/upsample skip-connection alignment would add complexity
  not worth it for a first baseline — `padding=1` keeps every layer at
  the input's exact spatial resolution instead.
- Three hidden 3×3 conv layers (default `hidden_channels=32`) + a final
  1×1 conv to collapse to one output channel. Effective receptive field
  is ~7×7 cells (~270km at this grid's 39km cell size) — likely too
  small to capture full supercell- or synoptic-scale context. Widening
  this (more layers, dilated convs, or a real U-Net with careful
  padding/cropping) is a natural next iteration once this baseline is
  confirmed to train end-to-end.
- **Outputs raw logits, not probabilities** — matches
  `training.FocalLoss`'s expectation (it applies sigmoid internally via
  `binary_cross_entropy_with_logits`, more numerically stable than
  training on already-sigmoided probabilities). Apply
  `torch.sigmoid(model(x))` for a probability map at inference time.
- Verified against the real pilot dataset: correct `(1, 1, 81, 138)`
  output shape from real pooled features, finite output, and a full
  gradient check — `FocalLoss` computed against the real labels for one
  sample backpropagates finite, nonzero gradients through every
  parameter (`tests/test_model.py::test_against_real_pilot_dataset`).
- No training run has happened yet — this ticket only covers the
  architecture. Wiring it into an actual training loop (optimizer,
  epochs, checkpointing) is the next step.

### Training loop (first run complete)

`training.train_model()` wires `DenseGridDataset` + `QuietMapDownsampler`
+ `FocalLoss` + a given model into an actual training loop, used by
`scripts/train_model.py`.

- **Determinism split across two seeding responsibilities.**
  `train_model()`'s only training-time source of randomness (which
  quiet maps get kept, batch shuffling) is controlled by its own
  `seed` argument, independent of torch's global RNG. Model weight
  *initialization* happens outside `train_model()` (in the caller), so
  full end-to-end reproducibility requires `torch.manual_seed(seed)`
  before constructing the model — `scripts/train_model.py` does this;
  verified in `tests/test_training.py::test_train_model_is_deterministic_given_same_seed`.
- **No train/val split.** At the pilot's 6-sample scale a formal split
  isn't meaningful — the script trains on all samples and reports
  training loss only. This is a mechanics check, not model evaluation.
- **Checkpoints are self-describing**: `model_state_dict`,
  `loss_history`, `training_hyperparameters`, `model_hyperparameters`
  (`in_channels`, `hidden_channels`), and `feature_names` (order
  matters for reconstructing input tensors) — enough for a future
  inference script to reconstruct the exact model without
  re-deriving anything from the dataset.
- **First real run** (`models/tornado_cnn_pilot.pt`, gitignored —
  regenerable via `scripts/train_model.py`): loss dropped from 0.0712
  to 0.0007 over 50 epochs on the pilot's 6 samples — the model fits
  this tiny dataset, as expected (memorization, not generalization;
  see "Consolidated training dataset" above for why the pilot is
  scoped this small).
- Verified with a real learning-progress test, not just a shape check:
  `tests/test_training.py::test_train_model_reduces_loss_on_real_pilot_dataset`
  asserts final loss < initial loss after 30 epochs on the real pilot
  data.

### Inference + sanity check (training-set fit only)

`src/tornado_predictor/inference.py` loads a checkpoint, predicts a
per-cell probability map for a sample, and sanity-checks it against
that sample's known positive labels, via `scripts/run_inference.py`.

- **`assert_feature_order_matches` is a real correctness guard, not
  boilerplate.** The dataset's feature channel order must exactly
  match the checkpoint's training-time order — a silent mismatch would
  feed the wrong physical field into the position the model learned to
  associate with e.g. CAPE, producing plausible-looking but meaningless
  predictions. Raises rather than allowing that to pass silently.
- **This is a training-set-fit check, not a generalization test** —
  the pilot's 6 samples are the same 6 the model trained on. A high
  probability at known-positive cells only confirms the model learned
  to associate *something* with its own training labels.
- **Empirical finding, not a bug**: positive-cell probability does
  *not* approach 1.0. Across all 6 pilot samples, mean probability at
  true positive cells lands around 0.26–0.34, while negative cells
  average 0.003–0.007 — a strong, consistent ~40–90x separation, but
  well short of confident (near-1.0) positive predictions, and in
  every sample at least one negative cell scores *higher* than the
  mean positive cell. This traces directly to `alpha=0.25` in
  `FocalLoss` down-weighting the positive class's loss contribution
  (per Lin et al. 2017's own formula) — which matters far more at this
  dataset's per-sample imbalance (7–46 positive cells among 11,178)
  than at RetinaNet's original object-detection imbalance. At the time
  this was written, `FocalLoss`'s own docstring speculated "a lower
  alpha" would help — **that guess was backwards** (alpha weights the
  positive class directly; a *higher* alpha gives it more weight, not
  less) and has since been corrected with real evidence — see "Alpha
  sweep" above, which found alpha=0.9 (the highest value tested) gave
  the best held-out separation of the values tried.
- **Visualization** (`outputs/inference_pilot.png`, gitignored):
  predicted probability heatmap (viridis, sequential/colorblind-safe)
  with true positive cells marked as a red X, one panel per sample,
  single shared colorbar. Confirms visually that the model produces a
  spatially coherent elevated-probability plume tightly localized
  around the true positive cells in every sample — not memorized
  noise scattered arbitrarily across the domain. `origin="lower"` so
  the plot reads north-up (grid row 0 is the domain's SW corner, per
  `grid.py`).

### Case study: prediction vs. detection (mechanics demo, not accuracy)

`notebooks/case_study_prediction_vs_detection.ipynb` — a self-contained,
**executed** notebook (real cached outputs, not just code) lining up
this project's prediction model against the sibling `tornet` detection
project's model, for the exact same real tornado.

- **Anchor event corrected from the pilot dataset's event.** Uses SPC
  `event_id 2021_2112101907-01` (AR→TN EF4, touchdown 2021-12-11
  01:07 UTC) — the tornado matching TorNet's own documented case study
  (catalog `event_id 997130`, KNQA radar) — **not**
  `2021_2112102054-01` (the separate Quad-State/Mayfield EF4 used to
  choose the pilot dataset's HRRR runs). Both are real EF4s from the
  same outbreak night; don't conflate them.
- **Only 3 of the pilot's 6 samples are genuine forecasts of this
  event** — the other 3 are positive at the same cell only because a
  *different* tornado that night also crossed it. Determined
  programmatically (`assign_valid_time_bin(init_time, touchdown) ==
  sample's bin_index`), not assumed.
- **Real cross-project number, not a stub**: the detection model's
  actual output was pulled directly from `tornet`'s own already-executed
  `notebooks/july_6th_result_analysis.ipynb` (cell 18) — logit 6.4960,
  probability 0.9985 (99.85%), on the radar frame valid ~01:35:30 UTC
  (~28 min after touchdown). Not re-run here — that needs `tornet`'s
  own environment/weights, out of scope. Its output figure is copied
  into `notebooks/tornet_detection_case_study.png` for reference.
- **Honest, not cherry-picked finding**: prediction probability at the
  touchdown cell does *not* increase monotonically as lead time
  shortens (3.12h→0.61, 2.12h→0.40, 0.12h→0.19) — flagged explicitly in
  the notebook as an expected symptom of training on 6 samples, not
  smoothed over.
- The 99.85% (detection, near-real-time, direct radar signature) vs.
  19–61% (prediction, hours out, 6-sample model) gap is framed as
  reflecting different problem difficulty, not just "prediction is
  worse" — detection confirms an existing phenomenon; prediction
  infers risk before it exists.
- Built and executed via `nbconvert`/`ipykernel`/`nbformat` (now
  dependencies) — `jupyter nbconvert --to notebook --execute --inplace
  notebooks/case_study_prediction_vs_detection.ipynb` to regenerate.

### Train/val split (locked in)

`src/tornado_predictor/split.py` splits a `run_bins` list into
train/val without leakage, via `scripts/build_training_dataset.py`
(now produces two files instead of one) and `training.train_model()`
(now optionally tracks val loss per epoch via `training.evaluate()`).

- **Adapted from the TorNet benchmark paper's actual methodology**
  (Veillette et al. 2024, verified by reading the paper directly, not
  from memory), not invented from scratch. TorNet groups by NOAA Storm
  Events Database "storm episode" and splits via `day-of-year mod 20 <
  17 -> train` (an 85/15 split chosen so both splits see the full
  seasonal cycle every year, not a lucky/unlucky year range), with a
  30-minute-and-0.25°-buffer removing training samples too close to
  test samples.
- **Split unit here is UTC calendar date of `init_time`**, not storm
  episode — the honest analog given our samples are full-domain maps,
  not storm-centered crops. Every `(run, bin)` sample from the same
  date is correlated regardless of which cell is being looked at, so
  the whole date goes to one split, never split across both. Same
  `mod 20 < 17` rule, same 85/15 ratio.
- **Buffer safeguard is temporal-only** (`DEFAULT_BUFFER_HOURS = 24`)
  — TorNet's spatial half doesn't apply since every sample already
  covers the whole domain. This exists because the mod-20 rule flips
  abruptly twice every 20-day cycle (verified concretely: 2021-12-02 →
  train, 2021-12-03 → val, consecutive calendar days, opposite
  splits) — roughly 10% of all days sit next to a boundary like this,
  not a rare edge case. A training sample is **dropped, not
  reassigned**, if within the buffer of any val sample — mirrors
  TorNet's own choice exactly.
- **Does NOT attempt tornadic/non-tornadic balance via the split rule**
  — TorNet achieves that at corpus-construction time (which samples
  exist at all, via its three sample categories), a decision this
  project makes separately when a training run/date list is assembled
  (ticket 3 in the modeling roadmap, not this module).
  `report_split_balance()` reports the resulting positive rate per
  split so an accidental imbalance is caught, not silently shipped —
  it does not correct for one.
- **Validated with a genuine two-outbreak split**, not just offline
  unit tests: the existing Dec 2021 pilot runs (all land on `train`,
  day 344 mod 20 = 4) plus real runs from the April 26–27, 2024
  outbreak (`val`, day 117 mod 20 = 17) — real IA/MO tornado activity,
  not a synthetic date. Result: 6 train / 4 val samples, 0 dropped by
  the buffer (the two outbreaks are months apart), 100% positive rate
  in both splits (`data/processed/training_dataset_train.nc`,
  `training_dataset_val.nc`).
- **Training with the real split surfaced a genuine, expected
  overfitting signature at this tiny scale**: val loss drops sharply
  then plateaus around epoch 10–15 (~0.0015) while train loss keeps
  improving to ~0.0006 — exactly the divergence a validation set exists
  to catch. Not addressed here (would need more data/regularization,
  future work); recorded because a working val split that reveals
  nothing informative would be a red flag, not a good sign.
- `training.evaluate()` runs the *entire* val set every epoch, no
  `QuietMapDownsampler` (downsampling exists to manage training-time
  cost/imbalance, not appropriate for evaluation) — restores whatever
  `model.training`/`model.eval()` state the caller had before it ran.

## Current status

Repo scaffolding, data collection, the spatial grid, time binning,
feature extraction, labeling, the class-imbalance training utilities, a
baseline CNN architecture, a first end-to-end training run,
inference/visualization, a prediction-vs-detection case study notebook,
a resolved, verified training-period decision (~Sept 2014 – Sept 2025),
a leakage-safe train/val split, a first stratified scale-up (44 train /
6 val samples across 25 dates), a data-exploration notebook
(`notebooks/data_exploration.ipynb`) that surfaced a real gap in the
first full-scale build's run-selection design, and now the
**corrected full-scale dataset build (v2)** (5,078 train / 980 val
samples, 63-64% active maps, 95.5% pull success after retries) are all
in place. The smaller scale-up's held-out validation results were the
clearest signal yet of what was still needed: significantly more
training data before generalization is real. That data now exists, and
**a full-scale training run confirms it worked**: a real, statistically
meaningful held-out generalization result (15.3x pooled separation,
probability-space Cohen's d 2.49 across 980 val samples — see
"Full-scale training run" above), not just a mechanics check on a
handful of samples. A 5-value **alpha sweep** (see "Alpha sweep"
above) corrected a backwards piece of earlier guidance (higher alpha
raises positive-cell confidence, not lower) and initially favored
alpha=0.9 by mean-separation metrics — but a follow-up **model
comparison summary** (see above, standardized AUC-ROC/AUC-PR/F1
metrics across every checkpoint, `scikit-learn` now a dependency)
reversed that: AUC-PR and best-threshold F1 both favor the *original*
alpha=0.25, and also revealed that AUC-ROC is misleadingly high
(0.92-0.98) at this imbalance while the honest metric, AUC-PR, sits
around 0.02-0.03 for every model tested — real signal (~100-150x the
no-skill baseline) but far more modest than ROC-AUC alone suggests.
**Project default remains alpha=0.25.** Architecture widening: ticket 1
built a reusable, tested evaluation module (`evaluate.py`); ticket 2's
width-vs-depth ablation found more receptive field (not more capacity)
helps, while channels-only widening and combining both axes did not;
ticket 3's dilated convolutions beat that with fewer parameters and no
overfitting; ticket 4's U-Net beat both — **AUC-PR 0.0387 (+36.7% over
the original 0.0283 baseline, +20.2% over ticket 3's dilated winner),
AUC-ROC 0.9882** — at 451,361 parameters (far more than ticket 3's
31,841) but still with no overfitting signature. **Ticket 5's
comparison notebook (`notebooks/architecture_comparison.ipynb`, see
"Architecture widening, ticket 5" above) confirms the U-Net as the
outright winner** across every metric and closes out this round of 5
tickets — `models/unet/tornado_unet.pt` is now this project's
best-performing checkpoint. Open questions for future work: no
architecture got its own hyperparameter tuning pass (all 5 shared the
original baseline's recipe), and the U-Net itself was one config, not
swept.

Following a literature-review roadmap's Tier 1 (see "Genuine held-out
test set" and "Gradient-boosted trees" above): a real 3-way
train/test/val split now exists (`split_run_bins_3way`,
`scripts/build_3way_split.py`), revealing that the U-Net's real,
never-touched-before test AUC-PR (**0.0836**) is more than double its
old, repeatedly-checked val number — a genuine finding about
single-split variance at this data scale, not a bug. A gradient-boosted
tree alternative was tried and **lost decisively** (test AUC-PR 0.0290
vs. the U-Net's 0.0836) — best explanation: GBT has zero spatial
receptive field, and every architecture ticket in this project found
receptive field is what actually drives improvement. The CNN/U-Net
direction remains the right one to keep pursuing. Ticket 1 (adding
updraft helicity features, the literature's single most-cited tornado
surrogate) also finished — the enriched dataset is now the canonical
one, but retraining the U-Net on it made AUC-PR *worse* (test: 0.0836 →
0.0491), a third instance of Cohen's d improving while AUC-PR dropped.
A follow-up ablation (user-proposed: retrain on UH25 alone, the one
layer with zero missing data, to rule out NaN-imputation as the cause)
confirmed UH itself — not the missing-data handling — is the dominant
driver of the regression: test AUC-PR only recovered to 0.0528, still
36.8% below the no-UH baseline (see "Updraft helicity features" above).
**All 3 Tier 1 tickets are now complete; none improved on the original
14-feature U-Net, which remains this project's best checkpoint.** The
recurring Cohen's d/AUC-PR divergence across three unrelated
experiments is itself the most actionable finding from this whole
round: trust AUC-PR, not mean-separation metrics, for every future
decision here.

A **12-fold leave-one-year-out cross-validation** (see "Year-holdout
cross-validation" above) resolved the val/test instability directly:
the trustworthy number for `models/unet/tornado_unet.pt` is **AUC-PR =
0.0486 ± 0.0180**, not the previously-headlined 0.0836 test figure,
which turned out to be an unusually favorable single draw (92nd
percentile of the 12 folds). The CV also surfaced a real, significant,
unexplained upward trend in skill by calendar year (r=0.79, p=0.002) —
plausibly tied to HRRR's own forecast-skill improvements over time
rather than anything in this pipeline, flagged for future investigation,
not yet resolved. Two follow-up tickets are queued next: scaling the
training data to full quiet-date coverage (1,682 additional quiet dates
beyond the current 300), and engineering a pressure-tendency feature
from HRRR's own forecast fields (motivated by the sibling `tornet`
project's experience with pressure-change features) — both written up
as Trello-style starter-prompt tickets rather than implemented yet.

- SPC tornado reports (2014–2025) downloaded and parsed to
  `data/processed/spc_tornado_reports_2014_2025.csv` (15,294 reports),
  including full track geometry (`path_wkt`) — see
  `scripts/download_spc_tornado_reports.py`. Range extended from an
  initial, arbitrarily-chosen 2012–2022 (picked to loosely match the
  sibling TorNet project) once the real HRRR-archive-availability
  question was actually investigated (see "Time binning" above).
- HRRR pulls via Herbie confirmed working (`scripts/smoke_test_herbie.py`).
- The ~40km grid is defined (see above): `src/tornado_predictor/grid.py`.
- The 0–8h per-run time-binning scheme is defined (see above):
  `src/tornado_predictor/time_bins.py`.
- HRRR feature extraction onto the grid is working end-to-end for a
  single `(init_time, fxx)` snapshot (see above) — not yet aggregated
  across a time bin's fxx values.
- Tornado report labeling is implemented (see above) and produces a
  sparse positive-label table at
  `data/processed/tornado_labels_2014_2025.csv` (147,440 rows, 111,580
  unique positives).
- A consolidated (sample, row, col) training dataset pipeline combining
  features + labels is working end-to-end (see above):
  `src/tornado_predictor/dataset.py`. Three builds exist: the original
  6-sample single-outbreak pilot (`training_dataset_pilot.nc`), a first
  stratified scale-up (44 train / 6 val across 25 dates,
  `training_dataset_train_scaled.nc`/`training_dataset_val_scaled.nc`),
  and now the corrected full-scale build v2 (5,078 train / 980 val
  across a report-covering set of runs, 2014–2025 — see "Full-scale
  build v2" above —
  `training_dataset_train_full.nc`/`training_dataset_val_full.nc`; the
  earlier v1 build is archived as
  `training_dataset_{train,val}_full_v1_superseded.nc`). Not yet
  aggregating full bins (single representative fxx per bin only) —
  this remains true at every scale built so far.
- The class-imbalance training utilities (focal loss, dataset wrapper,
  quiet-map downsampler) are implemented and tested (see above):
  `src/tornado_predictor/training.py`. `torch` is now a dependency.
- A baseline CNN architecture (`TornadoCNN`) is implemented and tested
  against the real pilot dataset, including a full gradient check (see
  above): `src/tornado_predictor/model.py`.
- A training loop (`training.train_model()`, `scripts/train_model.py`)
  is implemented and has been run end-to-end on the pilot dataset (see
  above) — loss dropped from 0.0712 to 0.0007 over 50 epochs.
- Inference + a training-set-fit sanity check + visualization are
  implemented and run end-to-end (see above):
  `src/tornado_predictor/inference.py`, `scripts/run_inference.py`,
  `outputs/inference_pilot.png`. Model consistently, meaningfully
  separates positive from negative cells (~40-90x), but absolute
  positive-class probabilities are modest (~0.3, not near-1.0) — traced
  to `FocalLoss`'s `alpha=0.25`, not yet retuned. No generalization
  evaluation exists — everything so far is training-set fit on a
  6-sample pilot. Scaling past that pilot is the next real step.
- A prediction-vs-detection case study notebook is built and executed
  (see above): `notebooks/case_study_prediction_vs_detection.ipynb`.
- A leakage-safe train/val split is implemented and validated on both a
  small two-outbreak example and the larger 25-date scale-up (see
  above): `src/tornado_predictor/split.py`. `training.train_model()`
  now optionally tracks val loss per epoch — surfaced a real
  overfitting signature at pilot scale, and at the scale-up's larger
  size, surfaced the more important finding that held-out
  positive/negative separation is still weak (~2.5–50x, not the
  training-set-only ~40–90x) — real evidence more data is needed.
- `requirements.txt` has data-handling deps (`herbie-data`, `xarray`,
  `numpy`, `pandas`, `netCDF4`, `matplotlib`, `shapely`, `pyproj`,
  `pytest`), `torch` as the first ML dependency, `nbconvert`/
  `ipykernel`/`nbformat` for building/executing notebooks, and
  `scikit-learn`/`scipy` for standardized AUC-ROC/AUC-PR/precision/
  recall/F1 metrics (see "Model comparison summary" above).

## Repo layout

- `src/tornado_predictor/` — package code (installed editable via `pyproject.toml`)
- `scripts/` — CLI entry points (data download/parsing, grid building, one-off verification)
- `data/raw/` — raw HRRR pulls, raw SPC report files (gitignored contents)
- `data/interim/` — regridded/aligned intermediate data (gitignored contents)
- `data/processed/` — final gridded sample tables ready for modeling (gitignored contents)
- `models/` — trained model checkpoints (gitignored contents, regenerable via `scripts/train_model.py`)
- `outputs/` — generated visualizations (gitignored contents, regenerable via `scripts/run_inference.py`)
- `notebooks/` — exploratory work
- `tests/`

## Environment

Python virtualenv at `.venv/`. Install with:
```
pip install -r requirements.txt
pip install -e .
```
The editable install (`pip install -e .`) is required for scripts to
`import tornado_predictor`.
