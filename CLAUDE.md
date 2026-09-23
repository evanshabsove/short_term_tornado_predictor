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
- **Known future constraint (not solved here):** HRRR's public
  archive availability starts well after 2012 (closer to 2014–2018
  depending on HRRR version/source reliability), while the SPC report
  range currently spans 2012–2022. The real usable training period
  will be the intersection of the two, narrower than the full SPC
  range — flagged for a future session, not addressed by this module.
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
  (`data/processed/tornado_labels_2012_2022.csv`) contains one row per
  `(event_id, init_time, bin_index, row, col)` — every row is an
  implicit label of 1; anything not present is an implicit 0. The full
  0-label space is not (and can't yet be) materialized — there's no
  defined training-run list until the HRRR-archive-availability gap
  (see "Time binning" above) is resolved.
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
  2012–2022 CSV, but not guaranteed for other date ranges.
- Validated against the real dataset: exactly 10 of 12,649 reports fall
  outside the grid domain (matches the known out-of-bounds count from
  grid validation), and the single outlier with >10 affected cells (13)
  is the December 10, 2021 Quad-State/Mayfield EF4 — the longest track
  in the dataset (~165 miles) — not a bug.

### Consolidated training dataset (pilot, not locked in)

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
  check. Scaling this to a real training-period date range is still
  blocked on the HRRR-archive-availability gap (see "Time binning"
  above) — not resolved by this pilot.

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

- Grounded in this project's real numbers (`tornado_labels_2012_2022.csv`):
  training on every hourly HRRR run 2012–2022 would give a per-cell
  positive rate of ~4.3e-05 (~1 in 23,000), but ~15% of `(run, bin)`
  grid-maps contain at least one positive cell — most of the imbalance
  is "positives are rare within an active map," not "active maps are
  rare."
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
  than at RetinaNet's original object-detection imbalance. Confirms
  `FocalLoss`'s own docstring note ("expect to need a lower alpha").
  Don't "fix" this by chasing near-1.0 probabilities without
  discussing with the user first — it may just need a lower `alpha`,
  or may be an inherent limit of training on 6 samples with a 3-layer,
  ~270km-receptive-field model; both are open, not yet investigated.
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

## Current status

Repo scaffolding, data collection, the spatial grid, time binning,
feature extraction, labeling, the class-imbalance training utilities, a
baseline CNN architecture, a first end-to-end training run,
inference/visualization, and a prediction-vs-detection case study
notebook are all in place; scaling to a real training period and true
generalization evaluation have not started yet.

- SPC tornado reports (2012–2022) downloaded and parsed to
  `data/processed/spc_tornado_reports_2012_2022.csv`, including full
  track geometry (`path_wkt`) — see `scripts/download_spc_tornado_reports.py`.
- HRRR pulls via Herbie confirmed working (`scripts/smoke_test_herbie.py`).
- The ~40km grid is defined (see above): `src/tornado_predictor/grid.py`.
- The 0–8h per-run time-binning scheme is defined (see above):
  `src/tornado_predictor/time_bins.py`.
- HRRR feature extraction onto the grid is working end-to-end for a
  single `(init_time, fxx)` snapshot (see above) — not yet aggregated
  across a time bin's fxx values.
- Tornado report labeling is implemented (see above) and produces a
  sparse positive-label table at
  `data/processed/tornado_labels_2012_2022.csv`.
- A pilot consolidated (sample, row, col) training dataset combining
  features + labels is working end-to-end (see above):
  `src/tornado_predictor/dataset.py`, `data/processed/training_dataset_pilot.nc`.
  Not yet scaled to a real training-period date range (still blocked
  on the HRRR-archive-availability gap) and not yet aggregating full
  bins (single representative fxx per bin only).
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
- `requirements.txt` has data-handling deps (`herbie-data`, `xarray`,
  `numpy`, `pandas`, `netCDF4`, `matplotlib`, `shapely`, `pyproj`,
  `pytest`), `torch` as the first ML dependency, and `nbconvert`/
  `ipykernel`/`nbformat` for building/executing notebooks.

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
