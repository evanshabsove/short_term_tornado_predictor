# Short-Term Tornado Predictor

## Goal

A short-range (0–8 hour lead time) tornado forecasting model driven by
HRRR (High-Resolution Rapid Refresh) numerical weather prediction fields,
trained against SPC (Storm Prediction Center) storm reports as ground truth.

This is a **separate but complementary** project to an existing CNN-based
tornado detection model built on radar data. That model detects tornadoes
in near-real-time from radar imagery; this project forecasts tornado
probability ahead of time from NWP model output.

## Methodology

This project follows a **dense-grid classification** approach, in the
style of Sobash et al. (2020), rather than a storm-centered approach
(e.g. TorNet): every `(grid cell, time bin)` pair is one sample, and the
domain/forecast period is labeled exhaustively rather than only near
observed storms.

The rest of this README documents exactly how that dataset is built —
grid definition, time bins, features, and label logic — so the pipeline
is reproducible from scratch and the design decisions (and why they were
made) are recorded somewhere other than one person's memory.

## Dataset Build

The pipeline has five stages, each a small, independently runnable
module + script under `src/tornado_predictor/` / `scripts/`. Every stage
is covered by offline unit tests in `tests/` except where noted.

### 1. Grid definition (`grid.py`, `scripts/build_grid.py`)

The ~40 km grid is **not** an independently reprojected or interpolated
grid. It's built by **block-aligned stride coarsening of HRRR's own
native grid**:

- HRRR's native grid is a Lambert Conformal Conic projection (standard
  parallels 38.5°N, central meridian -97.5°, spherical earth
  R=6,371,229 m, 3 km spacing, 1799×1059 points) — these constants are
  verified against a live HRRR grib file's `GRIB_*` keys by
  `scripts/verify_hrrr_grid_params.py`.
- Coarse cells are exact, contiguous 13×13 blocks of native HRRR pixels
  (13 × 3 km = **39 km**, the closest integer stride to "~40 km").
  Coarse grid shape is **(81, 138)** rows × cols; any leftover partial
  block at the north/east domain edge is dropped rather than kept as a
  ragged cell.
- The full native HRRR domain is used, not clipped to a CONUS land
  boundary — the grid extends over ocean/Mexico/Canada at the edges.
  The domain is *not* a simple lat/lon bounding box either: LCC
  projection curvature means the NW corner reaches as far as ~134°W
  despite the SW corner only being at ~123°W. This is genuine grid
  geometry (confirmed by computing all four native corners directly),
  not a bug.
- Because the grid is derived purely from projection math, building it
  requires no HRRR data download. It's saved to
  `data/processed/hrrr_coarse_grid_stride13.nc` — a self-contained
  NetCDF carrying cell-center lat/lon plus every projection parameter
  needed to reconstruct cell boundaries, so a loaded grid never depends
  on `grid.py`'s constants matching what built the file.

**Reproduce:** `python scripts/build_grid.py`

### 2. Time bins (`time_bins.py`)

Forecast lead time scope is **0–8h** (extended from an original 0–6h,
which doesn't divide evenly into 4-hour bins), split into two equal,
half-open 4-hour bins: bin 0 = `[0h, 4h)`, bin 1 = `[4h, 8h)`. This is
pure date/time arithmetic — no data download.

- **Every hourly HRRR run gets its own pair of bins.** All 24 hourly
  runs/day (not just synoptic 00/06/12/18Z) each define an independent
  0–8h timeline relative to *their own* init time, to maximize sample
  density.
- **Consequence — label multiplicity is intentional.** One absolute
  valid time (e.g. one tornado report) falls in a different bin
  depending on which run's timeline it's scored against, and becomes a
  positive label in the sample sets of *every* run whose
  `[init, init+8h)` window contains it — always exactly 8 different
  hourly runs (`candidate_run_inits`), split across their two bins.
  This is not deduplicated to one run per event; it's the correct
  behavior for a per-run sampling scheme.
- Two related functions serve two different inputs: `fxx_to_bin_index`
  maps an integer HRRR forecast hour (`fxx=8` itself is excluded from
  both bins under the half-open convention); `assign_valid_time_bin`
  maps an arbitrary absolute UTC timestamp (e.g. an SPC report, which
  isn't on the hour) using continuous lead-time hours.

**Training period (resolved, verified live — not assumed): ~Sept 2014 –
Sept 2025.** HRRR's native grid, required fields, and forecast length
were confirmed identical all the way back to Sept 2014 (the actual
start of the public AWS archive, `noaa-hrrr-bdp-pds`) — there's no
structural reason tied to HRRR's version history (v1–v4) to start
later. The upper bound comes from SPC, not HRRR: HRRR is produced
continuously to the present, but SPC's published report data was
confirmed live to currently end around 2025-09, lagging real time by
roughly a year. See `CLAUDE.md`'s "Time binning (locked in)" section
for the full verification detail.

### 3. Features (`features.py`, `scripts/extract_features.py`)

For one `(init_time, fxx)`, pulls these HRRR fields via
[Herbie](https://github.com/blaylockbk/Herbie) and pools each onto the
coarse grid:

| Field | Herbie search string |
|---|---|
| `cape` (surface-based CAPE) | `CAPE:surface` |
| `cin` (surface-based CIN) | `CIN:surface` |
| `srh_0_1km` (0–1km storm-relative helicity) | `HLCY:1000-0 m above ground` |
| `srh_0_3km` (0–3km storm-relative helicity) | `HLCY:3000-0 m above ground` |
| `shear_0_6km` (0–6km bulk shear magnitude) | computed as `hypot(u, v)` from `VUCSH:0-6000 m above ground` + `VVCSH:0-6000 m above ground` |
| `t2m` (2m temperature) | `TMP:2 m above ground` |
| `d2m` (2m dewpoint) | `DPT:2 m above ground` |

Search strings were verified against a live HRRR `.idx` file — a loose
`"CAPE:"` would also match HRRR's other CAPE variants (mixed-layer,
most-unstable), so these stay exact.

- **0–6km shear, not 0–8km.** HRRR has no native 0–8km shear field at
  all; 0–6km is what it natively provides and is the standard
  severe-weather bulk-shear layer in this literature anyway.
- **Pooling, not interpolation.** Reuses the grid's exact 13×13
  native-pixel block alignment (reshape + `mean`/`max` per block) —
  zero interpolation error. Every field gets both `_mean` and `_max`
  variables (14 output variables for 7 physical fields), to capture
  peak vs. typical conditions per cell.
- **One Herbie call per field**, not combined regex calls — avoids
  uncertainty about cfgrib's assigned variable names and Herbie's
  multi-hypercube merge behavior when combining different level-types
  in one call.
- This pulls a single `(init_time, fxx)` snapshot. Aggregating multiple
  `fxx` snapshots within one time bin into one bin-level feature set
  is a possible future refinement (see
  [Known Limitations](#known-limitations)).

**Reproduce:** `python scripts/extract_features.py --init-time "2024-05-06 00:00" --fxx 3`

### 4. Label logic (`labels.py`, `scripts/build_labels.py`)

Assigns positive tornado labels to `(grid cell, run init_time,
bin_index)` samples from SPC storm reports
(`data/processed/spc_tornado_reports_2014_2025.csv`, itself built by
`scripts/download_spc_tornado_reports.py` with full track geometry).

- **Full-track spatial matching, not touchdown-point-only.** Each
  report's track (`path_wkt`, a straight line from touchdown to
  lift-off) is buffered by half its damage-path width (`width_yd`,
  converted to meters) to build a corridor polygon, then intersected
  against the *exact* projected bounding box of each candidate grid
  cell — both the buffering and the intersection happen in the grid's
  native LCC meters, not lat/lon, for precision. This credits every
  cell a tornado crossed, not just where it touched down.
- **Temporal matching necessarily collapses to a single timestamp.**
  SPC's database has no per-point track timing or end-time field, so
  the *entire* buffered track is matched against the report's one
  touchdown `timestamp_utc`. Spatial matching uses the full track;
  temporal matching uses one instant. This is a real, permanent
  simplification given the source data, not an oversight.
- **Only positive labels are materialized.** The output table
  (`data/processed/tornado_labels_2014_2025.csv`) has one row per
  `(event_id, init_time, bin_index, row, col)` — every row is an
  implicit label of 1; anything absent is an implicit 0. The full
  0-label space isn't materialized (intractable without a defined
  training-run list — see [Known Limitations](#known-limitations)).
- Row granularity is per-(event, run, cell), **not deduplicated** —
  one report produces `(cells it touches) × 8` rows, preserving which
  report(s) caused which label. Collapsing to unique `(init_time,
  bin_index, row, col)` positives is a trivial `.drop_duplicates()` at
  training-table build time.
- A report's `width_yd` must be strictly positive — a zero-radius
  buffer of a line is an *empty* polygon in `shapely`, which would
  silently drop the report. `labels.buffer_radius_m` raises rather than
  allowing that to pass silently.

**Validated** against the real 2014–2025 dataset: 12 of 15,294 reports
fall outside the grid domain (proportionally consistent with the
original 2012–2022 pass's 10/12,649), and the one outlier touching >10
cells (13) is the December 10, 2021 Quad-State/Mayfield EF4 — the
longest track in the dataset (~165 miles), not a bug.

**Reproduce:** `python scripts/build_labels.py`

### 5. Consolidated training dataset (`dataset.py`, `scripts/build_training_dataset.py`)

Combines feature extraction and labeling across multiple HRRR runs into
one array-ready dataset.

- **One representative fxx per bin.** A bin spans 4 forecast hours, but
  each sample here uses only the bin's first hour (fxx=0 for bin 0,
  fxx=4 for bin 1) as a stand-in for the whole bin, keeping Herbie
  network cost proportional to sample count rather than 4× it.
- **Storage keeps `(sample, row, col)` structure** — one variable per
  field/statistic plus `label` — self-describing and consistent with
  every other artifact in this repo. `to_training_arrays(ds)` is a
  separate function that flattens this into literal `(sample, grid,
  feature)` + `(sample, grid)` numpy arrays for direct ML consumption;
  that flattening is lossy (loses which axis was row vs. col) and is
  deliberately not baked into the storage format.
- **Current build is a pilot, not a production dataset** — demonstrated
  on 3 hourly HRRR runs (2021-12-10 21:00/22:00/23:00 UTC) bracketing
  the December 10–11, 2021 outbreak, chosen because it's within the
  labeled SPC range *and* produces genuine positive labels (167
  positive sample/cell pairs across 6 samples), not an all-zero check.

**Reproduce:** `python scripts/build_training_dataset.py --init-times "2021-12-10 21:00" "2021-12-10 22:00" "2021-12-10 23:00"`

### Validation against a documented case (`scripts/validate_documented_case.py`)

Rather than a synthetic sanity check, the pipeline is spot-checked
against one specific, real, well-documented tornado: the EF4 that
touched down in Arkansas at 2021-12-11 01:07 UTC and tracked ~81 miles
into Tennessee (SPC `event_id 2021_2112101907-01`) — the same tornado
the sibling CNN radar-detection project already uses as a named case
study (TorNet catalog `event_id 997130`, KNQA radar), so both projects'
validation stories anchor on the same real event.

All checks pass: the touchdown point lands 9.5 km from its assigned
grid cell's center (well inside the ~19.5 km half-cell tolerance); the
buffered track's 6 cells include both touchdown and lift-off; the
precomputed labels table agrees with fresh recomputation for all 8
candidate HRRR runs; and pulled HRRR features show no NaNs, no
degenerate all-zero fields, and a physically realistic signature
leading up to touchdown — 0–1km and 0–3km storm-relative helicity more
than double in the final hour before the tornado, while CAPE stays
substantial (1200–2000 J/kg) throughout.

**Reproduce:** `python scripts/validate_documented_case.py`

## Class Imbalance (for modeling)

The dense-grid design makes for extreme, but structured, class
imbalance. Grounded in this project's real labels
(`tornado_labels_2014_2025.csv`): if every hourly HRRR run in this
period were used, the per-cell positive rate would be **~5.1×10⁻⁵ (~1
in 19,600)** — but **~17% of `(run, bin)` grid-maps contain at least one
positive cell**. Most of the imbalance is "positives are rare within an
active map," not "active maps are rare."

Two standard fixes were weighed:

- **Class weighting / focal loss** (loss-level) vs. **downsampling**
  (data-level).
- For a CNN over the dense grid (a spatial architecture, not an
  independent per-cell classifier), **per-cell downsampling isn't
  really executable** — you can't drop arbitrary pixels out of a grid
  map without breaking the convolutional receptive field. Downsampling
  can only sensibly happen at the *map* level (keep vs. drop entire
  `(run, bin)` grid-maps).

**Decision:** favor **focal loss** over a fixed class weight (a
1:23,000 weight ratio as a blanket multiplier tends to make training
unstable — a handful of positive pixels can dominate the gradient
noisily; focal loss down-weights *easy* examples dynamically instead),
paired with **light downsampling of fully-quiet maps only** (keep every
active map, keep a fraction of quiet ones, re-drawn each epoch) — for
training-speed reasons, not because weighting alone can't handle the
rest.

This is implemented and tested in `src/tornado_predictor/training.py`
(`FocalLoss`, `DenseGridDataset`, `QuietMapDownsampler`; `torch` is now
a dependency). `alpha=0.25, gamma=2.0` (RetinaNet defaults) and
`quiet_keep_fraction=0.2` are starting points, not tuned to this
dataset. Note `QuietMapDownsampler` is currently a no-op — the pilot
dataset's 6 samples are all "active," so there are no quiet maps yet to
downsample; it starts doing real work once the training set is scaled
beyond a single outbreak.

## Baseline CNN Architecture

`src/tornado_predictor/model.py` defines `TornadoCNN` — a minimal
first baseline (explicitly not a final architecture): input is
`(channels, row, col)` pooled HRRR features, output is a per-cell
`(row, col)` logit map.

- **Full-resolution convolutions only** — three hidden 3×3 conv layers
  (default `hidden_channels=32`) + a final 1×1 conv, all with
  `padding=1` so every layer stays at the input's exact spatial
  resolution. No pooling/upsampling: the grid's shape (81, 138) isn't
  a clean power of 2, so a U-Net's downsample/upsample skip-connection
  alignment would add complexity not worth it for a first baseline.
- Effective receptive field is ~7×7 cells (~270km) — likely too small
  to capture full supercell- or synoptic-scale context. Widening this
  (more layers, dilated convs, or a real U-Net) is a natural next
  iteration.
- **Outputs raw logits, not probabilities** — matches `FocalLoss`'s
  expectation (it applies sigmoid internally, more numerically stable
  than training on already-sigmoided values). Use
  `torch.sigmoid(model(x))` for a probability map at inference time.
- Verified end-to-end against the real pilot dataset: correct output
  shape from real pooled features, and a full gradient check —
  `FocalLoss` against real labels for one sample backpropagates
  finite, nonzero gradients through every parameter.
- No training run has happened yet — wiring this into an actual
  training loop (optimizer, epochs, checkpointing) is the next step.

## Training Loop

`training.train_model()` wires `DenseGridDataset` + `QuietMapDownsampler`
+ `FocalLoss` + a given model into an actual training loop, used by
`scripts/train_model.py`.

- **No train/val split.** At the pilot's 6-sample scale a formal split
  isn't meaningful — the script trains on all samples and reports
  training loss only. This is a mechanics check, not model evaluation.
- **Determinism** requires seeding in two places: `train_model()`'s
  `seed` argument controls the sampler (independent of torch's global
  RNG), but model weight *initialization* happens in the caller, so
  `torch.manual_seed(seed)` must be called before constructing the
  model too — `scripts/train_model.py` does this.
- **Checkpoints are self-describing**: `model_state_dict`,
  `loss_history`, `training_hyperparameters`, `model_hyperparameters`
  (`in_channels`, `hidden_channels`), and `feature_names` — enough to
  reconstruct the exact model later without re-deriving anything from
  the dataset. Saved to `models/` (gitignored, regenerable).
- **First real run**: loss dropped from 0.0712 to 0.0007 over 50
  epochs on the pilot's 6 samples — the model fits this tiny dataset,
  as expected (memorization, not generalization — see the Consolidated
  Training Dataset section above for why the pilot is scoped this small).

**Reproduce:** `python scripts/train_model.py`

## Train/Val Split

`src/tornado_predictor/split.py` splits a `run_bins` list into
train/val without leakage — wired into `scripts/build_training_dataset.py`
(now produces two files) and `training.train_model()` (now optionally
tracks val loss per epoch via `training.evaluate()`).

Adapted from the TorNet benchmark paper's actual methodology (Veillette
et al. 2024, read directly, not from memory):

- **Split unit is UTC calendar date** of a sample's `init_time` — the
  honest analog to TorNet's "storm episode" grouping, given our samples
  are full-domain maps rather than storm-centered crops. Every sample
  from the same date goes to the same split, never split across both.
- **Deterministic day-of-year-mod-20 rule** (`< 17` → train, an 85/15
  split), matching TorNet exactly — this guarantees both splits see the
  full seasonal cycle every year, rather than risking a lucky/unlucky
  contiguous year range.
- **Temporal buffer safeguard** (default 24h): drops — doesn't
  reassign — any training sample within the buffer of any val sample's
  `init_time`. Needed because the mod-20 rule flips abruptly twice
  every 20-day cycle (verified: 2021-12-02 → train, 2021-12-03 → val,
  consecutive days) — roughly 10% of all days sit next to a boundary
  like this, not a rare edge case.
- **Does not balance tornadic/non-tornadic representation** via the
  split rule itself — TorNet does that at corpus-construction time
  (a separate decision, made when the training run/date list is
  assembled). `report_split_balance()` reports the resulting positive
  rate per split so an accidental imbalance is caught, not silently
  shipped.
- **Validated with a genuine two-outbreak split**: the existing Dec
  2021 pilot runs (`train`) plus real runs from the April 26–27, 2024
  outbreak (`val`) — 6 train / 4 val samples, 0 dropped by the buffer
  (the outbreaks are months apart), 100% positive rate in both splits.
- **Training with the real split surfaced a genuine overfitting
  signature** even at this tiny scale: val loss plateaus around
  epoch 10–15 (~0.0015) while train loss keeps improving to ~0.0006 —
  exactly what a validation set exists to catch.

**Reproduce:** `python scripts/build_training_dataset.py --init-times "2021-12-10 21:00" "2021-12-10 22:00" "2021-12-10 23:00" "2024-04-26 20:00" "2024-04-26 22:00"`

## First Scale-Up

Beyond the 6-sample single-outbreak pilot, a **44 train / 6 val sample**
dataset was built from a stratified sample of 25 dates (2014–2025) —
deliberately sized to complete in one interactive session (~200 HRRR
pulls) while meaningfully exercising the pipeline at real scale, not
the full multi-year archive.

- **Date selection**: 12 real high-activity dates (the single busiest
  SPC-report day per year, 2014–2025) plus 13 genuine quiet dates
  (verified zero SPC reports, spread across years/seasons) — the first
  time this project has included real negative-class diversity; every
  earlier dataset was 100% positive.
- **A real bug caught before it silently corrupted the dataset**: one
  initial quiet-date pick (2014-02-10) predates the verified HRRR
  archive start (~Aug 15, 2014). The pull failed loudly rather than
  returning wrong data — fixed by swapping in a verified alternative
  (2014-11-10) before re-running.
- **Result**: 44 train samples (13 active / 31 quiet — `QuietMapDownsampler`
  finally does real work) / 6 val samples (3 active / 3 quiet).
- **The important, honest finding**: on the *held-out* val set,
  positive/negative probability separation is much weaker than the
  pilot ever showed — ranging ~2.5x to ~50x depending on the sample,
  versus the pilot's consistent ~40–90x on training data, and one
  positive cell scored essentially zero probability (a clear miss).
  Loss values themselves are lower than the pilot's (~0.0002–0.0003 vs
  ~0.0007–0.0017), but that's not a sign of a better fit — a
  mostly-quiet dataset naturally has lower mean loss regardless of
  discriminative quality. This is the first genuine generalization
  signal this project has produced, and it says plainly: **44 training
  samples still isn't enough.**

**Reproduce:** see `CLAUDE.md`'s "First scale-up" section for the full
25-date list and exact command.

## Inference + Sanity Check

`src/tornado_predictor/inference.py` loads a checkpoint, predicts a
per-cell probability map for a sample, and sanity-checks/visualizes it
against that sample's known positive labels, via `scripts/run_inference.py`.

- **`assert_feature_order_matches` is a real correctness guard.** The
  dataset's feature channel order must exactly match the checkpoint's
  training-time order — a silent mismatch would feed the wrong
  physical field into the position the model learned to associate with
  e.g. CAPE, producing plausible-looking but meaningless predictions.
  Raises rather than allowing that to pass silently.
- **This is a training-set-fit check, not a generalization test** — the
  pilot's 6 samples are the same 6 the model trained on.
- **Empirical finding**: positive-cell probability does *not* approach
  1.0. Across all 6 pilot samples, mean probability at true positive
  cells lands around 0.26–0.34 vs. 0.003–0.007 at negative cells — a
  strong, consistent ~40–90x separation, but well short of confident
  (near-1.0) predictions, and in every sample at least one negative
  cell scores *higher* than the mean positive cell. Traces to
  `FocalLoss`'s `alpha=0.25` down-weighting the positive class's loss
  contribution (per Lin et al. 2017's own formula), which matters more
  at this dataset's per-sample imbalance than RetinaNet's original use
  case — confirms `FocalLoss`'s own docstring note. Not yet retuned.
- **Visualization** (`outputs/inference_pilot.png`, gitignored):
  predicted probability heatmap (viridis — sequential, colorblind-safe)
  with true positive cells marked as a red X, one panel per sample.
  Confirms visually that the model produces a spatially coherent
  elevated-probability plume tightly localized around the true
  positive cells in every sample, not memorized noise scattered
  arbitrarily across the domain.

**Reproduce:** `python scripts/run_inference.py`

## Case Study: Prediction vs. Detection

`notebooks/case_study_prediction_vs_detection.ipynb` — a self-contained,
**executed** notebook (ships with real cached outputs) lining up this
project's prediction model against the sibling `tornet` detection
project's model, for the exact same real tornado: SPC `event_id
2021_2112101907-01` (AR→TN EF4, touchdown 2021-12-11 01:07 UTC),
matching TorNet's own documented case study (catalog `event_id 997130`,
KNQA radar). Note this is a **different** tornado than the one used to
choose the pilot dataset's HRRR runs (`2021_2112102054-01`, the
Quad-State/Mayfield EF4 from the same outbreak night).

**Framed explicitly as a mechanics demo, not an accuracy claim** — see
the notebook's own opening cell. Highlights:

- Only 3 of the pilot's 6 samples are genuine forecasts of this event
  (determined programmatically, not assumed); the other 3 are positive
  at the same cell because a *different* tornado that night also
  crossed it.
- The detection model's number (logit 6.4960, probability 99.85%) was
  pulled directly from `tornet`'s own already-executed notebook — not
  re-run here, since that needs a separate environment/weights.
- Reports, rather than hides, that prediction probability at the
  touchdown cell does *not* increase monotonically as lead time
  shortens — an expected symptom of training on 6 samples.

**Reproduce:** `jupyter nbconvert --to notebook --execute --inplace notebooks/case_study_prediction_vs_detection.ipynb`

## Reproducing the Full Pipeline

In order, from a clean clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/download_spc_tornado_reports.py      # SPC reports -> data/processed/
python scripts/build_grid.py                         # ~40km grid  -> data/processed/
python scripts/verify_hrrr_grid_params.py             # optional: live grid-param check (needs network)
python scripts/build_labels.py                        # labels      -> data/processed/
python scripts/extract_features.py --init-time "2024-05-06 00:00" --fxx 3   # single-snapshot smoke test
python scripts/build_training_dataset.py --init-times "2021-12-10 21:00" "2021-12-10 22:00" "2021-12-10 23:00" "2024-04-26 20:00" "2024-04-26 22:00"  # leakage-safe train/val split -> data/processed/
python scripts/validate_documented_case.py            # spot-check against a real event
python scripts/train_model.py --dataset data/processed/training_dataset_train.nc --val-dataset data/processed/training_dataset_val.nc  # train + track val loss -> models/
python scripts/run_inference.py                        # inference + sanity check -> outputs/
jupyter nbconvert --to notebook --execute --inplace notebooks/case_study_prediction_vs_detection.ipynb
pytest tests/                                          # offline unit tests
```

## Extending Beyond Short-Range

The design has specific extension points, and specific hard limits
worth knowing before assuming this scales straightforwardly:

- **More/longer time bins within HRRR's own range**: `time_bins.py`'s
  `BIN_EDGES_HOURS = (0, 4, 8)` is a plain tuple — extending to, say,
  `(0, 4, 8, 12, 16)` to use more of HRRR's forecast length is a small,
  contained change (the module's bisect-based logic already generalizes
  to more than 2 bins).
- **HRRR's own ceiling is the real limit for "longer range."** Most
  hourly HRRR runs only extend to 18h; only the four synoptic runs
  (00/06/12/18Z) extend to 48h. There is no way to get medium-range
  (multi-day) lead times from HRRR at all — that would require a
  different NWP source entirely (e.g. GFS for ~1–16 days, or an
  ensemble product like GEFS for probabilistic medium-range).
- **Swapping in a different NWP source is not a config change.**
  `grid.py`'s projection constants and `features.py`'s exact Herbie
  field/level strings are specific to HRRR's grid and grib output. A
  different model (different native grid, resolution, projection, and
  field-naming convention) would need its own grid-definition and
  feature-extraction modules, following the same design pattern
  (reuse the source model's native grid via block-alignment where
  possible; verify field names against a live `.idx`/inventory before
  hardcoding).
- **The labeling and time-binning logic is largely source-agnostic** —
  `labels.py` only depends on the grid object's interface, and
  `time_bins.py` is pure calendar arithmetic independent of any NWP
  source — so a medium-range extension would mainly mean a new
  grid/features pair, not a rebuild of the whole pipeline.

## Known Limitations

- **Training period is defined but not yet built at full scale.** The
  usable range (~Sept 2014 – Sept 2025) is resolved and verified, and
  SPC labels now cover it, but the actual dataset has only been built
  for a stratified 25-date, 50-sample scale-up within that range —
  scaling to the full period is future work.
- **Single representative fxx per bin**: feature extraction currently
  approximates a whole 4-hour bin with one hourly snapshot rather than
  aggregating all 4 hours in it.
- **Straight-line track approximation**: SPC provides only a start and
  end point per tornado, not intermediate waypoints, so `path_wkt` is a
  straight line — a simplification for any track with real curvature.
- **First scale-up's val results show real, not yet solved,
  generalization weakness** (see "First Scale-Up" above): held-out
  positive/negative separation is weak (~2.5–50x) compared to the
  pilot's training-set-only ~40–90x. 44 training samples is real
  evidence of insufficient data, not a tuning problem to paper over.
- **`FocalLoss`'s `alpha=0.25` may be poorly tuned for this dataset**:
  empirically, positive-cell probabilities plateau around 0.3 rather
  than approaching 1.0, consistent with `alpha` down-weighting the
  positive class more than this dataset's extreme per-sample imbalance
  calls for. Not yet retuned.

## Project Structure

```
short_term_tornado_predictor/
├── src/tornado_predictor/
│   ├── grid.py           # ~40km grid definition (pure projection math)
│   ├── time_bins.py       # 0-8h / two-bin scheme (pure date/time arithmetic)
│   ├── features.py        # HRRR field pull + pooling for one (init_time, fxx)
│   ├── labels.py           # SPC report -> (cell, run, bin) positive labels
│   ├── dataset.py          # combine features + labels across runs
│   ├── training.py         # class-imbalance handling + training loop
│   ├── split.py             # leakage-safe train/val split
│   ├── model.py             # baseline CNN architecture (TornadoCNN)
│   └── inference.py          # load checkpoint, predict, sanity-check, plot
├── scripts/                 # CLI entry points, one per pipeline stage (see above)
├── data/
│   ├── raw/                 # raw HRRR pulls, raw SPC report files
│   ├── interim/              # regridded / aligned intermediate data
│   └── processed/            # grid, labels, features, consolidated datasets
├── models/                    # trained model checkpoints (gitignored contents)
├── outputs/                    # generated visualizations (gitignored contents)
├── notebooks/                # exploratory analysis + case_study_prediction_vs_detection.ipynb
├── tests/                     # offline unit tests, one file per module above
├── requirements.txt
├── pyproject.toml              # editable install for src/tornado_predictor
└── CLAUDE.md                   # methodology/context notes for AI-assisted sessions
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Status

Grid, time binning, feature extraction, labeling, the class-imbalance
training utilities, a baseline CNN architecture, inference +
visualization, a prediction-vs-detection case study notebook, a
resolved training-period decision (~Sept 2014 – Sept 2025), a
leakage-safe train/val split, and a first stratified scale-up (44
train / 6 val samples, with real quiet-day negatives for the first
time) are all built and tested (see above). The scale-up's held-out
validation results are the clearest signal yet of what's needed next:
significantly more training data before generalization is real. Not
yet started: scaling further toward the full training period.
`requirements.txt` has data-handling deps (`herbie-data`, `xarray`,
`numpy`, `pandas`, `netCDF4`, `matplotlib`, `shapely`, `pyproj`,
`pytest`), `torch` as the first ML dependency, and
`nbconvert`/`ipykernel`/`nbformat` for
building/executing notebooks.
