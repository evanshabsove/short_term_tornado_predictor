# Locked decisions: full detail

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted. Contains the full text of the grid, time-binning (incl. training-period verification), feature-extraction, labeling and train/val-split decisions; CLAUDE.md keeps only the condensed rules. Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

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
  keys by `scripts/01_data_pipeline/verify_hrrr_grid_params.py`.
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
  `scripts/01_data_pipeline/build_grid.py`.
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
    range) via `scripts/01_data_pipeline/download_spc_tornado_reports.py` (defaults
    updated to `--start-year 2014 --end-year 2025`). The labels table
    was rebuilt to match: `data/processed/tornado_labels_2014_2025.csv`
    (147,440 rows; 111,580 unique positives) via `scripts/01_data_pipeline/build_labels.py`
    (defaults updated to match). All downstream default paths
    (`scripts/02_dataset_build/build_training_dataset.py`, `scripts/01_data_pipeline/validate_documented_case.py`,
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
`scripts/01_data_pipeline/extract_features.py`.

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
`scripts/01_data_pipeline/build_labels.py`.

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


### Train/val split (locked in)

`src/tornado_predictor/split.py` splits a `run_bins` list into
train/val without leakage, via `scripts/02_dataset_build/build_training_dataset.py`
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


