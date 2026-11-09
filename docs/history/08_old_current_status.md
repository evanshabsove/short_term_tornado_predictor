# Old 'Current status' section

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted. This section mostly restated the sections above it; kept for completeness. Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

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
`scripts/02_dataset_build/build_3way_split.py`), revealing that the U-Net's real,
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
decision here. *(Note: the UH instance of this pattern did not replicate in the multi-seed re-test, and the alpha-sweep instance is single-seed; "trust AUC-PR" remains sound, but treat "recurring divergence" as less certain than written.)*

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

- A country mask and Canadian-coverage analysis for the coarse grid exists (see "Country mask and Canadian coverage" above): `data/processed/hrrr_coarse_grid_stride13_countries.nc`. 1,243 Canadian cells; HRRR domain stops ~52.5N, so central Alberta is at/over the edge.
- SPC tornado reports (2014–2025) downloaded and parsed to
  `data/processed/spc_tornado_reports_2014_2025.csv` (15,294 reports),
  including full track geometry (`path_wkt`) — see
  `scripts/01_data_pipeline/download_spc_tornado_reports.py`. Range extended from an
  initial, arbitrarily-chosen 2012–2022 (picked to loosely match the
  sibling TorNet project) once the real HRRR-archive-availability
  question was actually investigated (see "Time binning" above).
- HRRR pulls via Herbie confirmed working (`scripts/01_data_pipeline/smoke_test_herbie.py`).
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
- A training loop (`training.train_model()`, `scripts/03_architecture_search/train_model.py`)
  is implemented and has been run end-to-end on the pilot dataset (see
  above) — loss dropped from 0.0712 to 0.0007 over 50 epochs.
- Inference + a training-set-fit sanity check + visualization are
  implemented and run end-to-end (see above):
  `src/tornado_predictor/inference.py`, `scripts/03_architecture_search/run_inference.py`,
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


