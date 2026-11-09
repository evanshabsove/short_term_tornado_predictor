# Pipeline mechanics: documented-case validation, class imbalance, baseline CNN, training loop, inference, case study

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Validation against a documented case

`scripts/01_data_pipeline/validate_documented_case.py` spot-checks the grid/time-bin/
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
`scripts/03_architecture_search/train_model.py`.

- **Determinism split across two seeding responsibilities.**
  `train_model()`'s only training-time source of randomness (which
  quiet maps get kept, batch shuffling) is controlled by its own
  `seed` argument, independent of torch's global RNG. Model weight
  *initialization* happens outside `train_model()` (in the caller), so
  full end-to-end reproducibility requires `torch.manual_seed(seed)`
  before constructing the model — `scripts/03_architecture_search/train_model.py` does this;
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
  regenerable via `scripts/03_architecture_search/train_model.py`): loss dropped from 0.0712
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
that sample's known positive labels, via `scripts/03_architecture_search/run_inference.py`.

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


