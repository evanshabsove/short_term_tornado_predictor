# First full-scale training, alpha sweep, model comparison, architecture tickets 2-5

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Full-scale training run (first statistically real generalization result)

`scripts/03_architecture_search/train_model.py --dataset training_dataset_train_full.nc
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

`scripts/03_architecture_search/sweep_alpha.py` trains 5 identical `TornadoCNN` runs on the
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
  `evaluate_checkpoint`) + `scripts/03_architecture_search/evaluate_model.py` (CLI, prints a
  summary and can save full metrics as JSON via `--out`). Architecture
  ticket 1 of the widening plan — every future architecture variant
  (tickets 2-5) should call this instead of re-deriving the metric
  computation. Verified to reproduce the comparison notebook's exact
  numbers bit-for-bit (`python scripts/03_architecture_search/evaluate_model.py` against the
  default full-scale/alpha=0.25 config prints AUC-ROC=0.9819,
  AUC-PR=0.0283, matching above precisely). `scikit-learn`'s
  `roc_auc_score`/`average_precision_score`/`precision_recall_curve`/
  `roc_curve` do the underlying computation; a zero-positives or
  zero-negatives `y_true` returns `nan` for the AUC fields instead of
  raising (`sklearn` itself raises `ValueError` on a single-class
  input) — same convention `inference.sanity_check_against_labels`
  already used. **`notebooks/model_comparison_summary.ipynb` and
  `scripts/03_architecture_search/sweep_alpha.py` were NOT retrofitted to call this module** —
  they keep their own inline copies of the same logic; only future work
  uses the extracted version.

### Architecture widening, ticket 2: width vs. depth ablation

`TornadoCNN` (`model.py`) generalized to accept `n_hidden_layers`
(default 3, exactly reproducing the original hardcoded architecture —
verified bit-identical against `tornado_cnn_full.pt`'s real predictions
before/after the change, so every existing checkpoint still loads
correctly with no migration). Added `receptive_field_cells(n)` =
`1 + 2*n`. `scripts/03_architecture_search/sweep_architecture.py` trained 4 new
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
`scripts/03_architecture_search/sweep_dilation.py` mirrors ticket 2's sweep script pattern.

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


