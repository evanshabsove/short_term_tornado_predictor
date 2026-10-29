# ConvLSTM tickets

Each ticket is a title, a description, and a starter prompt. Design and decisions: `docs/convlstm_spec.md`. Do them in order; T6 is a gate.
Every session should start by reading `CLAUDE.md` and the spec, and should activate `.venv` (bare `python` hits a broken pyenv shim; `tee`
hides exit codes, so use `set -o pipefail`).

---

## T1. Hourly sequence dataset

**Description.** A dataset that returns one run as input `(8, 14, 81, 138)` (fxx 0-7, 14 base channels, raw, checkpoint channel order) and
label `(2, 81, 138)` (bin 0, bin 1), read lazily per run from `data/interim/scaled_build_hourly/` with labels from `scaled_build_wholebin/`.
Reuses `split.py` year-holdout folds and the 24 h buffer so test samples match A's. Provides a run-level "quiet" flag (both bins all-negative)
and a flatten function back to A's `(run, bin)` sample order. Handles the 3 archive-gap runs the same way A did.

**Starter prompt.** Read `docs/convlstm_spec.md` sections 2 and 4 and `build_hourly.py`, `training.DenseGridDataset`, and `split.py`. Add
`DenseSequenceDataset` (new module, don't change existing ones). Tests: shapes and dtype, channel order equals the 14-feature checkpoint order
(`models/unet/tornado_unet.pt`), fxx 0 and 4 equal the v2-staged snapshots, labels equal the wholebin labels for a few real runs, the fold-2020
test sample order equals A's cached scores' order, and the run-quiet flag is correct on a known active and a known quiet run. Plan first, then
implement, and show me the plan before building.

---

## T2. ConvLSTM-U-Net model

**Description.** `TornadoConvLSTMUNet`: a shared U-Net encoder per timestep, a ConvLSTM (cell + sequence wrapper) over the 8 steps at the
bottleneck, and a U-Net decoder with 2 output channels, reusing `unet.py`'s pad-to-88x144 / crop-back logic and bilinear decoder. Raw logits
out. Registered in `inference.MODEL_REGISTRY` so `load_checkpoint` works with `model_class`. `base_channels=16` default.

**Starter prompt.** Read `unet.py`, `tests/test_unet.py`, and the spec section 2. Implement `ConvLSTMCell`, `TornadoConvLSTMUNet` and registry
entry. Decide skip-connection handling for the sequence (e.g. last-step encoder features, or ConvLSTM output only at the bottleneck) and
justify it before building. Tests: output `(B, 2, 81, 138)`, finite, gradient reaches every parameter, a permuted time order changes the output
(proves the recurrence is used), parameter count reported, and an existing U-Net checkpoint still loads unchanged. Plan first.

---

## T3. Training and scoring plumbing for sequence models

**Description.** A training path for run-level samples: FocalLoss averaged over both bins, a run-level `QuietMapDownsampler` (quiet = both bins
negative, keep 20%, redrawn each epoch), EMA 0.9999, val-loss tracking, a `--max-train-runs` / stratified subset option for the snippet (keep all
active runs), and a scoring function that flattens to `(run, bin)` and calls `evaluate.compute_metrics` / `diagnose_sample_mix.map_level`. A
`scripts/11_convlstm/` CLI that trains one fold and one seed and writes a self-describing checkpoint plus results JSON.

**Starter prompt.** Read `training.py`, `evaluate.py`, `scripts/05_year_cv_diagnostics/cross_validate_year.py`, and the spec. Add the sequence
training path without changing the existing one (existing tests must still pass, bit-identical for current callers). Tests: deterministic given
a seed, loss falls on a tiny real subset, EMA tracking does not change training, flatten order equals A's cached score order, subset keeps all
active runs. Plan first.

---

## T4. Timing and memory benchmark

**Description.** Measure real seconds per epoch and peak memory for the ConvLSTM-U-Net on fold 2020 at a few training-subset sizes, plus the
matched U-Net on whole-bin data. Pick the snippet size that lands a 10-epoch run in 1-2 hours, and project the full 9-run, 50-epoch cost.

**Starter prompt.** Using T1-T3, time 1-2 epochs at 3 subset sizes for both models (don't trust FLOP extrapolation: dilated convs were 2x worse
than predicted on this CPU). Report s/epoch, RSS, the chosen snippet size, and the projected full-study hours. Save to `models/convlstm/benchmark.json`.

---

## T5. Matched U-Net control on the snippet

**Description.** Train `TornadoUNet(base_channels=16)` on the whole-bin A data with the identical snippet (same runs, 10 epochs, seed 10, fold
2020, EMA). This is the only valid comparison for the snippet ConvLSTM.

**Starter prompt.** Reuse `cross_validate_year.py` with `--staging-dir scaled_build_wholebin` or the T3 path, restricted to exactly T4's chosen
subset (assert the same run list). Output to `models/convlstm/snippet_control/`. Report AUC-PR, best-F1, loss ratio, and the run list hash.

---

## T6. Snippet ConvLSTM run and gate

**Description.** Train the ConvLSTM-U-Net on the same snippet and evaluate the three gate checks from the spec: stable training, projected full
study <= ~3 days CPU (otherwise present options: smaller model, fewer epochs, vast.ai GPU), and AUC-PR no worse than ~10% below the control.
Write a short go / no-go note. Single seed, so describe the number with the +/-25% caveat.

**Starter prompt.** Run T3's CLI on T4's snippet at seed 10, then compare to T5's control. Report each gate check pass/fail with numbers and a
recommendation; if it fails, say what to debug first (gate biases, learning rate, gradient clipping, skip handling) rather than scaling up.
Don't start the full study.

---

## T7. Pre-registration (only after the gate passes)

**Description.** Write `docs/convlstm_preregistration.md` before any full-study ConvLSTM score exists, following
`docs/hourly_features_preregistration.md`: question, arms, the disclosed differences (sample unit, quiet-downsampling mix, seeds reused),
comparison and metric, decision rules ("helps" pooled E >= +8% and >= 2/3 folds; "confirmed" needs the paired interval to exclude 0 in >= 2 folds),
secondary metrics including the quiet-bins-of-active-runs false-alarm rate, the stated ~12% detectability limit, and my predictions with
honest priors. Record this file's sha256 in the results JSON.

**Starter prompt.** Read the spec section 3 and the hourly pre-registration. Draft the document and list what you inspected beforehand (only
data and snippet-gate facts, no full-study model output). Show it to me for review before any full run starts.

---

## T8. Full study: 3 folds x 3 seeds

**Description.** Train the ConvLSTM-U-Net on folds 2016 / 2020 / 2024, seeds 10-12, 50 epochs, EMA 0.9999, whole train set (9 runs), on CPU
or vast.ai per the gate decision. Then the evaluation script: 3-member ensemble per fold, `E = ConvLSTM / A - 1`, paired day-bootstrap
intervals, secondary metrics, JSON to `models/convlstm/full_study/`. Runner script activates the venv and fails on pipe errors.

**Starter prompt.** Follow `docs/convlstm_preregistration.md` exactly; do not change any rule after seeing a score. Build the runner
(`scripts/11_convlstm/run_full_study.sh`) and the evaluator, resumable per run. Report the scores against A's cached ensemble, and apply the
pre-registered rule mechanically.

---

## T9. Results write-up

**Description.** `docs/convlstm_results.md` plus a CLAUDE.md section: verdict by the pre-registered rule, per-fold table, secondary metrics,
which predictions held, caveats (3 seeds, detectability, sample-mix difference), and the explicit statement of what this does and does not
show. Negative or null results are written as plainly as positive ones.

**Starter prompt.** Read the pre-registration and the full-study JSON. Write the results doc and update CLAUDE.md in the repo's existing style
(sections with measured numbers and honest caveats). Do not reinterpret the rule.
