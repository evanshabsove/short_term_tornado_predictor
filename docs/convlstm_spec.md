# ConvLSTM on the hourly store: spec

Written 2026-10-28 after a grilling session with the user. Research-only; nothing here touches the API, the production bundle, or the
locked grid / labels / time binning. The pre-registration for the full study is a later ticket (T7) and is **not** this document.

## 1. Question

Arm A of the hourly-features study (`docs/hourly_features_results.md`) summarized each 4-hour bin by time-mean / time-max of the 4 hourly
snapshots and gained +22.5% AUC-PR over the single-snapshot baseline, with false alarms on quiet bins of active runs falling in every fold.
That supports the hypothesis that the model is weak on *when* within a severe day. Does a model that sees the whole 8-hour hourly
sequence and learns the temporal combination (a ConvLSTM) beat A's hand-built pooling?

Primary comparison: **ConvLSTM vs. arm A** (same 14 base channels, same recipe, same folds and seeds). Arm B (extra fields) is out of
scope. If the ConvLSTM wins, a 31-channel version is a separate follow-up.

## 2. Settled design

| item | decision |
|---|---|
| Inputs | existing hourly store `data/interim/scaled_build_hourly/` (4,917 per-run files, 23 GB; 14 base fields as `(fxx=8, row, col)`), 14 base channels, raw (no scaling), checkpoint channel order. No new HRRR pulls. |
| Sample unit | one **run**: input `(8, 14, 81, 138)` (fxx 0-7), output `(2, 81, 138)` (bin 0, bin 1). A uses one sample per (run, bin), so ~4,900 runs vs ~9,800 samples. |
| Labels | the per-bin labels already staged in `scaled_build_wholebin/` (identical to v4 labels on shared samples), stacked into `(2, 81, 138)`. |
| Architecture | hybrid: a shared U-Net encoder applied to each timestep, a ConvLSTM over the sequence at the bottleneck (11x18 after pad to 88x144), a U-Net decoder producing 2 output channels. Reuses `unet.py` pad-once / crop-once and bilinear-upsample decoder. Conservative width (`base_channels=16`) per the overfitting lessons. |
| Loss / recipe | identical to A: FocalLoss alpha 0.25 gamma 2.0, lr 1e-3, batch 2, EMA 0.9999 (EMA weights scored), 50 epochs (full study), `quiet_keep_fraction` 0.2. Loss averaged over both bins. |
| Quiet downsampling | a run counts as quiet only if **both** bins are all-negative; 20% of those kept per epoch, redrawn each epoch. Different effective mix from A (which downsamples per bin). **Disclosed known difference.** |
| Split | unchanged: year-holdout folds 2016 / 2020 / 2024, 24 h buffer, day-level assignment. Test samples identical to A's. |
| Scoring | flatten `(run, 2 bins)` back to A's `(run, bin)` sample order, then the existing `evaluate.py` / `diagnose_sample_mix.map_level` path. Asserted by comparing sample order against A's cached scores. |
| Compute | CPU first. If the projected full study exceeds ~3 days, decide then: smaller model, fewer epochs, or rent a GPU on vast.ai. Not assumed now. |

## 3. Staged plan

**Stage 1, snippet (dev scale).** Fold 2020, seed 10, 10 epochs (loss was flat by ~epoch 5 in earlier runs), a stratified subset of that
fold's training runs (all active runs kept plus a random share of quiet ones) sized from a measured seconds-per-epoch so a run lands in
**1-2 hours on CPU**. The snippet is for building, debugging, and the gate; it does **not** produce the headline verdict.

A **matched control** is trained on the identical snippet: the existing `TornadoUNet(base_channels=16)` on the whole-bin A data (same runs,
epochs, seed). The snippet ConvLSTM is judged only against this control, never against A's full-scale numbers.

**Gate before expanding.** All three must hold:
1. trains stably: final val/train loss ratio roughly 0.8-1.2, no NaNs;
2. measured cost projects the full 9-run study to about 3 days or less on CPU (else a GPU/shrink decision is taken explicitly);
3. snippet AUC-PR is no worse than about 10% below the matched control.

Beating the control is not required at snippet scale (10 epochs on a subset is too small to expect it). Failing check 3 means debug or
rethink the architecture, not scale up.

**Stage 2, full study.** Pre-registered *before any ConvLSTM score is loaded* (T7), in the style of `docs/hourly_features_preregistration.md`.
Folds 2016 / 2020 / 2024, seeds 10-12, 50 epochs, EMA 0.9999, whole train set.
- Primary metric: AUC-PR of the 3-member ensemble (mean predicted probability of seeds 10-12), per fold.
- Comparison: `E_fold = ConvLSTM ensemble / A ensemble - 1`, pooled E = mean of the three; paired day-bootstrap 95% interval (1,000
  resamples of test days) against A's matched seeds-10-12 ensemble; ConvLSTM's percentile reported against A's single-model spread.
- **"Helps"**: pooled E >= +8% and >= 2 of 3 folds E > 0. **"Hurts"**: pooled E <= -8% and >= 2 of 3 folds E < 0. Otherwise "no detectable effect".
  A "helps" is **confirmed** only if the paired interval excludes 0 in >= 2 folds, else "weak: inside noise".
- Secondary, descriptive (no decision rule): best-F1 with precision/recall, Cohen's d, map-level AUC-ROC, 1-cell-dilated AUC-PR, and the
  **false-alarm rate on quiet bins of active-day runs at 50% / 80% recall of active maps** (the timing hypothesis).
- **Stated up front:** with 3 seeds per fold the minimum detectable effect is ~12%, above the 8% threshold, so "no detectable effect"
  does not mean "no effect". A's 3-member ensemble is itself one draw; the noise band is the seed spread from `models/ema_study/`.
- Thresholds are the project's convention, copied from the UH and hourly studies, not derived.

## 4. Known risks and things to check

- **Cost**: ConvLSTM over 8 steps is much slower per epoch than the U-Net (47-78 min for 50 epochs). Dilated convs also cost more on CPU than
  FLOPs predict; measure, don't extrapolate.
- **Memory / IO**: one run is `8x14x81x138` float32 ~ 5 MB; the full train set is ~24 GB as float32. Lazy per-run loading from the existing
  hourly files (optionally a float16 cache) rather than holding it all in RAM.
- **Channel order**: `DenseGridDataset` takes channel order from the first staged file; assert against the checkpoint order
  (`inference.assert_feature_order_matches`), as the earlier pitfall showed.
- **Confounds**: the sample unit and quiet-downsampling mix differ from A (disclosed). Seeds 10-12 reuse the same seed numbers as A.
- **ConvLSTM-specific**: initial hidden state, gate-bias init (forget gate), and gradient clipping may matter; keep defaults first, change only
  after the gate, and record any change.
- **fxx 8**: the store has fxx 0-7 only; fxx 7 is bin 1's last hour. Nothing beyond.
- Single-seed rankings sit inside ~19% seed noise (project rule): the snippet gate uses a coarse 10% tolerance for that reason and decides
  nothing about accuracy.

## 5. Out of scope

Arm B features, input normalization, UH, production serving or live whole-bin pulls, changing the grid / labels / time bins, GPU work unless
the gate says so, hyperparameter tuning beyond the first comparison.
