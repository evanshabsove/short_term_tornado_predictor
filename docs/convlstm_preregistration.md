# ConvLSTM-U-Net vs. arm A (whole-bin U-Net): pre-registration

Written 2026-10-29, **before any full-study ConvLSTM model is trained and before any full-study ConvLSTM score exists.**
Spec: `docs/convlstm_spec.md`. The results JSON (`models/convlstm/full_study/`) records this file's sha256, so a later edit is visible.

**What I had seen beforehand, and only that:**
- Data facts: hourly store coverage (4,916 of 4,919 runs), sample counts, active shares (63% of fold-2020 training runs are active; 62% of
  bin-maps, vs. A's 79% on the same fold), channel order, label identity with A's whole-bin labels, test-sample order (asserted equal to A's
  cached score order).
- Cost: the T4 benchmark (ConvLSTM ~2.4 h per 50-epoch run, ~22 h for the 9-run study on this CPU).
- **The snippet gate (T6), fold 2020, seed 10, 10 epochs only:** ConvLSTM AUC-PR 0.0573 (plain weights) / 0.0526 (EMA) vs. a matched U-Net control
  0.0518 / 0.0463. This is one seed at 10 epochs, an input to the go/no-go decision and to my priors below, not a result. **Fold 2020 and seed 10 are
  part of the full study, so I have seen one early, short-training number on one of its nine cells.** The snippet models are not reused; the full
  study trains all nine runs from scratch at 50 epochs.
- A's scores (`models/hourly_study/wholebin/hourly_eval.json`, cached float16 scores in `data/interim/diag_scores/hourly_wholebin_ema0.9999_s*_*.npy`)
  were known beforehand: they are the comparison. A's 3-member ensemble AUC-PR: 0.0533 / 0.0689 / 0.0844 (2016 / 2020 / 2024).

## 1. Question

Arm A feeds the U-Net one time-collapsed image per 4-hour bin (time-mean of `_mean` channels, time-max of `_max` channels over the bin's hours).
That discards the order of the hours. Does a model that sees the 8 hourly steps in order, with memory across them, rank tornado cells better than A?
Hypothesis: yes, mainly on timing, i.e. on which of a run's two bins the tornadoes fall in. Only arm A is the comparator; arm B (extra fields) is out of scope.

## 2. Arms

| | A (exists) | ConvLSTM-U-Net (new) |
|---|---|---|
| input | per bin, 14 channels from whole-bin aggregation; two independent maps per run | one run: 8 hourly steps (fxx 0-7) x the same 14 channels, raw |
| model | `TornadoUNet(base_channels=16)`, 451,361 parameters, one map per sample | `TornadoConvLSTMUNet(base_channels=16, hidden_channels=64)`, 640,178 parameters, shared U-Net encoder per step, ConvLSTM at the 11x18 bottleneck, shared decoder run twice (after hour 3 for bin 0, after hour 7 for bin 1), one 1x1 head per bin; output `(2, 81, 138)` logits |
| sample unit | one `(run, bin)` map; batch 2 maps | one run (2 bin-maps); batch 1 run, so each step still sees 2 maps |
| data / folds / seeds | `scaled_build_wholebin`, folds 2016 / 2020 / 2024, seeds 10-12 | same runs, same folds, same 24 h buffer, same seeds 10-12 |

Shared recipe: focal loss alpha 0.25 gamma 2.0, lr 1e-3, `quiet_keep_fraction` 0.2, 50 epochs, EMA 0.9999 (the EMA weights are what is scored).
The 3 test-fold-free archive-gap runs are excluded identically in both.

**Known differences, disclosed (these are part of what "ConvLSTM vs. A" means here, not corrected for):**
1. **Quiet downsampling is run-level.** A run counts as quiet only if both bins are empty, so the ConvLSTM sees ~62% active bin-maps per epoch vs.
   A's ~79%. Earlier v4 work found extra quiet data made no detectable difference; not re-tested.
2. **Bin 1 also sees hours 0-3** through the recurrence, which A's bin 1 does not (A's bin 1 uses hours 4-7 only). Bin 0 uses hours 0-3 only,
   as in A (verified by a gradient test: bin 0 is exactly independent of hours 4-7). This is intended, but it is information A does not have.
3. **Seeds are reused by number only.** Seed 10 means the same split and sampler seed, but the models are different architectures, so there is
   no pairing of random draws.
4. **Parameter count** is 1.42x A's. Capacity was not matched.
5. Inputs are raw, as in A (no standardization), per the earlier decision.

## 3. Comparison and metric

Primary metric: **AUC-PR of the 3-member ensemble** (mean predicted probability of seeds 10-12), per fold, on the held-out year, pooled over
all `(run, bin)` samples in A's order.
- Effect `E_fold = ConvLSTM ensemble / A ensemble - 1`, with A's own 3-member (seeds 10-12) ensemble as the denominator.
  **Pooled E = mean of the three `E_fold`.**
- Reported: a paired day-bootstrap 95% interval (1,000 resamples of test days) of the ensemble difference vs. A's matched ensemble; the
  ConvLSTM ensemble's position against A's three single-seed models; per-model single-seed AUC-PR for both.
- Before any ConvLSTM number is shown, the evaluator reproduces A's per-fold ensemble AUC-PR from cached scores (0.0533 / 0.0689 / 0.0844).

## 4. Decision rules (fixed now)

- **"Helps"**: pooled E >= +8% **and** E > 0 in at least 2 of 3 folds.
- **"Hurts"**: pooled E <= -8% **and** E < 0 in at least 2 of 3 folds.
- Otherwise **"no detectable effect."**
- **Strength wording** (does not change the verdict): a "helps" is *confirmed* only if the paired interval excludes 0 in >= 2 folds; otherwise
  *weak: inside noise*.
- **No extension rule.** A has only three models per fold, so there is no 6-vs-6 follow-up. Nothing triggers extra seeds, folds, epochs or a
  re-analysis. Failed or diverged runs are re-run with the same seed only if the cause is a crash; a NaN loss or divergence is reported as a
  result, not retried with a new recipe.
- **Honest limit, stated in advance:** with 3 seeds per fold the minimum detectable effect is ~12% (measured earlier), above the 8% threshold.
  "No detectable effect" does not mean "no effect," and a "helps" without the *confirmed* qualifier should be read as suggestive. A's 3-member
  ensemble is itself one draw: single-model AUC-PR spread across seeds is ~19%, and the ensemble halves but does not remove it.
- The thresholds are the project's convention, copied from the UH and hourly studies, not derived. One model class, one hyperparameter recipe
  (A's, not tuned for the ConvLSTM): a null is a null for this recipe.

## 5. Secondary metrics (descriptive, no decision rule)

Best-F1 with its precision and recall, Cohen's d, map-level AUC-ROC, 1-cell-dilated AUC-PR, **AUC-PR per bin** (bin 0 and bin 1 separately),
and the hypothesis-specific **false-alarm rate on quiet bins of active runs at 50% and 80% recall of active maps** (`diagnose_sample_mix.map_level`).
A's values are in `hourly_eval.json` (e.g. 2016: false-alarm rate 0.193 at 50% recall and 0.609 at 80%). That rate has a
large seed spread, so only a large, consistent drop in >= 2 of 3 folds is read as support for the timing hypothesis.

## 6. Predictions, with my honest priors (fixed before results)

1. **AUC-PR:** most likely a modest gain, uncertain. P(no detectable effect) 45%, P(helps) 40%, P(hurts) 15%; central guess pooled E about +6%.
   The snippet's +10.6% on one cell at 10 epochs is weak evidence, and I know that cell, so I do not take it at face value.
2. **Confirmed:** given "helps", P(confirmed) 35%. The paired interval is wide with 3 seeds.
3. **Timing:** the quiet-bins-of-active-runs false-alarm rate at 50% recall falls by >= 10% (relative) in >= 2 of 3 folds: 50%.
4. **Where the gain sits:** if there is one, it is larger for bin 1 than for bin 0 (bin 1 sees all 8 hours; bin 0 sees the same 4 hours as A): 60%.
5. **Era dependence:** E is lowest in 2024: 40% (no particular reason; the prior is near chance).
6. **Stability:** no run diverges (no NaN, final val/train loss ratio between 0.4 and 1.3 in all nine runs): 85%.
7. **Calibration shape:** best-F1 thresholds differ from A's; I do not predict direction.

## 7. Checks before trusting results

- Each of the nine runs writes `results.json` with its run-list hash; all must equal the matched A hash for that fold, and the test-sample order
  must equal A's cached order.
- A reproduction check on A's cached scores, as in section 3, before any ConvLSTM score is read.
- Any deviation (re-run for a crash, recipe change, bug fix) is written in the results doc with its reason, not applied silently.

## 8. What happens after

Report every number as measured, including nulls and missed predictions, in `docs/convlstm_results.md`, and update `CLAUDE.md`. Production serving,
live hourly pulls, arm B features, input normalization and any ConvLSTM tuning are separate decisions. If the result is "helps," the next step is
a 12-fold or 6-seed confirmation, not deployment.
