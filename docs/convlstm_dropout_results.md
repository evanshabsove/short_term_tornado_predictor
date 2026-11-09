# Repaired ConvLSTM + dropout (Stage 1 selection, Stage 2 confirmation): results

Pre-registration: `docs/convlstm_dropout_preregistration.md` (sha256 `fc7dbb48...`, identical to the hash recorded at both launches; never edited). Numbers: `models/convlstm/dropout_select/selection.json` (Stage 1),
`models/convlstm/full_study_c3_dropout/convlstm_eval.json` and `gate_saturation.json` (Stage 2; evaluator `scripts/11_convlstm/convlstm_eval.py --study-dir models/convlstm/full_study_c3_dropout --prereg docs/convlstm_dropout_preregistration.md`).
Summary notebook: `notebooks/convlstm_study.ipynb`.
Recipe: C3 (`norm_bottleneck`, `pooled_skips`, grad clip 1.0, lr 1e-4) + `Dropout2d(0.1)` on the LSTM input and on the hidden state fed to the decoder; folds 2016/2020/2024, seeds 10-12, 50 epochs, EMA 0.9999.
Deviations: the first Stage 1 launch was killed by a terminal reset (no results saved) and re-run from scratch, detached; the evaluator's glob needed a one-line fix (it also matched a text file). Neither touches any result. All nine Stage 2 runs completed on the first attempt.

## Stage 1: rate selection (fold 2018, seeds 20/21, not used elsewhere)

| dropout | EMA AUC-PR (seeds 20, 21) | mean | val/train loss ratio | train loss |
|---|---|---|---|---|
| 0.0 (control) | 0.0422 / 0.0414 | 0.0418 | 2.45 | 4.7e-5 |
| **0.1** | 0.0425 / 0.0422 | 0.0424 | 2.13 | 5.2e-5 |
| 0.3 | 0.0359 / 0.0391 | 0.0375 | 1.79 | 5.5e-5 |
| 0.5 | 0.0342 / 0.0424 | 0.0383 | 1.42 | 5.9e-5 |

The fixed rule picked p=0.1 (+1.4% AUC-PR over the control, ratio 2.13 < 2.45) and said to proceed. That gain is far inside the ~19% seed noise: the rule passed on a tie, not on evidence of benefit.
Dropout lowered the overfitting ratio monotonically with the rate but did not raise AUC-PR at any rate.

## Stage 2 verdict by the pre-registered rule: **no detectable effect**

Pooled E = **+1.9%** (positive in 3 of 3 folds); paired day-bootstrap interval includes 0 in all three. "Helps" needed pooled >= +8%. Minimum detectable effect with 3 seeds is ~12%, so this is a tie inside the noise, not proof of equivalence.

| fold | dropout ensemble AUC-PR | arm A ensemble | E | paired 95% interval (difference) | C3 (no dropout) E | dropout vs. C3 ensemble |
|---|---|---|---|---|---|---|
| 2016 | 0.0542 | 0.0533 | +1.7% | [-0.0092, +0.0121] | -1.1% | +2.8% |
| 2020 | 0.0697 | 0.0689 | +1.2% | [-0.0118, +0.0146] | -5.7% | +7.4% |
| 2024 | 0.0866 | 0.0844 | +2.7% | [-0.0137, +0.0188] | +2.3% | +0.4% |
| pooled | | | **+1.9%** | | -1.5% | |

First attempt (no repair): -19.7%. Pooled over 2016/2024 only: +2.2% (C3: +0.6%). The dropout-vs-C3 column is descriptive (the same seeds and test samples, but I did not run a paired bootstrap between the two ConvLSTM arms); a 3.4-point pooled difference is well under the ~12% resolution, so it should not be read as dropout helping.

## What dropout did and did not change

- **Overfitting: reduced, not cured.** Final val/train loss ratio 1.87-4.02 (mean 2.68) vs. C3's 2.34-4.25 (mean 3.16); lower than C3 in all nine paired runs. Train loss 4.8e-5 (C3 ~4.5e-5). Still far outside the 0.4-1.3 healthy band that arm A sits in.
- **Single models stay weak:** mean single-model AUC-PR vs. A's mean single -26.5 / -23.7 / -22.3% (C3: -32 / -28 / -24%). A small improvement, nowhere near the "better than -15%" the prediction needed.
- **Ensemble gain over the mean single seed:** +59 / +52 / +51% (C3 +67 / +51 / +54%; A +15%). Members remain individually weak and very diverse; the ensemble tie is again diversity, not better members.
- **Timing metric (false alarms on quiet bins of active runs, 50% recall):** 0.118 / 0.111 / 0.088 vs. A's 0.193 / 0.216 / 0.231, i.e. -39 / -49 / -62% (C3: -45 / -63 / -60%). Map AUC-ROC higher than A in all folds (0.855 / 0.851 / 0.868 vs. 0.831 / 0.840 / 0.830).
- **Same cell-level cost as C3:** cell AUC-ROC below A in all folds, Cohen's d 1.57 / 1.63 / 1.59 vs. A's 2.90 / 3.19 / 2.96; per-bin AUC-PR is mixed (bin 1 beats A only in 2024); precision at best-F1 is about equal to A's, recall mixed.
- **Training health:** no NaN; gate pre-activation std 1.45-1.75, 2.1-4.1% beyond |4| (no saturation).

## Predictions, scored (`docs/convlstm_dropout_preregistration.md` section 4)

Stage 1:
1. Best rate lowers the val/train ratio below 1.5 (50%): **missed.** The best rate by AUC-PR (0.1) gave 2.13; only p=0.5 got below 1.5, and it was not the best rate.
2. Best rate beats the control's AUC-PR by >= 10% (40%): **missed** (+1.4%). Stage 2 proceeds (65%): **met**, via the rule's tie-break.
3. Best rate 0.1 / 0.3 / 0.5 at 30 / 40 / 30%: **0.1** (my 30% outcome), though only by a tie.

Stage 2:
4. Verdict helps 25% / no detectable effect 60% / hurts 15%, central +3%: **no detectable effect** (my 60% category); observed +1.9% vs. central +3%, a good calibration on this one.
5. Pooled E beats C3's -1.5% (60%): **met** (+1.9%). Beats no-dropout C3 in >= 2 of 3 folds (60%): **met** (3 of 3, but within noise; see above).
6. Single-model gap to A better than -15% in >= 2 folds (50%): **missed** (-26.5 / -23.7 / -22.3%).
7. Ensemble gain over the mean single seed below +35% in >= 2 folds (55%): **missed** (+59 / +52 / +51%). (The pre-registration quotes C3's gain as +52-63%; the correct figure is +51-67%. The pre-registration was not edited; the threshold is unaffected.)
8. Ratio within 0.4-1.3 in all nine (15%): **missed** (1.87-4.02). Ratio below C3's in all nine (70%): **met**.
9. Timing metric >= 10% below A's in >= 2 of 3 folds (70%): **met** (3 of 3, -39% to -62%).
10. No NaN and no gate saturation in all nine (90%): **met.**

## What this does and does not show

- Dropout at 0.1 made the repaired ConvLSTM slightly less overfit but did not change what it can do: it still ties arm A on AUC-PR (pooled +1.9%, all intervals include 0) and still beats A on the timing metric and map-level AUC-ROC while giving up cell-level separation.
- It does not show that dropout is useless: 0.1 is a small rate, picked by a rule that could not distinguish rates (Stage 1 had two seeds on one fold). The one rate that brought the ratio near the healthy band, 0.5, scored 8% below the control in Stage 1 (not distinguishable from it) and was not tested further.
- Selection caveat: the *idea* of dropout was chosen after seeing C3's overfitting on the study folds; the rate was chosen on separate data. A "helps" would have carried that caveat; none was found.
- Not shown, as before: that recurrence matters (no 3-seed no-recurrence ablation exists), and nothing about deployment.
- Lesson recorded for future studies: per-epoch held-out metrics (AUC-PR/ROC, precision, recall) were not logged, so the early-stopping question for C3 can only be argued from the loss curves (validation loss turns up around epoch 25 while training loss keeps falling).

## What happens next (not decided)

No extension rule. Options, each needing its own plan: an early-stopped / per-epoch-metric rerun of C3 (cheap, one fold), a 3-seed no-recurrence ablation, a hybrid feeding A's pooled fields to the decoder, a higher dropout rate with a proper multi-seed selection, or closing the ConvLSTM line here.
