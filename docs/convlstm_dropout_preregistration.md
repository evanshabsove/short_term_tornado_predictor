# ConvLSTM-U-Net with dropout (C3 + dropout) vs. arm A: pre-registration

Written 2026-11-03, **before any dropout model is trained and before any dropout score exists.** Two stages; Stage 1 selects a dropout rate on data outside the final comparison, Stage 2 is the confirmatory study.
Motivation: the repaired C3 (`docs/convlstm_c3_results.md`) tied arm A (pooled E -1.5%) but overfits: final val/train loss 2.3-4.2 in all nine runs (arm A ~0.7-0.8), every single model 24-31% below A's mean single model, ensemble gain over single +52-63%
(A +15%). The user's prior experience (a sibling project's MADIS model) is that dropout helped there; that is an expectation, not evidence for this model.

**What I had seen beforehand, and only that:** everything in `docs/convlstm_c3_results.md` and `docs/convlstm_diagnostic_ladder.md` (C3 and first-attempt scores on folds 2016/2020/2024, seeds 10-12; the fold-2020 ladder at 10 epochs). No score of any model with dropout, and no
score on fold 2018 for any ConvLSTM, exists yet.

## 1. Intervention (the only change)

`TornadoConvLSTMUNet(..., norm_bottleneck=True, pooled_skips=True, dropout=p)`: channel-wise `Dropout2d(p)` on the LSTM's input (normalized bottleneck features, independent mask per hour) and on the hidden state fed to the decoder at each stop. Off in eval mode, so the
EMA models and all scoring are unaffected; no parameters. Everything else is the C3 recipe: grad clip 1.0, lr 1e-4, focal alpha 0.25 gamma 2.0, `quiet_keep_fraction` 0.2, batch 1 run, 50 epochs, EMA 0.9999.
**Disclosed limits:** dropout is applied only around the LSTM; if the memorization lives in the encoder or decoder this may not reach it (the result would then say so). Only this one placement and one family of rates is tried; weight decay, early stopping and
other regularizers are not.

## 2. Stage 1: choose the rate (fold 2018, seeds 20 and 21 only)

Neither fold 2018 nor seeds 20-21 appear in Stage 2 or in any prior ConvLSTM or arm-A comparison, so the rate selection cannot leak into the confirmatory test. Runs: p in {0 (control, = C3), 0.1, 0.3, 0.5} x seeds {20, 21}, fold 2018, 50 epochs = 8 runs.
Metric: EMA 0.9999 held-out AUC-PR, mean of the two seeds, plus the final val/train loss ratio.
**Rule (fixed now):** among p in {0.1, 0.3, 0.5}, take the rate with the highest mean AUC-PR; go to Stage 2 with it **only if** its mean AUC-PR >= the control's and its mean val/train loss ratio is below the control's. Otherwise stop: report that dropout did not help on the
selection fold, and run nothing more. (Two seeds at ~19% single-seed spread give a mean good to ~13%, so this selection is partly noise; it is deliberately lax because Stage 2 is the real test.) Stage 1 is not scored against arm A (no A ensemble exists for 2018).

## 3. Stage 2: confirmatory study

C3 + the selected dropout, folds 2016 / 2020 / 2024, seeds 10-12, 50 epochs, EMA 0.9999: nine runs, same data, splits and seeds as the C3 study. Comparison, rules and thresholds are **unchanged from `docs/convlstm_c3_preregistration.md`**:
primary metric = 3-member-ensemble AUC-PR; `E_fold` = ensemble / A's 3-member ensemble - 1; pooled E = mean of the three; paired day-bootstrap 95% interval; "helps" = pooled E >= +8% and E > 0 in >= 2 folds; "hurts" = pooled E <= -8% and E < 0 in >= 2 folds; otherwise
"no detectable effect"; "confirmed" only if the interval excludes 0 in >= 2 folds. No extension rule: nothing triggers extra seeds, folds, epochs or rates. A crash is re-run with the same seed; divergence or NaN is reported as a result. Minimum detectable effect ~12% with 3 seeds.
The evaluator records this file's sha256. Reported alongside, descriptively: the same comparison against the no-dropout C3 ensembles (same seeds), the first attempt's -19.7%, a 2016/2024-only pooled E, and the secondary metrics of `docs/convlstm_c3_results.md`
(timing false-alarm rate, map AUC-ROC, per-bin AUC-PR, best-F1, dilated AUC-PR, Cohen's d), the val/train loss ratios, gate saturation, and the single-model vs. ensemble gap.
Selection caveat for Stage 2: the rate was chosen on 2018 / seeds 20-21 only, but the *idea* (dropout on the LSTM) was chosen after seeing C3's overfitting on the study folds. A "helps" would carry that caveat.

## 4. Predictions (my honest priors, fixed before results)

Stage 1:
1. The best rate lowers the mean val/train loss ratio below 1.5 (control ~3): 50%.
2. The best rate beats the control's mean AUC-PR by >= 10%: 40%. Stage 2 proceeds (rule in section 2): 65%.
3. Best rate is 0.1 / 0.3 / 0.5: 30% / 40% / 30%.

Stage 2 (if run):
4. Verdict vs. A: helps 25% / no detectable effect 60% / hurts 15%; central guess pooled E +3%.
5. Pooled E beats C3's -1.5%: 60%. Dropout ensemble beats the no-dropout C3 ensemble in >= 2 of 3 folds: 60%.
6. Mean single-model AUC-PR gap to A's mean single model shrinks from -24...-31% to better than -15% in >= 2 of 3 folds: 50%.
7. Ensemble gain over the mean single seed falls below +35% in >= 2 of 3 folds (C3: +52-63%; A: +15%): 55%.
8. Val/train loss ratio within 0.4-1.3 in all nine runs: 15% (ratio below C3's 2.3-4.2 in all nine: 70%).
9. Timing metric (false-alarm rate on quiet bins of active runs at 50% recall) at least 10% below A's in >= 2 of 3 folds: 70%.
10. No NaN, no gate saturation in all nine runs: 90%.

## 5. After

Report every number as measured, including nulls and missed predictions, in `docs/convlstm_dropout_results.md`; update `CLAUDE.md`. If Stage 2 says "helps", the next step is a 12-fold or fresh-seed confirmation, not deployment.
