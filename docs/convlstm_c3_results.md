# Repaired ConvLSTM (C3, lr 1e-4) vs. arm A: results

Pre-registration: `docs/convlstm_c3_preregistration.md` (sha256 `8f26facf…`, identical to the hash recorded at launch; never edited). Numbers:
`models/convlstm/full_study_c3/convlstm_eval.json` (evaluator `scripts/11_convlstm/convlstm_eval.py --study-dir models/convlstm/full_study_c3 --prereg docs/convlstm_c3_preregistration.md`, which
first reproduced A's ensemble AUC-PR and checked that all nine runs used A's training-run hash and test order). Gate saturation: `models/convlstm/full_study_c3/gate_saturation.json`.
Recipe: `TornadoConvLSTMUNet(base_channels=16, hidden_channels=64, norm_bottleneck=True, pooled_skips=True)`, grad clip 1.0, lr 1e-4, otherwise A's recipe; folds 2016/2020/2024, seeds 10-12, 50 epochs, EMA 0.9999.
No deviations: all nine runs completed on the first attempt with the launch recipe.

## Verdict by the pre-registered rule: **no detectable effect**

Pooled E = **-1.5%**, positive in 1 of 3 folds; the paired day-bootstrap interval includes 0 in all three. Neither "helps" (needs pooled >= +8%) nor "hurts" (pooled <= -8%) is met.
With 3 seeds the minimum detectable effect is ~12%, so this is a tie inside the noise, not proof of equivalence.

| fold | C3 3-member ensemble AUC-PR | A 3-member ensemble | E | paired 95% interval (difference) | first attempt E |
|---|---|---|---|---|---|
| 2016 | 0.0527 | 0.0533 | -1.1% | [-0.0113, +0.0106] | -29.2% |
| 2020 | 0.0650 | 0.0689 | -5.7% | [-0.0195, +0.0111] | -9.7% |
| 2024 | 0.0863 | 0.0844 | +2.3% | [-0.0138, +0.0199] | -20.3% |
| pooled | | | **-1.5%** | | **-19.7%** |

The first attempt's "hurts, -19.7%" verdict (`docs/convlstm_results.md`) stands as the verdict for that recipe; the repair removed the damage but did not produce a gain.
Descriptive, not a pre-registered check: pooled E over 2016 and 2024 only (the folds not used to pick lr 1e-4) is **+0.6%**; fold 2020, where the lr was chosen, is the worst fold, so the selection did not flatter the result.

## Single models vs. ensembles

Single models (EMA 0.9999), C3 vs. A:

| fold | C3 seeds 10 / 11 / 12 | A seeds 10 / 11 / 12 | mean C3 vs. mean A |
|---|---|---|---|
| 2016 | 0.0285 / 0.0341 / 0.0322 | 0.0447 / 0.0469 / 0.0480 | -30.6% |
| 2020 | 0.0374 / 0.0410 / 0.0503 | 0.0664 / 0.0533 / 0.0601 | -28.4% |
| 2024 | 0.0546 / 0.0552 / 0.0587 | 0.0595 / 0.0796 / 0.0821 | -23.8% |

Every C3 single model is below A's mean single model; in 2016 and 2020 nearly every C3 seed is below every A seed. The ensemble gain is far larger for C3 than for A:
3-member ensemble over mean single seed is +67 / +51 / +54% for C3 vs. +15 / +15 / +15% for A. The ensemble tie therefore comes from C3's members being individually weaker but
much more different from each other (higher model variance), not from better members. (Post hoc, descriptive.) Read with the overfitting result below.

## Training behaviour (pre-registered prediction 5: missed)

- No NaNs, and **no gate saturation**: LSTM gate pre-activation std 1.41-1.75 (first attempt: 3e3-9e5; untrained about 0.5) with 1.8-3.7% of values beyond |4| (first attempt 85-100%). The repair fixed the defect it targeted.
- **But the final val/train loss ratio is 2.34-4.25 in all nine runs**, far outside the pre-registered 0.4-1.3 band (2016: 2.94 / 2.88 / 2.34; 2020: 3.20 / 2.83 / 2.61; 2024: 4.25 / 3.59 / 3.83). Final train loss is
  4.1-4.8e-5, about a third of A's U-Net (~1.5e-4, ratio 0.72-0.83 in earlier records). The repaired model fits the training years much better than the held-out year: **overfitting**,
  the opposite of the first attempt's underfitting. Consistent with the ladder, where the ConvLSTM memorized 20 runs to train AUC-PR ~1.0 and a train-subset AUC-PR of 0.076-0.078 at lr 1e-4.
  Per the pre-registration this is reported, not retuned.

## Secondary metrics (descriptive, no decision rule)

- **Timing metric improved a lot.** False-alarm rate on quiet bins of active runs at 50% recall: 0.193 -> 0.106 (2016), 0.216 -> 0.080 (2020), 0.231 -> 0.093 (2024), i.e. -45 / -63 / -60%; at 80% recall
  0.609 -> 0.360, 0.481 -> 0.265, 0.516 -> 0.308 (-41 / -45 / -40%). Map-level AUC-ROC higher in all folds (0.851 vs. 0.831, 0.858 vs. 0.840, 0.869 vs. 0.830). The first attempt moved the same metric in the same direction but less.
- **Cell-level ranking is worse on some metrics.** Cell AUC-ROC is lower in all folds (0.980 vs. 0.991, 0.989 vs. 0.995, 0.990 vs. 0.993); Cohen's d is about half of A's (1.40 / 1.51 / 1.54 vs. 2.90 / 3.19 / 2.96); 1-cell-dilated AUC-PR is -0.4% / -10% / -1.4%.
  Best-F1 is mixed (0.137 vs. 0.128, 0.148 vs. 0.156, 0.173 vs. 0.168); precision at best-F1 is higher in all three folds, recall mixed (0.202 vs. 0.193, 0.190 vs. 0.229, 0.200 vs. 0.259).
- **Per bin:** bin 0 AUC-PR is below A in all folds (-5.7 / -7.8 / -4.8%); bin 1 is -25.9 / -18.4 / +6.5%. Bin 1 (which sees all eight hours) is not where C3 gains.

## Predictions, scored (`docs/convlstm_c3_preregistration.md`)

1. **Helps 40% / no detectable effect 45% / hurts 15%, central guess +4%:** outcome is the 45% category; the central guess was ~5.5 points too high (observed -1.5%). Not a miss on category.
2. **Confirmed given helps (30%):** not applicable, no "helps".
3. **Better than -19.7% in all three folds (90%): met** (-1.1 / -5.7 / +2.3%).
4. **Fold 2020 has the highest E (50%): not met.** It has the lowest; 2024 is highest.
5. **No NaN, loss ratio 0.4-1.3 and no gate saturation in all nine runs (85%): not met.** No NaN and no saturation, but the loss ratio is 2.3-4.2 in all nine.
6. **Timing metric falls by >= 10% in >= 2 of 3 folds (50%): met** (3 of 3, -45% to -63% at 50% recall).

## What this does and does not show

- A ConvLSTM that sees the eight hourly steps, once its saturation defect and learning rate are repaired, **ties** arm A's hand-built time pooling on the primary metric (3 seeds, ~12% resolvable). It does not beat it.
- It does **not** show that recurrence matters: the earlier no-recurrence snippet (fold 2020, one seed) scored the same as the recurrent model, and the study varies GroupNorm, pooled skips, clipping and lr together with the architecture. The repair's gain over the first attempt is not attributable to the LSTM.
- The false-alarm-on-quiet-bins improvement is large and consistent but arrives with worse cell-level separation and clear overfitting; it is supportive of the timing hypothesis, not a win on the primary metric.
- Caveats: lr 1e-4 was chosen on fold 2020 seeds 10-11, which are in this study (see the 2016/2024-only figure above); one recipe, not tuned for the ConvLSTM (no regularization, early stopping or fewer epochs was tried); parameter count 1.44x A's.
- The evaluator printed numpy `divide by zero / overflow / invalid value in matmul` warnings from the bootstrap (`ap_from_weights`). I checked them: on random data of the same shape the `matmul` result equals `einsum` to 1e-15 and is finite, so these are spurious
  floating-point flags from numpy 2.0.2's matmul on this machine, not wrong values; the evaluator also reproduced A's ensemble AUC-PR before reading any C3 score. I did not re-run the bootstrap with `einsum`.

## What happens next (not decided)

Per the pre-registration there is no extension rule. Open options, each needing its own plan: a regularized or shorter-training C3 (the train/val gap suggests the model is overfitting, and an earlier-stopped model might beat A), a no-recurrence ablation at 50 epochs with 3 seeds to isolate recurrence,
a hybrid with A's pooled fields feeding the decoder, or stopping here. Not deployed; production serving is untouched.
