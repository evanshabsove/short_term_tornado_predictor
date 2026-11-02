# Repaired ConvLSTM-U-Net (C3, lr 1e-4) vs. arm A: pre-registration

Written 2026-11-01, **before any full-study score of this arm exists.** Second attempt at the question in `docs/convlstm_preregistration.md`
(same comparator, same rules, same thresholds). The first attempt's verdict, **hurts, pooled E -19.7%** (3/3 folds negative), stands and is reported
alongside this one; this study does not replace it. Diagnosis: `docs/convlstm_diagnostic_ladder.md`, `docs/convlstm_repair_plan.md`.
The user waived a review pause before launch.

## What I had seen beforehand
- First-attempt full-study results (all nine runs, evaluator output) and its saturation diagnosis.
- Fold 2020 only, 10 epochs, single-fold snippets: C3 at lr 1e-3 failed the gate (train-subset AUC-PR 0.0443 vs. control 0.0464); the ladder then found lr 1e-3 too
  aggressive. **C3 at lr 1e-4: held-out AUC-PR 0.0687 (seed 10) and 0.0817 (seed 11), plain weights; EMA 0.0674 / 0.0783**, gate stats fine (pre-activation std ~1.2, <1% beyond |4|).
  Matched U-Net (arm A's model on whole-bin data) at lr 1e-4: 0.0485 / 0.0612 plain, 0.0520 / 0.0455 EMA; at lr 1e-3, 0.0518 / 0.0463.
  **C3 without recurrence at lr 1e-4 (one seed): 0.0661 / 0.0688, train-subset AUC-PR equal to the recurrent run's.** So I already know fold 2020 and seeds 10 and 11 look
  good for this arm at 10 epochs; these cells are part of the full study, which retrains all nine from scratch at 50 epochs.
- A's scores are the comparison (3-member ensemble AUC-PR 0.0533 / 0.0689 / 0.0844 for 2016 / 2020 / 2024).
- Not seen: any 2016 or 2024 score, any 50-epoch score, any ensemble score of this arm.

## Arm and recipe
`TornadoConvLSTMUNet(base_channels=16, hidden_channels=64, norm_bottleneck=True, pooled_skips=True)`, grad clip norm 1.0, **lr 1e-4** (A: 1e-3), batch 1 run, focal loss alpha 0.25
gamma 2.0, `quiet_keep_fraction` 0.2 (run-level), 50 epochs, EMA 0.9999 (EMA weights are the scored model), folds 2016 / 2020 / 2024, seeds 10-12, same runs, buffer and
test-sample order as before. Disclosed differences from A (all of the first study's five, plus): lr, GroupNorm, pooled skips, clipping; parameter count ~651k vs. A's 451k.
Whether recurrence matters is **not** isolated by this study; the no-recurrence snippet suggests it may not.

## Comparison and rules (unchanged from the first pre-registration)
Primary metric: 3-member-ensemble AUC-PR per fold; E_fold = ensemble / A's 3-member ensemble - 1; pooled E = mean of the three.
"Helps": pooled E >= +8% and E > 0 in >= 2 of 3 folds. "Hurts": pooled E <= -8% and E < 0 in >= 2 folds. Otherwise "no detectable effect".
"Confirmed" only if the paired day-bootstrap interval (1,000 resamples) excludes 0 in >= 2 folds, else "weak: inside noise".
No extension rule. A crash is re-run with the same seed; NaN or divergence is a result, not retried. Minimum detectable effect ~12% with 3 seeds, above the 8% threshold.
One tuned choice (lr 1e-4) was selected on fold 2020 seeds 10-11, which are inside this study: that is selection on the test fold, so a "helps" is read with that caveat
(a 2016/2024-only pooled E is reported as a descriptive check, no rule attached). Secondary metrics as before (best-F1, Cohen's d, map AUC-ROC, dilated AUC-PR,
per-bin AUC-PR, quiet-bins-of-active-runs false-alarm rate at 50% / 80% recall). Also reported: saturation check (gate pre-activation std and fraction beyond |4|) on the nine final models.

## Predictions (fixed before results)
1. P(helps) 40%, P(no detectable effect) 45%, P(hurts) 15%; central guess pooled E about +4%. (The snippet's single-seed margin over the matched U-Net was ~+37%, but the
   U-Net control is not A's 50-epoch ensemble and the snippet is 10 epochs.)
2. Given "helps", P(confirmed) 30%.
3. Pooled E better than the first attempt's -19.7% in all three folds: 90%.
4. Fold 2020 has the highest E (it is the fold the lr was chosen on): 50%.
5. No NaN and final val/train loss ratio between 0.4 and 1.3 in all nine runs: 85%. Gate saturation (std < 10, < 20% beyond |4|) in all nine: 85%.
6. Timing metric (quiet-bin false alarms at 50% recall) falls by >= 10% vs. A in >= 2 of 3 folds: 50%.

## After
Report every number including misses in `docs/convlstm_c3_results.md`, update CLAUDE.md. Next step if "helps": 12-fold or 6-seed confirmation plus a no-recurrence ablation at 50 epochs, not deployment.
