# Pre-registration: early-stopped ConvLSTM vs. early-stopped arm A

Written 2026-11-07, before any run of this study. The launch script records this file's sha256 in `models/convlstm/early_stop/prereg_sha_at_launch.txt`. Nothing below is changed after launch; a change of rule or settings needs a new pre-registration.

## Question
The repaired ConvLSTM (C3) tied arm A at a fixed 50 epochs (pooled E -1.5%; with dropout +1.9%). A one-fold, one-seed rerun with per-epoch held-out metrics (`docs/convlstm_per_epoch_plan.md`) showed C3's held-out AUC-PR peaking around epoch 15-20 and falling to ~0.47 of its peak by epoch 50, while arm A held ~0.92 of its peak. So the 50-epoch comparison may have favored A.
**If each arm is allowed to stop at its own best epoch, chosen on data that is not the test year, does the ConvLSTM beat arm A?**

## Design (identical for both arms)
- **Early stopping rule.** After every epoch, score the **EMA 0.9999** weights on a validation set; the monitored metric is **validation AUC-PR** (pooled per-cell, the project's primary metric). The best epoch is the one with the highest value (a later epoch must be strictly higher to replace it). Stop when at least **10 epochs** are done and **8 epochs** have passed since the best epoch; hard cap **50 epochs**. At the end the raw weights and the EMA weights are **restored to the best epoch**, and the held-out test year is scored **once**, on the restored weights. Loss is not used (arm A's held-out loss is flat, so a loss rule is unsuitable).
- **Validation set.** Carved from the **training years only**, by the project's day-of-year mod-20 date rule (`split.split_run_bins`: whole UTC dates, ~15% of dates, spread over every season and year), with the usual 24 h buffer removing training samples within 24 h of a validation sample. The test year is never used for any stopping decision. Real sizes (bin-samples, fold 2016 / 2020 / 2024): train 7,028 / 6,972 / 6,892, validation 1,358 / 1,338 / 1,326, test 782 / 876 / 962. **Both arms therefore train on ~78% of the runs the earlier studies used** (the 15% validation plus ~7% buffer loss).
- **Arms.** A = `TornadoUNet(base_channels=16)` on the whole-bin features (the established arm A recipe, `cross_validate_year.py --early-stopping`). C3 = `TornadoConvLSTMUNet(base_channels=16, hidden_channels=64, norm_bottleneck=True, pooled_skips=True)`, grad clip 1.0, lr 1e-4, **no dropout** (dropout was not shown to help; this isolates the effect of stopping). Both: focal loss alpha 0.25 gamma 2.0, quiet_keep_fraction 0.2, EMA 0.9999, same training and validation run lists (the evaluator asserts the hashes), same test samples in the same order.
- **Folds and seeds.** Held-out years 2016 / 2020 / 2024, seeds 10-12, 3-member ensembles of the EMA models (best-epoch weights) per arm and fold.
- Code: opt-in `train_model(early_stop_fn=...)` (tests verify it leaves training up to the stopping epoch unchanged and restores the best epoch's weights), runner `scripts/11_convlstm/run_early_stop_study.sh`, evaluator `scripts/11_convlstm/convlstm_eval.py --a-dir`.

## Primary comparison and decision rule
Per fold, **E = AUC-PR(C3 early-stopped 3-member ensemble) / AUC-PR(arm A early-stopped 3-member ensemble) - 1** on the same test samples. The comparison is against the **new** early-stopped A, not the old 50-epoch A (the old A trained on 100% of the runs and did not stop). Rule, as in every study here: **helps** = pooled E >= +8% and E > 0 in >= 2 folds; **hurts** = pooled E <= -8% and E < 0 in >= 2 folds; otherwise **no detectable effect**. Paired day-bootstrap 95% intervals are reported; a "helps" is **confirmed** only if the paired interval excludes 0 in >= 2 folds. With 3 seeds the smallest resolvable effect is ~12%.

## Secondary, descriptive (no decision rule)
Stopping epochs and best epochs of all 18 runs; validation vs. test AUC-PR at the chosen epoch per run (selection optimism and year-to-year difference); each arm's early-stopped single and ensemble AUC-PR vs. its own old 50-epoch result (this mixes the effect of stopping with the ~22% smaller training set and cannot be separated); the timing metric (false-alarm rate on quiet bins of active runs at 50% and 80% recall), map AUC-ROC, cell AUC-ROC, Cohen's d, best-F1 with precision and recall; ensemble gain over the mean single seed. Note: `final_train_loss`/`final_val_loss`/`val_over_train_loss` in the run results refer to the last epoch trained (the stopping epoch, which is 8+ epochs after the restored best epoch), not to the restored weights.

## Known limits, stated now
- The patience (8), minimum epochs (10) and validation rule were chosen after seeing the fold-2020 seed-10 per-epoch curves, which were scored on the 2020 test year; that run reappears in this study (fold 2020, seed 10). The settings were not tuned beyond a replay of the rule on those curves, and fold 2020's result is reported separately; the pooled 2016/2024 effect is reported too.
- A validation set of ~670 runs gives a noisy AUC-PR (a day-block bootstrap moves a year's AUC-PR by ~16%), so a run may stop early or late for noise reasons. Patience 8 and the EMA smoothing are the only protection. Not tuned.
- Both arms lose ~22% of their training runs, so absolute AUC-PR is expected to be somewhat lower than in earlier studies; only the matched comparison is the verdict.
- One recipe per arm; the ConvLSTM's other overfitting remedies (dropout, smaller model) are not combined with stopping here.
- Three seeds and three folds; the study cannot show equivalence.

## Predictions (my honest priors, written before any result)
1. Verdict: **helps 25% / no detectable effect 60% / hurts 15%**; central guess for pooled E +4%.
2. C3 stops early (stopped_early true and stopped epoch <= 40) in at least 7 of its 9 runs: 80%.
3. Arm A stops early (before epoch 50) in at least 6 of its 9 runs: 65%.
4. The median best epoch of C3's nine runs is lower than arm A's: 75%.
5. C3's early-stopped single-model test AUC-PR beats the same seed and fold's old 50-epoch C3 single in at least 7 of 9 runs: 80%.
6. Arm A's early-stopped single models are within +/-10% of the old A singles on average (stopping gain offset by less data): 60%.
7. C3's timing metric (false alarms on quiet bins of active runs at 50% recall) stays at least 10% below A's in at least 2 of 3 folds: 70%.
8. All 18 runs finish without NaN, within ~9 h wall clock: 75%.

## What each outcome means
- **Helps (confirmed or not):** the ConvLSTM was handicapped by fixed-length training; the next study is the 3-seed no-recurrence ablation at the stopped epoch, then a hybrid with A's pooled fields.
- **No detectable effect:** the ConvLSTM line is closed; arm A stays the feature design; the stopping rule is kept as the default for future studies.
- **Hurts:** close the ConvLSTM line.
