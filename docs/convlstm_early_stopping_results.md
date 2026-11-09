# Early-stopped ConvLSTM (C3) vs. early-stopped arm A: results

Pre-registration: `docs/convlstm_early_stopping_preregistration.md` (sha256 `a84d40a3...`, identical at launch and at scoring; never edited). Numbers:
`models/convlstm/early_stop/c3/convlstm_eval.json` (evaluator `scripts/11_convlstm/convlstm_eval.py --a-dir models/convlstm/early_stop/arm_a --study-dir models/convlstm/early_stop/c3 --prereg ...`;
it asserts that both arms' training and validation run lists equal the expected mod-20 split). Per-run stopping history is in each `results.json` under `early_stopping`.
Design: both arms stop on validation AUC-PR of the EMA 0.9999 weights (validation = mod-20 date split carved from the training years, never the test year), patience 8, at least 10 epochs, cap 50,
best-epoch weights restored; test year scored once. Folds 2016/2020/2024, seeds 10-12. C3 = the repaired recipe (GroupNorm, pooled skips, clip 1.0, lr 1e-4), no dropout.
Both arms train on ~78% of the earlier training runs (the rest are validation or buffer), so **E here is against the NEW early-stopped A, not the old 50-epoch A**.

## Verdict by the pre-registered rule: **helps, confirmed**

Pooled E = **+29.4%**, positive in 3 of 3 folds; the paired day-bootstrap interval excludes 0 in 2 (2016, 2024).

| fold | C3 3-member ensemble AUC-PR | A 3-member ensemble (early-stopped) | E | paired 95% interval (difference) | earlier 50-epoch C3 E (vs old A) |
|---|---|---|---|---|---|
| 2016 | 0.0706 | 0.0520 | +35.8% | [+0.0067, +0.0292] | -1.1% |
| 2020 | 0.0847 | 0.0802 | +5.7% | [-0.0035, +0.0157] | -5.7% |
| 2024 | 0.1150 | 0.0784 | +46.7% | [+0.0234, +0.0489] | +2.3% |
| pooled | | | **+29.4%** | | -1.5% |

Fold 2020 is inside the noise (and below the ~12% resolvable with 3 seeds); the pooled number is carried by 2016 and 2024. Against the OLD 50-epoch A ensembles (0.0533 / 0.0689 / 0.0844) the early-stopped C3 is +32 / +23 / +36%.
The 2024 value (0.1150) is the highest ensemble AUC-PR this project has recorded on any fold; I checked that it is not a pipeline artifact as far as I can see: the validation set comes from training years only, the run-list hashes were asserted, and
A's early-stopped singles on the same fold (0.069-0.077) match the old A singles (0.060-0.082).

## Single models

| fold | C3 singles (seeds 10 / 11 / 12) | A singles | mean C3 vs. mean A |
|---|---|---|---|
| 2016 | 0.0588 / 0.0587 / 0.0678 | 0.0451 / 0.0476 / 0.0442 | +35.4% |
| 2020 | 0.0799 / 0.0814 / 0.0735 | 0.0656 / 0.0705 / 0.0750 | +11.3% |
| 2024 | 0.1153 / 0.1006 / 0.1030 | 0.0692 / 0.0747 / 0.0774 | +44.1% |

Every C3 single beats every A single in 2016 and 2024; in 2020 the ranges overlap. Early-stopped C3 singles beat the same seed and fold's 50-epoch C3 single in 9 of 9 runs (e.g. fold 2020 seed 10: 0.0374 -> 0.0799).
The ensemble gain over the mean single seed is now similar for both arms (C3 +14 / +8 / +8%, A +14 / +14 / +6%): the earlier "weak but diverse members" pattern (C3 +51-67%) was the overfit tail; stopped models are individually strong.

## Stopping behaviour

- **C3:** stopped early in 9 of 9 runs, best epochs 17-30 (median 20), stopped at epochs 25-38. Loss ratio val/train at the stopping epoch 0.77-1.27 (vs. 2.3-4.2 at epoch 50 in the earlier study).
- **Arm A:** stopped early in 8 of 9 runs (seed 10 on fold 2020 ran all 50), best epochs 17-46 (median 29). A's validation AUC-PR is nearly flat, so its best epoch is noisy; the rule is doing little for A except choosing among near-equal epochs.
- Early-stopped A singles vs. the old 50-epoch A singles: -2.0 / +17.4 / +0.1% per fold (mean +5.2%), so stopping on 78% of the data did not hurt A.
- No NaN. Compute was ~19 h of C3 training alone; wall clock from launch to completion was ~25 h (includes a ~4 h machine sleep and slow, contended epochs).

## Secondary metrics (descriptive, no decision rule), 3-member ensembles, C3 vs. A

- **Timing metric:** false-alarm rate on quiet bins of active runs at 50% recall 0.081 vs 0.236 (-66%), 0.074 vs 0.161 (-54%), 0.071 vs 0.247 (-71%); at 80% recall -48 / -43 / -40%. Map AUC-ROC higher in all folds (+6.6 / +3.5 / +6.4%).
- **Cell-level:** best-F1 higher in all folds (+27.7 / +12.2 / +27.4%); 1-cell-dilated AUC-PR +30.5 / +8.9 / +35.6%; per-bin AUC-PR higher in 5 of 6 (bin 1 in 2020 -9.1%).
  Precision at best-F1 higher in all folds; recall higher in 2016 and 2020, lower in 2024 (-16%). Cell AUC-ROC is the same (+0.0-0.2%).
- **Cohen's d** is lower for C3 in two folds (-7.4%, -7.4%) and higher in 2024 (+3.7%): the old "half of A's" gap is gone.

## Predictions, scored (`docs/convlstm_early_stopping_preregistration.md`)

1. **Helps 25% / no detectable effect 60% / hurts 15%; central guess +4%:** outcome is the 25% category; the central guess was off by ~25 points. My priors came from the 50-epoch comparisons and from the one-fold curve replay (+9% at best epochs), which understated the effect on 2016 and 2024.
2. **C3 stops early with stopped epoch <= 40 in >= 7 of 9 (80%): met** (9 of 9, stopped 25-38).
3. **A stops early in >= 6 of 9 (65%): met** (8 of 9).
4. **Median best epoch of C3 below A's (75%): met** (20 vs. 29).
5. **Early-stopped C3 single beats the old 50-epoch single in >= 7 of 9 (80%): met** (9 of 9).
6. **A's early-stopped singles within +/-10% of the old A singles on average (60%): met** (+5.2% mean, though one fold is +17.4%).
7. **C3 timing metric >= 10% below A in >= 2 of 3 folds (70%): met** (3 of 3, -54 to -71%).
8. **No NaN and within ~9 h wall clock (75%): not met.** No NaN, but ~25 h wall clock (~19 h of C3 compute alone). My 9 h estimate assumed ~80 s epochs and uncontended cores.

## What this does and does not show

- With a fair stopping rule applied to both arms, the repaired ConvLSTM-U-Net **beats** arm A's hand-built time pooling on the primary metric: pooled +29.4%, strongest on 2016 and 2024, not distinguishable on 2020.
  The earlier "no detectable effect" for C3 (and for C3 + dropout) was largely an artifact of comparing a model that peaked at ~epoch 20 and then overfit against an arm that does not overfit.
- It does **not** show that recurrence is the cause. The study varies the whole C3 recipe (GroupNorm, pooled skips, clipping, lr, ConvLSTM); the earlier no-recurrence snippet matched the recurrent model at 10 epochs. A 3-seed no-recurrence ablation under the same stopping rule is the way to find out.
- Caveats: three folds and three seeds (about 12% resolvable); fold 2020's interval includes 0; the stopping rule and patience were fixed in advance and not tuned, but C3's lr 1e-4 was chosen earlier on fold 2020 seeds 10-11; early stopping on validation AUC-PR from ~1,300 bin-samples is itself noisy (A's best epoch ranges 17-46);
  the comparison is on the HRRR dense-grid/hourly store only; nothing here touches the serving bundle, which still uses single-snapshot features.
- Not done: no-recurrence ablation, hybrid with A's pooled fields, other folds (12-fold confirmation), whether the gain survives the deployment-like 20Z stratum, and a retrained serving bundle (whole-bin/hourly features need fxx 0-7 live pulls).

## What happens next (not decided)

Per the pre-registration's "helps" branch: the 3-seed no-recurrence ablation at the stopped epoch, then a hybrid with A's pooled fields. Another option is a 12-fold confirmation of early-stopped C3 vs. A, since three folds is a small sample for a result this size.
