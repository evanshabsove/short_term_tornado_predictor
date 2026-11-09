# Tuned arm-A U-Net control vs. the early-stopped ConvLSTM (C3): results

Pre-registration: `docs/unet_control_preregistration.md` (sha256 `5f931a16...`, identical at launch of both stages and at scoring; never edited). Numbers: `models/unet_control/select/selection.json` (Stage 1),
`models/unet_control/tuned_a/convlstm_eval.json` (Stage 2; `scripts/11_convlstm/convlstm_eval.py --a-dir models/unet_control/tuned_a --study-dir models/convlstm/early_stop/c3 --prereg ... --out ...`, which asserts that both arms used the same
training/validation run lists). Runners: `scripts/11_convlstm/run_unet_control_select.sh`, `unet_control_select_eval.py`, `run_unet_control_study.sh`.
Question: the earlier +29.4% (`docs/convlstm_early_stopping_results.md`) compared C3 (lr 1e-4, clip 1.0, partly tuned on fold 2020) with arm A at its never-tuned recipe (lr 1e-3, no clip). How much survives once arm A is tuned?

## Stage 1: selection on fold 2018 (outside the study folds), seeds 20/21, early stopping as before

| setting | EMA test AUC-PR (seed 20 / 21) | mean | vs. U0 | best epoch | val/train loss |
|---|---|---|---|---|---|
| U0: lr 1e-3, no clip (existing recipe) | 0.0590 / 0.0481 | 0.0535 | n/a | 25 / 21 | 0.51 / 0.50 |
| **U1: lr 3e-4, clip 1.0** | 0.0615 / 0.0635 | **0.0625** | **+16.6%** | 27 / 24 | 0.76 / 0.74 |
| U2: lr 1e-4, clip 1.0 | 0.0617 / 0.0618 | 0.0617 | +15.2% | 24 / 23 | 0.85 / 0.81 |

Rule (fixed beforehand): a non-default setting is selected only if its mean is >= 5% above U0's. **U1 selected; Stage 2 run.** U1 and U2 are effectively tied; two seeds cannot separate them, so U1 is "a better recipe than U0", not a located optimum.
U0's two seeds differ by ~20%, U1's and U2's by ~3%. The old arm-A recipe was not overfitting (loss ratio ~0.5); the lower-lr settings sit at 0.75-0.85.

## Stage 2: U1 on folds 2016/2020/2024, seeds 10-12 (9 runs), scored vs. early-stopped C3

### Verdict by the pre-registered rule: **helps, confirmed**

Pooled E = **+25.8%**, positive in 3 of 3 folds, the paired day-bootstrap interval excludes 0 in all three.

| fold | C3 3-member ensemble AUC-PR | tuned-A 3-member ensemble | E | paired 95% interval (difference) | E vs. untuned early-stopped A (previous study) |
|---|---|---|---|---|---|
| 2016 | 0.0706 | 0.0528 | +33.5% | [+0.0095, +0.0280] | +35.8% |
| 2020 | 0.0847 | 0.0733 | +15.7% | [+0.0014, +0.0228] | +5.7% |
| 2024 | 0.1150 | 0.0898 | +28.1% | [+0.0151, +0.0367] | +46.7% |
| pooled | | | **+25.8%** | | +29.4% |

The C3 side is the same set of models as in the previous study (identical numbers); only the arm-A side changed. The 2020 interval now excludes 0 only narrowly (lower end +0.0014), so that fold is weak evidence.

### The effect of tuning arm A (secondary, descriptive)

- **Ensembles:** tuned A 0.0528 / 0.0733 / 0.0898 vs. untuned early-stopped A 0.0520 / 0.0802 / 0.0784: +1.7 / -8.6 / +14.6%, mean ensemble +2.6% pooled. Inside the noise.
- **Single models (mean of 3 seeds):** 0.0483 / 0.0691 / 0.0836 vs. 0.0457 / 0.0705 / 0.0739: +5.7 / -1.9 / +13.2%.
- So **the ~17% Stage 1 gain on fold 2018 mostly did not carry to the study folds**; tuning is worth about 3.6 points of the earlier +29.4% (to +25.8%), within what ~12% resolvability can see. Stage 1's two-seed screen on one fold was a coarse selector, and the selected margin shrank on new folds.
- Stopping: tuned A best epochs 19-26 (median 24, stopped 27-34), vs. the untuned A's 17-46 (median 29); the best epoch is less scattered. C3's median best epoch is 20-22.
- No NaN in any run (no NaN in logs; all final losses finite).

### Secondary metrics, C3 vs. tuned A (3-member ensembles, descriptive)

- **Timing metric:** false-alarm rate on quiet bins of active runs at 50% recall 0.081 vs 0.236 (-66%), 0.074 vs 0.136 (-45%), 0.071 vs 0.258 (-72%); at 80% recall -46 / -42 / -45%. Tuning did not change A's timing metric materially (0.236 / 0.136 / 0.258 vs. 0.236 / 0.160 / 0.247).
  Map AUC-ROC higher for C3 in all folds (+7.2 / +3.3 / +7.1%).
- **Cell level:** best-F1 +19.2 / +17.4 / +15.7%; 1-cell-dilated AUC-PR +22.3 / +12.1 / +19.6%; per-bin AUC-PR higher in all six cells of the table (bin 0 +8.9 to +14.4%, bin 1 +18.8 to +45.2%); cell AUC-ROC equal (within 0.1%);
  Cohen's d -4.7 / -5.5 / +1.8% (about equal).
- **Single models:** C3 singles are above the tuned A singles in every comparison but the ranges overlap in 2020 (C3 0.0735-0.0814 vs. A 0.067-0.072).

## Predictions, scored (`docs/unet_control_preregistration.md`)

1. **Stage 1 selects a non-default setting (40%): met** (U1, +16.6%).
2. **Stage 2 verdict helps 50% / no detectable effect 42% / hurts 8%; central pooled E +12%:** outcome is the 50% category; the central guess was ~14 points low (+25.8% observed).
3. **Tuned A's pooled AUC-PR within +/-8% of the previous early-stopped A's (55%): met** (+2.6%; one fold, 2020, is -8.6%).
4. **C3 timing metric >= 10% below tuned A in >= 2 of 3 folds (80%): met** (3 of 3, -45 to -72%).
5. **If a lower-lr setting is selected, its median best epoch exceeds the previous A's 29 (70%): not met** (median 24; the lower-lr runs stopped slightly earlier, not later).
6. **No NaN (95%): met.**

## What this does and does not show

- With arm A tuned over lr and clipping (matching what C3's repair got), the ConvLSTM-U-Net still beats it by a pooled +25.8% on all three folds; the difference is not an artifact of A's untuned optimizer. Most of the +29.4% (about 88%) survives.
- It does **not** show that recurrence is the cause: C3 also has GroupNorm before the LSTM and pooled-information skips, neither of which the U-Net has, and the no-recurrence snippet matched at 10 epochs.
  A GroupNorm U-Net control and the 3-seed no-recurrence ablation under the same stopping rule are the way to find out.
- Limits: only lr and clipping tuned, on a 3-point grid (two seeds, one fold); C3 was not re-tuned, and its lr 1e-4 was chosen partly on fold 2020 seeds 10-11 inside the study folds (2016/2024 alone are the cleaner evidence, and are the strongest folds here);
  three folds x three seeds (~12% resolvable); fold 2020's interval barely excludes 0; the hourly store only, nothing here touches the serving bundle (single-snapshot features).
- Not done: GroupNorm U-Net control, C3 re-tuning, no-recurrence ablation, hybrid with A's pooled fields, 12-fold confirmation, deployment-like (20Z) stratum, serving on hourly features.

## What happens next (not decided)

Per the pre-registration's "C3 still helps" branch: the 3-seed no-recurrence ablation under the same stopping rule (and optionally a GroupNorm U-Net control), then the hybrid with A's pooled fields. A 12-fold confirmation is the other option.
