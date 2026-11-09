# Pre-registration: a tuned arm-A U-Net control for the early-stopped ConvLSTM

Written before any run of this study. Follows `docs/convlstm_early_stopping_results.md`, where the early-stopped repaired ConvLSTM (C3) beat early-stopped arm A by a pooled +29.4% (3/3 folds positive, paired interval excludes 0 in 2016 and 2024).

## Question
That comparison was not like for like. C3's learning rate (1e-4) and gradient clipping (1.0) were chosen during its repair, in part on fold 2020 seeds 10-11 (inside the study), while arm A kept its original recipe (lr 1e-3, no clipping) and was never tuned. How much of the +29.4% survives once arm A gets an equivalent tuning budget?

## Design
Two stages, both with the same early-stopping rule as the previous study (validation = mod-20 date split of the TRAINING years, 24 h buffer; stop on validation AUC-PR of the EMA 0.9999 weights, patience 8, min 10 epochs, cap 50, best weights restored; test year scored once). Arm A architecture, features (whole-bin store), batch size, loss and sampling are unchanged; only the optimizer settings below vary.

**Stage 1 (selection, fold 2018, outside the confirmatory folds, seeds 20 and 21).** Three optimizer settings, two seeds each (6 runs):
- U0: lr 1e-3, no clipping (the existing arm-A recipe; control)
- U1: lr 3e-4, gradient clip 1.0
- U2: lr 1e-4, gradient clip 1.0 (the optimizer settings C3 uses)

Fixed selection rule: the setting with the highest mean EMA test AUC-PR over the two seeds is selected, but a non-default setting is selected only if its mean is at least **5% above U0's**; otherwise U0 stands. (Fold 2018 is not a study fold, as in the dropout study's Stage 1. Two seeds carry ~13% noise, so this is a coarse screen, not a precise optimum.)
If U0 stands, arm A is considered tuned within this grid: the existing early-stopped arm-A runs ARE the tuned control, Stage 2 is not run, and the C3-vs-A result of the previous study stands as the answer to this question.

**Stage 2 (confirmatory, only if a non-default setting is selected).** The selected setting trained with the same early stopping on folds 2016 / 2020 / 2024, seeds 10-12 (9 runs), then scored against the early-stopped C3 ensembles from the previous study with the existing evaluator (`convlstm_eval.py --a-dir <new arm-A dir> --study-dir models/convlstm/early_stop/c3 --out ...`), which checks that both arms used the same training/validation run lists and test order.

## Decision rule (unchanged from earlier studies)
E = (C3 3-member ensemble AUC-PR / tuned-A 3-member ensemble AUC-PR) - 1, per fold; pooled E is the mean over folds.
- **helps**: pooled E >= +8% and E > 0 in at least 2 folds. **hurts**: pooled E <= -8% and E < 0 in at least 2 folds. Otherwise **no detectable effect**. With 3 seeds ~12% is resolvable.
- "Confirmed" requires the paired day-bootstrap interval of the difference to exclude 0 in at least 2 folds.
- Secondary, descriptive: tuned A vs. the previous early-stopped A (the effect of tuning), quiet-bin false-alarm rate at 50% recall, map AUC-ROC, best-F1, Cohen's d, stopping epochs.

## Outcome meanings
- **C3 still helps (confirmed or not):** the ConvLSTM design has an edge beyond arm A's untuned recipe; the next study is the 3-seed no-recurrence ablation, then the hybrid with A's pooled fields.
- **No detectable effect:** most of the earlier gain was the optimizer recipe; arm A stays the production feature design and the ConvLSTM line closes (or is reopened only with a new idea).
- **hurts (tuned A better than C3):** close the ConvLSTM line and adopt the tuned recipe for arm A.
- If U0 stands in Stage 1: the +29.4% stands unchanged, with the added statement that a three-point lr/clip grid did not help arm A on fold 2018.

## What this does not cover (stated in advance)
- Only learning rate and clipping are tuned. C3 also has GroupNorm and pooled skips, which have no direct arm-A counterpart (arm A already pools over time at the input; the U-Net has no normalization layers). A GroupNorm variant of the U-Net would be a further, separate control.
- A three-point grid on one fold with two seeds is a modest tuning budget, though it is at least as much as C3's repair got; C3 itself is not re-tuned here.
- Three folds, three seeds: ~12% resolvable; per-fold results can disagree (fold 2020 was inside the noise last time).

## Predictions (my honest priors, written before any result)
1. Stage 1 selects a non-default setting (U1 or U2) under the +5% rule: 40%.
2. If Stage 2 runs: verdict **helps 50% / no detectable effect 42% / hurts 8%**; central guess for pooled E +12%.
3. If Stage 2 runs: the tuned A's pooled AUC-PR is within +/-8% of the previous early-stopped A's: 55%.
4. If Stage 2 runs: C3's timing metric (quiet-bin false alarms at 50% recall) stays at least 10% below tuned A's in at least 2 of 3 folds: 80%.
5. If a lower-lr setting is selected, the median best epoch of its nine runs is higher than the previous arm A's (29): 70%.
6. All runs finish without NaN: 95%.

## Cost
Stage 1: 6 runs in 3 parallel lanes, ~1.5-3 h. Stage 2: 9 runs in 3 parallel lanes, ~2.5-4 h. Plug in; the Mac sleeps on lid close.

## Code
`cross_validate_year.py` gained `--grad-clip-norm` (default off, behaviour unchanged). Runners: `scripts/11_convlstm/run_unet_control_select.sh` (Stage 1), `scripts/11_convlstm/unet_control_select_eval.py` (selection rule), `scripts/11_convlstm/run_unet_control_study.sh` (Stage 2, takes the selected setting from `selection.json`).
