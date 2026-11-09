# Hourly features, ConvLSTM, C3, dropout, per-epoch, early stopping, tuned U-Net control

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Hourly features: whole-bin aggregation (A) and extra fields (B) -- closed

Pre-registered (`docs/hourly_features_preregistration.md`), results in `docs/hourly_features_results.md`, numbers in
`models/hourly_study/{wholebin,extras}/hourly_eval.json`. Code: `src/tornado_predictor/build_hourly.py`, `scripts/02_dataset_build/build_hourly_dataset.py`
and `build_wholebin_datasets.py`, `scripts/10_hourly_features/`. Hourly HRRR (fxx 0-7, plus 9 extra fields) is staged for 4,916 of 4,919 runs
(`data/interim/scaled_build_hourly{,_extra}/`; 3 permanent AWS archive gaps in 2015); derived whole-bin staging in `scaled_build_wholebin{,_extra}/`.
Setup: `TornadoUNet(base_channels=16)`, EMA 0.9999, folds 2016/2020/2024, seeds 10-12, 3-member-ensemble AUC-PR vs. the mean of the 20 baseline
3-member ensembles; "helps" = pooled E >= +8% and >= 2/3 folds. The 17 new channels in B are z-scored with training-split stats (stored as
`input_scaling` in checkpoints, applied by `inference.apply_input_scaling`); the base 14 stay raw.

- **A (all 4 hours per bin, same 14 channels, time-mean of `_mean`, time-max of `_max`): helps, confirmed.** Pooled **+22.5%** (+31.0/+19.3/+17.1%),
  above all 20 baseline ensembles in every fold, paired intervals exclude 0 in 3/3. Ensemble AUC-PR 0.0533/0.0689/0.0844 vs. baseline 0.0407/0.0577/0.0720.
  The false-alarm rate on quiet bins of active runs at 50% recall fell in every fold (0.323->0.193, 0.259->0.216, 0.258->0.231) and map AUC-ROC rose in
  every fold: supportive of the timing hypothesis, not proof (the change also alters value ranges, e.g. time-max, and bin 1 moves from fxx 4 to fxx 4-7).
- **B (A + MLCAPE/MLCIN, MUCAPE/MUCIN, SB LCL, 0-1 km shear, REFC, surface pressure, derived `pres_sfc_tend`): helps weakly versus A.** B vs. A pooled **+9.7%**
  (+4.9/+20.2/+4.0%), 3/3 folds positive, but the paired interval excludes 0 only in 2020, and two folds are under the ~12% resolvable with 3 seeds:
  "weak: inside noise" by the pre-registered wording. B vs. baseline +34.2%. B's timing metric does not improve consistently over A.
  **Nothing here attributes the gain to pressure or its tendency** (B bundles all extras; no ablation was run). Predictions: A timing metric met; A size far
  above my central guess; B-vs-A prior (60% no effect) narrowly missed; era-dependence prediction not met; training stability met (loss ratios 0.945-1.066).
- **Not in serving:** the production bundle and live path still use single-snapshot features. Whole-bin features would need fxx 0-7 pulls live and a retrained bundle.
- **Not done:** drop-one-group ablation of B, 12-fold confirmation of A, standardizing the base 14, deployment-like (20Z) stratum, and the ConvLSTM on the hourly store
  (the user's next, separate project). Pitfall: study runners must activate `.venv` (bare `python` hits a broken pyenv shim; `tee` hides the exit code).

### ConvLSTM-U-Net on the hourly store vs. arm A -- negative result, with a diagnosed defect

Pre-registered (`docs/convlstm_preregistration.md`), results in `docs/convlstm_results.md`, numbers in `models/convlstm/full_study/convlstm_eval.json`. Code: `sequence_dataset.py`, `convlstm_unet.py`
(registered as `TornadoConvLSTMUNet`), `sequence_study.py`, `scripts/11_convlstm/`. Input one run (8 hourly steps x the same 14 raw channels), shared U-Net encoder per hour, ConvLSTM at the 11x18 bottleneck, shared decoder
run at hour 3 (bin 0) and hour 7 (bin 1), one head per bin; A's recipe, batch 1 run, run-level quiet downsampling (~62% active maps vs. A's 79%). Folds 2016/2020/2024, seeds 10-12, 50 epochs, EMA 0.9999.

- **Verdict by the rule: hurts.** 3-member ensemble AUC-PR vs. A's: 0.0378 / 0.0622 / 0.0673 vs. 0.0533 / 0.0689 / 0.0844; E -29.2 / -9.7 / -20.3%, pooled **-19.7%**, 3/3 folds negative, paired interval excludes 0 in 2.
- **Mixed secondary picture:** timing metric (false alarms on quiet bins of active runs at 50% recall) fell in all folds and map AUC-ROC rose in all folds, but best-F1, per-bin AUC-PR and dilated AUC-PR fell in all folds.
- **Post-hoc diagnosis (exploratory):** the ConvLSTM also fits the training data worse (not overfitting); the LSTM gates are saturated in all nine trained models (pre-activation std 3e3-9e5, 85-100% beyond |4|; fine at init),
  appearing within the first epoch (gradient norms stay <= 0.1, so clipping is not the fix; likely raw unscaled inputs with no normalization before the LSTM); and the skip connections come from a single hour, unlike A's whole-bin pooled fields. So this tests a defective configuration, not the idea; the verdict is not revised.
- **Not done:** a repaired arm (GroupNorm before the LSTM, pooled-information skips; clipping is inert) needs its own pre-registration and snippet gate (gate saturation as a check). Pitfall: raw-scale inputs and saturating gates do not mix.
  The pre-registered number to cite for the ConvLSTM is the -19.7%, with the defect caveat. (The repair was then done and tested: see the next section.)

### Repaired ConvLSTM (C3) vs. arm A -- no detectable effect

Diagnosis in `docs/convlstm_diagnostic_ladder.md`, gate check `models/convlstm/repair_gate_check.json`, pre-registration `docs/convlstm_c3_preregistration.md` (written and committed before launch), results `docs/convlstm_c3_results.md`,
numbers `models/convlstm/full_study_c3/convlstm_eval.json` and `gate_saturation.json`. Runner `scripts/11_convlstm/run_c3_study.sh`; `train_convlstm.py` and `cross_validate_year.py` gained `--lr`, `convlstm_eval.py` gained `--prereg`/`--study-dir`,
`diagnose_gates.py` gained `--study-dir`. C3 = `norm_bottleneck=True` + `pooled_skips=True` + grad clip 1.0 + **lr 1e-4** (lr 1e-3 was too aggressive), otherwise the first study's recipe; folds 2016/2020/2024, seeds 10-12, 50 epochs, EMA 0.9999.

- **Ladder (fold 2020, 10 epochs, held-out plain AUC-PR):** at lr 1e-4 C3 scored 0.0687 / 0.0817 (seeds 10/11) vs. a matched-lr U-Net 0.0485 / 0.0612; C3 without recurrence scored 0.0661. The ConvLSTM can memorize 20 runs (train AUC-PR ~1.0), so it is not an optimization failure.
- **Verdict by the rule: no detectable effect.** Ensemble AUC-PR 0.0527 / 0.0650 / 0.0863 vs. A's 0.0533 / 0.0689 / 0.0844; E -1.1 / -5.7 / +2.3%, **pooled -1.5%** (first attempt -19.7%, which stands for that recipe); paired intervals include 0 in all folds; minimum detectable effect ~12%.
  Pooled over 2016/2024 only (not used to pick lr): +0.6%. lr 1e-4 was chosen on fold 2020 seeds 10-11, which are inside the study.
- **Single models are weaker, ensembles tie:** every C3 single seed is below A's mean (mean single -31 / -28 / -24%), but the 3-member ensemble gains +51-67% over the mean single seed (A +15%): members are individually weaker and more diverse.
- **Training:** no NaN and no gate saturation (pre-activation std 1.4-1.75, 2-4% beyond |4|), but **final val/train loss ratio 2.3-4.2 in all nine runs** (pre-registered band 0.4-1.3, prediction 5 missed): the repaired model overfits (train loss ~4.5e-5, about a third of A's).
- **Secondary:** quiet-bin-of-active-runs false-alarm rate at 50% recall -45 / -63 / -60% vs. A and map AUC-ROC higher in all folds (timing hypothesis supported), but cell AUC-ROC lower, Cohen's d about half of A's, bin-1 AUC-PR not better.
  Predictions: 3 and 6 met, 4 and 5 missed, 1 landed in the 45% category with a central guess ~5 points too high.
- **Not shown:** that recurrence matters (the no-recurrence snippet matches; GroupNorm, skips, clipping and lr changed together). Not done: a regularized/earlier-stopped C3, a 3-seed no-recurrence ablation, a hybrid with A's pooled fields, 12-fold confirmation.
  Pitfall: numpy 2.0.2 on this Mac prints spurious `divide by zero / overflow in matmul` warnings in the bootstrap (values verified equal to einsum).

### Repaired ConvLSTM + dropout -- no detectable effect (study closed)

Pre-registration `docs/convlstm_dropout_preregistration.md` (sha256 `fc7dbb48...`, unchanged), results `docs/convlstm_dropout_results.md`, summary notebook `notebooks/convlstm_study.ipynb`, numbers `models/convlstm/dropout_select/selection.json` and
`models/convlstm/full_study_c3_dropout/{convlstm_eval,gate_saturation}.json`. Code: `TornadoConvLSTMUNet(dropout=p)` (Dropout2d on the LSTM input and decoder hidden state; p=0 reproduces C3 exactly), `train_convlstm.py --dropout`, runners `run_dropout_select.sh` / `run_dropout_study.sh`.

- **Stage 1 (fold 2018, seeds 20/21):** dropout lowered the val/train loss ratio monotonically (2.45 at p=0 -> 2.13 / 1.79 / 1.42 at 0.1 / 0.3 / 0.5) but AUC-PR did not improve (0.0418 control; 0.0424 / 0.0375 / 0.0383). The fixed rule picked p=0.1 on a +1.4% tie.
- **Stage 2 (p=0.1, folds 2016/2020/2024, seeds 10-12): no detectable effect.** Ensemble AUC-PR 0.0542 / 0.0697 / 0.0866 vs. A's 0.0533 / 0.0689 / 0.0844; E +1.7 / +1.2 / +2.7%, **pooled +1.9%**, paired intervals include 0 in every fold (C3 without dropout -1.5%; first attempt -19.7%).
  The 3.4-point gap to no-dropout C3 is inside the ~12% resolvable, so dropout is not shown to help.
- **Overfitting reduced, not cured:** val/train ratio 1.87-4.02 (mean 2.68) vs. C3's 2.34-4.25 (3.16), lower in all nine pairs, still far above arm A's ~0.6. Single models still -22 to -27% vs. A's mean single; ensemble gain over the mean single +51-59% (A +15%).
  Same secondary picture as C3: timing false-alarm rate -39 / -49 / -62% vs. A and map AUC-ROC higher in all folds, cell AUC-ROC and Cohen's d lower. No NaN, no gate saturation.
- **Predictions:** 4 (no detectable effect, central +3% vs. +1.9% observed), 5, 8b, 9, 10 met; Stage 1 predictions 1-2 and Stage 2 predictions 6, 7, 8a missed.
- **Correction:** C3's ensemble gain over the mean single seed is +67 / +51 / +54% (an earlier note said +63 / +52 / +54%; the dropout pre-registration still quotes +52-63% and was deliberately not edited).
- **Not done:** a 3-seed no-recurrence ablation, an early-stopped C3, a hybrid with A's pooled fields, a higher dropout rate with multi-seed selection, 12-fold confirmation. **Lesson: log per-epoch held-out metrics (AUC-PR/ROC, precision, recall) in future studies** -- only per-epoch loss was recorded here, so early stopping can only be argued from loss curves
  (C3's validation loss turns up around epoch 25 while training loss keeps falling).

### Per-epoch held-out metrics (one fold, one seed) -- the ConvLSTM overfits, arm A does not

Plan and scored predictions: `docs/convlstm_per_epoch_plan.md`; figure `notebooks/convlstm_study.ipynb` (per-epoch subsection); numbers `models/convlstm/per_epoch/{arm_a,c3}/results.json`. `training.train_model(epoch_metrics_fn=...)` (opt-in, tests verify
it leaves loss history and weights unchanged), `cross_validate_year.py --per-epoch-metrics`, `train_convlstm.py --per-epoch-metrics`; runner `scripts/11_convlstm/run_per_epoch_study.sh`. Fold 2020, seed 10, 50 epochs, AUC-PR/AUC-ROC/precision/recall after every epoch on raw and EMA weights.
Final-epoch values reproduce the full study exactly (A 0.0664, C3 0.0374).

- **Arm A (EMA):** AUC-PR peaks 0.0723 at epoch 24, epoch 50 = 0.0664 (0.92 of peak); held-out loss flat from ~epoch 10; AUC-ROC never degrades. **C3 (EMA):** peaks 0.0791 at epoch 20, epoch 50 = 0.0374 (**0.47 of peak**); cell AUC-ROC falls 0.995 -> 0.976;
  held-out loss minimum at epoch 15, doubles by epoch 50 while training loss keeps falling. **The 50-epoch "tie" compared A near its best with C3 well past its best.** C3's best epoch is +9% over A's best: one seed each, inside the noise, and selected on the test year.
- EMA curves start near zero (the average starts at the random initial weights; ~5-epoch ramp) -- not a training failure. Predictions: 1, 5, 6 met; 2 mixed; 3 not met; 4 met numerically (I had 25%).
- **Caveat:** these curves are scored on the held-out year itself, so choosing an epoch from them selects on the test data. A fair early-stopping study needs a separate validation set (see below).
- **Next study (run, see below):** early stopping for both arms on a mod-20 validation split of the training dates, patience 8 on EMA validation AUC-PR (warm-up 10 epochs, cap 50, best weights restored), folds 2016/2020/2024, seeds 10-12. A loss-based rule is unsuitable because A's loss is flat. (Done; then a tuned arm-A control was run -- see the two sections below.)

### Early-stopped ConvLSTM (C3) vs. early-stopped arm A -- helps (confirmed)

Pre-registration `docs/convlstm_early_stopping_preregistration.md` (sha256 `a84d40a3...`, unchanged at scoring), results `docs/convlstm_early_stopping_results.md`, numbers `models/convlstm/early_stop/c3/convlstm_eval.json` (per-run stopping history in each run's `results.json`).
Code: `training.train_model(early_stop_fn=, early_stop_patience=8, early_stop_min_epochs=10, early_stop_decay=)` (opt-in; monitors validation AUC-PR of the EMA weights, restores the best epoch), `--early-stopping/--patience/--min-epochs` on `cross_validate_year.py` and `train_convlstm.py`,
`convlstm_eval.py --a-dir`, runner `scripts/11_convlstm/run_early_stop_study.sh`. Validation = mod-20 date split of the TRAINING years (24h buffer), never the test year; both arms train on ~78% of the earlier runs, so E is against the new early-stopped A, not the old 50-epoch A.

- **Verdict by the rule: helps, confirmed.** 3-member ensemble AUC-PR C3 0.0706 / 0.0847 / 0.1150 vs. A 0.0520 / 0.0802 / 0.0784; E +35.8 / +5.7 / +46.7%, **pooled +29.4%**, 3/3 folds positive, paired interval excludes 0 in 2016 and 2024 (2020's includes 0). The 2024 value is the highest ensemble AUC-PR recorded on any fold.
  Earlier 50-epoch C3 was -1.5% and C3 + dropout +1.9% vs. the old A: those "no detectable effect" results compared an overfit-after-epoch-20 model against an arm that does not overfit.
- **Single models:** early-stopped C3 beats its own 50-epoch single in 9/9 runs; C3 singles are +35 / +11 / +44% over A's mean single; ensemble gain over the mean single is now similar for both arms (~+6-14%), so the earlier "weak but diverse members" pattern was the overfit tail.
- **Stopping:** C3 stopped early in 9/9 runs (best epochs 17-30, median 20); A in 8/9 (best epochs 17-46, median 29; its AUC-PR curve is nearly flat so its best epoch is noisy). Val/train loss ratio at stop 0.77-1.27 (was 2.3-4.2 at epoch 50). Early-stopped A singles are +5% vs. the old A singles (stopping on 78% of the data did not hurt).
- **Secondary:** quiet-bin-of-active-runs false-alarm rate at 50% recall -66 / -54 / -71% vs. A; map AUC-ROC higher in all folds; best-F1 +28 / +12 / +27%; Cohen's d now about equal (-7 / -7 / +4%), cell AUC-ROC equal.
- **Predictions:** 2-7 met; 1 landed in the 25% "helps" category with a central guess +4% vs. +29.4% observed; 8 missed (no NaN, but ~25 h wall clock incl. a ~4 h sleep, not ~9 h).
- **Not shown:** that recurrence is the cause (C3 bundles GroupNorm, pooled skips, clipping, lr 1e-4 and the LSTM; the earlier no-recurrence snippet matched at 10 epochs). Not done: 3-seed no-recurrence ablation under the same stopping rule, hybrid with A's pooled fields, 12-fold confirmation, deployment-like (20Z) stratum, a serving bundle on hourly features (live path would need fxx 0-7 pulls).
  Fold 2020 is inside the noise and ~12% is resolvable with 3 seeds; the pooled number is carried by 2016 and 2024. Pitfall: wall clock was ~25 h because the Mac slept on lid close (`caffeinate -i` does not stop that) and epochs slowed under contention.

### Tuned arm-A U-Net control vs. early-stopped C3 -- helps (confirmed)

Pre-registration `docs/unet_control_preregistration.md` (sha256 `5f931a16...`, unchanged at launch of both stages and at scoring), results `docs/unet_control_results.md`, numbers `models/unet_control/select/selection.json` and `models/unet_control/tuned_a/convlstm_eval.json`.
Code: `cross_validate_year.py --grad-clip-norm` (default off, behaviour unchanged), runners `scripts/11_convlstm/run_unet_control_select.sh` / `run_unet_control_study.sh`, selection rule `unet_control_select_eval.py`. Question: the +29.4% compared C3 (lr 1e-4, clip 1.0, partly tuned on fold 2020) with arm A at its never-tuned recipe (lr 1e-3, no clip).

- **Stage 1 (fold 2018, seeds 20/21, early stopping):** mean EMA AUC-PR U0 (lr 1e-3, no clip) 0.0535, **U1 (lr 3e-4, clip 1.0) 0.0625 (+16.6%)**, U2 (lr 1e-4, clip 1.0) 0.0617 (+15.2%). The >=5% rule selected U1; U1 vs U2 is a tie (two seeds). U0's seeds differ by ~20%, U1/U2's by ~3%.
- **Stage 2 (U1, folds 2016/2020/2024, seeds 10-12) vs. the early-stopped C3 models:** tuned-A ensemble AUC-PR 0.0528 / 0.0733 / 0.0898; E +33.5 / +15.7 / +28.1%, **pooled +25.8%**, 3/3 folds positive, paired interval excludes 0 in all three (2020 only narrowly): **helps, confirmed**.
  Tuning moved arm A's ensembles by +1.7 / -8.6 / +14.6% (pooled +2.6%) and singles by +5.7 / -1.9 / +13.2%: the Stage 1 gain mostly did not carry to the study folds. About 88% of the +29.4% survives. Tuned A's best epochs 19-26 (median 24, vs. 29 untuned).
- **Secondary:** C3's timing false-alarm rate at 50% recall is -66 / -45 / -72% vs. tuned A (tuning left A's own timing numbers unchanged); map AUC-ROC +7.2 / +3.3 / +7.1%; best-F1 +19 / +17 / +16%; Cohen's d about equal; cell AUC-ROC equal.
- **Predictions:** 1, 3, 4, 6 met; 2 landed in the 50% "helps" category but the central +12% was ~14 points low; 5 missed (median best epoch 24, not above 29).
- **Not shown / not done:** that recurrence is the cause (C3 also has GroupNorm and pooled skips; no GroupNorm U-Net control, no C3 re-tuning; C3's lr 1e-4 was chosen partly inside the study folds, though 2016/2024 are the strongest). Next per the prereg: 3-seed no-recurrence ablation under the same stopping rule, hybrid with A's pooled fields, 12-fold confirmation.


