# ConvLSTM repair arm "C": plan and snippet gate

Written 2026-10-31 after the first full study returned "hurts" (`docs/convlstm_results.md`) and diagnostics found (post hoc) saturated LSTM gates and single-hour skips.
The snippet gate below is fixed **before any repaired model has a score.** The first study is reported alongside whatever follows; this is a second attempt after a failed first one.

## Changes (all off by default; with them off the model is bit-identical to the first study's)
1. `norm_bottleneck`: GroupNorm(8) on the bottleneck features before the LSTM. The encoder still takes A's raw inputs.
2. `pooled_skips`: each decoder skip = mean and max over the bin's own hours (bin 0: hours 0-3, bin 1: hours 4-7) of the encoder features at that resolution, merged by a 1x1 conv shared by both stops.
3. `grad_clip_norm=1.0`. A pre-flight diagnostic (`models/convlstm/diagnostics/training_diagnostic.json`) found gradient norms <= 0.1, so the clip never engages; it is kept for safety but cannot be credited with anything.

Pre-flight result (2 epochs, 800 runs): gate pre-activation std 8 -> 32 for the original configuration, ~1 for every variant with the GroupNorm.

## Snippets (fold 2020, seed 10, 10 epochs, full training set, EMA 0.9999; fold 2020 seed 10 is also a study cell, as before)
C1 = norm + clip; C2 = pooled skips only; C3 = all three. C3 goes to the full study if it passes the gate; C1 and C2 only say which repair matters and do not choose the arm.

## Gate for C3 (all must hold; judged on the plain final weights, since 10 epochs is too short for EMA 0.9999)
1. Gates unsaturated on held-out runs: pre-activation std < 10 and < 20% of values beyond |4| (`diagnose_gates.py` logic).
2. AUC-PR on 500 stratified training runs is not below the matched U-Net control's on the same runs.
3. Held-out AUC-PR >= the control's 0.0518 (the first snippet's bar was 10% below it, and it did not catch the problem).
4. Stable training: no NaNs, final val/train loss ratio about 0.5-1.2.
Failing means stop and diagnose, not scale up. Single seed, ~19% noise: passing is permission to run the full study, not evidence it will win.

## Full study if the gate passes
Folds 2016/2020/2024, seeds 10-12, 50 epochs, EMA 0.9999, CPU (~22 h), with its own pre-registration written before any score: the same rules and thresholds as `docs/convlstm_preregistration.md`
against A's matched 3-member ensemble ("helps" pooled E >= +8% and >= 2 of 3 folds positive; "hurts" mirror), plus: any "helps" claim needs the "confirmed" standard (paired interval excludes 0 in >= 2 folds),
the first attempt's -19.7% is reported next to it, and the saturation check is repeated on the trained models.
