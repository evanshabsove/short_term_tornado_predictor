# ConvLSTM-U-Net vs. arm A: results

Pre-registration: `docs/convlstm_preregistration.md` (sha256 `52ec918f…`, identical to the hash recorded at launch; never edited). Numbers:
`models/convlstm/full_study/convlstm_eval.json` (evaluator `scripts/11_convlstm/convlstm_eval.py`, which first reproduced A's ensemble AUC-PR and checked that all nine runs used A's
training-run hash and test order). Code: `sequence_dataset.py`, `convlstm_unet.py`, `sequence_study.py`, `scripts/11_convlstm/`.

## Verdict by the pre-registered rule: **hurts**

Pooled E = **-19.7%**, negative in 3 of 3 folds (rule: pooled <= -8% and >= 2 folds negative). Not "helps", so "confirmed" does not apply.

| fold | ConvLSTM 3-member ensemble AUC-PR | A 3-member ensemble | E | paired day-bootstrap 95% interval (difference) |
|---|---|---|---|---|
| 2016 | 0.0378 | 0.0533 | -29.2% | [-0.0255, -0.0074] excludes 0 |
| 2020 | 0.0622 | 0.0689 | -9.7% | [-0.0146, +0.0029] includes 0 |
| 2024 | 0.0673 | 0.0844 | -20.3% | [-0.0259, -0.0085] excludes 0 |

Single models (EMA 0.9999): 2016 ConvLSTM 0.0328 / 0.0285 / 0.0349 vs. A 0.0447 / 0.0469 / 0.0480; 2020 0.0523 / 0.0478 / 0.0584 vs. 0.0664 / 0.0533 / 0.0601;
2024 0.0590 / 0.0644 / 0.0562 vs. 0.0595 / 0.0796 / 0.0821. In 2016 every ConvLSTM seed is below every A seed.

## Secondary metrics (descriptive)

- **Timing metric improved.** False-alarm rate on quiet bins of active runs at 50% recall: 0.193 -> 0.161 (2016), 0.216 -> 0.130 (2020), 0.231 -> 0.176 (2024); at 80% recall
  0.609 -> 0.522, 0.481 -> 0.407, 0.516 -> 0.456. Map-level AUC-ROC rose in all three folds (0.831 -> 0.847, 0.840 -> 0.850, 0.830 -> 0.840).
- **Cell-level ranking got worse.** Best-F1 fell in every fold (0.128 -> 0.109, 0.156 -> 0.143, 0.168 -> 0.147), and precision fell in every fold; recall fell in 2016 and 2020 and rose in
  2024 (0.259 -> 0.276). 1-cell-dilated AUC-PR fell in every fold. Cohen's d fell in every fold. AUC-PR per bin fell in every fold and bin (e.g. 2016 bin 0 0.066 -> 0.042, bin 1 0.041 -> 0.027);
  the gap is somewhat smaller for bin 1 than bin 0 in all three folds (2020: -22% vs. -2%).
- **Training was well behaved by the pre-registered checks:** no NaNs; final val/train loss ratios 0.60-0.69 (2016), 0.60-0.63 (2020), 0.84-0.86 (2024); held-out loss similar to A's
  (2016 0.72 vs. 0.64, 2020 0.68 vs. 0.66, 2024 0.90 vs. 0.90, seed 10; loss does not rank models in this project).

## Predictions, scored

Missed: P(hurts) was 15% (my central guess was +6%); 2024 was not the weakest fold (2016 was); "confirmed" never applied. Held: timing metric fell by >= 10% in 3 of 3 folds (prediction 50%);
no run diverged (85%). Not cleanly scorable: "gain larger on bin 1" assumed a gain. The snippet gate's +10.6% (one seed, 10 epochs, one cell of this study) did not predict the result.

## Post-hoc diagnostics (exploratory, run after seeing the verdict; not part of the pre-registered analysis)

Prompted by the user's doubt that a model with strictly more information could be this much worse.
1. **Not overfitting; the ConvLSTM fits the training data worse too.** AUC-PR on 500 stratified training runs (seed 10, EMA): 2016 ConvLSTM 0.0545 vs. A 0.0750; 2020 0.0386 vs. 0.0517 -- the
   same ~25-27% deficit as held-out. Only two models were checked.
2. **The LSTM gates are saturated in all nine trained models** (`scripts/11_convlstm/diagnose_gates.py`, `models/convlstm/full_study/gate_saturation.json`): gate pre-activations have a
   standard deviation of 2.8e3 to 8.8e5, and 85-100% of them lie beyond |4| where sigmoid/tanh are flat. **An untrained model is fine** (std ~0.5, 0% beyond |4|, with raw or z-scored input), so
   training drove it there. The saturation appears within the first epoch (a 2-epoch diagnostic, `scripts/11_convlstm/diagnose_training.py`, `models/convlstm/diagnostics/training_diagnostic.json`:
gate pre-activation std ~8 by step 25 and 30-50 by epoch 2 on an 800-run subset, vs. ~1 with a GroupNorm before the LSTM). An earlier guess in this document that large early gradients caused it is
**wrong**: global gradient norms stay <= 0.1, so gradient clipping never engages. The likely mechanism (not isolated): raw-scale inputs (CAPE to ~6,600, helicity to ~1,800), an encoder with no normalization,
and Adam's step size being roughly independent of gradient magnitude, so the encoder's bottleneck activations grow steadily until the saturating gates stop carrying usable memory. A plain U-Net (A) is
immune because ReLU convs have no saturating nonlinearity in the path.
3. **A structural difference unrelated to scaling.** At all three decoder resolutions the skip connections come from the single stop hour (hour 3 for bin 0, hour 7 for bin 1). Information from the other
   hours reaches the decoder only through the 11x18 bottleneck memory, while A receives the whole bin's time-mean/time-max at full resolution. The design docstring's statement that the temporal
   combination is "the only thing that differs from arm A" is therefore too strong; this limit would remain even with healthy gates.

## What this does and does not show

- **Shows:** for this exact configuration (A's recipe, raw inputs, no clipping or normalization, single-hour skips), the ConvLSTM is worse than A at cell-level ranking by about 20%, in every fold, beyond
  day-resampling noise in two. It is better on a few map-level and timing metrics.
- **Does not show:** that a ConvLSTM cannot beat A. The saturation finding means the recurrent part was not functioning as designed, so this is a test of a defective configuration, not of the idea.
  The pre-registered verdict stays "hurts" and is not revised.
- **Not isolated:** how much of the deficit is gate saturation vs. single-hour skips vs. the run-level quiet mix (62% vs. 79% active) vs. no tuning. No ablation was run.
- **Disclosure:** I saw one single-model score (seed 10, fold 2016: EMA 0.0328) before running the evaluator, by printing a log. It did not influence any rule.

## Next, if wanted (not started; needs its own pre-registration)

A repaired arm "C": normalization or scaling before the LSTM (keeping A's raw encoder inputs), gradient clipping, and skips that carry whole-bin pooled information. First a 10-epoch snippet on fold 2020 with the
gate-saturation check as an added gate item, then a new pre-registered full study, labeled as a second attempt after a failed first one, with this study reported alongside it.
