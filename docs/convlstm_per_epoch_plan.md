# Per-epoch held-out metrics: control U-Net (arm A) vs. repaired ConvLSTM (C3)

Written 2026-11-06, **before the runs**. Exploratory, descriptive: one fold (2020), one seed (10), no decision rule and no verdict. Purpose: see the held-out AUC-PR / AUC-ROC / precision / recall curves
per epoch (the user's standard way to judge training health and early stopping), which the earlier studies did not record.

Design: `train_model(epoch_metrics_fn=...)` (default off, training unchanged, tested) scores the held-out year after every epoch for the raw weights and for the EMA 0.9999 weights, recording AUC-PR, AUC-ROC, and precision / recall / F1
at the best-F1 threshold. Runs: arm A (`TornadoUNet(base_channels=16)`, whole-bin data, `cross_validate_year.py --per-epoch-metrics`) and C3 (`train_convlstm.py --per-epoch-metrics`, lr 1e-4, norm_bottleneck, pooled_skips, grad clip 1.0),
fold 2020, seed 10, 50 epochs, EMA 0.9999 -- the same recipe and seed as the corresponding full-study runs, so the final epoch should reproduce them.
Output: `models/convlstm/per_epoch/{arm_a,c3}/`. Runner `scripts/11_convlstm/run_per_epoch_study.sh`.

Caveat fixed now: the held-out year here is the year being scored, as in every study in this project. The curves describe training; **picking an epoch from them is selecting on the test year**, so any early-stopping epoch read off these curves would
need a separate validation split before it could be reported as a result.

## Predictions (my honest priors)
1. C3's raw-weight held-out AUC-PR peaks at or before epoch 35 and is lower at epoch 50 than at its peak: 65%.
2. Arm A's AUC-PR is flat after about epoch 10 (epoch-50 value within 10% of its maximum): 75%.
3. C3's AUC-PR at epoch 50 is within 15% of its own maximum (ranking degrades much less than the validation loss does): 55%.
4. C3's best epoch beats arm A's best epoch on AUC-PR (this fold/seed: C3 single 0.0374 vs. A 0.0664 in the full study, so I expect not): 25%.
5. The final-epoch metrics reproduce the full study's same-seed run (A 0.0664, C3 0.0374 EMA AUC-PR) to within 5%: 85% (the hook only reads the model; threads/BLAS ordering may cause small differences).
6. C3's EMA curve is smoother than its raw curve (lower epoch-to-epoch standard deviation of AUC-PR over epochs 10-50): 80%.

## Results (appended after the runs; the plan above is unchanged, its sha256 4cd99161... matches the hash recorded at launch)

One run each, fold 2020, seed 10, 50 epochs. Figure: `outputs/per_epoch_curves.png`. Numbers: `models/convlstm/per_epoch/{arm_a,c3}/results.json`.

| | peak AUC-PR (epoch) | epoch 50 | epoch 50 / peak | val loss minimum (epoch) |
|---|---|---|---|---|
| A, EMA | 0.0723 (24) | 0.0664 | 0.92 | 43 (flat from ~epoch 10) |
| A, raw | 0.0696 (23) | 0.0549 | 0.79 | |
| C3, EMA | 0.0791 (20) | 0.0374 | 0.47 | 15 (1.4e-4 at epoch 50, roughly double its minimum) |
| C3, raw | 0.0772 (15) | 0.0371 | 0.48 | |

C3 cell AUC-ROC peaks at 0.9947 (epoch 19, EMA) and falls to 0.9763 by epoch 50; A's rises to 0.9948 and stays.

Predictions: 1 met (C3 raw peak epoch 15, lower at 50). 2 mixed: A is within 10% of its max at epoch 50 on EMA (0.92) but not raw (0.79), and it is not flat after epoch 10 (still climbing until ~20). 3 not met (0.47-0.48 of max). 4 met numerically (C3 best 0.0791 vs. A best 0.0723, +9%) but that is one seed each, inside the ~19-25% noise, and I had put it at 25%. 5 met exactly (A EMA 0.0664, C3 EMA 0.0374 match the full study). 6 met (mean absolute epoch-to-epoch change over epochs 10-50: C3 raw 0.0046 vs. EMA 0.0014; A 0.0057 vs. 0.0014).

Caveat: the curves are scored on the held-out year itself, so "C3's best epoch" is selected on the test data and is optimistic. Choosing an early-stopping epoch for a reported result needs a separate validation split.
