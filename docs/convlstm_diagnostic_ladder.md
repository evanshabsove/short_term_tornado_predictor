# ConvLSTM diagnostic ladder (exploratory, not pre-registered for a verdict)

Written 2026-11-01, **before any ladder run.** Follows the repair snippets (C3 failed gate check 2 by 4.6%: train-subset AUC-PR 0.0443 vs. the control's 0.0464)
and a literature review (four agents reading primary sources; findings in the chat, unverified items flagged there). Single seed (10), fold 2020, 10 epochs,
so these results choose what to look at next. They support no claim and change no verdict. Code: `scripts/11_convlstm/diagnose_ladder.py`.

| step | what | question |
|---|---|---|
| 1 | Overfit 20 active runs (300 epochs, no quiet sampling, no EMA): original ConvLSTM, C3, matched U-Net (per-bin maps, batch 2), early-fusion U-Net (112 ch) | can the model memorize at all? An optimization problem shows as C3 stalling while the U-Net does not |
| 2 | Early-fusion U-Net: 8 hours x 14 raw channels stacked (112 ch), 2 output channels, ConvLSTM recipe (run batch 1, run-level quiet sampler, EMA), full fold-2020 train, 10 epochs | does time order help at all, and is the gap from the sampling/batching or from the recurrent path? |
| 3 | C3 with the LSTM replaced by a per-hour 1x1 conv + tanh (no recurrence; norm, pooled skips, clip kept) | does the recurrence contribute anything? |
| 4a | C3 at batch 4 runs, lr 2e-3 (4x fewer, less noisy steps) | is step noise / batch 1 binding? |
| 4b | C3 at batch 1, lr 1e-4 (published recurrent models use 1e-4 to 8e-5) | is lr 1e-3 too aggressive? |

Metrics for steps 2-4: held-out AUC-PR (plain and EMA), train-subset AUC-PR on the same 500 stratified runs used in the gate check, val/train loss ratio, gate
saturation where there is an LSTM. References: matched control 0.0464 train-subset / 0.0518 held-out; C3 0.0443 / 0.0656; original 0.0423 / 0.0573.

**Predictions, written before running (my honest priors):**
1. Step 1: the U-Net and early-fusion U-Net memorize clearly better (higher final train AUC-PR) than C3 and the original ConvLSTM: 55%. All four reach similar fit: 35%. C3 clearly best: 10%.
2. Step 2: early-fusion held-out AUC-PR within +/-15% of the control's 0.0518 (the 1-bin/2-bin difference and 8-vs-4 hours are small effects): 55%; above C3's 0.0656: 25%.
3. Step 3: the no-recurrence variant is within +/-15% of C3 on both held-out and train-subset AUC-PR: 60% (i.e. the recurrence adds nothing visible at 10 epochs).
4. Step 4: neither optimizer change lifts C3's train-subset AUC-PR above the control's 0.0464 by more than 10%: 70%. These single-seed 10-epoch runs have ~19-25% noise, so only a large jump counts as a signal.

---
## Round 1 results (read 2026-11-01) and round 2 (predictions written before running)

Round 1: both ConvLSTM variants memorize 20 runs (train AUC-PR 0.9999 / 1.0000) better than the U-Net (0.837) and early-fusion U-Net (0.983), so predictions 1 was wrong.
Early fusion held-out 0.0479 plain / 0.0518 EMA (prediction 2 held). No-recurrence C3 = C3 (0.0652 vs 0.0656; prediction 3 held). C3 batch 4 lr 2e-3 worse (0.0439 / EMA 0.0260).
**C3 at lr 1e-4 (seed 10): held-out 0.0687 / 0.0674 EMA, train-subset 0.0782 / 0.0690 (control 0.0464), gates healthy; prediction 4 wrong for lr 1e-4.**

Round 2 (same fold 2020, 10 epochs, lr 1e-4, batch 1, clip 1.0, norm + pooled skips):
- 2a: C3, seed 11 (is the lr 1e-4 fit gain reproducible?).
- 2b: C3 with no recurrence, seed 10 (does memory across hours matter once the optimizer is right?).

Predictions: (1) seed 11's train-subset AUC-PR is above the control's 0.0464 by >=20% (>= 0.0557): 65%; held-out above the control's 0.0518: 75%.
(2) 2b's train-subset AUC-PR is within +/-15% of seed 10's C3 value (0.0782): 40%; below by more than 15%: 45%; above by more than 15%: 15%.
Seed-to-seed spread is ~19-25%, so a gap smaller than that between 2a and the seed-10 run is not evidence of anything.

## Round 2 results (recorded before round 3 was run)

Fold 2020, 10 epochs, lr 1e-4, clip 1.0, batch 1. Held-out AUC-PR plain / EMA; train-subset (500 runs) plain.
- C3 seed 10 (round 1): 0.0687 / 0.0674; train 0.0782. C3 seed 11: 0.0817 / 0.0783; train 0.0760 (+64% over control's 0.0464; prediction >= +20% held).
- C3 no-recurrence: 0.0661 / 0.0688; train 0.0781 (within +-15% of seed 10; prediction 40% held). Gate stats fine (preact std 1.19, 0.84% beyond |4|).
- Reading: lr 1e-4 reproduces across two seeds; recurrence shows no added value at this budget (one seed).

## Round 3: control for the learning rate (written before it runs)

Question: how much of the lr 1e-4 gain is the learning rate itself rather than the ConvLSTM design? Run arm A's `TornadoUNet(base_channels=16)`
on the whole-bin data, fold 2020, 10 epochs, EMA 0.9999, identical to the snippet control except `--lr 1e-4` (no grad clipping, as the original control),
seeds 10 and 11 (matching the two ConvLSTM seeds). Compare held-out AUC-PR (plain) with the lr 1e-3 control (0.0518) and C3's two seeds (mean 0.0752).
Train-subset AUC-PR on the same 500 runs is computed afterwards with `gate_check_c3.py`-style scoring.

Interpretation rule (fixed now; seed noise ~19-25% means a two-seed mean is still coarse):
- U-Net lr 1e-4 two-seed mean >= 0.0639 (within 15% of C3's mean): **the learning rate explains most of the gain**; the ConvLSTM design adds nothing detectable.
- U-Net mean <= 0.0560 (C3 at least ~25% higher): the repaired ConvLSTM-U-Net is genuinely ahead of a U-Net at the same lr.
- In between: inconclusive; needs more seeds before any full study.

Predictions: P(mean >= 0.0622, i.e. +20% over the lr 1e-3 control) 45%. P(rule 1, >= 0.0639) 35%. P(rule 2, <= 0.0560) 35%. P(inconclusive) 30%.
My central guess is a modest lr gain for the U-Net too (~0.058-0.062), because the U-Net has no recurrence and 1e-3 was shown to be too aggressive only for the sequence models; I could be wrong on that.
