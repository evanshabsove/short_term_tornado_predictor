# Hourly-features experiments A and B: pre-registration

Written 2026-10-27, **before any model was trained on the whole-bin data and before any score from it exists.**
What was inspected beforehand, and only that: how many runs were downloaded and which failed; sample counts, class balance,
channel order, NaN counts and per-channel value ranges of the whole-bin datasets; label identity with the v4 dataset
(identical on every shared sample); and the extra fields' raw value ranges on seven dates. No model output on this data.
The baseline models' scores (`models/ema_study/`, seeds 10-15) were known beforehand: they are the comparison.
`models/hourly_study/*/hourly_eval.json` records this file's sha256, so a later edit is visible.

## 1. Question

The diagnostics found the U-Net is decent at *which* days are dangerous and weak at *when* within a day: false alarms
concentrate on the quiet bins of active-day runs. Every model so far describes a 4-hour bin with one forecast hour (fxx 0 for
bin 0, fxx 4 for bin 1). Hypothesis: using all 4 hours of each bin helps, mainly on timing. A second question: do extra
HRRR fields chosen from the literature (docs/feature_candidates_research.md), including the user's pressure idea, add skill?
This line of work then stops (conclusion for knowledge); a ConvLSTM on the hourly store is a separate, later project.

## 2. Arms

| arm | features | scaling | data |
|---|---|---|---|
| **baseline** (exists) | 14 base, one snapshot per bin | raw | `scaled_build_v2` (4,919 runs); EMA 0.9999 models, seeds 10-15 |
| **A: whole-bin** | the same 14 channels, each built from all 4 hours of its bin: time-mean of each `_mean` channel, time-max of each `_max` channel | raw | `scaled_build_wholebin` (4,916 runs) |
| **B: + extras** | A's 14, + 16 extra channels (mixed-layer and most-unstable CAPE and CIN, surface-based LCL, 0-1 km shear, simulated composite reflectivity, surface pressure; mean and max, aggregated like A), + 1 derived channel `pres_sfc_tend` (cell-mean surface pressure at the bin's last hour minus its first) | the 17 new channels z-scored with training-split statistics; the base 14 stay raw | `scaled_build_wholebin_extra` |

Decided with the user: the base 14 stay raw (the model has never normalized inputs; changing that is a separate ticket and would
confound these comparisons). The new channels are scaled because surface pressure (~96,000 Pa) would otherwise swamp the
network (shown in `tests/test_input_scaling.py`: first-layer outputs >100x larger unscaled).

Everything else is identical across arms: `TornadoUNet(base_channels=16)`, focal loss alpha 0.25 gamma 2.0, lr 1e-3, batch 2,
`quiet_keep_fraction` 0.2, 50 epochs, EMA 0.9999 (the EMA weights are what is scored), year-holdout folds **2016, 2020, 2024**
with the same 24 h buffer, seeds **10, 11, 12** (same seed numbers as the matching baseline models).

**Known differences, disclosed:** arms train on 4,916 runs vs. the baseline's 4,919 (3 archive-gap runs from 2015, ~6 of ~9,100
training samples; none is in a test fold, so test samples are identical; asserted by comparing sample order with the cached
baseline scores). Pressure and other near-surface fields have a documented HRRR Great Lakes elevation bug from 2020-12-02 to
2022-07-19 (SCN 22-68); the 2024 fold's training data and the 2020 fold's test data are partly inside that window. REFC has a
documented distribution break at HRRRv4 (2020-12). Neither is corrected for.

**Data note (checked before training, no model involved):** `pres_sfc_tend` is heavy-tailed. Median |value| is 92 Pa per 3 h,
99th percentile 426 Pa, 99.99th 900 Pa, but the extremes reach -3,295 / +4,120 Pa (33-41 hPa in 3 h). They are real: the largest
samples are hurricane landfall dates (Ida 2021-08-29, Helene 2024-09-27, Florence 2018-09-12, Ian 2022-09-28), and the 4,327
cells beyond 1,000 Pa (0.005% of cells, 335 samples) are mostly over water or Canada/the NE domain edge. They are kept: no
clipping or winsorizing is applied, so after z-scoring these cells sit many standard deviations out. The other extra channels are
likewise z-scored without clipping.

## 3. Comparisons and metric

Primary metric: **AUC-PR of the 3-member ensemble** (mean predicted probability of seeds 10-12), per fold, on the held-out year.
- **Comparison 1 (A vs baseline) and 2 (B vs baseline):** effect `E_fold = arm ensemble / mean of the 20 three-member baseline
  ensembles (all 3-subsets of seeds 10-15) - 1`; **pooled E = mean of the three E_fold.**
- **Comparison 3 (B vs A), the decisive one for the extras:** `E_fold = B ensemble / A ensemble - 1`, pooled the same way.
- Reported for every comparison: the matched baseline ensemble (seeds 10-12) with a paired day-bootstrap 95% interval (1,000
  resamples of test days), the arm's percentile within the 20 baseline ensembles, per-model single-seed AUC-PR.

## 4. Decision rules (fixed now, mirror the UH studies' Part 8)

For each comparison: **"helps"** if pooled E >= +8% **and** at least 2 of 3 folds have E > 0. **"Hurts"**: pooled E <= -8% and at
least 2 of 3 folds E < 0. Otherwise **"no detectable effect"**.
- **Extension:** if an arm's pooled E is in [+4%, +8%) with >= 2 of 3 folds positive, that arm gets seeds 13-15 (6 members vs. the 6
  baseline models, the 6-vs-6 comparison of Part 8) and the rule is applied once more to the 6-vs-6 result. Nothing else triggers extra
  seeds, extra folds or a re-analysis.
- **Strength wording** (does not change the verdict): a "helps" may be called *confirmed* only if the arm ensemble exceeds the
  baseline ensembles' 90th percentile in >= 2 folds (comparisons 1-2) or the paired interval excludes zero in >= 2 folds (all); otherwise
  it is reported as *weak: inside the baseline's noise*.
- **Honest limit, stated in advance:** with 3 seeds per fold the minimum detectable effect is ~12% (measured earlier), larger
  than the 8% threshold. "No detectable effect" therefore does not mean "no effect", and a "helps" without the strength qualifier
  should be read as suggestive. The thresholds are my choice, copied from the previous studies.
- A and B are never pooled, and B's channels are never tested individually in this round (B bundles all extras; a positive B would
  need a drop-one-group follow-up to attribute it, and that only starts if the user asks).

## 5. Secondary metrics (descriptive, no decision rule)

Best-F1 and its precision/recall, Cohen's d, map-level AUC-ROC, 1-cell-dilated AUC-PR, and the hypothesis-specific
**false-alarm rate on quiet bins of active-day runs at 50% and 80% recall of active maps** (`diagnose_sample_mix.map_level`). That rate
has a large seed spread (a quiet-date version had CV ~53%), so only a large, consistent drop would be mentioned as support for the timing
hypothesis.

## 6. Predictions, with my honest priors (fixed before results)

1. **A, AUC-PR:** most likely a small gain. P(no detectable effect) 50%, P(helps) 40%, P(hurts) 10%; central guess pooled E about +5%.
2. **A, timing metric:** the quiet-bins-of-active-runs false-alarm rate at 50% recall falls by >= 10% (relative) in >= 2 of 3 folds: 40%.
   (If A helps AUC-PR but not this, the gain is not about timing.)
3. **B vs A:** most likely no detectable extra effect. P(no detectable) 60%, P(helps) 30%, P(hurts) 10%. The extras add 17 channels to a
   model that has overfit before when given capacity; 3 seeds cannot resolve a small gain.
4. **Reflectivity is the likeliest contributor** if B helps (the only extra aimed at timing; the literature's explicit-storm fields were
   the most important inputs in the closest HRRR study), but it cannot be tested alone here.
5. **Pressure tendency:** I expect no visible effect (no study tested forecast pressure tendency; the closest HRRR CNN ranked pressure
   level among its least important inputs). B cannot isolate it; a null for B does not rule out a tendency effect, and a positive B does not
   credit it.
6. **Era dependence:** B's effect is larger in 2016/2020 than in 2024 (REFC break, HRRRv4 era): 55%.
7. **Stability:** B trains without divergence (final train loss within 25% of A's in every run): 90%.

## 7. Checks before trusting results

- A one-fold smoke run (`--years 2016`, few epochs) per arm to confirm the pipeline, the checkpoint's `input_scaling`, and
  that scored test samples line up with the cached baseline scores.
- The evaluation reproduces a baseline number from cached scores (baseline matched-ensemble AUC-PR equals the previously reported
  per-fold value) before any arm is scored.
- Any deviation from this document (a failed run re-run with the same seed, a changed recipe, a bug fix) is written in the results
  doc with the reason, not applied silently.

## 8. What happens after

Report every result exactly as measured, including nulls and any prediction that missed, in `docs/hourly_features_results.md`, then
update `CLAUDE.md` and the walkthrough. The line of work then closes; the hourly stores (8 forecast hours per run, plus the extras) stay on
disk as the input for the planned ConvLSTM.
