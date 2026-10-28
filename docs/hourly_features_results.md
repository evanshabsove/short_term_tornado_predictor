# Hourly-features experiments A and B: results

Pre-registration: `docs/hourly_features_preregistration.md` (written before any training; sha256 recorded in both eval JSONs).
Numbers: `models/hourly_study/{wholebin,extras}/hourly_eval.json`. Scoring: `scripts/10_hourly_features/hourly_eval.py`.
Setup as registered: `TornadoUNet(base_channels=16)`, EMA 0.9999, folds 2016/2020/2024, seeds 10-12, primary metric AUC-PR of the
3-member ensemble, effect E against the mean of the 20 three-member baseline ensembles (`models/ema_study`, seeds 10-15).
No deviation from the pre-registration in recipe, folds, seeds or rules. (The first launch of A failed instantly because the runner used
a bare `python` that hit a broken pyenv shim; the runners were fixed to activate the venv and relaunched. No model had trained.)

## Verdicts (pre-registered rules)

| comparison | pooled E | per fold (2016 / 2020 / 2024) | rule | strength qualifier |
|---|---|---|---|---|
| **A vs baseline** | **+22.5%** | +31.0% / +19.3% / +17.1% | **helps** | **confirmed**: above baseline p90 in 3/3 folds (100th percentile in all), paired interval excludes 0 in 3/3 |
| **B vs baseline** | +34.2% | +37.4% / +43.4% / +21.7% | helps | confirmed (3/3, 3/3) |
| **B vs A** (the extras alone) | **+9.7%** | +4.9% / +20.2% / +4.0% | helps (>= +8%, 3/3 folds positive) | **weak: inside noise**, paired interval excludes 0 in only 1/3 folds (2020) |

No extension (seeds 13-15) was triggered: no comparison fell in the [+4%, +8%) band.

## Ensemble AUC-PR

| held-out year | baseline (mean of 20) | A | B |
|---|---|---|---|
| 2016 | 0.0407 | 0.0533 | 0.0559 |
| 2020 | 0.0577 | 0.0689 | 0.0828 |
| 2024 | 0.0720 | 0.0844 | 0.0877 |

Single-model (EMA) AUC-PR, A: 2016 0.045/0.047/0.048, 2020 0.066/0.053/0.060, 2024 0.059/0.080/0.082. Baseline singles (seeds 10-15):
2016 0.027-0.040, 2020 0.045-0.055, 2024 0.046-0.066. In 2016 every A single beats every baseline single; in 2020 and 2024 the ranges overlap
(A seed 10 in 2024 is 0.059, below five of six baseline singles), so the fold-level gain rests on the ensemble, not on every member.

## Reading the results

**A (whole-bin aggregation) is a real, large gain.** Using all four forecast hours of each bin (time-mean of the `_mean` channels, time-max of the
`_max` channels) instead of the one start-of-bin snapshot lifts the 3-member ensemble by about a fifth, in every fold, above every one of the 20
baseline ensembles. This is the largest effect found since ensembling itself, and it is far above the ~12% a 3-seed comparison can resolve.

**Timing evidence is supportive, not conclusive.** The false-alarm rate on quiet bins of active-day runs (the diagnosed weak spot), A vs. the matched
baseline ensemble:

| | 2016 | 2020 | 2024 |
|---|---|---|---|
| at 50% recall | 0.323 -> 0.193 | 0.259 -> 0.216 | 0.258 -> 0.231 |
| at 80% recall | 0.826 -> 0.609 | 0.630 -> 0.481 | 0.571 -> 0.517 |

It falls in every fold (relative drop at 50% recall: 40%, 17%, 11%), and map-level AUC-ROC rises in every fold (0.780->0.831, 0.808->0.840,
0.817->0.830). That fits the timing explanation. This metric has a large seed spread (a quiet-date version had CV ~53%), and the experiment changed
more than timing information (see limits), so it does not prove the mechanism.

**B (extra fields) adds a smaller, uncertain amount on top of A.** B vs. A is +9.7% pooled, but it is driven by 2020 (+20.2%); 2016 and 2024 are
+4-5%, each below the ~12% resolvable with 3 seeds, with paired intervals that include 0. By the pre-registered wording this is a weak "helps".
The timing metric does not improve consistently from A to B at 50% recall (0.199 vs 0.193, 0.167 vs 0.216, 0.269 vs 0.231 for B vs A in
2016/2020/2024): the extras' contribution, if real, is not visibly about timing.

**Pressure.** B bundles 17 channels (MLCAPE/MLCIN, MUCAPE/MUCIN, SB LCL, 0-1 km shear, REFC, surface pressure, and the derived pressure tendency).
Nothing here isolates pressure or its tendency, so **no claim about pressure is supported either way.** A drop-one-group ablation would be
needed (not run; not started unless asked).

## Predictions scored (from the pre-registration)

1. A, AUC-PR: priors 50% no effect / 40% helps / 10% hurts; central guess about +5%. **"Helps" happened, but the size (+22.5%) was far above my
   central guess**; my prior put 40% on helps and very little on an effect this large.
2. A, timing metric (>= 10% relative drop at 50% recall in >= 2/3 folds): **met** (40%, 17%, 11%).
3. B vs A, priors 60% no effect / 30% helps / 10% hurts: **missed in the margin** (weak "helps", one fold carries it).
4. REFC the likeliest contributor if B helps: **untested**, B cannot isolate it.
5. Pressure tendency: no visible effect expected: **untestable here**, no evidence either way.
6. Era dependence (B's effect larger in 2016/2020 than 2024, 55%): **not met**: +4.9% / +20.2% / +4.0%, no clean pattern; 2016 and 2024 look alike.
7. Stability (B's final train loss within 25% of A's in every run, 90%): **met**: ratios 0.945-1.066 across all nine runs.

## Limits (state these when citing)

- 3 seeds x 3 folds: minimum detectable effect about 12%; pooled B vs A (+9.7%) is under it, with two of three folds at +4-5%.
- A changes more than "timing": it replaces one snapshot with a 4-hour mean/max summary, which also changes the value ranges (time-max of
  the `_max` channels is more extreme) and, for bin 1, shifts from fxx 4 to the fxx 4-7 window. The experiment does not separate these. In
  particular a time-max of CAPE or SRH over four hours may carry signal a single snapshot misses even without any "timing" benefit.
- Not tested: the deployment-like 20Z stratum, other folds (only 2016/2020/2024), and whether the A gain holds on the 12-fold design.
- The 2020 fold's test period includes Dec 2020 (HRRRv4 introduction, REFC distribution break, start of the documented Great Lakes near-surface
  elevation bug). B's largest gain is in this fold; I did not test whether that is related. 2024 and 2020 training sets are partly inside the bug window.
- The bootstrap code printed numpy `matmul` divide-by-zero/overflow warnings; the intervals are finite and consistent with the point
  estimates, which suggests spurious floating-point flags from the macOS math library, not traced.
- **Serving implication, not acted on:** the production ensemble and the live feature path use the old single-snapshot features (fxx 0 and 4).
  Using A would need fxx 0-7 pulls live (about 4x the pulls) and retraining the bundle. Not done.

## What closes and what remains

Closed, as agreed: whole-bin aggregation (A) and the bundled extras (B). Open, not started: the ConvLSTM on the hourly store, standardizing
the base 14, a drop-one-group ablation of B, a 12-fold confirmation of A, and A's effect in the serving path.
