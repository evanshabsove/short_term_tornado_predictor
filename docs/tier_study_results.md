# Tier study (forecast API ticket 4): results

Run 2026-10-20 with `scripts/09_serving/tier_study.py` (2,000 day-block bootstrap draws, seed 0) exactly as written in
`docs/tier_study_preregistration.md` (sha256 `c2aaaa90a557...`, recorded in `models/tiers/tier_study.json`, which holds
every number below). Scores are a **proxy**: A = 2-model leave-one-year-out ensemble, 12 years; B = 5-member EMA
ensembles on 2016/2020/2024. The served ensemble has no held-out scores. Primary sample: the init-20Z stratum
(4,410 samples, 2,205 dates). Hit rates are per cell: the share of a tier's cells that had a tornado touch the cell or
one of its 8 neighbors (~40 km), in that 4-hour bin.

Before the real run I did a 20-draw smoke run and saw its log lines (chosen operator, cutoffs). The rules were fixed
and did not change. I also checked the table machinery against brute-force numpy/sklearn on three real years: AUC-PR
identical to 15 digits, tier area/capture/hit rate identical.

## Decisions the pre-registered rules produced

| decision | result |
|---|---|
| Operator | **max, 3x3.** AUC-PR vs k=1 labels 0.0782 (max) vs 0.0812 (mean); paired difference +0.0030, 95% interval [-0.0008, +0.0070] includes 0, so the default (max) applies. |
| High cutoff (best F1) | **0.190** (F1 0.164, precision 0.139, dilated recall 0.201) |
| Elevated cutoff (60% capture) | **0.122** (the 15% area cap did not bind) |
| Transfer to a 5-member EMA ensemble | **Holds**, so the served cutoffs are A's fixed scores. Median area ratio B/A at the same cutoffs: Elevated 1.14, High 1.24 (limits 0.75-1.33). |
| Calibrated "% chance" | **Not defensible by the rule**; `calibrated_percent` stays `null`. C1 failed; C2, C3, C4 passed (details below). |
| `tier_set_id` | `ts-e2c497b6` |

## What each tier measures (A, primary sample, with 95% intervals)

| tier | area share of cells | share of tornado cells it holds | hit rate |
|---|---|---|---|
| Low | 99.77% | **39.9%** [35.7, 44.9] | 0.012% [0.010, 0.014] |
| Elevated (exclusive) | 0.200% [0.186, 0.214] | 33.7% [29.4, 38.0] | 3.9% [3.3, 4.4] |
| High | 0.035% [0.031, 0.040] | 26.3% [21.4, 31.2] | **13.9%** [11.7, 16.0] |
| Elevated or High | 0.235% [0.218, 0.254] | 60.1% [55.1, 64.3] | 5.4% [4.7, 6.1] |

- **Low is not safe: it contains 40% of tornado cells.** The model misses a lot; the wording rules below follow from this.
- Area shares are tiny because they count every cell of every map, and 89% of 20Z maps are quiet. About **10.8%** of
  20Z maps contain at least one High cell [9.7, 11.9]; of those, **44.6%** [39.8, 49.5] had a tornado somewhere in the
  covered area in that bin (the base rate for any map is ~11%). That is a map-level statement and a different one from the per-cell hit rate.
- **Robustness:** the prior-shift-weighted full mix gives the same picture (High hit rate 15.1% [14.1, 16.1], Elevated
  3.8%; High area 0.082%); the two agree within the pre-registered 2x. On B's three folds at the same cutoffs the High hit
  rate is 13.9% [10.3, 17.6] and the High tier holds 34% of tornado cells [26, 43].
- **The pooled hit rate hides a large era shift.** High-tier hit rate is **10.3% in 2015-2020 vs 19.8% in 2021-2025**
  (per-year 7.5-12.5% vs 16-22%). This is the same calendar trend CLAUDE.md already flags (partly confounded with the
  tornado population changing); it is not new evidence about HRRR. A single pooled hit rate overstates the early years and understates the recent ones.

## Stability

- **S1, model noise (upper bound; 3-member ensembles, 10 disjoint 3-vs-3 splits x 3 folds):** two disjoint
  ensembles flag the same High cells **55%** of the time (Jaccard 0.554, range 0.52-0.58 across splits), the same
  Elevated-or-High cells 71%. They agree on whether a map has any High cell in 95.6% of maps, but only 69% of maps
  flagged by either are flagged by both. So **which** cells are High is noticeably model-dependent even though the
  amount of High is not. The served ensemble has 5 members, so it should be somewhat steadier; untested.
- **S2, hour to hour (anecdotal; 20 run pairs x 2 bins = 40 rows):** 0-2.1% of covered cells change tier between
  consecutive runs; where both runs have High cells, the overlap ranges from 0 to 0.80 (several 0.0). Far too few pairs to say more. The live archive
  is the place to measure this.

## Calibration (leave-one-year-out isotonic on the smoothed score, k=1 dilated target, 7 score-quantile bins)

Reliability is **much better than I expected**: predicted/observed ratios of 1.03, 1.01, 0.98, 1.02, 1.02, 1.00 in bins 1-6
(bin 0 has only 16 positives). The top bin (the top 0.1% of cells) has an observed ~8.8% dilated rate and the
calibrated prediction is 8.8%.

| criterion | result |
|---|---|
| C1: top 4 bins, ratio in [0.7, 1.4] **and** bootstrap interval contains 1 | **FAIL.** Ratios are 0.98-1.02, but the interval excludes 1 in bin 3 (0.979 [0.953, 0.997]) and bin 5 (1.016 [1.003, 1.037]) by less than 0.01. |
| C2: High tier within [0.5, 2.0] in >= 9 eligible years | pass (11 of 11 eligible) |
| C3: A's map on B's 5-member scores, ratios in [0.7, 1.4] | pass (High 0.93, Elevated 1.04) |
| C4: weighted full mix, High ratio in [0.5, 2.0] | pass (0.99) |

By the pre-registered rule (all four must pass) a calibrated percentage is **not** defensible, and I have not made one.
That outcome rests on a strict interval-contains-1 condition that the near-perfect point ratios narrowly miss; I am not
changing the rule after seeing the result. Reasons to stay cautious regardless of that: (1) the High-tier ratio **drifts
with year**: 1.1-1.9 in 2015-2020 (the map over-forecasts) and 0.58-0.80 in 2021-2025 (it under-forecasts); Spearman
-0.73 (p = 0.011) -- a map fit on all years would be off by 30-50% in either era; (2) the target is a 4-hour bin and ~40 km, not SPC's daily 25-mile
definition; (3) the scores are a proxy ensemble and one init-hour stratum. If a percentage is ever wanted, the era drift is the
problem to solve first, and a revised criterion should be set before looking, not now.

## Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 | no significant max-vs-mean difference, so max | exactly that (interval includes 0) | met. Note mean was ahead in 6 of 6 B subsets (AUC-PR ~0.091 vs 0.088; subsets overlap, so this is not 6 independent votes) and by 3.8% on A. |
| P2 | High area 0.3-3%, hit 12-35%, capture 10-35% | area **0.035%**, hit 13.9%, capture 26.3% | **partly missed:** hit rate and capture were in range, area was ~10x smaller than I predicted. |
| P3 | the 15% Elevated area cap binds | Elevated captured 60% at 0.235% area; the cap did not bind | **wrong.** I expected Elevated to be 8-15% of the map; it is 0.2%. |
| P4 | fixed cutoffs do not transfer to the 5-member EMA ensemble | they transfer (High ratio 1.24 against a 1.33 limit) | **wrong**, though High is close to the edge. |
| P5 | calibration fails at least one criterion, most likely C3 or C4 | fails C1 only; C3 and C4 pass | right on "fails", wrong on which. |
| P6 | S1 High Jaccard 0.35-0.70 | 0.554 | met |

## What this does and does not support

- The tiers are real and ordered: hit rate rises about 1,200-fold from Low (0.012% [0.010, 0.014]) to High (13.9%), and the High
  tier's 13.9% is consistent across the A, weighted-mix and B checks.
- It does **not** support calling any tier "likely": about 1 in 7 High cells verified in the 4-hour bin, and 4 in 10 tornado cells sat in Low.
- Tier areas are small. Expect most hours to show no High cells at all (about 11% of 20Z maps have one).
- Not measured: the served ensemble itself, real hour-to-hour flicker, other init hours, any Canadian skill.
- Wording and the filled `GET /tiers` numbers are in `api/README.md`.
