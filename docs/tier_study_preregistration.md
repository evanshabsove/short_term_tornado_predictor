# Tier study (forecast API ticket 4): pre-registration

Written 2026-10-20, **before any model score from `data/interim/diag_scores/` was loaded.**
What was inspected beforehand, and only that: sample counts per year and init hour, label counts, array shapes,
the number of consecutive-hour run pairs, and SPC's probability definition (below). Nothing about how any model
scores. `models/tiers/tier_study.json` records this file's sha256, so a later edit is visible.

Goal: turn the ensemble's raw per-cell probabilities into Low / Elevated / High tiers on a neighborhood-smoothed
score, say what each tier measurably means, measure how stable the tiers are, and decide whether a calibrated
"% chance" is defensible. Report everything as measured, including bad news. No retraining, no label changes.

## 1. Data (decided with the user: "A, plus a B check")

The served model is a 5-member all-years ensemble with **no held-out scores**, so every number here comes from proxies.

| name | scores | years | role |
|---|---|---|---|
| **A** | mean of the `old` and `v4` single-seed leave-one-year-out models (a 2-model ensemble) | all 12 (2014-2025), 9,838 (run, bin) samples | fit the cutoffs, set the operator, measure the table |
| **B** | EMA 0.9999 models, seeds 10-15 (the production recipe), leave-one-year-out | 2016, 2020, 2024 only (2,620 samples) | check that A's results transfer to a 5-member EMA ensemble |

B's 5-member ensembles are the 6 leave-one-out subsets of the 6 seeds (they overlap, so they are not independent;
the spread across them is a lower bound on seed noise). Scores are float16 on disk; ensembles are averaged in float32.

**Everything below is for a proxy ensemble, not the served one.** The served ensemble trains on all 12 years and has
more data per member than any A or B model; its score scale is assumed to match B's (same recipe), nothing more.

## 2. Fixed definitions (not decisions)

- **Coverage:** cells with country code water (0) or US (1) per `cell_mask.us_trusted_mask` (9,269 cells). Every other
  cell is tier -1 and excluded from every statistic. Water cells are inside the denominator (secondary figure: US land only).
- **Smoothing** is applied to the full-grid scores (including non-US cells, whose outputs are real forecasts of weather
  there), then coverage-masked. Max filter: square window `(2r+1)^2`, `mode="nearest"` at the grid edge. Mean filter:
  average over the in-grid cells of the window only (no zero padding).
- **Dilated label (radius k):** a cell is positive if a tornado touched it or any cell in the `(2k+1)^2` window around it.
  Zero padding outside the grid. This is the window `diagnose_sample_mix.py` already used for k=1,2.
- **Product radius = 1 cell (3x3 window, ~39 km).** SPC defines its tornado probabilities as the chance of a tornado
  within 25 miles (~40 km) of a point ([SPC probabilistic outlook info](https://www.spc.noaa.gov/products/outlook/archive/2004/probinfo.html),
  checked 2026-10-20). One cell is the nearest match. Caveats: SPC's window is a whole outlook day, ours is a 4-hour bin,
  and a 3x3 window reaches ~55 km diagonally. Radius 2 is reported as a sensitivity result only; it is **not** chosen by
  AUC-PR, because dilated prevalence rises ~6x (k=1) and ~14x (k=2), so AUC-PR across radii compares prevalences.
- **Statistics per tier** (pooled over (sample, cell) pairs, equal weight per sample, tier by cumulative cutoff "at or
  above" and also exclusive):
  - `area_share`: fraction of coverage cells in the tier.
  - `capture_share`: fraction of undilated tornado cells (coverage cells with a positive label) inside the tier.
  - `hit_rate`: fraction of the tier's cells that are dilated-positive (k=1). The undilated rate is reported beside it.
  - Map-level, High tier only: fraction of maps with at least one High cell, and of those the fraction with a tornado
    anywhere in coverage.
- **Uncertainty:** day-block bootstrap (resample UTC dates, 2,000 draws, seed 0, 95% percentile intervals), as in
  `scripts/05_year_cv_diagnostics/bootstrap_stability.py`. Consecutive days are correlated, so intervals are somewhat too narrow.
- `tier_set_id` = `ts-` + first 8 hex of sha256 of the JSON `{operator, radius, cutoffs rounded to 6 significant figures}`.

## 3. Primary sample and sensitivity

- **Primary: the init-20Z stratum** (4,410 samples, 493 active maps (11%), 1,581 tornado cells over 12 years). CLAUDE.md
  identifies it as the closest available proxy to an all-hourly-runs stream. It is small, so tier hit rates for rare
  tiers will have wide intervals; that is reported, not hidden.
- **Sensitivity: full mix with a prior-shift weight.** All 9,838 samples, active maps weighted `0.17/0.432`, quiet maps
  `0.83/0.568`, so the active-map share matches the ~17% of an all-hourly-runs stream (CLAUDE.md "Class imbalance
  strategy"). Reported beside the primary; it never replaces it. If the two disagree by more than 2x on any tier's
  hit rate, say so plainly.

## 4. Decision 1: smoothing operator

Candidates at radius 1: **max** and **mean**. Comparison on A, primary sample, pooled over all 12 years, metric =
AUC-PR of the smoothed score against **k=1 dilated labels** (same labels for both, so the prevalence is identical and
the comparison is fair), paired day-block bootstrap of (mean - max).
**Rule:** if the 95% interval of the difference is entirely above 0, use mean; entirely below 0, use max; otherwise use
**max** (simplest; it is the neighborhood-max verification the project already ran, and it reads as "within ~25 miles").
Also reported, no decision attached: raw (no smoothing) vs dilated labels, radius 2 pair, and whether the ordering
reverses on B's three folds.

## 5. Decision 2: tier cutoffs (A, primary sample, chosen operator, radius 1)

Two cutoffs on the smoothed score `s`, found as operating points rather than round numbers:
- **High** = the best-F1 threshold, where precision and recall are both computed against k=1 dilated labels
  (`dilated_recall` here is only the quantity used to find this point).
- **Elevated** = the highest threshold at which the cumulative `capture_share` reaches **60%**, subject to two guards:
  (a) if that implies a cumulative area share above **15%** of coverage cells, use the 15%-area quantile instead;
  (b) if it lands at or above the High cutoff (High alone already captures 60%), use the cutoff whose cumulative area
  share is 3x High's.
- Low = everything covered and below the Elevated cutoff. A cell with score exactly equal to a cutoff goes up a tier.
- The cutoffs are recorded both as fixed scores and as the area shares they imply.

## 6. Decision 3: does A's table transfer to a 5-member EMA ensemble? (the B check)

On B's three folds pooled, primary sample, for each of the 6 five-member subsets:
apply **A's fixed-score cutoffs** to its smoothed score and measure the cumulative area share of Elevated and of High.
Reference = the area share of the same cutoffs on **A restricted to those same 3 folds** (so year mix does not confuse it).
- **Transfer holds** if, for both tiers, the median over the 6 subsets of (B area share / reference) is in **[0.75, 1.33]**.
  Then serving cutoffs = A's fixed scores, and the published table is A's.
- **Otherwise** serving cutoffs are **quantile-matched on B**: for each subset, the score whose cumulative area share
  equals the reference share; take the median over subsets. The published table is then re-measured on B (3 folds, wider
  intervals) and A's is shown beside it.
Either way, the table's A and B versions are both written to the results JSON. The served ensemble's own scale remains
unverified; the live archive's High-tier area share (a prospective check) is the monitor for that, and is out of this ticket.

## 7. Stability

- **S1, model noise (decision-free):** on B, the 10 disjoint 3-vs-3 splits of the 6 members x 3 folds. Each 3-member
  ensemble gets the same cumulative area-share cutoffs (High and Elevated, as in section 5) applied by quantile on the
  pooled B primary sample, so only ranking disagreement is measured, not scale. Report: Jaccard overlap of the High sets and
  of the (at-or-above Elevated) sets, the share of the union flagged by only one side, and map-level agreement on "any High
  cell". 3-member ensembles are noisier than the served 5-member, so this is an **upper bound** on model-noise instability.
- **S2, real hour-to-hour change:** the staged data holds only **58 consecutive-hour run pairs in total (20 in B's folds)**,
  too few to measure. Reported for those 20 only, labelled anecdotal, no decision. The real measurement is deferred to the
  live archive `data/live/issued/` once it holds enough hourly pairs (target: 14+ days); that is not part of this ticket.

## 8. Calibration check: is a "% chance" defensible?

Target definition: probability of a tornado within one cell (~40 km, SPC's 25 miles) in the 4-hour bin, i.e. the k=1
dilated label. Procedure: **leave-one-year-out isotonic regression** (fit on 11 years of A, primary sample, applied to the
held-out year, pooled), on the chosen smoothed score. Reliability by 7 bins with edges at the 0, 0.5, 0.9, 0.97, 0.99,
0.997, 0.999 and 1.0 quantiles of the pooled coverage-cell score distribution (see Errata 1).
A calibrated percentage is defensible only if **all four** hold:
- **C1:** in each of the top 4 bins, pooled predicted/observed ratio is in **[0.7, 1.4]** and its day-bootstrap 95%
  interval contains 1.
- **C2:** in the High tier, the ratio is in **[0.5, 2.0] in at least 9 of the 12** held-out years (years with fewer than
  10 observed positives in the tier do not count; if fewer than 9 years are eligible, C2 fails).
- **C3:** A's isotonic map applied to B's 5-member scores (through the Decision 3 rule's cutoff mapping if it fired) gives
  High and Elevated ratios in **[0.7, 1.4]**.
- **C4:** on the prior-shift-weighted full mix, the High ratio is in **[0.5, 2.0]**.
If any fails: `calibrated_percent` stays `null`, tiers are the only display, and the failed criterion is reported.
Even on a pass, any percentage would carry the "~40 km, 4-hour window" wording, not SPC's daily definition.

## 9. Wording rules (draft; final text after results)

- High = "highest relative risk in this forecast", never "likely"; a High cell does not usually verify.
- Low is **not** "safe" and not "low chance": tornadoes occur in Low cells; say "lower relative risk".
- Canada/Mexico/other land = "no model coverage", never Low.
- Hit rates, if shown, read "historically, about X% of High cells had a tornado within ~40 km in the 4-hour window",
  with the interval, the sample (proxy ensemble, 20Z stratum) and the 4-hour-versus-daily caveat.

## 10. Predictions (written before looking; scored in the results)

- **P1:** no significant max-vs-mean difference, so max is chosen by default (~60% confident).
- **P2:** High cumulative area share 0.3-3% of coverage cells; k=1 hit rate 12-35%; capture share 10-35%.
- **P3:** the Elevated 15%-area cap binds (~55%); otherwise Elevated area is 8-15%.
- **P4:** fixed-score cutoffs do **not** transfer to the 5-member EMA ensemble (the rule fires, quantile matching is
  used) (~65%); I expect the 5-member EMA mean to have fewer extreme cells than the 2-model mean.
- **P5:** calibration fails at least one of C1-C4 (~85%), most likely C3 or C4.
- **P6:** S1 High-tier Jaccard overlap between disjoint 3-v-3 ensembles is 0.35-0.70.

## 11. What this study cannot tell us

- The served ensemble's own tier shares and hit rates (it has no held-out scores): only the proxies'.
- Real hour-to-hour tier flicker (S2 is deferred to the live archive).
- Skill in Canada (no labels exist there).
- Whether 20Z-stratum behavior carries to every hour of the day; the weighted full mix is only a rough check.

## 12. Outputs

`src/tornado_predictor/tiers.py` (smoothing, tier assignment, tier stats; unit-tested), `scripts/09_serving/tier_study.py`,
`models/tiers/tier_study.json` (all numbers above, this file's sha256), a short conclusion in CLAUDE.md, and the filled-in
`GET /tiers` numbers in `api/README.md` once the results are in.

## Errata (made before any score was loaded; each is a correction of my own text, not a change of a decision)

1. **Section 8 said "8 bins" but listed 8 edges, which define 7 bins.** The bins are `[0,0.5)`, `[0.5,0.9)`, `[0.9,0.97)`,
   `[0.97,0.99)`, `[0.99,0.997)`, `[0.997,0.999)`, `[0.999,1.0]` (score quantiles), and "the top 4 bins" of C1 are the four
   highest: from the 0.97 quantile up. Because scores are float16 with many exact ties, a level is assigned to the bin by
   its cumulative area share counted from the top (the first edge that is at least that share).
2. **C3 implementation detail:** A's isotonic map is fit on the 9 years that are *not* B's folds (all years except 2016,
   2020, 2024), so the label years being scored are not also used to fit the map. This is stricter than fitting on all 12.
3. **Scores are quantized to float16 after smoothing** (relative resolution ~1e-3), and cutoffs are float16 levels; a cell
   goes up a tier when its score is at or above the cutoff. Tables are built on these quantized scores, so every
   number in the results JSON uses the same quantization.
