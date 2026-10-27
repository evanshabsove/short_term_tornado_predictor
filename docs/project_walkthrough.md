# Project Walkthrough: How This Got Built, and Why

A guided tour of this project's full arc — every major piece, in the order
it was built, with the reasoning behind each decision and the real bugs/
findings along the way. If `CLAUDE.md` is the reference manual, this is
the story that explains why the manual says what it says.

*Last updated 2026-10-22. Sections 1-9 cover the build-out and the first
round of modeling; sections 10-14 cover everything since (cross-validation,
seed noise and ensembles, diagnostics, the Canada groundwork, and the
forecast API). Section 15 is the short "where are we" summary.*

## 1. What this project actually is

**Goal**: predict whether a tornado will touch down in a given ~40km
grid cell, 0-8 hours in the future, using HRRR (a numerical weather
model) forecast fields as input.

**Not to be confused with** the sibling `tornet` project, which
*detects* tornadoes happening right now from live radar. Different
problem: forecasting ahead of time (this project, using coarse
atmospheric fields) vs. confirming something already visible (radar
signatures).

**Long-term direction**: a forecaster-aid tool first (now partly built, see
section 13), then possibly a transfer to Canada (groundwork in section 12).

**Methodology**: "dense-grid classification," following a real paper
(Sobash et al.). The key design choice, locked in from day one: every
`(grid cell, time window)` pair is one training sample — exhaustively,
across the whole domain — not just cells near known storms. This is
different from `tornet`'s approach (crops built around storm objects).
It means the model has to learn to say "no" correctly across millions
of boring, storm-free cells, which is where most of this project's
hard problems come from (see section 7).

## 2. The spatial grid

HRRR's native grid is ~3km resolution over the continental US (and
parts of Canada/Mexico/ocean at the edges). That's too fine-grained and
too large to train on directly, so it gets coarsened to ~40km cells.

**The decision that mattered**: coarsen by *exact pixel-block averaging*
(13×13 native pixels → one 39km cell), not by reprojecting/interpolating
onto an independently-designed grid. This keeps pooling mathematically
exact — a cell's value really is the mean/max of the native pixels it
contains, zero interpolation error. The tradeoff: the grid's shape is
whatever 13×13 blocks happen to tile the native domain (81×138 cells,
not a clean number), and it extends over ocean/Mexico/Canada at the
edges rather than being clipped to a US land boundary. That "weird"
NW-corner-reaches-134°W behavior looked like a bug the first time it
showed up — it isn't; it's genuine map-projection curvature, verified
by computing the domain's actual corners directly rather than assuming.

## 3. Time binning

Each HRRR run forecasts 0-8 hours ahead, split into two 4-hour windows
(bin 0: hours 0-4, bin 1: hours 4-8).

**The non-obvious decision**: *every single hourly HRRR run* gets its
own pair of bins — not just the 4 standard "synoptic" runs a day
(00/06/12/18Z). This was deliberate, to maximize how many training
samples exist, and it has a real consequence: one tornado report can
become a positive label in up to 8 different runs' sample sets
(whichever runs' 0-8h windows happen to contain it). That's not
double-counting or a bug — it's the intended per-run sampling design,
and it's proven mathematically (`candidate_run_inits` always returns
exactly 8 runs for any report).

**Training period**: ~Sept 2014 onward. HRRR's AWS archive technically
starts mid-August 2014, but the first ~2 weeks are missing CAPE, CIN,
helicity and shear (only a reduced ~57-field set exists), so the real
usable start is **September 1, 2014**. The upper end is set by SPC's
labels, not HRRR; the SPC file originally ended 2025-09-09 and now runs
to 2025-12-29.

## 4. Labels: turning SPC reports into 1s and 0s

Ground truth comes from SPC (Storm Prediction Center) tornado reports —
touchdown point, track length, damage-path width. 15,294 reports for
2014-2025.

**The decision**: credit a report to *every* cell its damage path
crosses, not just the touchdown cell. A tornado's track gets buffered
(by half its damage width) into a corridor polygon, then intersected
against each candidate cell's exact projected boundary. A long-track
tornado (the Dec 2021 Quad-State/Mayfield EF4, ~165 miles) ends up
touching 13 cells this way, not 1.

**The permanent simplification**: SPC has no per-point track timing —
just one touchdown timestamp for the whole track. So spatial matching
uses the full track, but temporal matching collapses to that one
instant. There's no way to fix this without inventing a fake
track-speed model, so it's documented as a real, permanent limitation
rather than something to silently paper over.

**Storage**: only positive labels are stored (a sparse table). Anything
not listed is an implicit 0 — the full 0-space (every cell × every run
× every bin) is billions of rows and was never worth materializing
directly.

## 5. Features: what the model actually sees

HRRR forecast fields, pulled via a library called Herbie, pooled onto
the coarse grid (mean *and* max per field — max catches a sharp local
spike that mean would dilute).

**The 7 base fields** (14 channels after mean/max): CAPE, CIN, 0-1km
and 0-3km storm-relative helicity, 0-6km bulk shear, 2m temperature,
2m dewpoint. Every search string was verified against a *live* HRRR
index file before being trusted — a loose `"CAPE:"` search, for
example, would also match HRRR's other CAPE variants, so exactness
mattered. (Shear is 0-6km, not 0-8km, because HRRR simply has no 0-8km
field.)

**A real scope tradeoff, not an oversight**: each 4-hour bin is
represented by just its *first* hour's snapshot (fxx 0 for bin 0, fxx 4
for bin 1), not an aggregate over all 4 hours. Pulling and averaging all
4 would be 4x the network cost for the whole dataset — deliberately
deferred, and later diagnostics (section 10) suggest it may be part of why
the model struggles with *when* during a severe day.

**Updraft helicity (UH) was added, tested thoroughly, and dropped** — the
full story is in section 11. The 14 base features are what every current
model uses.

## 6. Scaling the dataset up

The path: a 6-sample pilot → a 50-sample stratified scale-up → full build v1
→ a *corrected* full build v2 → a quiet-date expansion, v4 (the current one).

**v1's real bug, found by actually looking at the data, not assumed
away**: v1 pulled exactly one HRRR run per active date, fixed at 20:00
UTC. Auditing found this missed a real tornado on **27.2% of active
dates entirely** — most days with a tornado have multiple reports
spread across many hours, and a single fixed 8-hour window often just
doesn't cover when it happened.

**v2's fix**: a greedy algorithm that covers every report on a date
with the minimum number of runs needed (most dates still need just 1
run; busy days get 2-3). The dataset build script now operates on
"runs" as the unit of work, not "dates." v2 gave 5,078 train / 980 val
samples with ~63% of maps active.

**v4's expansion**: v2 only had 300 randomly sampled quiet dates. v4
added *every* remaining quiet date — 4,919 runs total, 7,652 train / 1,478
val samples, active maps down to ~43-44% (expected), per-cell positive rate
1.44e-4. After a retry pass 131 runs still failed (mostly the 2014-2018
archive, including 107 whose `.idx` file is simply absent from AWS).
**Result: the ~1,700 added quiet runs gave no measurable gain** — and no
measurable harm either (see section 9 for how that was established).

## 7. Handling extreme class imbalance

Positive cells are rare — roughly 1 in 7,000 cells at v4 scale (1 in
4,700 at v2 scale), even though a large share of *maps* (grid × time
window) contain at least one positive cell. That distinction shaped the
whole strategy:

- **FocalLoss**, not plain cross-entropy — down-weights "easy" examples
  (the millions of obviously-quiet cells) so the loss focuses on the
  genuinely hard/rare positive cases.
- **Map-level downsampling**, not per-cell — you can't drop arbitrary
  pixels out of a spatial CNN's input without breaking its
  convolutional receptive field, so instead: keep every map with at
  least one positive cell, and only train on a fraction of the
  all-negative maps (re-chosen each epoch).

**A real direction-of-reasoning bug**: an early note said the fix for
under-confident positive predictions would be a *lower* `alpha` in
FocalLoss. That turned out to be backwards — `alpha` directly weights the
positive class, so a *higher* alpha raises positive-class confidence. A
5-run sweep confirmed it. But then a *second* correction: judging "best
alpha" by mean-separation (Cohen's d) said 0.9 won — until AUC-PR found
the *opposite*: the original `alpha=0.25` was best and 0.9 the *worst*.
**The default is still alpha=0.25.** Both corrections are kept in the docs
because the disagreement is itself a lesson (see section 9). One more
caveat added later: that sweep was single-seed, and section 9 shows
single-seed differences of that size are inside the noise.

## 8. Models tried, in order, and what each one taught

| model | what it is | key result |
|---|---|---|
| **Baseline CNN** | 3 plain conv layers, 32 channels, no pooling | First working end-to-end model. Tiny "receptive field" (~270km). AUC-PR 0.0283. |
| **Width/depth ablation** | Same architecture, wider and deeper | Widening channels alone *didn't help*. Adding depth *did* — receptive field, not raw capacity. The 128-channel/5-layer variant overfit (the only train/val divergence in the project). |
| **Dilated convolutions** | Same layer count, but layers skip pixels at increasing spacing | Bigger receptive field (~1,200km) for *fewer* parameters than the deeper CNN, no overfitting. AUC-PR 0.0322. |
| **U-Net** | Downsample → compress → upsample with skip connections, 451K parameters | **Best architecture.** AUC-PR 0.0387 on the original val set, despite ~14x more parameters than the dilated CNN. |
| **Gradient-boosted trees** | No CNN — per-grid-cell tabular rows | **Lost decisively** (2-3x worse than the U-Net). A per-cell model has *zero* receptive field by construction. |

The thread running through all of it: **receptive field mattered every
time it was tested; raw parameter count, on its own, mostly didn't.**

**An honest caveat that arrived later (section 9)**: the *ordering* of the
three CNN steps (+12% to +37% each) came from single-seed runs, and
single-seed noise turned out to be ~19-25%. Large effects survived
re-reading (U-Net vs. GBT, U-Net vs. physics baselines, U-Net vs.
climatology); the small rungs of the architecture ladder are suggestive,
not proven.

## 9. How results get measured — and the long road to a number worth trusting

Lessons, in the order they were learned:

1. **Accuracy is meaningless here.** Every model scores >99.8%, including
   one that always predicts "no tornado."
2. **AUC-ROC is misleadingly optimistic** (0.92-0.99 for everything).
   **AUC-PR (average precision)** is the honest metric; it ranges ~0.02-0.09.
   Always judge decisions by AUC-PR, not mean-separation metrics like Cohen's d.
3. **One validation set, checked repeatedly, becomes optimistic.** So a
   genuine train/test/val 3-way split was built. The U-Net scored **0.0836
   on the new test set vs. 0.0398 on val** — a >2x gap with no leakage.
4. **Year-holdout cross-validation explained the gap.** 12 leave-one-year-out
   folds (2014-2025) gave **AUC-PR = 0.0486 ± 0.0180** for the same model.
   The 0.0836 was a lucky draw (92nd percentile of the folds), not a stable
   number. **This CV number, not any single split, is what to cite.** The v4
   data gives about the same (0.0442 ± 0.0142; statistically
   indistinguishable once scored on identical samples).
5. **Seed noise is as big as most of the effects being chased.** Retraining
   the *identical* recipe with different seeds moves a fold's AUC-PR by
   ~19% (nine seeds per fold; an earlier 3-seed estimate of ~24-28% was
   too high, and was corrected). A single-seed, 12-fold comparison can
   only detect an effect of ~21%; three seeds per fold gets that to ~12%.
   Day-block bootstrapping confirmed the models are genuinely *unstable*
   run to run, not just noisily measured.
6. **Ensembling is the biggest gain found in the whole project.** Averaging
   the predictions of 3 seeds beats the average single seed by ~30% AUC-PR
   (the old + v4 two-model ensemble beat both members in 12/12 folds:
   0.0532 vs. 0.0461/0.0442). Marginal gains fade after ~4-5 members.
7. **EMA weight averaging helps stability, less than it first looked.**
   Averaging weights over training (EMA 0.9999) first looked like it cut
   run-to-run spread by ~64%; with more seeds, the verdict was revised
   from "works" to "partial" (~34% reduction, pooled +5.5% AUC-PR). Three
   averaged members meet the pre-registered stability bar, narrowly; it
   does not raise the accuracy ceiling.

The working rules that fell out of this: report ensembles (>=3 seeds) as
the headline; cite single-seed AUC-PR only with a ±25% caveat; require
>=3 seeds (or ensemble vs. ensemble) before calling any comparison a
finding. Decisions and predictions were increasingly **written down before
looking at results** (pre-registration), including the tier study in
section 13.

## 10. Diagnostics: what the model is and isn't good at

All evaluation-only (no retraining):

- **It really beats simple baselines.** vs. the best untrained physics
  composite (an STP-like index, 5x5-smoothed): ~3-4x better AUC-PR in
  12/12 years. vs. per-cell monthly climatology: ~30x.
- **The "skill rises with calendar year" trend (r=0.79) is confounded.**
  It was first guessed to reflect HRRR improving. But over 2016-2025 mean SPC
  track length rose 2.8 → 4.9 miles and the weak-EF0 share fell
  0.54 → 0.30, and AUC-PR tracks those closely. Controlling for track length
  shrinks the year effect a lot. 12 points can't settle it: don't cite the
  trend as HRRR skill gains.
- **Sample-mix artifacts exist** (positives skew to short lead because
  active runs are anchored at the earliest report; 70% of quiet samples
  are 20Z). Neither a time-of-day shortcut nor a short-lead inflation was
  supported as the source of skill. But raw AUC-PR depends heavily on
  prevalence, so the **deployment-like expectation is ~0.02-0.03, not
  0.044**.
- **Map-level discrimination is the weak link.** Max predicted probability
  separates active from quiet maps only ~0.79-0.81 AUC-ROC. False alarms
  concentrate on the quiet *bins of active-day runs* — the model mostly
  rejects ordinary quiet days but struggles with *when* during a severe
  day, consistent with the single-snapshot-per-bin simplification.
- **Spatial near-misses are common.** Counting a hit if a tornado touched the
  cell *or a neighbor*, AUC-PR goes 0.044 → 0.148 (1 cell) → 0.210 (2
  cells); precision at best-F1 goes 8% → 20%. Mostly prevalence, but about
  12% of flagged cells are adjacent misses. (Verification only — the
  training labels are unchanged.)
- **The extra quiet data didn't reduce quiet-day false alarms**, even on the
  metric built to detect it.

## 11. Updraft helicity: a cautionary tale about single results

Literature review said UH is the most-cited tornado/supercell surrogate.
Findings along the way:

- It's available on HRRR (`MXUPHL`), but the 0-2km and 0-3km layers only
  from **2018-07-13** (found by binary search; matches the HRRRv3
  upgrade). Only 2-5km covers the whole archive. Gaps were stored as honest
  `NaN` plus a `uh_layers_available` indicator, imputed to 0 only inside
  `DenseGridDataset`.
- **First result (single seed, single split): UH made AUC-PR *worse*** (test
  0.0836 → 0.0491), while Cohen's d went up. This was written up as the third
  case of "Cohen's d up, AUC-PR down."
- **Multi-seed averaged ensembles did not reproduce it.** Over two rounds
  (up to 6 UH vs. 6 baseline models per fold, plus the full 21-feature
  set): no detectable pooled effect (+4.4%, +1.0%; "helps" needed >=+8%).
  A scary-looking 2024 result (-17.5%) came from one weak seed and
  vanished with more seeds. The Cohen's d divergence didn't replicate.
- **Decision (pre-registered rule): keep the 14 base features; UH testing
  is closed.** Untested: the newest HRRR-v4-era folds, other UH treatments.
- **A pitfall found on the way:** channel order in `DenseGridDataset` follows
  the first staged file, and differently-sourced files order their UH
  columns differently. It never affected training or the 14-feature
  studies, but `assert_feature_order_matches` caught it and the CV and UH
  scripts now align order explicitly.

The lesson isn't "UH is useless"; it's that the first, single-seed answer
was not a reliable negative — and neither would a single-seed positive
have been.

## 12. Canada groundwork and the US-only baseline

Motivated by the long-term Canada goal. Nothing here changes the US model's
scope.

- **Country mask** (`hrrr_coarse_grid_stride13_countries.nc`): classifies all
  11,178 cells — 5,148 US, 1,243 Canada, 659 Mexico, 4,121 water. HRRR's
  domain stops ~52.5N, so ~17% of Canada by area is covered, and central
  Alberta sits at the edge. *Real bug caught by looking at the plot*: the
  first build assumed Natural Earth country polygons exclude the Great
  Lakes; they don't, so lake cells came out as land. Fixed by subtracting
  the lakes layer.
- **Label noise**: SPC only reports US tornadoes, so Canadian/Mexican cells
  (~17% of the grid) are always labeled 0 — essentially noise.
- **Does it matter? Not detectably, so far.** Stage A (rescoring existing
  models with those cells excluded from the *metric*): AUC-PR +0.4-0.7%.
  Stage B (retraining with those cells excluded from the *loss* too, 3
  seeds x 3 folds): no detectable effect (pooled +1.7%). It's a valid clean
  US-only baseline, not a better model. Masked-trained models give higher
  probability over Canadian cells — expected, but not a measure of skill
  there (no Canadian labels to check).
- **Next, not started**: Northern Tornadoes Project labels and HRDPS
  (Canada's convection-allowing model; different projection, 4 runs/day, a
  format change in 2023) — see `docs/canada_data_availability.md`. An access
  inquiry to NTP was sent; no reply recorded.

## 13. The forecast API (side track toward the forecaster-aid tool)

Goal: serve the U-Net ensemble to the sibling Rails app
(`../tornado_prediction_radar`) through a Flask API modeled on `tornet`'s.
Eleven tickets (`docs/forecast_api_tickets.md`); **eight are done**.

| ticket | what it is | state |
|---|---|---|
| 1 | Production bundle: 5-member EMA ensemble (seeds 20-24) on all years | done |
| 2 | Live HRRR feature extraction, with a train/serve parity test | done |
| 3 | Disk-backed forecast store (raw probabilities, never only tiers) | done |
| 4 | Risk tiers + a calibration decision (pre-registered study) | done |
| 5 | Hourly scheduler job | done |
| 6 | API contract (`api/README.md`) | done |
| 7 | Flask API (read-only, :5060) | done |
| 8 | Replay of past cases + SPC reports endpoint | done |
| 9 | Rails forecast page | **not started** |
| 10 | Rails verification overlay | **not started** |
| 11 | Packaging and deployment (cron/launchd, country mask at startup) | **not started** |

Things worth knowing:

- **The production ensemble has no held-out score.** It trains on every
  year, so the only honest expectation is the 12-fold CV of the same recipe
  (single seed ~0.044-0.049, deployment-like ~0.02-0.03). Mechanics checks
  only: training loss matched the CV runs, members agree (map correlation
  0.90-0.95).
- **Live path verified against training.** The live-extracted features
  reproduce the staged training features for 2019- and 2023-era runs
  (rtol 1e-6). HRRR timing: a run is usable ~1h after init, so bin 0 is
  already ~1h old when issued.
- **Three tiers, no percentages.** Low / Elevated (>=0.122) / High
  (>=0.190) on a 3x3 max-smoothed ensemble score, plus a grey "no model
  coverage" state for Canada/Mexico. Measured on proxy scores per cell per
  4-hour bin: **High hits 13.9% [11.7, 16.0]**, Elevated 3.9%, Low 0.012% —
  but **Low still contains ~40% of tornado cells** (the model misses a lot),
  and Elevated+High together hold ~60%. High-tier hit rate shifted from 10%
  (2015-2020) to 20% (2021-2025). The percent-chance calibration check
  *narrowly failed* its pre-registered rule (a bootstrap interval missed 1
  by <0.01 in two bins), and the rule was not relaxed after the fact, so
  `calibrated_percent` stays `null`.
- **Pre-registered predictions: mixed.** 2 of 6 met; the predicted size of the High
  tier was off by ~10x (observed 0.035% of cells).
- **The API only serves what's stored.** Replays are computed offline
  (`run_replay.py`), not in a request handler — a deliberate departure from
  the ticket's compute-on-request idea. Six replays exist (Mayfield
  2021-12-10 at 21/22/23Z; 2024-04-26 at 18/20/22Z). Those dates are
  *inside the training years*, so their good-looking hit rates (83-100% of
  touchdowns near an Elevated/High cell) are plumbing checks, **not
  skill**.
- **Verified over real HTTP**: ~2-28 ms responses, ~157 MB RSS, 200 KB per
  forecast. The two real live forecasts so far were quiet-night maps with no
  Elevated/High cells, so tier boundaries are covered by synthetic tests,
  not a real severe live case.
- Nothing is on a timer yet, so `/status` reports `stale` until the job is
  scheduled.

## 14. Honest limitations, in one place

- Absolute skill is low: AUC-PR ~0.02-0.05 (about 100-300x the no-skill
  baseline), and ~40% of tornado cells fall in the Low tier.
- No held-out score for the served ensemble itself.
- Single representative forecast hour per 4-hour bin; timing within a
  severe day is the weak spot.
- Most pre-ensemble comparisons (alpha, architecture rungs, v4 vs. old) are
  within seed noise; no architecture has had its own hyperparameter tuning.
- The temporal label simplification (one timestamp per track) is permanent.
- Canada/Mexico are unlabeled; Canadian skill is unmeasured.
- Calendar-year trend and tier hit-rate drift (10% → 20%) are not
  understood, partly tornado-population composition.

## 15. Where things stand right now

- **Done**: sections 2-13 above.
- **Dataset**: v4, 4,919 runs, 14 features (canonical files are
  `training_dataset_{train,val}_full_v4.nc`; the UH columns inside are
  mostly NaN, so drop them when training).
- **Best model**: still the original 14-feature U-Net recipe, headlined as
  an **ensemble**. Cited performance: single seed 0.044-0.049 (±25%), 2-model
  ensemble 0.053 (12 folds), deployment-like ~0.02-0.03.
- **Served**: a 5-member EMA ensemble (bundle `3c02381f1611`) behind a
  read-only Flask API; hourly job written and verified, not yet scheduled.
- **Next, on the API track**: Rails forecast page (9), verification overlay
  (10), deployment (11).
- **Open modeling questions** (none started): neighborhood-aware features
  for the GBT, aggregating all 4 forecast hours per bin (the likeliest fix
  for the timing weakness), per-architecture hyperparameter tuning and
  U-Net width/depth sweeps, pressure-tendency features, Canadian labels
  (NTP) and HRDPS features.

## 16. The meta-lesson, if there's one

Almost every real correction in this project's history came from
*checking a claim against live data* instead of trusting a plausible-
sounding assumption: the grid's "wrong-looking" NW corner, the HRRR
archive's true start date, the alpha-direction mix-up (twice, in two
different directions), the UH archive gap, the run-selection bug that
quietly dropped 27% of active dates, the Great Lakes missing from the
country mask, and the lucky 0.0836 test score.

The second half of the project added a companion lesson: **a number from
one run isn't a finding.** The UH "hurts" result, the alpha ranking, the
architecture ladder and the rising-skill-over-time trend each looked solid
from a single seed or split and weakened or reversed once there were more
seeds, folds, or a closer look at confounds. Pre-registering the rules
beforehand, averaging seeds, and writing down misses (including
predictions that turned out wrong) is what turned the later results from plausible
stories into things worth standing behind.
