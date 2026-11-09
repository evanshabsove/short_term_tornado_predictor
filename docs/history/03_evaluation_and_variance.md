# Held-out test, GBT, UH, year-holdout CV, diagnostics, seed variance, ensembles, EMA

> Archived verbatim from CLAUDE.md (as of commit 6d2eaed, 2026-11-08) when CLAUDE.md was compacted.  Wording, numbers and any "see above/below" references are as originally written; read them as pointers into the sections of CLAUDE.md that now live in docs/history/ (see docs/history/README.md for the map). Where a later section supersedes an earlier one, the original text says so.

### Genuine held-out test set (Tier 1 roadmap, ticket 2)

Every AUC-PR reported through ticket 5 (including the U-Net's 0.0387)
came from `training_dataset_val_full.nc` — a set checked repeatedly
across 5 architecture-selection rounds. `split.py` gained
`assign_date_split_3way`/`split_run_bins_3way`/`report_split_balance_3way`
(added alongside the existing 2-way functions, not replacing them —
nothing in tickets 1-5 needed to change), and `build_scaled.py` gained
`finalize_3way`, to build a genuine third bucket: **train (day-of-year
mod 20: 0-13, 70%) / test (14-16, 15%, new) / val (17-19, 15%)**. val's
*date-assignment rule* is unchanged from the 2-way split (mod 17-19,
same as always); test is carved from what the 2-way split called
train — days never touched by any past architecture decision, since
train was only ever used for gradient updates. No new HRRR pulls
needed — built entirely from the existing v2 staged runs via
`scripts/02_dataset_build/build_3way_split.py`, output to clearly-distinct
`training_dataset_{train,test,val}_3way.nc` files (the canonical
`_full.nc` files tickets 1-5 depend on are untouched).

- **A real inconsistency in the plan was caught by the actual numbers,
  not assumed away.** The plan claimed val's composition would stay
  "unchanged" — true for the *date-assignment rule* (verified exactly:
  with `buffer_hours=0`, the 3-way val is byte-identical to the 2-way
  val, both 980 samples) but **false once the buffer safeguard is
  applied**: protecting test from indirect val-leakage (dropping any
  val sample within 24h of a test sample) removes 182 of those 980
  val samples (18.6%), leaving **798**. User chose the more rigorous
  option: keep the buffer drop, and re-evaluate the U-Net on the new,
  smaller val too, rather than silently weakening test's protection to
  preserve exact historical comparability.
- **Real split sizes** (3,183 completed v2 runs, buffer_hours=24):
  train 4,174 / test 922 / val 798 / dropped 472 (7.4%, vs. the 2-way
  split's 6.5% — expected, there are now two boundaries needing buffer
  protection instead of one). Positive rate is consistent across all
  three (train 63.2%, test 62.6%, val 62.9% active maps) — no
  accidental imbalance.
- **U-Net evaluated on all three, via the identical `evaluate.py` code
  path**:

  | split | samples | AUC-PR | AUC-ROC | Cohen's d |
  |---|---|---|---|---|
  | val (new, 798) | 798 | 0.0398 | 0.9880 | 2.27 |
  | val (old, 980, for reference) | 980 | 0.0387 | — | — |
  | **test (new, 922, never touched before)** | 922 | **0.0836** | 0.9924 | 2.59 |

- **The new val's AUC-PR (0.0398) is reassuringly close to the old
  val's (0.0387, +2.8%)** — removing the 182 test-adjacent samples
  didn't meaningfully change the number, a good sign the buffer-drop
  isn't introducing some weird bias.
- **The genuinely new, never-touched test set scores notably
  *higher*** (0.0836, more than double val's ~0.04) — not a leakage
  red flag (leakage would show up as test scoring *worse* than a val
  set the architecture search implicitly fit to; scoring *better* on
  fresh data is the reassuring direction). Checked whether this is a
  trivial composition artifact before taking the number at face value:
  active-map rate (62.6% test vs. 62.9% val) and per-cell positive
  rate (2.01e-04 vs. 2.06e-04) are both nearly identical between the
  two splits — so the gap isn't explained by one split being an
  obviously easier class-balance mix. **The honest conclusion: at this
  project's current scale (~2,344 total dates, split 3 ways by a
  deterministic day-of-year rule), which specific storms land in which
  partition carries real variance** — a single train/val/test split
  can swing AUC-PR by 2x depending on date luck, not just model
  quality. This is itself a genuine, useful finding, not noise to
  explain away: it's direct evidence that year-based cross-validation
  (already flagged as a Tier 3 roadmap item) would give a much more
  trustworthy performance estimate than any single split, including
  this new one.
- **Not done in this ticket** (explicitly out of scope, per the plan):
  whether to promote the new, smaller (70%) train as canonical for
  future training — a separate decision, not folded in silently. No
  model has been retrained on the new train split; only the
  already-trained U-Net was evaluated against the new val/test.

### Gradient-boosted trees (Tier 1 roadmap, ticket 3) — CNN family wins, decisively

Literature found an HGBT beat a U-Net on a closely analogous CAM-grid
task — this ticket tested whether that holds here too.
`src/tornado_predictor/gbt.py` (`flatten_for_gbt`,
`downsample_negative_cells`) + `scripts/04_baselines/train_gbt.py` trained
`sklearn.ensemble.HistGradientBoostingClassifier` on
`training_dataset_train_3way.nc` flattened to one row per grid cell,
keeping every positive cell and downsampling negatives to 20:1 (9,992
positive + 199,840 negative = 209,832 rows, vs. 46.66M total — per-cell
downsampling a spatial CNN structurally can't do, see "Class imbalance
strategy" above). Evaluated on val and test, full/non-downsampled, via
the same `evaluate.compute_metrics` code path as every CNN variant.
Fit took **1.9 seconds** (HGBT's whole design point is scaling past
this dataset's size easily).

| model | val AUC-PR | test AUC-PR | val Cohen's d | test Cohen's d |
|---|---|---|---|---|
| U-Net (ticket 4/5) | 0.0398 | **0.0836** | 2.27 | 2.59 |
| **GBT (this ticket)** | 0.0186 | **0.0290** | **3.86** | **4.04** |

- **The literature's finding did not replicate here — the U-Net beats
  GBT by 2.1x (val) to 2.9x (test), decisively, not a close call.**
  Reported exactly as found, not spun: this is a real, useful answer to
  "should this project keep chasing bigger CNNs" — yes, for now.
- **Best-supported explanation, consistent with this project's own
  throughline**: every architecture-widening ticket (2-4) found that
  *receptive field* — seeing neighboring cells, not just more
  per-cell capacity — is what drove real improvement. A per-cell GBT
  has **zero** receptive field by construction: each grid cell is an
  i.i.d. tabular row with no access to its neighbors at all. The
  literature's own HGBT success likely came from richer, already
  spatially-aware engineered predictors (Sobash et al. 2020 used 174
  engineered predictors, not 14 raw pooled fields) and/or
  neighborhood-radius label matching — this project's GBT had neither.
- **A second, independent confirmation that Cohen's d and AUC-PR can
  sharply disagree** (first found comparing alpha-sweep configs, see
  "Model comparison summary" above) — this time *across model
  families*: GBT's Cohen's d (3.86-4.04) is actually *larger* than the
  U-Net's (2.27-2.59), and its mean positive-cell probability is far
  higher (~0.88 vs. the U-Net's much more modest, better-calibrated
  confidence) — yet its AUC-PR is far worse. GBT is confident and
  well-separated *on average*, but ranks enough of the ~9-10 million
  negative cells above true positives somewhere in the distribution's
  tail to hurt precision badly at any real operating point — exactly
  the failure mode a per-cell model with no spatial disambiguation
  would be expected to have (an isolated favorable-looking cell
  surrounded by unfavorable neighbors scores high but is often a false
  alarm; the U-Net can tell the difference, GBT structurally cannot).
- **Feature importance (permutation, on val) validates the literature's
  CAPE+shear+helicity emphasis**: `cape_mean`, `srh_0_3km_max`,
  `cape_max`, and `shear_0_6km_max` are the top 4 of 14, `cin_mean` is
  dead last — consistent with the STP-style severe-weather literature
  and with this project's own earlier Cohen's d findings (data
  exploration notebook: CIN was the weakest single-feature
  discriminator there too).
- **Two concrete, well-motivated follow-ups, not done here**: (1)
  engineer neighborhood-aware features for GBT (e.g. a 3x3-cell local
  average/spread per field) to give it *some* spatial context without
  building a full CNN — would directly test whether the gap is really
  about receptive field or something else; (2) re-run once ticket 1's
  UH-enriched dataset is ready — updraft helicity was the literature's
  single most-cited tornado surrogate and wasn't available for this
  run.
- Model: `models/gbt/tornado_gbt.joblib`. Results:
  `models/gbt/results.json`.

### Updraft helicity features (Tier 1 roadmap, ticket 1) — did not help

> **SUPERSEDED IN PART (see "UH re-test with averaged ensembles" below, notebook Part 7):** the result below was a single seed on a single split; with ~19% seed noise it is not a reliable negative. The multi-seed averaged-ensemble re-test found no detectable pooled effect of UH 2-5 km, and the Cohen's d / AUC-PR divergence described here did not replicate. Kept as recorded at the time.

The data pull (section above, `scripts/07_updraft_helicity/augment_dataset_fields.py`)
finished successfully: 3,182 of 3,183 runs augmented (one sustained
network outage mid-job cost 851 runs, recovered via `--retry-failed`
once connectivity returned — down to 1 genuine remaining failure).
Verified before use: 22 variables, `uh_0_2km`/`uh_0_3km` NaN fraction
(~31%) matches the known 2018-07-13 archive boundary exactly,
`uh_2_5km` has zero NaN as expected. Promoted to canonical filenames
(`training_dataset_{train,val}_full.nc` and the 3-way equivalents),
archiving the prior 14-feature v2 files as `_v2_superseded` — same
precedent as the v1→v2 promotion.

The U-Net (21 features now: the original 14 + 6 UH mean/max + the
`uh_layers_available` indicator) was retrained on the enriched 3-way
train set and evaluated on both val and test, identical recipe
(`alpha=0.25`, 50 epochs, seed=0) to every prior run:

| model | val AUC-PR | test AUC-PR | val Cohen's d | test Cohen's d |
|---|---|---|---|---|
| U-Net, 14 features (no UH) | 0.0398 | **0.0836** | 2.27 | 2.59 |
| U-Net, 16 features (UH25 only, zero NaN) | 0.0377 | 0.0528 | 2.97 | 3.01 |
| U-Net, 21 features (+UH, NaN-imputed) | 0.0359 | 0.0491 | 2.77 | 2.86 |

- **Adding the literature's single most-cited tornado surrogate made
  AUC-PR *worse*** — val -9.8%, test -41.3% — despite Cohen's d
  *improving* on both splits. Reported exactly as found: this is the
  literature's actual answer not replicating, same honest treatment as
  the GBT result above, not spun positive because it was expected to
  help.
- **A third, independent instance of the same Cohen's d / AUC-PR
  divergence** (alpha sweep; GBT vs. U-Net; now UH features) — a
  genuinely recurring pattern in this project at this point, not a
  one-off. Mean separation between positive and negative cells keeps
  improving across these three cases while rank-based precision against
  the full negative population gets worse or stays flat — a strong
  argument that any future hyperparameter/feature decision in this
  project should be judged by AUC-PR specifically, never by Cohen's d
  alone, no exceptions.
- **No overfitting signature** (final val loss 0.000114 vs. train loss
  0.000146, the same healthy pattern as every non-overfit run in this
  project) — the regression isn't explained by the model fitting worse
  to begin with.
- **Resolved by a targeted ablation (user-proposed): UH itself is the
  dominant cause, not the NaN-imputation handling.** Retrained with
  *only* `uh_2_5km` (the field available for the whole archive, zero
  NaN anywhere — eliminates the missing-data/`uh_layers_available`
  explanation entirely): test AUC-PR = 0.0528, still 36.8% below the
  no-UH baseline (0.0836) and barely above the full 21-feature version
  (0.0491, +7.5% relative). If missing-data confusion had been the main
  driver, this clean-data version should have landed close to 0.0836 —
  it didn't. The small residual gap (0.0491 → 0.0528) shows the NaN
  handling was a real but secondary contributor, not the main story.
  **This is also the cleanest instance yet of the Cohen's d / AUC-PR
  divergence**: Cohen's d rises monotonically as UH is added (2.59 →
  2.86 → 3.01, UH25-only alone reaching the *highest* separation of
  all three configs) while AUC-PR falls at every step. Likely
  explanation: UH25 is a general supercell/rotation surrogate — it's
  elevated on many strong non-tornadic storms too, so it widens the
  average positive/negative gap (truly tornadic cells have even higher
  UH) while also creating more confidently-wrong false positives among
  rotating-but-non-tornadic cells, which is precisely what hurts
  precision-sensitive AUC-PR without hurting mean separation.
- Checkpoints: `models/unet_v3features/tornado_unet_v3features.pt` (21
  features), `models/unet_uh25only/tornado_unet_uh25only.pt` (16
  features, UH25-only ablation). Results:
  `models/unet_v3features/results.json`,
  `models/unet_uh25only/results.json`. **The project's best checkpoint
  remains the original 14-feature U-Net** (`models/unet/tornado_unet.pt`,
  test AUC-PR 0.0836) — neither UH variant beats it; both are
  documented negative results, and UH (at least these layers, at this
  training budget) is not recommended as a feature for this task.

### Year-holdout cross-validation (resolves the val/test instability)

Every AUC-PR reported above came from one fixed train/val(/test) split —
and the 3-way val (0.0398) vs. test (0.0836) numbers for the identical
frozen U-Net checkpoint differed by >2x, with no way to tell whether
either was "the truth." `src/tornado_predictor/split.py` gained
`assign_year`, `available_years`, `split_run_bins_year_holdout` (additive,
same pattern as the 2-way/3-way functions); `build_scaled.py`'s duplicated
`combine()` closure was extracted into a shared, reusable
`combine_staged_runs` helper (with a real `.load()` correctness fix —
the original inline version only worked because its callers happened to
call `to_netcdf` before closing source file handles). `scripts/05_year_cv_diagnostics/cross_validate_year.py`
runs 12 leave-one-year-out folds (2014-2025) on the identical established
recipe (`TornadoUNet(base_channels=16)`, alpha=0.25, 50 epochs, seed=0),
reusing the already-staged v2 runs — no new HRRR pulls. Full results and
plots: `notebooks/cv_year_holdout_results.ipynb`.

| held-out year | test samples | AUC-PR | AUC-ROC | Cohen's d |
|---|---|---|---|---|
| 2014 | 88 | 0.0314 | 0.9881 | 2.80 |
| 2015 | 528 | 0.0276 | 0.9773 | 1.95 |
| 2016 | 484 | 0.0309 | 0.9765 | 2.37 |
| 2017 | 582 | 0.0466 | 0.9896 | 2.55 |
| 2018 | 524 | 0.0427 | 0.9887 | 2.86 |
| 2019 | 644 | 0.0362 | 0.9868 | 2.98 |
| 2020 | 524 | 0.0554 | 0.9879 | 2.43 |
| 2021 | 572 | 0.0447 | 0.9895 | 3.05 |
| 2022 | 564 | 0.0740 | 0.9902 | 2.80 |
| 2023 | 616 | 0.0526 | 0.9879 | 2.85 |
| 2024 | 660 | 0.0486 | 0.9810 | 2.66 |
| 2025 | 580 | 0.0925 | 0.9912 | 3.06 |

- **The trustworthy performance number for `models/unet/tornado_unet.pt`
  is AUC-PR = 0.0486 ± 0.0180** (12 folds, coefficient of variation
  ~37%, min 0.0276, max 0.0925) — this, not any single split, is what
  should be cited as "how good is this model" going forward.
- **The old test number (0.0836) was an unusually favorable draw, not a
  stable baseline**: z = +1.94 relative to the CV distribution, and 11 of
  12 independent year-folds score *below* it. The old val number (0.0398)
  was, by contrast, fairly representative (z = -0.49, 4/12 folds below
  it). This directly explains the >2x val/test swing that motivated this
  ticket — both numbers were just different draws from a genuinely wide
  underlying distribution; test happened to land around the 92nd
  percentile of it, not because of leakage or a stronger model, just split
  luck.
- **An unexpected, statistically significant finding**: AUC-PR correlates
  strongly with calendar year (Pearson r = 0.79, p = 0.002, still r = 0.77,
  p = 0.005 excluding the small-sample 2014 fold) — 2020-2025 averages
  0.0613 vs. 2015-2019's 0.0359, a 71% relative increase. Not something
  this project changed (identical model/features/recipe every fold) —
  leading, unconfirmed hypothesis is that it reflects HRRR's own forecast
  skill improving over its operational history (notably the HRRRv3→v4
  upgrade, operational since Dec 2020, right at this trend's inflection
  point) rather than anything about this pipeline. SPC labeling-practice
  changes over time are an uneliminated alternative explanation. Worth
  investigating before the next architecture/feature round, since it may
  mean recent years are systematically easier to predict on for reasons
  outside this project's control.
- Checkpoints: `models/cv_year_holdout/tornado_unet_heldout_<year>.pt`.
  Results: `models/cv_year_holdout/results.json`.
- **Not yet done**: re-running this CV on the quiet-date-scaled dataset or
  with pressure features once those tickets land — a natural follow-up,
  not required immediately. This CV describes the dataset as it stood
  before both.


### Post-v4 diagnostics: baselines, trend confound, and sample-mix artifacts

Cheap, evaluation-only analyses run after the v4 CV (scripts:
`scripts/04_baselines/physics_baselines.py`, `scripts/04_baselines/climatology_baseline.py`; results
`models/cv_year_holdout_v4/{physics_baselines,climatology_baseline}.json`).
Baselines use the v4 train+val files, i.e. ~7% fewer samples than the CV test
sets (buffer-dropped samples) -- comparison is approximate, not sample-identical.

- **U-Net vs. untrained baselines (per-year AUC-PR, 12 folds):** best simple
  physics baseline (STP-like composite, no LCL term, 5x5-smoothed max fields)
  averages 0.016; unsmoothed STP 0.012-0.013; CAPE x SRH 0.0075; single fields
  <=0.002; leave-one-year-out climatology (per cell, per month) ~0.0013. The v4
  U-Net (0.044) beats the best physics baseline in 12/12 years by ~3-4x, and
  beats climatology by ~30x. The U-Net/baseline ratio shows no trend with year
  (r=0.13), i.e. the model's value-add is stable.
- **The calendar-year skill trend is largely confounded with tornado-population
  composition, not cleanly attributable to HRRR:** mean SPC track length rose
  2.8 -> 4.9 mi and the EF0 share fell 0.54 -> 0.30 over 2016-2025 (year vs EF0
  share r=-0.91). Per-year AUC-PR correlates with mean track length (rho
  0.85-0.87) and EF0 share (rho -0.80 to -0.90). Controlling for track length,
  the year effect's partial r falls from 0.79 to 0.40 (p=0.19) for the old
  models and 0.87 to 0.61 (p=0.04) for v4. The untrained smoothed-STP baseline
  also drifts upward with year (r~0.55). A linear trend fits slightly better
  than a step at an HRRR upgrade boundary. 12 points cannot separate the
  explanations; do not cite the trend as evidence of HRRR forecast-skill gains.
- **Run-selection design shapes the sample mix (a sample-selection artifact,
  not a labeling bug):** 70% of quiet samples are init 20Z (the fixed quiet-date
  hour) vs. 11% of active samples, so P(active | init hour) is ~11% at 20Z but
  60-70% at 21Z/00Z. Active runs are anchored at the earliest report's hour, so
  positives skew to short lead: bin 0 has 2x the positive cells of bin 1 (9,778
  vs 4,924; active frac 60.5% vs 26.6%), and per-cell positive rate is 1.4e-4
  vs ~5e-5 over all hourly runs. Quantified below
  (`scripts/05_year_cv_diagnostics/diagnose_sample_mix.py` -> `models/cv_year_holdout_v4/
  sample_mix_diagnostics.json`; per-fold cached scores in
  `data/interim/diag_scores/`; 12 folds, both model sets, full v4 test years).

### Sample-mix, map-level, and neighborhood diagnostics (evaluation-only)

- **Raw AUC-PR is strongly prevalence-dependent; compare lift (AUC-PR / positive
  rate) within a stratum, and never across strata.** Mean per-year (v4 models):
  all samples AUC-PR 0.0442 (pos rate 1.39e-4, lift 321x); init 20Z only 0.0217
  (3.1e-5, 689x); init not-20Z 0.0481 (2.3e-4, 206x); bin 0 0.0578 (1.85e-4,
  328x); bin 1 0.0341 (9.3e-5, 368x). Lift is *not* prevalence-invariant (it falls
  as prevalence rises: dilation k=0/1/2 -> 321x/182x/111x), so it is a
  within-stratum yardstick only.
- **Time-of-day shortcut: not supported as the source of skill.** Restricting to
  init 20Z (where active and quiet maps are both ordinary) does not reduce skill
  relative to base rate -- lift is higher (689x vs 321x), not lower. Old and v4
  models behave the same. (Not a proof the shortcut is unused, only that skill
  does not collapse where it is unavailable.)
- **Short-lead skew: not supported as inflating skill either.** Bin 1 lift (368x)
  is >= bin 0's (328x); the raw AUC-PR gap (0.058 vs 0.034) is prevalence (2x
  fewer positives in bin 1), not worse ranking. The model's skill is apparently
  environment-scale, not nowcast-lead-dependent.
- **Deployment-like estimate:** the 20Z stratum (11% active maps, positive rate
  3.1e-5) is the closest available proxy to an all-hourly-runs stream (17%
  active, ~5e-5): **raw AUC-PR ~0.02 there, vs. 0.044 on the CV mix.** Quote
  ~0.02-0.03 as the deployment-like expectation, not 0.044. Proxy is imperfect.
- **Map-level discrimination is the weak link (mean over 12 years):** max
  predicted probability separates active from quiet maps with AUC-ROC only
  ~0.79-0.81 and map AUC-PR 0.71-0.72 vs. an active fraction of 0.42 (lift 1.7x).
  At 50% recall of active maps, 13.5% (old) / 14.9% (v4) of quiet maps
  false-alarm; at 80% recall, 34% / 36%. The false alarms concentrate on the
  quiet bins of *active* runs (31-33% at 50% recall, 66-67% at 80%) rather than
  on quiet-date runs (5.5-6.2% at 50% recall, 19-20% at 80%): the model mostly
  rejects ordinary quiet days but struggles with *when* during a severe day
  (timing within an active run), consistent with using a single start-of-bin
  snapshot (fxx 0/4) per 4-hour bin. Within active maps, spatial AUC-PR is 0.054
  (lift ~162-168x).
- **The extra quiet training data did not improve quiet-day false alarms**, even
  on the metric designed to see it: quiet-date-run false-alarm rate at 50%
  recall is 5.5% (old) vs 6.2% (v4); map AUC-ROC 0.805 vs 0.794 (v4 wins 4/12,
  p=0.27). No direction is significant. Quiet-date negatives are not where the
  model's errors are.
- **Neighborhood-dilated verification (scores unchanged; positive if a tornado
  touched the cell or any cell within k cells, k=1 ~ 39 km, k=2 ~ 78 km):** v4
  AUC-PR 0.044 -> 0.148 (k=1) -> 0.210 (k=2); best-F1 0.115 -> 0.237 -> 0.295;
  precision at best-F1 0.084 -> 0.199 -> 0.259; recall 0.194 -> 0.298 -> 0.349.
  Positive rate rises 6x (k=1) and 14x (k=2), so most of the AUC-PR rise is
  prevalence (lift falls 321x -> 182x -> 111x); the informative part is that
  precision goes 8% -> 20%, i.e. roughly 12% of flagged cells are adjacent
  misses within one cell. v4 vs. old stays flat (-2.0% at k=1, p=0.67; -1.4% at
  k=2, p=0.74). This is verification only -- changing the *training* labels to a
  neighborhood definition would alter the locked labeling method and needs a
  discussion first.
- **Documented in `notebooks/sample_mix_diagnostics.ipynb`** (Part 1: the diagnostics
  above; Part 2: the seed-variance study and ensembles below, both executed).

### Seed-variance study and ensembles (changes how every earlier comparison should be read)

v4 14-feature U-Net CV recipe retrained with seeds 1 and 2 on folds 2016/2020/2024
(seed 0 = the existing v4 CV model); scripts `scripts/06_seed_ensemble_ema/run_seed_study.sh`,
`scripts/06_seed_ensemble_ema/seed_study_eval.py`, `scripts/06_seed_ensemble_ema/ensemble_followup.py` (`cross_validate_year.py`
gained `--seed`); results in `models/seed_study/`. Predictions were pre-registered in the
notebook before results.

- **Seed noise is about as large as the signal in every single-seed comparison made so
  far.** Within a fold, AUC-PR of the *identical* recipe varies by a mean CV of **24%**
  across seeds (16-34%); max-min range is 32-69% of the fold mean (2024: 0.0638 / 0.0307 /
  0.0501). Pooled seed SD 0.0110 (27.9% of mean); excluding seed 0 (the member affected
  by fold selection) gives 26.5%. Estimated from 3 folds x 3 seeds (6 d.f.; 95% interval
  on the SD ~0.64x-2.2x), and the folds were selected for large v4-vs-old gaps.
- **Detectability:** a single-seed, 12-fold paired comparison can only detect an effect of
  ~**32%** (80% power); 3 seeds per fold ~18%. About half of the observed fold-to-fold
  variance of the single-seed 12-fold CV (SD 0.0149) may be seed noise (implied true
  between-fold SD ~0.010).
- **v4 vs. old, re-read:** seed 0's apparent +33% / -26% / +36% vs. the old model on the
  three folds becomes +7% / -23% / +3% (mean -4.4%) with the 3-seed v4 mean -- the per-fold
  swings were largely seed luck; the 12-fold null stands.
- **Ensembling (mean predicted probability) is the largest gain found in this project.**
  3-seed ensemble vs. the mean single seed: AUC-PR **+36%** (+33/+45/+30% per fold), best-F1
  +23%, map AUC-ROC +3.5%; it matches or beats the best single seed in 2 of 3 folds
  (+6/+23/-2%). **Across all 12 folds, the old+v4-seed-0 two-model ensemble scores AUC-PR
  0.0532 vs. 0.0461 (old) / 0.0442 (v4)** -- +18% over the mean of its members, beats both
  members in 12/12 folds, paired p=0.0002 / 0.0016; best-F1 0.124 vs 0.114/0.115.
- **Old-recipe and v4-recipe models are exchangeable:** pair-ensemble gain is +26.0%
  (old+seed) vs. +26.5% (seed+seed) (9 pairs each), reinforcing that the quiet-data
  expansion changed nothing detectable.
- **What this does to earlier conclusions (unresolved, not wrong):** single-seed rankings
  within about +/-30% -- the alpha sweep, the architecture-widening ladder (+12% to +37%
  steps), UH ablations (single split + single seed), v4 vs. old, and Part 1's map-level and
  false-alarm comparisons (quiet-date false-alarm rate has a seed CV of ~53%) -- are inside
  the noise band and cannot be called findings without repeats. Large effects survive:
  U-Net vs. GBT (2.1-2.9x), U-Net vs. physics baselines (~3-4x) and climatology (~30x).
  The calendar-year skill trend should be re-checked on ensembles (single-seed trend r=0.79;
  2-model ensemble r=0.86).
- **Instability vs. metric noise (day-block bootstrap, `scripts/05_year_cv_diagnostics/bootstrap_stability.py`,
  `scripts/06_seed_ensemble_ema/ensemble_stability.py`; notebook Part 3; decision rule written before looking):
  the models are genuinely unstable -- not "stable".** Resampling the test *dates* with
  one fixed model moves a year's AUC-PR by ~16% (relative SD; ~+/-30% 95% intervals; 2014's
  95 days far noisier) -- this metric noise is real and is NOT reduced by ensembling. But the
  seed spread (CV 16-34%) is larger and survives resampling (mean seed CV under resampling
  23/18/35%), and **7 of 9 seed pairs differ beyond day-resampling noise** (most with the
  same sign in all 2,000 resamples); old vs. v4-seed-0 differs significantly in 7/12 folds
  with mixed signs (3 +, 4 -), i.e. instability not a data effect. Exceptions: 2016 seed 1 vs 2
  and 2020 seed 0 vs 2 are within noise (2020's per-model sampling noise is ~30%).
- **Ensembles halve the instability but 2 members are not "stable":** disjoint 2-member
  ensembles disagree by ~13% vs. ~27% for single-model pairs (differ beyond noise in 4/9 vs
  12/18; one 2020 split still differs by 33%). The 2-model old+v4 ensemble beats each member
  beyond resampling noise in 8-9 of 12 folds and is never significantly worse. Caveats: days
  resampled as independent (consecutive-day correlation makes intervals somewhat too narrow);
  float16-quantized scores; ensemble stability measured only at 2 members.
- **Plan implied:** keep ensembling as the baseline (only thing tested that reduces model
  variance) but use more members (>=5 is a reasonable target; the number needed is untested);
  report 12-fold pooled means with bootstrap intervals, not single years; consider reducing
  instability at the source (EMA/SWA weight averaging, LR decay, earlier stopping -- loss
  flattened by ~epoch 5 in earlier runs -- gentler loss); verify with two disjoint 3-member
  ensembles (3 more seeds on the 3 seed folds, ~7h).
- **Ensemble-size study (notebook Part 4; seeds 3/4/5 added on folds 2016/2020/2024 ->
  six seeds per fold; `scripts/06_seed_ensemble_ema/ensemble_size.py`, results `models/seed_study/ensemble_size.json`).**
  Disjoint-ensemble disagreement (mean relative AUC-PR gap) falls **25.5% (single) -> 13.0% (2
  members) -> 9.5% (3 members)**; the share of splits differing beyond day-resampling noise falls
  56% -> 33% -> 20% (2024 alone: 50% at k=3). **Pre-registered criterion (gap <=8% AND <=20% of
  splits beyond noise) is narrowly NOT met at 3 members** (gap 9.5%). Gap x size is roughly constant
  (25.5/26.0/28.5), so by extrapolation (NOT a direct test -- disjoint splits stop at 3+3 with six
  seeds) 4 members ~6-7%, 5-6 members ~4-5%.
- **Accuracy vs. size:** mean gain over a single seed +22% (2), +30% (3), +35% (4), +37% (5), +39%
  (6); marginal gain of the k-th member +22%, +6.8%, +3.4%, +2.0%, +1.3% -- 5 members capture ~95% of
  the gain in this pool. Test-sampling noise is identical at every ensemble size (~19% bootstrap rel.
  SD): ensembling reduces model variance only. 6-seed ensembles: AUC-PR 0.042 / 0.062 / 0.070
  (2016/2020/2024), +6-12% over the 3-seed ensemble and +19% to +54% over the original single-seed CV
  models; each still has wide day-bootstrap intervals (+/-25-45% relative). All four pre-registered
  predictions held (6-member gain 39% vs. predicted 40-45%, marginally under).
- **Decision (per the rule written beforehand): use a >=5-member ensemble as the project baseline;
  variance reduction at the source (EMA/SWA, LR decay, earlier stopping) is the next experiment.**
  A 12-fold 5-member baseline would cost ~22-25h CPU (the old and v4-seed-0 models already give 2
  members per fold and were found exchangeable).
- **Weight-averaging (EMA) study (notebook Part 5; `training.train_model(ema_decays=...)`, default off, tests
  verify tracking the average does not change training; `cross_validate_year.py --ema-decays`;
  `scripts/06_seed_ensemble_ema/run_ema_study.sh`, `scripts/06_seed_ensemble_ema/ema_eval.py`, results `models/ema_study/ema_eval.json`).** Seeds 10/11/12
  on folds 2016/2020/2024; plain and averaged weights saved from the same run (paired). **Pre-registered
  outcome "works" for EMA decay 0.9999 (~4-epoch horizon):** C1 seed CV 10.4% -> 3.7% (ratio 0.36) PASS;
  C2 pooled AUC-PR +2.9% PASS; C4 disjoint-pair gap 12.7% -> 4.7% (ratio 0.37; pairs beyond test noise
  4/9 -> 0/9) PASS; **C3 FAIL (EMA >= raw in only 5/9 runs, threshold 6)** -- averaging moves models toward a common
  value (per-run changes -15% to +19%) rather than improving each one. EMA 0.999 (~0.4-epoch horizon) mostly
  does not help (CV 10.4% -> 7.7%). Prediction check: I predicted "partial"; the paired result was stronger.
- **RECALIBRATION (supersedes the ~24% / ~32% figures above and in Part 2-3 text):** pooled over **nine plain
  seeds per fold** (0-5, 10-12), plain-model seed spread of AUC-PR is **~19%** (18.8%; 19.5% excluding the
  selection-affected seed 0), not ~24-28%: the first estimate used 3 seeds on folds selected for large v4-vs-old
  gaps, and 3-seed CVs range 9%-34% across seed groups. Minimum detectable effect for a 12-fold paired
  comparison: ~21% with one seed per fold, ~12% with three. About a third (not half) of the single-seed 12-fold CV
  variance is seed noise.
- **Direct 4-member plain-ensemble test (8 unselected seeds 1-5,10-12; `scripts/06_seed_ensemble_ema/ensemble_size.py --extra-raw-seeds
  10 11 12 --exclude-s0 --max-k 4`, `models/seed_study/ensemble_size_8seeds.json`): near miss, and my extrapolation
  was too optimistic.** Disjoint plain ensembles disagree by 22.5% / 11.3% / 9.2% / 8.2% at 1/2/3/4 members, with
  50% / 30% / 26% / 25% of splits beyond test noise (2024 ~49% at k=4) -- the gap flattens, not 1/k. Gain over a
  single seed +19% / +25% / +29% at 2/3/4 members and +33-36% for all 8 (the earlier 6-seed figures
  +22/+30/+35/+37/+39% and the "5 members capture ~95%" statement were relative to that small pool and slightly
  high). A single EMA(0.9999) model's paired stability (ratio 0.37 -> ~8% when scaled to the pool's 22.5%) is
  roughly that of a 4-member plain ensemble, from one model (a scaled comparison, not a direct test).
- **Decision:** adopt EMA(0.9999) for every ensemble member (negligible cost). Untested: whether 3 EMA members
  meet the stability criterion (needs six EMA models per fold for a disjoint 3-vs-3 test: seeds 13-15 with
  EMA, ~7h). Plain ensembles need >=5 and still sit near the criterion at 4.
- **Averaged-ensemble stability test (notebook Part 6; seeds 13/14/15 with EMA 0.9999 added on folds
  2016/2020/2024 -> six averaged and six plain models per fold from the same runs, seeds 10-15;
  `scripts/06_seed_ensemble_ema/ema_eval.py --seeds 10..15`, `scripts/06_seed_ensemble_ema/ensemble_size.py --emastudy-variant {raw,ema0.9999}`; outputs
  `models/ema_study/ema_eval_6seeds.json`, `ens_size_6seeds_{raw,ema0.9999}.json`).** Disjoint-ensemble
  criterion (gap <=8% AND <=20% of splits beyond day-resampling noise), pooled over 3 folds:
  **averaged 3-vs-3 MEETS it (gap 6.9%, 6/30 = 20.0% beyond noise -- exactly at the line); plain 3-vs-3 does not
  (8.5%, 8/30 = 26.7%).** Averaged 1v1 12.7% / 2v2 7.4% (24% beyond noise, fails); plain 1v1 19.5% / 2v2 9.7%.
  Accuracy ceiling NOT raised: 6-model ensemble AUC-PR averaged 0.042/0.060/0.076 vs plain 0.045/0.065/0.073
  (pooled ratio 0.973); averaged ensembles gain less from ensembling (3-member +21/+16/+20% vs plain +27/+26/+23%).
  Predictions: 1v1 gap 5-8% NOT confirmed (12.7%); 3v3 result confirmed (narrowly); ceiling within +/-5% only pooled;
  smaller ensembling gain confirmed in direction.
- **REVISED Part 5 verdict ("works" -> "partial"), from re-evaluating EMA 0.9999 vs plain on 18 runs (six seeds
  per fold):** seed CV 16.3% -> 10.8% (ratio 0.66; C1 FAIL), pair gap 19.5% -> 12.7% (ratio 0.65; C4 FAIL), pooled
  AUC-PR +5.5% (C2 pass), EMA >= plain in 13/18 runs (C3 pass), single-model pairs beyond test noise 17/45 plain
  vs 16/45 averaged (no difference). The 9-run ratios (0.36/0.37; 4/9 -> 0/9) were optimistic -- seeds 13-15 added
  low outliers; a 3-seed spread estimate is unreliable (flagged at the time, now borne out). I originally
  predicted "partial".
- **Decision (updated):** use averaged (EMA 0.9999) members; 3 averaged members is the minimum that meets the bar
  (thin margin, thresholds are my choice), 4-5 for a safety margin; expect no higher accuracy ceiling than a plain
  ensemble of the same size. Cost of a 12-fold baseline: the 9 folds without averaged models need fresh training
  (plain old/v4 models cannot be converted): 3 members ~27 trainings (~21h), 5 members ~45 (~36h).
- **Pitfall found during the UH re-test setup (feature channel order):** `DenseGridDataset` takes channel order from the
  variable order of the *first staged run* in the combined set, and staged files from different sources order their
  variables differently (UH columns come after shear in the `scaled_build_v3` augmented copies but before it in
  natively pulled files). So a train set and a test set built from the same merged staging can disagree on channel order
  (the 14-feature studies are unaffected: with UH dropped the order is identical). `inference.assert_feature_order_matches`
  caught this in `uh_eval.py` (which sorts runs; the CV script builds its run list in manifest order). Checked directly
  for seed 10 / fold 2016: the CV script's own in-script AUC-PR (0.0326 plain / 0.0366 averaged) equals the correctly
  aligned value, whereas swapped channels would give 0.0293 / 0.0305 -- so the UH runs' in-script numbers were valid, but only
  by luck of ordering. `cross_validate_year.py` now aligns the test set to the training order (a no-op when they match) and
  `uh_eval.py` aligns each model's scoring set to its checkpoint's order, then asserts. Training itself was never affected.
- **UH re-test with averaged ensembles (notebook Part 7; UH 2-5 km only = 14 base + `uh_2_5km` mean/max = 16 features,
  EMA 0.9999, seeds 10-12, folds 2016/2020/2024; `scripts/07_updraft_helicity/run_uh_study.sh`, `scripts/07_updraft_helicity/uh_eval.py`, exploratory
  `scripts/07_updraft_helicity/uh_pairs_exploratory.py`; results `models/uh_study/uh_eval.json`, `uh_pairs_exploratory.json`) -- trained on the
  same 4,919 runs/splits as the baseline averaged models, so only the UH features differ. The original "UH hurts" result is
  NOT reproduced.** Pre-registered rule outcome: **no detectable pooled effect** (E = +0.3%; neither "hurts" nor "helps"
  met). Per fold the UH 3-member ensemble vs. the 20 baseline 3-member ensembles: **2016 +12.6%** (z=+2.7, above all 20, paired
  interval excludes 0; every UH single and pair beats baseline), **2020 +5.7%** (95th pct, n.s.), **2024 -17.5%** (z=-4.4, below
  all 20, paired interval excludes 0) -- but the 2024 loss is driven by ONE weak member (UH seed 11, AUC-PR 0.027 vs ~0.06 for
  the other two); the pair without it scores 0.0687 vs baseline 0.0690 (-0.4%) (exploratory, post hoc). A pooled zero is an
  average of opposite fold-level results; hypothesis (untested): UH from older HRRR versions carries signal that does not
  transfer to the HRRR-v4-era 2024 fold. Predictions: (1) no pooled effect confirmed, not per fold; (2) "slightly negative"
  not confirmed; (3) **Cohen's d higher for UH in >=2/3 folds NOT confirmed (1/3: +0.8%, -3.8%, -8.1%) -- the old "mean
  separation rises while AUC-PR does not" pattern did not replicate**; (4) secondary metrics partly within 10%.
  **Decision (per the pre-registered rule): stop; keep the 14 base features** (UH not shown to matter, may help in earlier
  years). Not tested: 0-2/0-3 km UH layers and the full 21-feature set. If revisited: 3 more UH seeds on 2016/2024 (6 vs 6) and
  the HRRR-v4-era folds (2022-2025).
- **UH testing finished (notebook Part 8; 3 more UH 2-5 km seeds 13-15 -> 6 UH vs the 6 baseline averaged models, plus the full 21-feature
  set (all UH layers, 0-2/0-3 km zero-filled + `uh_layers_available`), seeds 10-12; folds 2016/2020/2024; `scripts/07_updraft_helicity/run_uh_all.sh`,
  `scripts/07_updraft_helicity/run_uh21_study.sh`, `scripts/07_updraft_helicity/run_uh_part8_eval.sh`, `scripts/07_updraft_helicity/uh_eval.py --study-dir/--keep-uh-columns/--tag`; results
  `models/uh_study/uh_eval_6seeds.json`, `uh_eval_13_15.json`, `models/uh21_study/uh21_eval.json`).** Pre-registered rule outcome for both:
  **no detectable effect** (8A 6v6 pooled +4.4%, folds +9.5/+10.1/-6.3%; new seeds alone +7.1/+4.0/+1.0%; 8B 21 features +1.0%, folds
  +10.0/+1.3/-8.2%; "helps" needed pooled >= +8%). **Part 7's fold-level results do not hold up:** the 2024 -17.5% came from one weak UH
  member (seed 11); with three more seeds 2024 is +1.0% (new seeds) / -6.3% (all 6), and 2016's +12.6% shrinks to +7-10% with intervals
  including 0. Seed 11 is weak on 2024 in both UH variants (0.027, 0.033) but not in the baseline (0.063) -- unexplained. The 0-2/0-3 km layers
  add nothing detectable over 2-5 km (-2%/-4%/+11% from one weak seed). Cohen's d is the same for UH and baseline ensembles. **UH testing is closed;
  keep the 14 base features.** Untested: HRRR-v4-era folds 2022-2025 (older folds lean slightly positive, 2024 slightly negative -- hypothesis only),
  other UH treatments.
- **Working rules going forward:** report ensembles (>=3 seeds) as the headline model and
  cite single-seed AUC-PR only with a +/-25% caveat; require >=3 seeds per configuration
  (or ensemble-vs-ensemble) before treating any comparison as a finding. Cited performance
  for the 14-feature U-Net: single seed ~0.044-0.049 (+/-25%); 2-model ensemble 0.053 (12
  folds). Not measured: ensemble-vs-ensemble variance, weighted/calibrated ensembling,
  seed sensitivity of other architectures.
- **Open caveats:** these are each model's own best-F1 thresholds, 12 single-seed
  folds, v4 test years only; baselines were not re-scored under dilation or per
  stratum.


