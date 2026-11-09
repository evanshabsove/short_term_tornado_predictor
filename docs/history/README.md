# History archive index

Everything that used to live in CLAUDE.md (as of commit 6d2eaed, 2026-11-08), moved verbatim. CLAUDE.md holds the current rules, conclusions and a "where to look" table; this folder holds the full evidence (tables, numbers, predictions scored, superseded findings).

To find something: `grep -rn "keyword" docs/history/` or open the file below whose headings match. Pre-registrations and results for the later studies are separate files in `docs/` (see CLAUDE.md).

## [00_locked_decisions_detail.md](00_locked_decisions_detail.md)

- Methodology (decided, follow unless told otherwise)
- Grid definition (locked in)
- Time binning (locked in)
- Feature extraction (locked in)
- Labeling (locked in)
- Train/val split (locked in)

## [01_dataset_builds.md](01_dataset_builds.md)

- Consolidated training dataset (pilot + first scale-up)
- First scale-up (44 train / 6 val samples)
- Full-scale build v1 (superseded — see v2 below)
- Full-scale build v2 (report-covering runs, canonical)
- Full-scale build v4 (expanded quiet dates + one retry pass)

## [02_architecture_search.md](02_architecture_search.md)

- Full-scale training run (first statistically real generalization result)
- Alpha sweep (resolved — corrects earlier "lower alpha" guidance)
- Model comparison summary (standardized metrics, all runs)
- Architecture widening, ticket 2: width vs. depth ablation
- Architecture widening, ticket 3: dilated convolutions (new best result)
- Architecture widening, ticket 4: a real U-Net (new best result)
- Architecture widening, ticket 5: full comparison (concludes this round)

## [03_evaluation_and_variance.md](03_evaluation_and_variance.md)

- Genuine held-out test set (Tier 1 roadmap, ticket 2)
- Gradient-boosted trees (Tier 1 roadmap, ticket 3) — CNN family wins, decisively
- Updraft helicity features (Tier 1 roadmap, ticket 1) — did not help
- Year-holdout cross-validation (resolves the val/test instability)
- Post-v4 diagnostics: baselines, trend confound, and sample-mix artifacts
- Sample-mix, map-level, and neighborhood diagnostics (evaluation-only)
- Seed-variance study and ensembles (changes how every earlier comparison should be read)

## [04_hourly_and_convlstm.md](04_hourly_and_convlstm.md)

- Hourly features: whole-bin aggregation (A) and extra fields (B) -- closed
- ConvLSTM-U-Net on the hourly store vs. arm A -- negative result, with a diagnosed defect
- Repaired ConvLSTM (C3) vs. arm A -- no detectable effect
- Repaired ConvLSTM + dropout -- no detectable effect (study closed)
- Per-epoch held-out metrics (one fold, one seed) -- the ConvLSTM overfits, arm A does not
- Early-stopped ConvLSTM (C3) vs. early-stopped arm A -- helps (confirmed)
- Tuned arm-A U-Net control vs. early-stopped C3 -- helps (confirmed)

## [05_pipeline_mechanics.md](05_pipeline_mechanics.md)

- Validation against a documented case
- Class imbalance strategy (active)
- Baseline CNN architecture (implemented, first pass)
- Training loop (first run complete)
- Inference + sanity check (training-set fit only)
- Case study: prediction vs. detection (mechanics demo, not accuracy)

## [06_canada_and_us_only.md](06_canada_and_us_only.md)

- Country mask and Canadian coverage (Canada-transfer groundwork, not a modeling change)
- US-only baseline (mask non-US cells from loss and metrics): Stage A and Stage B done

## [07_forecast_api.md](07_forecast_api.md)

- Forecast API for the Rails app (side track; tickets 1-8 of 11 done)

## [08_old_current_status.md](08_old_current_status.md)

- Current status

