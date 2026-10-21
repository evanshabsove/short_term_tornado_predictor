# scripts/

CLI entry points, grouped by project phase (numbering follows the order the
work was done; see CLAUDE.md for the findings from each phase). Run everything
from the repo root inside the project venv, e.g.
`python scripts/02_dataset_build/build_scaled_dataset.py`.

| dir | phase | scripts |
|---|---|---|
| `01_data_pipeline/` | SPC reports, grid, HRRR feature extraction, labels, one-case validation | `download_spc_tornado_reports`, `build_grid`, `verify_hrrr_grid_params`, `smoke_test_herbie`, `extract_features`, `build_labels`, `validate_documented_case` |
| `02_dataset_build/` | Pilot -> scaled -> full datasets and the train/val/test splits | `build_training_dataset`, `build_scaled_dataset`, `build_3way_split` |
| `03_architecture_search/` | Training, inference, evaluation, alpha/depth/dilation sweeps, U-Net | `train_model`, `run_inference`, `evaluate_model`, `sweep_alpha`, `sweep_architecture`, `sweep_dilation`, `train_unet` |
| `04_baselines/` | Non-CNN comparisons: GBT, physics composites, climatology | `train_gbt`, `physics_baselines`, `climatology_baseline` |
| `05_year_cv_diagnostics/` | Leave-one-year-out CV and sample-mix/bootstrap diagnostics | `cross_validate_year`, `compare_cv_old_vs_v4`, `diagnose_sample_mix`, `bootstrap_stability` |
| `06_seed_ensemble_ema/` | Seed-variance, ensemble-size/stability and EMA weight-averaging studies | `seed_study_eval`, `ensemble_followup`, `ensemble_size`, `ensemble_stability`, `ema_eval`, `run_seed_study.sh`, `run_ema_study.sh` |
| `07_updraft_helicity/` | UH data augmentation and the UH ablation studies | `augment_dataset_fields`, `run_uh_study.sh`, `run_uh21_study.sh`, `run_uh_all.sh`, `run_uh_part8_eval.sh`, `uh_eval`, `uh_pairs_exploratory` |
| `08_canada_transfer/` | Country mask and Canadian domain coverage (long-term transfer-learning goal) | `build_country_mask`, `plot_country_mask` |
| `09_serving/` | Production ensemble and live HRRR inputs for the forecast API (`api/README.md`) | `train_production_ensemble`, `live_features_smoke`, `run_forecast_job`, `run_replay`, `tier_study` |

Notes:
- Scripts in `06_` and `07_` import helpers (`diagnose_sample_mix`,
  `bootstrap_stability`) from `05_year_cv_diagnostics/`; each adds that
  directory to `sys.path` itself.
- Several scripts without `argparse` (`smoke_test_herbie`, `validate_documented_case`,
  `verify_hrrr_grid_params`, `physics_baselines`, `climatology_baseline`,
  `compare_cv_old_vs_v4`, `diagnose_sample_mix`, `ensemble_followup`,
  `ensemble_stability`, `uh_pairs_exploratory`) run immediately, so `--help`
  executes them. Don't use it to probe them.
