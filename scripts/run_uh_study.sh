#!/usr/bin/env bash
# UH re-test (see notebooks/sample_mix_diagnostics.ipynb, Part 7): v4 U-Net CV recipe with the 14 base features PLUS uh_2_5km mean/max
# (16 features; zero missing values in every run), EMA weight averaging at decay 0.9999 (the plain final model is saved too), seeds 10-12,
# folds 2016/2020/2024. Trained on data/interim/scaled_build_uh_merged/ -- symlinks to the UH-augmented copy (scaled_build_v3/) for the 3,182
# original runs and to the native 21-feature files (scaled_build_v2/) for the 1,737 newer runs: the SAME 4,919 runs, splits and 14 base
# feature values as the baseline averaged models (models/ema_study/, seeds 10-15), so the comparison isolates the UH features.
# ~50 min per run x 9 = ~8h. Resumable. Usage: caffeinate -i bash scripts/run_uh_study.sh   (repo root, project venv)
set -e
DROP="uh_0_2km_mean uh_0_2km_max uh_0_3km_mean uh_0_3km_max uh_layers_available"
mkdir -p models/uh_study
for SEED in ${SEEDS:-10 11 12}; do
  mkdir -p models/uh_study/seed$SEED
  python -u -W ignore scripts/cross_validate_year.py --seed $SEED --years 2016 2020 2024 --ema-decays 0.9999 \
    --staging-dir data/interim/scaled_build_uh_merged --manifest data/interim/scaled_build_uh_merged/manifest.json \
    --out-dir models/uh_study/seed$SEED --results models/uh_study/seed$SEED/results.json \
    --drop-columns $DROP 2>&1 | tee -a models/uh_study/seed$SEED/run.log
done
