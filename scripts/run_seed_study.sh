#!/usr/bin/env bash
# Seed study: retrain the v4 14-feature U-Net CV recipe with seeds 1 and 2
# on three folds (2016, 2020, 2024; seed 0 already exists in
# models/cv_year_holdout_v4/). ~47 min per fold -> ~4.7h total. Resumable
# (cross_validate_year.py skips folds already in each seed's results.json).
# Run from the repo root inside the project venv. Usage: caffeinate -i bash scripts/run_seed_study.sh
# Seeds default to "1 2"; override with e.g.  SEEDS="3 4 5" bash scripts/run_seed_study.sh  (ensemble-size study, 3 more seeds).
set -e
UH="uh_0_2km_mean uh_0_2km_max uh_0_3km_mean uh_0_3km_max uh_2_5km_mean uh_2_5km_max uh_layers_available"
mkdir -p models/seed_study
for SEED in ${SEEDS:-1 2}; do
  mkdir -p models/seed_study/seed$SEED
  python -u -W ignore scripts/cross_validate_year.py --seed $SEED --years 2016 2020 2024 \
    --out-dir models/seed_study/seed$SEED --results models/seed_study/seed$SEED/results.json \
    --drop-columns $UH 2>&1 | tee -a models/seed_study/seed$SEED/run.log
done
