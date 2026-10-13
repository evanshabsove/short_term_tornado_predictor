#!/usr/bin/env bash
# UH re-test, full 21-feature variant (notebook Part 8): 14 base + all UH columns (uh_0_2km, uh_0_3km, uh_2_5km mean/max + uh_layers_available).
# Missing 0-2/0-3 km values (all runs before 2018-07-13) are zero-filled by DenseGridDataset, with uh_layers_available as the indicator (same handling as the
# original 21-feature run). EMA 0.9999, seeds 10-12, folds 2016/2020/2024, same merged staging/splits as run_uh_study.sh. ~50 min per run x 9 = ~8h. Resumable.
# Usage: caffeinate -i bash scripts/run_uh21_study.sh   (repo root, project venv)
set -e
mkdir -p models/uh21_study
for SEED in ${SEEDS:-10 11 12}; do
  mkdir -p models/uh21_study/seed$SEED
  python -u -W ignore scripts/cross_validate_year.py --seed $SEED --years 2016 2020 2024 --ema-decays 0.9999 \
    --staging-dir data/interim/scaled_build_uh_merged --manifest data/interim/scaled_build_uh_merged/manifest.json \
    --out-dir models/uh21_study/seed$SEED --results models/uh21_study/seed$SEED/results.json \
    2>&1 | tee -a models/uh21_study/seed$SEED/run.log
done
