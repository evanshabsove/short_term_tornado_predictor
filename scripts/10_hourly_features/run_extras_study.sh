#!/usr/bin/env bash
# Experiment B (docs/hourly_features_preregistration.md): experiment A's 14 whole-bin base features (raw) + 16 extra-field channels
# (mixed-layer/most-unstable CAPE/CIN, SB LCL, 0-1 km shear, REFC, surface pressure; mean/max) + 1 derived pressure-tendency channel,
# the 17 new channels z-scored with training-split statistics (--standardize-extras; stored in each checkpoint as input_scaling).
# Same recipe, seeds 10-12, folds 2016/2020/2024, EMA 0.9999. Trained on data/interim/scaled_build_wholebin_extra/. ~8 h. Resumable.
# Usage (repo root, project venv):  caffeinate -i bash scripts/10_hourly_features/run_extras_study.sh
set -eo pipefail   # pipefail: tee must not hide a failing python
cd "$(dirname "$0")/../.."
source .venv/bin/activate   # bare `python` hits a broken pyenv shim outside the venv
OUT=models/hourly_study/extras
for SEED in ${SEEDS:-10 11 12}; do
  mkdir -p $OUT/seed$SEED
  python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $SEED --years 2016 2020 2024 --ema-decays 0.9999 \
    --staging-dir data/interim/scaled_build_wholebin_extra --manifest data/interim/scaled_build_wholebin_extra/manifest.json \
    --out-dir $OUT/seed$SEED --results $OUT/seed$SEED/results.json --standardize-extras 2>&1 | tee -a $OUT/seed$SEED/run.log
done
