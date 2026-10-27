#!/usr/bin/env bash
# Experiment A (docs/hourly_features_preregistration.md): the 14 base features, each built from all 4 forecast hours of its bin
# (time-mean of _mean, time-max of _max), raw inputs, EMA weight averaging at 0.9999 (the plain model is saved too), seeds 10-12,
# folds 2016/2020/2024, the v4 U-Net recipe -- identical to the baseline averaged models (models/ema_study/, seeds 10-15) except for
# the features. Trained on data/interim/scaled_build_wholebin/ (4,916 runs). ~50 min per training x 9 = ~8 h. Resumable.
# Both studies share data/interim/cv_scratch, so run A and B one after the other, never together.
# Usage (repo root, project venv):  caffeinate -i bash scripts/10_hourly_features/run_wholebin_study.sh
#   extension seeds (only if the pre-registered rule calls for it):  SEEDS="13 14 15" bash scripts/10_hourly_features/run_wholebin_study.sh
set -eo pipefail   # pipefail: tee must not hide a failing python
cd "$(dirname "$0")/../.."
source .venv/bin/activate   # bare `python` hits a broken pyenv shim outside the venv
OUT=models/hourly_study/wholebin
for SEED in ${SEEDS:-10 11 12}; do
  mkdir -p $OUT/seed$SEED
  python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $SEED --years 2016 2020 2024 --ema-decays 0.9999 \
    --staging-dir data/interim/scaled_build_wholebin --manifest data/interim/scaled_build_wholebin/manifest.json \
    --out-dir $OUT/seed$SEED --results $OUT/seed$SEED/results.json 2>&1 | tee -a $OUT/seed$SEED/run.log
done
