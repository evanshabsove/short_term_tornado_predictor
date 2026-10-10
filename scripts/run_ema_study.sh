#!/usr/bin/env bash
# Weight-averaging (EMA) study: v4 14-feature U-Net CV recipe, seeds 10-12, folds 2016/2020/2024, with an exponential moving
# average of the weights tracked at decay 0.9999 (primary, ~10k-step horizon ~ 4 epochs) and 0.999 (exploratory, ~1k steps).
# Each run saves the plain final model AND the averaged models, so raw-vs-EMA is a paired comparison on identical training runs.
# ~47 min per run x 9 = ~7h. Resumable. Usage: caffeinate -i bash scripts/run_ema_study.sh   (repo root, project venv)
# Override seeds/decays: SEEDS="13 14 15" DECAYS="0.9999" bash scripts/run_ema_study.sh
set -e
UH="uh_0_2km_mean uh_0_2km_max uh_0_3km_mean uh_0_3km_max uh_2_5km_mean uh_2_5km_max uh_layers_available"
mkdir -p models/ema_study
for SEED in ${SEEDS:-10 11 12}; do
  mkdir -p models/ema_study/seed$SEED
  python -u -W ignore scripts/cross_validate_year.py --seed $SEED --years 2016 2020 2024 --ema-decays ${DECAYS:-0.9999 0.999} \
    --out-dir models/ema_study/seed$SEED --results models/ema_study/seed$SEED/results.json \
    --drop-columns $UH 2>&1 | tee -a models/ema_study/seed$SEED/run.log
done
