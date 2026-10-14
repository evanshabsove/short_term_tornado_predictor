#!/usr/bin/env bash
# US-only baseline (Stage B of the US-only ticket): v4 14-feature U-Net CV recipe, EMA 0.9999, seeds 10-12, folds 2016/2020/2024,
# but with Canada/Mexico/other-land cells EXCLUDED from the loss (and the in-script metrics) via --cell-mask us_trusted. Everything else is
# identical to scripts/06_seed_ensemble_ema/run_ema_study.sh, so models/ema_study/ (unmasked loss, same seeds) is the paired baseline.
# ~47 min per run x 9 = ~7h. Resumable. Usage: caffeinate -i bash scripts/08_canada_transfer/run_us_only_study.sh   (repo root, project venv)
# Override seeds: SEEDS="13 14 15" bash scripts/08_canada_transfer/run_us_only_study.sh
set -e
UH="uh_0_2km_mean uh_0_2km_max uh_0_3km_mean uh_0_3km_max uh_2_5km_mean uh_2_5km_max uh_layers_available"
mkdir -p models/us_only_study
for SEED in ${SEEDS:-10 11 12}; do
  mkdir -p models/us_only_study/seed$SEED
  python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $SEED --years 2016 2020 2024 --ema-decays 0.9999 \
    --cell-mask us_trusted --out-dir models/us_only_study/seed$SEED --results models/us_only_study/seed$SEED/results.json \
    --drop-columns $UH 2>&1 | tee -a models/us_only_study/seed$SEED/run.log
done
