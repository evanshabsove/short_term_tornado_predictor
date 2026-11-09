#!/usr/bin/env bash
# Stage 1 of docs/unet_control_preregistration.md: optimizer-setting selection for arm A on fold 2018 (outside the study folds), seeds 20 and 21,
# early stopping as in the early-stopping study. Three lanes (one per setting, 4 threads each), each running seed 20 then 21. Resumable. Plug in; launch detached under caffeinate -i.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=models/unet_control/select
mkdir -p $OUT
shasum -a 256 docs/unet_control_preregistration.md >> $OUT/prereg_sha_at_launch.txt
lane() { NAME=$1; shift
  for SEED in 20 21; do
    D=$OUT/$NAME/seed$SEED; [ -f $D/results.json ] && continue; mkdir -p $D
    OMP_NUM_THREADS=4 python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $SEED --years 2018 --ema-decays 0.9999 --early-stopping --patience 8 --min-epochs 10 "$@" \
      --staging-dir data/interim/scaled_build_wholebin --manifest data/interim/scaled_build_wholebin/manifest.json \
      --scratch-dir data/interim/cv_scratch_uc_${NAME}_s$SEED --out-dir $D --results $D/results.json 2>&1 | tee -a $D/run.log
  done; }
lane U0 --lr 1e-3 &
lane U1 --lr 3e-4 --grad-clip-norm 1.0 &
lane U2 --lr 1e-4 --grad-clip-norm 1.0 &
wait
echo UNET_CONTROL_SELECT_DONE | tee -a $OUT/run.log
