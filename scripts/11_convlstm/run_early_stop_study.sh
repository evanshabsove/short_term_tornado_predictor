#!/usr/bin/env bash
# docs/convlstm_early_stopping_preregistration.md: arm A and the repaired ConvLSTM (C3, no dropout), BOTH with early stopping on a mod-20 validation split of the
# training dates (patience 8 on EMA 0.9999 validation AUC-PR, min 10 epochs, cap 50, best weights restored), folds 2016/2020/2024 x seeds 10-12.
# Three parallel lanes (one per seed, 4 threads each); within a lane, per fold: arm A then C3. Resumable (a finished run is skipped). Plug in.
# Launch detached: python subprocess.Popen(..., start_new_session=True) under caffeinate -i.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=models/convlstm/early_stop
mkdir -p $OUT
shasum -a 256 docs/convlstm_early_stopping_preregistration.md >> $OUT/prereg_sha_at_launch.txt
lane() { SEED=$1
  for FOLD in 2016 2020 2024; do
    DA=$OUT/arm_a/seed$SEED; mkdir -p $DA
    OMP_NUM_THREADS=4 python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $SEED --years $FOLD --ema-decays 0.9999 --early-stopping --patience 8 --min-epochs 10 \
      --staging-dir data/interim/scaled_build_wholebin --manifest data/interim/scaled_build_wholebin/manifest.json \
      --scratch-dir data/interim/cv_scratch_es_s$SEED --out-dir $DA --results $DA/results.json 2>&1 | tee -a $DA/run.log
    DC=$OUT/c3/seed$SEED/fold$FOLD; [ -f $DC/results.json ] && continue; mkdir -p $DC
    OMP_NUM_THREADS=4 python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold $FOLD --seed $SEED --epochs 50 --ema-decays 0.9999 --early-stopping --patience 8 --min-epochs 10 \
      --norm-bottleneck --pooled-skips --grad-clip-norm 1.0 --lr 1e-4 --out-dir $DC 2>&1 | tee -a $DC/run.log
  done; }
lane 10 & lane 11 & lane 12 &
wait
echo EARLY_STOP_STUDY_DONE | tee -a $OUT/run.log
