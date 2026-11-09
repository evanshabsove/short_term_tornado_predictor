#!/usr/bin/env bash
# Per-epoch held-out metrics, arm A vs. repaired ConvLSTM (docs/convlstm_per_epoch_plan.md): fold 2020, seed 10, 50 epochs, two parallel lanes (6 threads each).
# Resumable (skips a lane whose results.json exists). Plug in. Launch detached: python subprocess.Popen(..., start_new_session=True).
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
OUT=models/convlstm/per_epoch
mkdir -p $OUT/arm_a $OUT/c3
shasum -a 256 docs/convlstm_per_epoch_plan.md >> $OUT/plan_sha_at_launch.txt
lane_a() { [ -f $OUT/arm_a/results.json ] && return 0
  OMP_NUM_THREADS=6 python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed 10 --years 2020 --ema-decays 0.9999 --per-epoch-metrics \
    --staging-dir data/interim/scaled_build_wholebin --manifest data/interim/scaled_build_wholebin/manifest.json \
    --scratch-dir data/interim/cv_scratch_per_epoch --out-dir $OUT/arm_a --results $OUT/arm_a/results.json 2>&1 | tee -a $OUT/arm_a/run.log; }
lane_c3() { [ -f $OUT/c3/results.json ] && return 0
  OMP_NUM_THREADS=6 python -u -W ignore scripts/11_convlstm/train_convlstm.py --fold 2020 --seed 10 --epochs 50 --ema-decays 0.9999 --per-epoch-metrics \
    --norm-bottleneck --pooled-skips --grad-clip-norm 1.0 --lr 1e-4 --out-dir $OUT/c3 2>&1 | tee -a $OUT/c3/run.log; }
lane_a & lane_c3 &
wait
echo PER_EPOCH_DONE | tee -a $OUT/run.log
