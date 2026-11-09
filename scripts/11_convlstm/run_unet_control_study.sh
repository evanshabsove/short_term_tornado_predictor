#!/usr/bin/env bash
# Stage 2 of docs/unet_control_preregistration.md: the selected arm-A optimizer setting (from models/unet_control/select/selection.json) with early stopping, folds 2016/2020/2024 x seeds 10-12.
# Three lanes (one per seed, 4 threads each); resumable. Plug in; launch detached under caffeinate -i.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
SEL=models/unet_control/select/selection.json
read RUN LR CLIP < <(python - <<'P'
import json
s = json.load(open("models/unet_control/select/selection.json"))
g = s["selected_settings"]
print(int(s["run_stage2"]), g["lr"], g["grad_clip_norm"] if g["grad_clip_norm"] is not None else "none")
P
)
[ "$RUN" = "1" ] || { echo "Stage 2 not run: U0 stands (see $SEL)"; exit 0; }
EXTRA=(--lr $LR); [ "$CLIP" != "none" ] && EXTRA+=(--grad-clip-norm $CLIP)
OUT=models/unet_control/tuned_a
mkdir -p $OUT
shasum -a 256 docs/unet_control_preregistration.md >> $OUT/prereg_sha_at_launch.txt
echo "selected: lr=$LR clip=$CLIP" | tee -a $OUT/run.log
lane() { SEED=$1
  for FOLD in 2016 2020 2024; do
    D=$OUT/seed$SEED; mkdir -p $D
    OMP_NUM_THREADS=4 python -u -W ignore scripts/05_year_cv_diagnostics/cross_validate_year.py --seed $SEED --years $FOLD --ema-decays 0.9999 --early-stopping --patience 8 --min-epochs 10 "${EXTRA[@]}" \
      --staging-dir data/interim/scaled_build_wholebin --manifest data/interim/scaled_build_wholebin/manifest.json \
      --scratch-dir data/interim/cv_scratch_uct_s$SEED --out-dir $D --results $D/results.json 2>&1 | tee -a $D/run.log
  done; }
lane 10 & lane 11 & lane 12 &
wait
echo UNET_CONTROL_STUDY_DONE | tee -a $OUT/run.log
