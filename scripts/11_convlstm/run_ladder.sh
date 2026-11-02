#!/usr/bin/env bash
# Diagnostic ladder (docs/convlstm_diagnostic_ladder.md): steps 1-4 one after another; resumable (a step with its json is skipped).
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
D=models/convlstm/diagnostics/ladder; mkdir -p $D
[ -f $D/overfit.json ] && grep -q '"unet"' $D/overfit.json || python -u -W ignore scripts/11_convlstm/diagnose_ladder.py overfit 2>&1 | tee -a $D/run.log
for n in early_fusion c3_norecur c3_b4_lr2e3 c3_lr1e4; do
  [ -f $D/$n.json ] && { echo "skip $n"; continue; }
  python -u -W ignore scripts/11_convlstm/diagnose_ladder.py full $n 2>&1 | tee -a $D/run.log
done
echo LADDER_DONE | tee -a $D/run.log
