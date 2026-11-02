#!/usr/bin/env bash
# Ladder round 2 (docs/convlstm_diagnostic_ladder.md): lr 1e-4 on another seed, and without recurrence. Resumable.
set -eo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
D=models/convlstm/diagnostics/ladder
for n in c3_lr1e4_s11 c3_norecur_lr1e4; do
  [ -f $D/$n.json ] && { echo "skip $n"; continue; }
  python -u -W ignore scripts/11_convlstm/diagnose_ladder.py full $n 2>&1 | tee -a $D/run2.log
done
echo LADDER2_DONE | tee -a $D/run2.log
