#!/usr/bin/env bash
# Part 8 evaluation: 8A 6 UH seeds (10-15), 8A replication triple (13-15), 8B 21-feature triple (10-12). Evaluation only; scores are cached.
set -e
python -u -W ignore scripts/07_updraft_helicity/uh_eval.py --seeds 10 11 12 13 14 15 --out models/uh_study/uh_eval_6seeds.json
python -u -W ignore scripts/07_updraft_helicity/uh_eval.py --seeds 13 14 15 --out models/uh_study/uh_eval_13_15.json
python -u -W ignore scripts/07_updraft_helicity/uh_eval.py --study-dir models/uh21_study --keep-uh-columns --tag uh21study --seeds 10 11 12 --out models/uh21_study/uh21_eval.json
