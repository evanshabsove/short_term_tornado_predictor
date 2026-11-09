"""
Stage 1 decision of docs/unet_control_preregistration.md: reads models/unet_control/select/U*/seed*/results.json, prints the table and applies the fixed rule
(highest mean EMA test AUC-PR on fold 2018 over seeds 20/21; a non-default setting is chosen only if its mean is >= 5% above U0's, else U0 stands and Stage 2 is not run).
Output: models/unet_control/select/selection.json
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "models/unet_control/select"
SETTINGS = {"U0": {"lr": 1e-3, "grad_clip_norm": None}, "U1": {"lr": 3e-4, "grad_clip_norm": 1.0}, "U2": {"lr": 1e-4, "grad_clip_norm": 1.0}}
rows = {}
for name in SETTINGS:
    rs = [json.load(open(f))["folds"][0] for f in sorted((D / name).glob("seed*/results.json"))]
    if rs:
        es = [r["early_stopping"] for r in rs]
        rows[name] = {"n": len(rs), "auc_pr": [r["ema_metrics"]["0.9999"]["auc_pr"] for r in rs], "plain_auc_pr": [r["auc_pr"] for r in rs],
                      "best_epoch": [e["best_epoch"] for e in es], "stopped_epoch": [e["stopped_epoch"] for e in es], "stopped_early": [e["stopped_early"] for e in es],
                      "val_over_train": [r["final_val_loss"] / r["final_train_loss"] for r in rs]}
for name, r in rows.items():
    r["mean_auc_pr"] = float(np.mean(r["auc_pr"]))
    print(f"{name} {SETTINGS[name]}: n={r['n']} EMA AUC-PR {[round(x, 4) for x in r['auc_pr']]} mean {r['mean_auc_pr']:.4f} | best epoch {r['best_epoch']} stopped {r['stopped_epoch']} "
          f"| val/train {[round(x, 2) for x in r['val_over_train']]}")
out = {"settings": SETTINGS, "rows": rows}
if set(rows) == set(SETTINGS) and all(r["n"] == 2 for r in rows.values()):
    best = max(("U1", "U2"), key=lambda n: rows[n]["mean_auc_pr"])
    gain = rows[best]["mean_auc_pr"] / rows["U0"]["mean_auc_pr"] - 1
    chosen = best if gain >= 0.05 else "U0"
    out.update(best_non_default=best, gain_vs_u0=gain, selected=chosen, run_stage2=chosen != "U0", selected_settings=SETTINGS[chosen])
    print(f"best non-default {best}: {gain:+.1%} vs U0 -> selected {chosen}; Stage 2 {'runs' if chosen != 'U0' else 'is NOT run (U0 stands)'}")
else:
    print("incomplete: need U0/U1/U2 with 2 seeds each")
(D / "selection.json").write_text(json.dumps(out, indent=1))
