"""
Stage 1 decision of docs/convlstm_dropout_preregistration.md: reads models/convlstm/dropout_select/p*/seed*/results.json, prints the table and applies the fixed rule
(best mean EMA AUC-PR among p>0; proceed only if it is >= the control's and its mean val/train loss ratio is below the control's). Output: <dir>/selection.json
"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "models/convlstm/dropout_select"
rows = {}
for pdir in sorted(x for x in D.glob("p[0-9]*") if x.is_dir()):
    p = float(pdir.name[1:])
    rs = [json.load(open(f)) for f in sorted(pdir.glob("seed*/results.json"))]
    if rs:
        rows[p] = {
            "n": len(rs), "auc_pr": [r["metrics"]["ema0.9999"]["auc_pr"] for r in rs], "plain_auc_pr": [r["metrics"]["raw"]["auc_pr"] for r in rs],
            "ratio": [r["val_over_train_loss"] for r in rs], "train_loss": [r["final_train_loss"] for r in rs],
        }
for p, r in rows.items():
    r["mean_auc_pr"], r["mean_ratio"] = float(np.mean(r["auc_pr"])), float(np.mean(r["ratio"]))
    print(f"p={p}: n={r['n']} EMA AUC-PR {[round(x, 4) for x in r['auc_pr']]} mean {r['mean_auc_pr']:.4f} | plain {[round(x, 4) for x in r['plain_auc_pr']]} | "
          f"val/train {[round(x, 2) for x in r['ratio']]} mean {r['mean_ratio']:.2f} | train loss {np.mean(r['train_loss']):.2e}")
out = {"rows": rows}
if 0.0 in rows and any(p > 0 for p in rows) and all(r["n"] == 2 for r in rows.values()):
    best = max((p for p in rows if p > 0), key=lambda p: rows[p]["mean_auc_pr"])
    ok = rows[best]["mean_auc_pr"] >= rows[0.0]["mean_auc_pr"] and rows[best]["mean_ratio"] < rows[0.0]["mean_ratio"]
    out.update(best_rate=best, proceed_to_stage2=bool(ok),
               gain_vs_control=rows[best]["mean_auc_pr"] / rows[0.0]["mean_auc_pr"] - 1)
    print(f"best rate {best}: AUC-PR {out['gain_vs_control']:+.1%} vs control; ratio {rows[best]['mean_ratio']:.2f} vs {rows[0.0]['mean_ratio']:.2f} -> proceed to Stage 2: {ok}")
else:
    print("incomplete: need control and >= 1 dropout rate, 2 seeds each")
(D / "selection.json").write_text(json.dumps(out, indent=1))
