"""Per-year AUC-PR of a leave-one-year-out climatology baseline (smoothed per-cell, and per-cell-per-month, positive frequency) plus init-hour / bin diagnostics of the v4 sample mix. Output: models/cv_year_holdout_v4/climatology_baseline.json. Run from repo root."""
import json, numpy as np, pandas as pd, xarray as xr
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score
L=[];T=[];B=[]
for n in ("train","val"):
    ds=xr.open_dataset(f"data/processed/training_dataset_{n}_full_v4.nc")
    L.append(ds["label"].values.astype("float32")); T.append(ds["init_time"].values); B.append(ds["bin_index"].values)
y=np.concatenate(L); t=pd.to_datetime(np.concatenate(T)); b=np.concatenate(B)
year=t.year.values; month=t.month.values; hour=t.hour.values
act=y.sum((1,2))>0
print("samples",len(y),"active frac %.3f"%act.mean())
print("init-hour dist ACTIVE:",np.bincount(hour[act],minlength=24).tolist())
print("init-hour dist QUIET :",np.bincount(hour[~act],minlength=24).tolist())
print("fraction of QUIET samples at 20Z: %.3f ; fraction of ACTIVE at 20Z: %.3f"%((hour[~act]==20).mean(),(hour[act]==20).mean()))
for bi in (0,1):
    m=b==bi; print(f"bin {bi}: samples {m.sum()}, active frac {act[m].mean():.3f}, pos cells {int(y[m].sum())}, pos rate {y[m].mean():.2e}")
# climatology baseline
res={}
for Y in sorted(set(year)):
    tr=year!=Y; te=year==Y
    out={}
    for name,key in (("clim_cell",None),("clim_cell_month","m")):
        if key is None:
            rate=gaussian_filter(y[tr].sum(0)/tr.sum(),1.5)
            sc=np.broadcast_to(rate,(te.sum(),)+rate.shape)
        else:
            sc=np.empty((te.sum(),)+y.shape[1:],dtype="float32")
            for mo in range(1,13):
                mm=tr&(month==mo)
                r=gaussian_filter(y[mm].sum(0)/max(mm.sum(),1),1.5) if mm.sum() else np.zeros(y.shape[1:])
                sc[month[te]==mo]=r
        out[name]=float(average_precision_score(y[te].ravel(),np.asarray(sc).ravel()))
    out["pos_rate"]=float(y[te].mean()); res[int(Y)]=out; print(Y,{k:round(v,5) for k,v in out.items()},flush=True)
json.dump(res,open("models/cv_year_holdout_v4/climatology_baseline.json","w"),indent=1)
