"""Per-year AUC-PR of untrained physics baselines (STP-like composite, single fields, spatially smoothed) on the v4 train+val files (excludes the ~7% buffer-dropped samples). Output: models/cv_year_holdout_v4/physics_baselines.json. Run from repo root."""
import json, numpy as np, xarray as xr
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score
V=["cape_mean","cape_max","cin_max","srh_0_1km_mean","srh_0_1km_max","srh_0_3km_mean","srh_0_3km_max","shear_0_6km_mean","shear_0_6km_max","label"]
parts=[]
for n in ("train","val"):
    ds=xr.open_dataset(f"data/processed/training_dataset_{n}_full_v4.nc")
    parts.append(({v:ds[v].values.astype("float32") for v in V}, ds["init_time"].values))
D={v:np.concatenate([p[0][v] for p in parts]) for v in V}
t=np.concatenate([p[1] for p in parts]); year=(t.astype("datetime64[Y]").astype(int)+1970)
print("samples",len(year),flush=True)
def stp(cape,srh,shear,cin):
    sh=np.where(shear<12.5,0,np.minimum(shear,30)/20)
    c=np.clip((200+cin)/150,0,1)
    return (cape/1500)*(srh/150)*sh*c
S={}
for k,(cp,sr,shr) in {"mean":("cape_mean","srh_0_1km_mean","shear_0_6km_mean"),"max":("cape_max","srh_0_1km_max","shear_0_6km_max")}.items():
    S[f"stp_{k}"]=stp(D[cp],D[sr],D[shr],D["cin_max"])
S["cape_max"]=D["cape_max"]; S["srh01_max"]=D["srh_0_1km_max"]; S["srh03_max"]=D["srh_0_3km_max"]; S["shear06_max"]=D["shear_0_6km_max"]
S["cape_x_srh03_max"]=D["cape_max"]*np.maximum(D["srh_0_3km_max"],0)
for k in ("stp_max","stp_mean"):
    for w in (5,9):
        S[f"{k}_smooth{w}"]=uniform_filter(S[k],size=(1,w,w),mode="nearest")
y=D["label"]
res={}
for yr in sorted(set(year)):
    m=year==yr
    res[int(yr)]={"pos_rate":float(y[m].mean()),"n":int(m.sum())}
    for k,v in S.items():
        res[int(yr)][k]=float(average_precision_score(y[m].ravel(),v[m].ravel())) if y[m].sum()>0 else float("nan")
    print(yr,{k:round(v,4) for k,v in res[int(yr)].items() if k not in("n",)},flush=True)
json.dump(res,open("models/cv_year_holdout_v4/physics_baselines.json","w"),indent=1)
