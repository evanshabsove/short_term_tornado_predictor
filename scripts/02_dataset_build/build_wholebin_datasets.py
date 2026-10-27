"""
Build train/val datasets from the whole-bin staging (build_hourly_dataset.py
--derive-only) with the same 2-way split rule as v4 (mod-20 days, 24 h buffer),
then check them against the v4 single-snapshot datasets:

  - split sizes and dropped count
  - active-map share and per-cell positive rate (should match v4 closely)
  - zero NaN; the 14 base channels come first and in the baseline checkpoints' order
  - labels identical to v4's for every (init_time, bin_index) sample both contain
  - per-channel mean/min/max vs v4 (whole-bin max/mean features differ by design)

Usage:
    python scripts/02_dataset_build/build_wholebin_datasets.py            # base 14
    python scripts/02_dataset_build/build_wholebin_datasets.py --extra    # 14 + 16 extra
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from tornado_predictor.build_scaled import finalize, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM = REPO_ROOT / "data" / "interim"
PROCESSED = REPO_ROOT / "data" / "processed"
BASELINE_CKPT = REPO_ROOT / "models" / "unet" / "tornado_unet.pt"


def summarize(ds: xr.Dataset, name: str) -> dict:
    y = ds["label"].values
    active = (y.sum(axis=(1, 2)) > 0)
    out = {"samples": int(ds.sizes["sample"]), "active_frac": float(active.mean()), "pos_rate": float(y.mean())}
    print(f"{name}: {out['samples']} samples, {out['active_frac']:.1%} active maps, per-cell positive rate {out['pos_rate']:.3g}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra", action="store_true", help="use the 30-channel (base + extra) staging")
    args = parser.parse_args()

    tag = "wholebin_extra" if args.extra else "wholebin"
    staging = INTERIM / f"scaled_build_{tag}"
    out_train = PROCESSED / f"training_dataset_train_full_{tag}.nc"
    out_val = PROCESSED / f"training_dataset_val_full_{tag}.nc"

    manifest = load_manifest(staging / "manifest.json")
    runs = sorted(pd.Timestamp(k) for k, v in manifest.items() if v["status"] == "done")
    print(f"{len(runs)} runs in {staging.name}")
    print("finalize:", finalize(staging, runs, out_train, out_val))

    baseline_features = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)["feature_names"]
    for split in ("train", "val"):
        new = xr.open_dataset(PROCESSED / f"training_dataset_{split}_full_{tag}.nc").load()
        old = xr.open_dataset(PROCESSED / f"training_dataset_{split}_full_v4.nc").load()
        print(f"\n== {split}")
        summarize(old, "  v4      ")
        summarize(new, f"  {tag}")

        names = [v for v in new.data_vars if v != "label"]
        print("  channels:", len(names), "| first 14 match baseline checkpoint order:", names[:14] == baseline_features)
        print("  NaNs:", sum(int(np.isnan(new[v].values).sum()) for v in names))

        key = lambda ds: {(pd.Timestamp(t), int(b)): i for i, (t, b) in enumerate(zip(ds["init_time"].values, ds["bin_index"].values))}
        k_new, k_old = key(new), key(old)
        common = sorted(set(k_new) & set(k_old))
        same = sum(
            np.array_equal(new["label"].values[k_new[k]], old["label"].values[k_old[k]]) for k in common
        )
        print(f"  samples only in v4: {len(set(k_old) - set(k_new))}, only in {tag}: {len(set(k_new) - set(k_old))}")
        print(f"  labels identical to v4 on {same}/{len(common)} common samples")

        print("  channel          v4 mean   new mean |  v4 max    new max")
        for v in names[:14]:
            print(f"  {v:16s} {float(old[v].mean()):9.4g} {float(new[v].mean()):9.4g} | {float(old[v].max()):9.4g} {float(new[v].max()):9.4g}")
        if args.extra:
            print("  extra channel    new mean     new min    new max")
            for v in names[14:]:
                print(f"  {v:16s} {float(new[v].mean()):9.4g} {float(new[v].min()):10.4g} {float(new[v].max()):10.4g}")


if __name__ == "__main__":
    main()
