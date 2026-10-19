"""
Loads and runs the production model bundle: a fixed set of trained
checkpoints (members) described by a manifest JSON, combined by a plain
mean of their per-cell probabilities.

Why a plain mean: it is the combination every ensemble study in this
project measured (see CLAUDE.md "Seed-variance study and ensembles").
Weighted or calibrated ensembling is untested and deliberately not here.
Ensembling reduces model variance only -- it does not raise the accuracy
ceiling and it does not calibrate the output.

The per-cell spread (population std of the members' probabilities) is
returned alongside the mean as a disagreement signal. Its meaning is
untested, so callers must not present it as a confidence interval.

A manifest is self-describing and relocatable: member paths are stored
relative to the manifest's own directory, and each member's sha256 is
checked on load, so a swapped or truncated checkpoint raises instead of
silently changing what the API serves. Every member must share the same
`feature_names` and `model_class`; `predict_ensemble` additionally
checks the caller's feature order against the bundle's (the same guard
inference.assert_feature_order_matches gives a single checkpoint).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tornado_predictor.inference import assert_feature_order_matches, load_checkpoint

DEFAULT_BUNDLE_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models" / "production" / "manifest.json"


@dataclass(frozen=True)
class EnsemblePrediction:
    """mean: plain mean of member probabilities; spread: population std
    (ddof=0) across members. Same shape as the input without its channel
    axis: (H, W) for a (C, H, W) input, (N, H, W) for (N, C, H, W)."""

    mean: np.ndarray
    spread: np.ndarray


@dataclass(frozen=True, eq=False)
class EnsembleBundle:
    manifest: dict
    models: list[nn.Module]
    feature_names: list[str]
    bundle_id: str


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_bundle_id(member_sha256s: list[str], recipe: dict) -> str:
    """12-hex-char id from the sorted member hashes plus the training
    recipe -- changes whenever any member's weights or the recipe do."""
    payload = json.dumps({"members": sorted(member_sha256s), "recipe": recipe}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_bundle(manifest_path: Path | str = DEFAULT_BUNDLE_MANIFEST_PATH) -> EnsembleBundle:
    """Reads the manifest, verifies each member's sha256 and the bundle
    id, loads every member via inference.load_checkpoint, and requires
    one shared feature order and model class. Raises ValueError on any
    mismatch."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    members = manifest["members"]
    if not members:
        raise ValueError(f"{manifest_path}: manifest lists no members")

    models, feature_names, model_classes = [], None, set()
    for member in members:
        path = manifest_path.parent / member["path"]
        actual = file_sha256(path)
        if actual != member["sha256"]:
            raise ValueError(
                f"{path}: sha256 {actual} does not match the manifest's {member['sha256']} "
                "-- the checkpoint changed since the bundle was built"
            )
        model, checkpoint = load_checkpoint(str(path))
        names = list(checkpoint["feature_names"])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"{path}: feature order {names} differs from the other members' {feature_names}")
        model_classes.add(checkpoint.get("model_class", "TornadoCNN"))
        models.append(model)

    if len(model_classes) != 1:
        raise ValueError(f"members use different model classes: {sorted(model_classes)}")
    if feature_names != manifest["feature_names"]:
        raise ValueError(
            f"manifest feature_names {manifest['feature_names']} differ from the checkpoints' {feature_names}"
        )

    expected_id = compute_bundle_id([m["sha256"] for m in members], manifest["recipe"])
    if manifest["bundle_id"] != expected_id:
        raise ValueError(f"manifest bundle_id {manifest['bundle_id']} does not match the recomputed {expected_id}")

    return EnsembleBundle(manifest=manifest, models=models, feature_names=feature_names, bundle_id=manifest["bundle_id"])


def predict_ensemble(bundle: EnsembleBundle, X: np.ndarray, *, feature_names: list[str]) -> EnsemblePrediction:
    """X: (C, H, W) or (N, C, H, W) float array in `feature_names` order.
    feature_names is required (not optional) so a caller can never skip
    the channel-order check."""
    assert_feature_order_matches(list(feature_names), {"feature_names": bundle.feature_names})
    X = np.asarray(X, dtype=np.float32)
    single = X.ndim == 3
    if X.ndim not in (3, 4):
        raise ValueError(f"X must be (C, H, W) or (N, C, H, W), got shape {X.shape}")
    if X.shape[-3] != len(bundle.feature_names):
        raise ValueError(f"X has {X.shape[-3]} channels, bundle expects {len(bundle.feature_names)}")
    if single:
        X = X[None]

    with torch.no_grad():
        x = torch.from_numpy(X)
        member_probs = torch.stack([torch.sigmoid(m(x)).squeeze(1) for m in bundle.models])  # (members, N, H, W)
        mean = member_probs.mean(dim=0).numpy()
        spread = member_probs.std(dim=0, unbiased=False).numpy()

    if single:
        mean, spread = mean[0], spread[0]
    return EnsemblePrediction(mean=mean, spread=spread)
