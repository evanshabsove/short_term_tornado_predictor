from pathlib import Path

import pytest
import torch

from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.training import DenseGridDataset, FocalLoss
from tornado_predictor.unet import TornadoUNet

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_pilot.nc"

GRID = build_coarse_grid()


def test_output_shape_matches_grid():
    """The real 81x138 grid isn't a power of 2 -- this is the actual
    shape the pad-to-88x144/crop-back-to-81x138 pipeline needs to get
    right, not just any arbitrary shape."""
    model = TornadoUNet(in_channels=14)
    x = torch.randn(2, 14, GRID.n_rows, GRID.n_cols)
    logits = model(x)
    assert logits.shape == (2, 1, GRID.n_rows, GRID.n_cols)


@pytest.mark.parametrize(
    "shape",
    [(81, 138), (10, 12), (16, 16), (1, 1), (7, 200), (88, 144)],  # includes odd sizes, tiny sizes, and the exact padded size itself
)
def test_output_shape_for_various_grid_sizes(shape):
    model = TornadoUNet(in_channels=3, base_channels=4)
    x = torch.randn(2, 3, *shape)
    logits = model(x)
    assert logits.shape == (2, 1, *shape)


@pytest.mark.parametrize("base_channels", [4, 16, 32])
def test_output_shape_for_various_base_channels(base_channels):
    model = TornadoUNet(in_channels=14, base_channels=base_channels)
    x = torch.randn(1, 14, GRID.n_rows, GRID.n_cols)
    logits = model(x)
    assert logits.shape == (1, 1, GRID.n_rows, GRID.n_cols)


def test_output_is_finite():
    model = TornadoUNet(in_channels=14)
    x = torch.randn(1, 14, GRID.n_rows, GRID.n_cols)
    logits = model(x)
    assert torch.isfinite(logits).all()
    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_gradients_flow_through_every_parameter():
    """A real, non-redundant check here -- the skip-connection wiring
    has more ways to accidentally leave a branch disconnected than
    TornadoCNN's straight stack."""
    model = TornadoUNet(in_channels=14, base_channels=8)
    x = torch.randn(2, 14, 10, 12, requires_grad=False)
    y = (torch.rand(2, 10, 12) > 0.5).float()

    logits = model(x).squeeze(1)
    loss = FocalLoss()(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"
        assert (param.grad != 0).any(), f"all-zero gradient in {name}"


def test_against_real_pilot_dataset():
    if not PILOT_DATASET_PATH.exists():
        pytest.skip("pilot dataset not present")
    dataset = DenseGridDataset(str(PILOT_DATASET_PATH))
    model = TornadoUNet(in_channels=len(dataset.feature_names))

    X, y = dataset[0]
    x_batch = torch.from_numpy(X).unsqueeze(0)  # (1, 14, 81, 138)
    logits = model(x_batch)

    assert logits.shape == (1, 1, GRID.n_rows, GRID.n_cols)
    assert torch.isfinite(logits).all()

    # confirm it's actually trainable end-to-end against real labels
    target = torch.from_numpy(y).unsqueeze(0)
    loss = FocalLoss()(logits.squeeze(1), target)
    loss.backward()
    assert torch.isfinite(loss)
