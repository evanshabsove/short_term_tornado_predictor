from pathlib import Path

import pytest
import torch

from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.model import TornadoCNN
from tornado_predictor.training import DenseGridDataset, FocalLoss

PILOT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "training_dataset_pilot.nc"

GRID = build_coarse_grid()


def test_output_shape_matches_grid():
    model = TornadoCNN(in_channels=14)
    x = torch.randn(2, 14, GRID.n_rows, GRID.n_cols)
    logits = model(x)
    assert logits.shape == (2, 1, GRID.n_rows, GRID.n_cols)


@pytest.mark.parametrize("in_channels,hidden_channels", [(1, 4), (7, 16), (14, 32)])
def test_output_shape_for_various_channel_configs(in_channels, hidden_channels):
    model = TornadoCNN(in_channels=in_channels, hidden_channels=hidden_channels)
    x = torch.randn(3, in_channels, 10, 12)
    logits = model(x)
    assert logits.shape == (3, 1, 10, 12)


def test_output_is_finite():
    model = TornadoCNN(in_channels=14)
    x = torch.randn(1, 14, GRID.n_rows, GRID.n_cols)
    logits = model(x)
    assert torch.isfinite(logits).all()
    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_gradients_flow_through_every_parameter():
    model = TornadoCNN(in_channels=14)
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
    model = TornadoCNN(in_channels=len(dataset.feature_names))

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
