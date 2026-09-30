from pathlib import Path

import pytest
import torch

from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.inference import load_checkpoint, predict_proba
from tornado_predictor.model import TornadoCNN, receptive_field_cells
from tornado_predictor.training import DenseGridDataset, FocalLoss

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_pilot.nc"
FULL_CHECKPOINT_PATH = REPO_ROOT / "models" / "tornado_cnn_full.pt"
FULL_VAL_DATASET_PATH = REPO_ROOT / "data" / "processed" / "training_dataset_val_full.nc"
H32_L5_CHECKPOINT_PATH = REPO_ROOT / "models" / "architecture_sweep" / "tornado_cnn_h32_l5.pt"

GRID = build_coarse_grid()


def test_output_shape_matches_grid():
    model = TornadoCNN(in_channels=14)
    x = torch.randn(2, 14, GRID.n_rows, GRID.n_cols)
    logits = model(x)
    assert logits.shape == (2, 1, GRID.n_rows, GRID.n_cols)


@pytest.mark.parametrize(
    "in_channels,hidden_channels,n_hidden_layers",
    [(1, 4, 3), (7, 16, 3), (14, 32, 3), (14, 64, 3), (14, 128, 3), (14, 32, 5), (14, 128, 5)],
)
def test_output_shape_for_various_channel_configs(in_channels, hidden_channels, n_hidden_layers):
    model = TornadoCNN(in_channels=in_channels, hidden_channels=hidden_channels, n_hidden_layers=n_hidden_layers)
    x = torch.randn(3, in_channels, 10, 12)
    logits = model(x)
    assert logits.shape == (3, 1, 10, 12)


@pytest.mark.parametrize("n_hidden_layers,expected", [(1, 3), (2, 5), (3, 7), (4, 9), (5, 11)])
def test_receptive_field_cells_formula(n_hidden_layers, expected):
    assert receptive_field_cells(n_hidden_layers) == expected


@pytest.mark.parametrize(
    "dilations,expected",
    [([1, 1, 1], 7), ([1, 2, 4], 15), ([1, 2, 4, 8], 31), ([2, 2, 2], 13)],
)
def test_receptive_field_cells_formula_with_dilations(dilations, expected):
    assert receptive_field_cells(len(dilations), dilations=dilations) == expected


@pytest.mark.parametrize(
    "hidden_channels,dilations",
    [(32, [1, 2, 4]), (32, [1, 2, 4, 8]), (16, [2, 2, 2])],
)
def test_output_shape_for_dilated_configs(hidden_channels, dilations):
    model = TornadoCNN(in_channels=14, hidden_channels=hidden_channels, n_hidden_layers=len(dilations), dilations=dilations)
    x = torch.randn(2, 14, 40, 50)
    logits = model(x)
    assert logits.shape == (2, 1, 40, 50)  # padding=dilation must preserve spatial size at every dilation rate


def test_dilations_length_mismatch_raises():
    with pytest.raises(ValueError):
        TornadoCNN(in_channels=14, n_hidden_layers=3, dilations=[1, 2])


def test_default_architecture_matches_original_hardcoded_3_layer_design():
    """Regression guard: generalizing TornadoCNN to accept
    n_hidden_layers must not change the default (n_hidden_layers=3)
    architecture -- same layer types/shapes as the original hardcoded
    nn.Sequential(conv,relu,conv,relu,conv,relu,conv)."""
    model = TornadoCNN(in_channels=14, hidden_channels=32)

    assert len(model.net) == 7  # 3x (Conv2d, ReLU) + final Conv2d
    conv_layers = [layer for layer in model.net if isinstance(layer, torch.nn.Conv2d)]
    relu_layers = [layer for layer in model.net if isinstance(layer, torch.nn.ReLU)]
    assert len(conv_layers) == 4
    assert len(relu_layers) == 3

    assert conv_layers[0].in_channels == 14 and conv_layers[0].out_channels == 32
    assert conv_layers[1].in_channels == 32 and conv_layers[1].out_channels == 32
    assert conv_layers[2].in_channels == 32 and conv_layers[2].out_channels == 32
    assert conv_layers[3].in_channels == 32 and conv_layers[3].out_channels == 1
    assert all(layer.kernel_size == (3, 3) for layer in conv_layers[:3])
    assert conv_layers[3].kernel_size == (1, 1)

    assert receptive_field_cells(3) == 7  # matches this project's documented "~7x7" baseline claim


def _measured_receptive_field(model: TornadoCNN, grid_shape: tuple[int, int], center: tuple[int, int]) -> tuple[int, int]:
    """Empirically measures a model's receptive field by backprop from
    one output cell through a random (requires_grad) input, and
    checking which input cells received nonzero gradient. Random, not
    all-zero, input matters: an all-zero input makes every hidden
    conv's pre-activation spatially constant (bias only, since input
    contributes nothing), which can make a ReLU uniformly dead across
    the whole map by pure chance on the bias's sign and silently
    collapse the measured field to nothing -- confirmed by hitting
    exactly this failure mode while writing this test. `center` must be
    far enough from every edge that the true receptive field can't be
    clipped by the input boundary, or this undercounts."""
    x = torch.randn(1, 1, *grid_shape, requires_grad=True)
    out = model(x)
    out[0, 0, center[0], center[1]].backward()
    influenced = x.grad[0, 0] != 0
    rows = influenced.any(dim=1).nonzero().flatten()
    cols = influenced.any(dim=0).nonzero().flatten()
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


@pytest.mark.parametrize(
    "n_hidden_layers,dilations",
    [(3, None), (5, None), (4, [1, 2, 4, 8])],
)
def test_receptive_field_formula_matches_gradient_based_measurement(n_hidden_layers, dilations):
    """Empirically proves the receptive-field claim this ticket is
    built around, rather than trusting the formula alone -- same
    verify-don't-assume standard applied to every other geometric claim
    in this project (e.g. grid.py's corner verification). Uses
    hidden_channels=32 (a real config's width, not an artificially tiny
    one) so the chance of every channel being simultaneously ReLU-dead
    at the one measured position is astronomically small, not just
    unlikely -- verified reliable across 5 seeds before being written
    as a fixed-seed test."""
    torch.manual_seed(0)
    model = TornadoCNN(in_channels=1, hidden_channels=32, n_hidden_layers=n_hidden_layers, dilations=dilations)
    model.eval()

    grid_shape = (81, 138)
    center = (40, 69)  # far enough from every edge for even the largest tested receptive field to fit fully
    measured_height, measured_width = _measured_receptive_field(model, grid_shape, center)

    expected = receptive_field_cells(n_hidden_layers, dilations=dilations)
    assert measured_height == expected
    assert measured_width == expected


def test_real_full_checkpoint_predictions_unchanged_after_generalizing_model(tmp_path):
    """Backward-compatibility regression test against a real artifact:
    models/tornado_cnn_full.pt was saved before n_hidden_layers existed
    (its model_hyperparameters has no such key), so it must still load
    via the default and produce identical predictions to before this
    change."""
    if not FULL_CHECKPOINT_PATH.exists() or not FULL_VAL_DATASET_PATH.exists():
        pytest.skip("real full-scale checkpoint or val dataset not present")

    model, checkpoint = load_checkpoint(str(FULL_CHECKPOINT_PATH))
    assert "n_hidden_layers" not in checkpoint["model_hyperparameters"]
    assert len(model.net) == 7  # reconstructed the original 3-hidden-layer design

    dataset = DenseGridDataset(str(FULL_VAL_DATASET_PATH))
    X, _ = dataset[0]
    proba = predict_proba(model, X)

    # known-good values, captured from this exact checkpoint/sample
    # before model.py was generalized to accept n_hidden_layers
    assert proba.sum() == pytest.approx(37.13296, abs=1e-3)
    assert proba[0, 0] == pytest.approx(4.6911555e-07, abs=1e-9)
    assert proba[40, 60] == pytest.approx(0.0005584206, abs=1e-8)


def test_real_h32_l5_checkpoint_predictions_unchanged_after_adding_dilations():
    """Same backward-compatibility guard as the test above, for a
    checkpoint saved by ticket 2 (has n_hidden_layers but no dilations
    in its model_hyperparameters) -- must still load via the dilations
    default and produce identical predictions after this ticket's
    change."""
    if not H32_L5_CHECKPOINT_PATH.exists() or not FULL_VAL_DATASET_PATH.exists():
        pytest.skip("real architecture-sweep checkpoint or val dataset not present")

    model, checkpoint = load_checkpoint(str(H32_L5_CHECKPOINT_PATH))
    assert "dilations" not in checkpoint["model_hyperparameters"]
    assert len(model.net) == 11  # 5x (Conv2d, ReLU) + final Conv2d

    dataset = DenseGridDataset(str(FULL_VAL_DATASET_PATH))
    X, _ = dataset[0]
    proba = predict_proba(model, X)

    # known-good values, captured from this exact checkpoint/sample
    # before model.py was generalized to accept dilations
    assert proba.sum() == pytest.approx(41.288822, abs=1e-3)
    assert proba[0, 0] == pytest.approx(3.4750363e-16, abs=1e-18)
    assert proba[40, 60] == pytest.approx(0.0012632699, abs=1e-8)


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
