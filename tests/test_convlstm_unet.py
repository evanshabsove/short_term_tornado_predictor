from pathlib import Path

import numpy as np
import pytest
import torch

from tornado_predictor.convlstm_unet import DEFAULT_STOP_STEPS, ConvLSTMCell, TornadoConvLSTMUNet
from tornado_predictor.grid import build_coarse_grid
from tornado_predictor.inference import MODEL_REGISTRY, load_checkpoint
from tornado_predictor.training import FocalLoss
from tornado_predictor.unet import TornadoUNet

REPO_ROOT = Path(__file__).resolve().parents[1]
UNET_CKPT = REPO_ROOT / "models" / "unet" / "tornado_unet.pt"
GRID = build_coarse_grid()


def small(**kw):
    return TornadoConvLSTMUNet(in_channels=3, base_channels=4, hidden_channels=8, **kw)


def test_default_stops_are_the_last_hour_of_each_bin():
    assert DEFAULT_STOP_STEPS == (3, 7)


def test_output_shape_on_the_real_grid():
    model = TornadoConvLSTMUNet(in_channels=14)
    x = torch.randn(2, 8, 14, GRID.n_rows, GRID.n_cols)
    assert model(x).shape == (2, 2, GRID.n_rows, GRID.n_cols)


@pytest.mark.parametrize("shape", [(81, 138), (10, 12), (16, 16), (1, 1), (7, 200), (88, 144)])
def test_output_shape_for_various_grid_sizes(shape):
    out = small()(torch.randn(2, 8, 3, *shape))
    assert out.shape == (2, 2, *shape) and torch.isfinite(out).all()


def test_gradients_reach_every_parameter():
    model = small()
    x = torch.randn(2, 8, 3, 10, 12)
    y = (torch.rand(2, 2, 10, 12) > 0.5).float()
    FocalLoss()(model(x), y).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"
        assert (p.grad != 0).any(), f"all-zero gradient in {name}"


def test_reversing_time_changes_the_output_so_the_recurrence_is_used():
    torch.manual_seed(0)
    model = small().eval()
    x = torch.randn(1, 8, 3, 10, 12)
    with torch.no_grad():
        a, b = model(x), model(x.flip(1))
    assert not torch.allclose(a, b, atol=1e-5)


def test_bin_0_cannot_see_hours_after_its_stop_but_bin_1_sees_earlier_hours():
    torch.manual_seed(0)
    model = small().eval()
    x = torch.randn(1, 8, 3, 10, 12)
    late = x.clone()
    late[:, 4:] += torch.randn_like(late[:, 4:])  # change hours 4-7
    with torch.no_grad():
        base, l = model(x), model(late)
    assert torch.allclose(base[:, 0], l[:, 0], atol=1e-6)  # bin 0 unaffected by hours 4-7 (causal)
    assert not torch.allclose(base[:, 1], l[:, 1], atol=1e-5)  # bin 1 uses its own hours

    # Sensitivity via gradients (an untrained net's outputs sit near 0.5, so a value diff on the
    # earlier hours would fall under any sensible tolerance): d bin / d hour, summed over the hour.
    def sensitivity(bin_index):
        xg = x.clone().requires_grad_(True)
        model(xg)[:, bin_index].sum().backward()
        return xg.grad.abs().sum(dim=(0, 2, 3, 4))  # per hour

    s0, s1 = sensitivity(0), sensitivity(1)
    assert torch.all(s0[:4] > 0) and torch.all(s0[4:] == 0)  # bin 0: exactly hours 0-3
    assert torch.all(s1 > 0)  # bin 1: its own hours AND the memory of hours 0-3 (disclosed difference)


def test_the_two_bins_are_decoded_separately():
    """Same weights everywhere but different stops and heads -> bins differ."""
    torch.manual_seed(0)
    out = small().eval()(torch.randn(1, 8, 3, 10, 12))
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_too_few_steps_raises_and_extra_steps_are_ignored():
    model = small().eval()
    with pytest.raises(ValueError, match="at least 8 steps"):
        model(torch.randn(1, 7, 3, 10, 12))
    x = torch.randn(1, 8, 3, 10, 12)
    with torch.no_grad():
        assert torch.allclose(model(x), model(torch.cat([x, torch.randn(1, 2, 3, 10, 12)], dim=1)), atol=1e-6)


def test_bad_stop_steps_rejected():
    with pytest.raises(ValueError):
        small(stop_steps=(7, 3))
    with pytest.raises(ValueError):
        small(stop_steps=(3, 3))


def test_forget_gate_bias_starts_at_one_and_state_starts_at_zero():
    cell = ConvLSTMCell(4, 6)
    bias = cell.x_gates.bias
    assert torch.all(bias[6:12] == 1.0) and torch.all(bias[:6] == 0) and torch.all(bias[12:] == 0)


def test_parameter_count_is_reported(capsys):
    n = sum(p.numel() for p in TornadoConvLSTMUNet(in_channels=14).parameters())
    n_unet = sum(p.numel() for p in TornadoUNet(in_channels=14).parameters())
    with capsys.disabled():
        print(f"\nTornadoConvLSTMUNet(14): {n:,} parameters (TornadoUNet(14): {n_unet:,})")
    assert n > n_unet


def test_checkpoint_roundtrip_through_the_registry(tmp_path):
    assert MODEL_REGISTRY["TornadoConvLSTMUNet"] is TornadoConvLSTMUNet
    torch.manual_seed(0)
    model = small(stop_steps=(3, 7)).eval()
    hp = dict(in_channels=3, base_channels=4, hidden_channels=8, stop_steps=[3, 7])
    path = tmp_path / "m.pt"
    torch.save({"model_class": "TornadoConvLSTMUNet", "model_state_dict": model.state_dict(), "model_hyperparameters": hp,
                "feature_names": ["a", "b", "c"]}, path)
    loaded, _ = load_checkpoint(str(path))
    x = torch.randn(1, 8, 3, 10, 12)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


@pytest.mark.skipif(not UNET_CKPT.exists(), reason="U-Net checkpoint not present")
def test_existing_unet_checkpoint_still_loads_unchanged():
    model, ckpt = load_checkpoint(str(UNET_CKPT))
    assert isinstance(model, TornadoUNet)
    with torch.no_grad():
        out = model(torch.zeros(1, 14, GRID.n_rows, GRID.n_cols))
    assert out.shape == (1, 1, GRID.n_rows, GRID.n_cols) and np.isfinite(out.numpy()).all()
