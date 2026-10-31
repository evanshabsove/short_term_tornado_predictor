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


REPAIRS = [dict(norm_bottleneck=True), dict(pooled_skips=True), dict(norm_bottleneck=True, pooled_skips=True)]


def test_default_model_has_no_repair_parameters_so_old_checkpoints_load():
    keys = set(small().state_dict())
    assert not any(k.startswith(("norm_bottleneck", "skip_merge")) for k in keys)
    assert any(k.startswith("skip_merge") for k in small(pooled_skips=True).state_dict())
    assert any(k.startswith("norm_bottleneck") for k in small(norm_bottleneck=True).state_dict())
    small().load_state_dict(small().state_dict())  # strict load of the default layout


@pytest.mark.parametrize("kw", REPAIRS)
def test_repaired_variants_have_the_same_output_shape_and_every_parameter_gets_gradient(kw):
    torch.manual_seed(0)
    model = TornadoConvLSTMUNet(in_channels=3, base_channels=8, hidden_channels=8, **kw)  # base 8: GroupNorm(8, 8 * 8) needs 64 channels
    x = torch.randn(2, 8, 3, 10, 12)
    out = model(x)
    assert out.shape == (2, 2, 10, 12) and torch.isfinite(out).all()
    FocalLoss()(out, (torch.rand(2, 2, 10, 12) > 0.5).float()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and (p.grad != 0).any(), name


@pytest.mark.parametrize("kw", REPAIRS)
def test_repaired_variants_keep_the_causality_contract(kw):
    torch.manual_seed(0)
    model = TornadoConvLSTMUNet(in_channels=3, base_channels=8, hidden_channels=8, **kw).eval()
    x = torch.randn(1, 8, 3, 10, 12)

    def sensitivity(bin_index):
        xg = x.clone().requires_grad_(True)
        model(xg)[:, bin_index].sum().backward()
        return xg.grad.abs().sum(dim=(0, 2, 3, 4))

    s0, s1 = sensitivity(0), sensitivity(1)
    assert torch.all(s0[:4] > 0) and torch.all(s0[4:] == 0)  # bin 0: exactly hours 0-3
    assert torch.all(s1 > 0)  # bin 1: hours 4-7 plus the memory of hours 0-3


def test_bin_1_pooled_skips_use_only_its_own_hours():
    """With the memory path cut (zero hidden state), bin 1's pooled skips must not depend on hours 0-3."""
    torch.manual_seed(0)
    model = TornadoConvLSTMUNet(in_channels=3, base_channels=8, hidden_channels=8, pooled_skips=True).eval()
    for p in list(model.lstm.parameters()) + list(model.hidden_to_bottleneck.parameters()):
        torch.nn.init.zeros_(p)  # no information through the LSTM, only through the skips
    x = torch.randn(1, 8, 3, 10, 12)
    early = x.clone()
    early[:, :4] += torch.randn_like(early[:, :4])
    with torch.no_grad():
        a, b = model(x), model(early)
    assert torch.allclose(a[:, 1], b[:, 1], atol=1e-6)  # bin 1 ignores hours 0-3 when only skips carry data
    assert not torch.allclose(a[:, 0], b[:, 0], atol=1e-6)  # bin 0 uses them


def test_bottleneck_norm_keeps_the_gates_out_of_saturation_when_encoder_scale_drifts():
    """The failure seen in the first full study: training grew the encoder's activations until the gate pre-activations reached +-1e5.
    (An untrained model is fine, so the drift is simulated by scaling up the last bottleneck conv.)"""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 3, 10, 12)

    def gate_std(model):
        with torch.no_grad():
            conv = model.bottleneck.net[2]
            conv.weight.mul_(1e4)
            conv.bias.mul_(1e4)
        seen = []
        orig = model.lstm.forward
        model.lstm.forward = lambda xx, st: (seen.append((model.lstm.x_gates(xx) + model.lstm.h_gates(st[0])).detach()), orig(xx, st))[1]
        with torch.no_grad():
            model.eval()(x)
        return torch.stack(seen).std().item()

    plain = gate_std(TornadoConvLSTMUNet(in_channels=3, base_channels=8, hidden_channels=8))
    normed = gate_std(TornadoConvLSTMUNet(in_channels=3, base_channels=8, hidden_channels=8, norm_bottleneck=True))
    assert plain > 100 and normed < 10
