"""
Baseline CNN family for dense-grid tornado forecasting: takes pooled
HRRR feature channels (channels, row, col) and outputs a per-cell
tornado logit map (row, col).

Deliberately simple: full-resolution convolutions only, no
pooling/upsampling. The grid's shape (81, 138) isn't a clean power of
2, so a U-Net's downsample/upsample skip-connection alignment would add
complexity not worth it for a first baseline -- padding=1 keeps every
layer at the input's exact spatial resolution instead.

`n_hidden_layers` (default 3), `hidden_channels` (default 32), and
`dilations` (default `None` -> all 1s, i.e. no dilation) are all
configurable -- the original, hardcoded-3-layer/32-channel/no-dilation
design is exactly the default, not a special case: every existing
checkpoint (none of which have `n_hidden_layers`/`dilations` in their
saved `model_hyperparameters`, since these parameters didn't exist when
they were trained) still loads and reconstructs identically via
`inference.load_checkpoint`'s
`TornadoCNN(**checkpoint["model_hyperparameters"])`. Widening this
(more layers, more channels, dilated convs, or a real U-Net with
careful padding/cropping) was always the flagged next iteration once a
first baseline trained end-to-end -- see CLAUDE.md's "architecture
widening" tickets for the evaluated variants and results
(`models/architecture_sweep/results.json`,
`models/dilation_sweep/results.json`).

Receptive field: each stacked 3x3 conv with dilation `d` (stride 1)
adds `2*d` to the per-dimension receptive field (a plain, undilated
conv is the `d=1` special case), so a model with per-layer dilations
`[d_1, ..., d_L]` sees `1 + 2*sum(d_i)` cells in each dimension --
7x7 cells (~270km at this grid's 39km cell size) at the original
all-1s default of 3 layers, 11x11 (~430km) at 5 plain layers, and much
larger for genuinely dilated schedules (e.g. `[1,2,4,8]` -> 31x31
cells, ~1209km -- computed exactly, not estimated) without adding
layers or channels. Dilated convs are how this project reaches
synoptic-scale context cheaply, since stacking more plain 3x3 layers
(ticket 2) grows receptive field only linearly with layer count (and
therefore with parameter count/compute) -- see CLAUDE.md's "dilated
convolutions" ticket for the real cost comparison. `padding=dilation`
(not the fixed `padding=1` a plain conv uses) is what keeps spatial
dimensions exactly unchanged at any dilation rate.

Outputs raw logits, not probabilities -- this matches
training.FocalLoss's expectation (it calls sigmoid internally, via
binary_cross_entropy_with_logits, which is more numerically stable
than training on already-sigmoided probabilities). Apply
torch.sigmoid(model(x)) to get a probability map at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def receptive_field_cells(n_hidden_layers: int, dilations: list[int] | None = None) -> int:
    """Per-dimension receptive field (in grid cells) of a TornadoCNN
    with n_hidden_layers stacked 3x3 convs at the given per-layer
    dilation rates (default: all 1s, i.e. no dilation)."""
    dilations = dilations if dilations is not None else [1] * n_hidden_layers
    return 1 + 2 * sum(dilations)


class TornadoCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        n_hidden_layers: int = 3,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1] * n_hidden_layers
        if len(dilations) != n_hidden_layers:
            raise ValueError(
                f"dilations must have length n_hidden_layers ({n_hidden_layers}), got {len(dilations)}: {dilations}"
            )

        layers: list[nn.Module] = []
        channels = in_channels
        for dilation in dilations:
            layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            channels = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, row, col) -> logits: (batch, 1, row, col)"""
        return self.net(x)
