"""
Minimal baseline CNN for dense-grid tornado forecasting: takes pooled
HRRR feature channels (channels, row, col) and outputs a per-cell
tornado logit map (row, col).

Deliberately simple: full-resolution convolutions only, no
pooling/upsampling. The grid's shape (81, 138) isn't a clean power of
2, so a U-Net's downsample/upsample skip-connection alignment would add
complexity not worth it for a first baseline -- padding=1 keeps every
layer at the input's exact spatial resolution instead.

This is a first baseline, not a final architecture. Its receptive
field is small (three stacked 3x3 convs -> ~7x7 cells, ~270km at this
grid's 39km cell size) -- likely too small to capture full supercell-
or synoptic-scale context. Widening this (more layers, dilated convs,
or a real U-Net with careful padding/cropping) is a natural next
iteration once this baseline is confirmed to train end-to-end.

Outputs raw logits, not probabilities -- this matches
training.FocalLoss's expectation (it calls sigmoid internally, via
binary_cross_entropy_with_logits, which is more numerically stable
than training on already-sigmoided probabilities). Apply
torch.sigmoid(model(x)) to get a probability map at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TornadoCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, row, col) -> logits: (batch, 1, row, col)"""
        return self.net(x)
