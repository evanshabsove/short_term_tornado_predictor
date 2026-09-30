"""
A real U-Net for dense-grid tornado forecasting -- downsampling to
build a large receptive field cheaply, then upsampling back to full
resolution with skip connections so the dense per-cell output doesn't
lose fine spatial detail. Same shape TorNet's own benchmark backbone
uses (4 downsampling blocks, growing channels).

The grid (81x138) isn't a power of 2 -- model.py's docstring has
flagged this since the first baseline as the reason a U-Net was
deferred. Resolved here with the simpler of two options: pad to a
compatible shape once, crop back once (not per-block asymmetric
padding, which needs bookkeeping padding offsets at every skip
connection). 3 downsample stages need H, W divisible by 8: 81 -> 88
(+7), 138 -> 144 (+6). Padding right/bottom only (via
F.pad(x, (0, 6, 0, 7)), PyTorch's (left, right, top, bottom) order)
puts the real 81x138 region in the padded canvas's top-left corner, so
cropping back is a trivial output[..., :81, :138] -- no offset
bookkeeping needed. The downsample sequence is then exact, no further
rounding: 88x144 -> 44x72 -> 22x36 -> 11x18 (bottleneck).

Channel width is deliberately conservative (base_channels=16, not
TorNet's 64): tickets 2-3 (see CLAUDE.md's "architecture widening"
sections) found growing receptive field helps, but growing raw
parameter count without more spatial context doesn't and can actively
overfit (a 607K-parameter flat CNN already showed a real train/val loss
gap). A naive U-Net at TorNet-like widths would land around ~1.8M
parameters by rough estimate -- well past that point. Starting narrow
applies that lesson on purpose rather than re-learning it.

Bilinear upsampling + a 1x1 conv (not ConvTranspose2d) is used for each
decoder step, avoiding the checkerboard-artifact failure mode
transposed convolutions are known for.

Outputs raw logits, not probabilities -- same convention as
TornadoCNN, see model.py's docstring.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DOWNSAMPLE_STAGES = 3
DOWNSAMPLE_FACTOR = 2 ** DOWNSAMPLE_STAGES


class ConvBlock(nn.Module):
    """Two (Conv3x3 pad=1, ReLU) pairs at a fixed resolution -- the
    standard U-Net "double conv" block, matching TornadoCNN's existing
    no-batchnorm, minimal style."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Bilinear 2x upsample + 1x1 channel-reduce, concat with the
    matching encoder skip, then a ConvBlock back down to out_channels."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.conv = ConvBlock(skip_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TornadoUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, row, col) -> logits: (batch, 1, row, col)
        -- output spatial shape exactly matches input, any (row, col)."""
        _, _, h, w = x.shape
        pad_h = (-h) % DOWNSAMPLE_FACTOR
        pad_w = (-w) % DOWNSAMPLE_FACTOR
        x = F.pad(x, (0, pad_w, 0, pad_h))

        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        bottom = self.bottleneck(self.pool(skip3))

        d3 = self.up3(bottom, skip3)
        d2 = self.up2(d3, skip2)
        d1 = self.up1(d2, skip1)

        out = self.head(d1)
        return out[..., :h, :w]
