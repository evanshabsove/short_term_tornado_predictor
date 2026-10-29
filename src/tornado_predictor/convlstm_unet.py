"""
ConvLSTM-U-Net for the hourly-sequence study (docs/convlstm_spec.md, ticket T2).

Input is one whole run, `(batch, 8, channels, row, col)`: the 8 hourly HRRR snapshots
(fxx 0-7). Output is both 4-hour bins at once, `(batch, 2, row, col)` raw logits.

Design, in the order data flows:

1. **Shared encoder per timestep.** The three `unet.py` encoder blocks plus its bottleneck
   run on every hour with the same weights (time is folded into the batch axis, so it is one
   big batched call, not a Python loop).
2. **ConvLSTM over the bottleneck features** (11x18 after the same pad-to-88x144). It
   carries a memory forward through the hours; the hidden state at hour t summarizes hours
   0..t.
3. **Two decode stops, one per bin** ("Option B" in the T2 plan). Bin b is decoded at the
   LAST hour of its own window (hour 3 for bin 0, hour 7 for bin 1) from the ConvLSTM hidden
   state at that hour plus the encoder skip features from that same hour. The decoder trunk
   is shared; each bin has its own 1x1 output head. So bin 0 sees exactly hours 0-3, the
   hours arm A pooled over, and the only thing that differs from arm A is that the temporal
   combination is learned instead of hand-pooled. A recurrence is causal, so hours 4-7
   cannot influence bin 0 at all (tested). **Disclosed difference:** bin 1's memory also
   contains hours 0-3, which arm A's bin 1 cannot see.

ConvLSTM cell: input-to-gate is a 1x1 conv (the encoder has already done the spatial work),
hidden-to-gate is a 3x3 conv. The forget-gate bias starts at 1 so memory is kept early in
training; initial hidden and cell state are zero. A 1x1 conv projects the hidden state back
to the bottleneck width for the decoder.

Reuses `unet.ConvBlock` / `unet.UpBlock` and the pad-once / crop-once handling of the
non-power-of-2 grid. Outputs raw logits like every other model here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tornado_predictor.time_bins import N_BINS, fxx_values_for_bin
from tornado_predictor.unet import DOWNSAMPLE_FACTOR, ConvBlock, UpBlock

# Last hour of each bin's window: the step at which that bin is decoded.
DEFAULT_STOP_STEPS: tuple[int, ...] = tuple(fxx_values_for_bin(b)[-1] for b in range(N_BINS))


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.x_gates = nn.Conv2d(in_channels, 4 * hidden_channels, kernel_size=1)
        self.h_gates = nn.Conv2d(hidden_channels, 4 * hidden_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.x_gates.bias.zero_()
            self.x_gates.bias[hidden_channels : 2 * hidden_channels] = 1.0  # forget gate (gate order i, f, g, o)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.x_gates(x) + self.h_gates(h)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class TornadoConvLSTMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        hidden_channels: int = 64,
        stop_steps: tuple[int, ...] | list[int] = DEFAULT_STOP_STEPS,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stop_steps = tuple(int(s) for s in stop_steps)
        if list(self.stop_steps) != sorted(set(self.stop_steps)) or min(self.stop_steps) < 0:
            raise ValueError(f"stop_steps must be distinct, increasing, non-negative: {self.stop_steps}")

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.lstm = ConvLSTMCell(c4, hidden_channels)
        self.hidden_to_bottleneck = nn.Conv2d(hidden_channels, c4, kernel_size=1)

        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.heads = nn.ModuleList([nn.Conv2d(c1, 1, kernel_size=1) for _ in self.stop_steps])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, steps, in_channels, row, col) -> logits (batch, n_stops, row, col)."""
        b, t, ch, h, w = x.shape
        if t <= self.stop_steps[-1]:
            raise ValueError(f"need at least {self.stop_steps[-1] + 1} steps, got {t}")
        x = x[:, : self.stop_steps[-1] + 1]  # later hours cannot matter to any stop
        t = x.shape[1]
        pad_h = (-h) % DOWNSAMPLE_FACTOR
        pad_w = (-w) % DOWNSAMPLE_FACTOR
        x = F.pad(x.reshape(b * t, ch, h, w), (0, pad_w, 0, pad_h))

        # shared encoder over every hour at once (time folded into the batch axis)
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        bottom = self.bottleneck(self.pool(skip3))

        bh, bw = bottom.shape[-2:]
        bottom_seq = bottom.reshape(b, t, -1, bh, bw)
        state = (bottom.new_zeros(b, self.lstm.hidden_channels, bh, bw), bottom.new_zeros(b, self.lstm.hidden_channels, bh, bw))
        hidden_at_stop = {}
        for step in range(t):
            state = self.lstm(bottom_seq[:, step], state)
            if step in self.stop_steps:
                hidden_at_stop[step] = state[0]

        def at(skip: torch.Tensor, step: int) -> torch.Tensor:
            return skip.reshape(b, t, *skip.shape[1:])[:, step]

        outputs = []
        for head, step in zip(self.heads, self.stop_steps):
            d3 = self.up3(self.hidden_to_bottleneck(hidden_at_stop[step]), at(skip3, step))
            d2 = self.up2(d3, at(skip2, step))
            d1 = self.up1(d2, at(skip1, step))
            outputs.append(head(d1))
        return torch.cat(outputs, dim=1)[..., :h, :w]
