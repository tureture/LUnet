"""UNet++ (Nested U-Net) – 3D encoder-decoder with dense skip connections.

This module provides two model families:

1. **UNet++** — a faithful 3D re-implementation of the original UNet++
   architecture by Zhou et al.  The original 2D Keras/TensorFlow code is
   available at https://github.com/MrGiovanni/UNetPlusPlus and is used here
   as a reference.  See "License" section below.

2. **LUNet++** (Lean UNet++) — a *novel* architecture proposed in this work.
   LUNet++ retains the nested dense-skip topology of UNet++ but replaces the
   conventional doubling filter schedule (e.g. [4, 8, 16, 32, 64]) with a
   *constant* filter width across all encoder/decoder levels
   (e.g. [4, 4, 4, 4, 4]).  This dramatically reduces parameter count and
   memory footprint while preserving multi-scale feature aggregation, making
   the model particularly suited to resource-constrained 3D medical-image
   segmentation tasks.

Modifications to the original UNet++ code
------------------------------------------
  - Converted from 2D (Keras/TensorFlow) to a 3D PyTorch implementation.
  - Made the number of filters per level, convolutions per block, and
    dropout probability fully configurable.
  - Added a dedicated segmenter head (conv without dropout + 1×1 projection).
  - Introduced preset factory functions (``UNetPP_*`` / ``LUNetPP_*``).

Architecture (shared by UNet++ and LUNet++)
-------------------------------------------
Backbone:  [ConvBlock + MaxPool] × (L-1) + Bottleneck ConvBlock
Nested:    Dense skip-connection grid — node (i, j) aggregates *j* same-level
           features and one upsampled deeper-level feature via concatenation.
Final:     HalfBlock + SegmenterBlock + 1×1 Conv + Sigmoid

Each ConvBlock contains ``n_convs_per_block`` units of Conv3d → ReLU → BN →
Dropout.  The final output node splits the last block: (n_convs_per_block - 1)
convs with dropout followed by 1 conv without dropout ("segmenter block"),
matching the reference UNet convention.

Reference
---------
Z. Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
Segmentation," DLMIA/ML-CDS 2018.
https://arxiv.org/abs/1807.10165

License (original UNet++ code)
------------------------------
Copyright 2019 Arizona Board of Regents for and on behalf of Arizona State
University, a body corporate. Patent Pending in the United States of America.

The original Work is licensed solely for **non-commercial / academic research**
use. Derivative Works (including this file) may be reproduced and distributed
under the same non-commercial terms, provided that:
  (a) Recipients receive a copy of the original license;
  (b) Modified files carry prominent notices of changes (see above);
  (c) All original copyright and patent notices are retained.

For the full license text see:
https://github.com/MrGiovanni/UNetPlusPlus/blob/master/LICENSE

Any commercial use requires a separate written agreement with Arizona State
University.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class UNetPlusPlus(nn.Module):
    """3D UNet++ with configurable filters per level, convolutions per block,
    and dropout.

    Use doubling filter values (e.g. ``[4, 8, 16, 32, 64]``) for standard
    UNet++, or constant values (e.g. ``[4, 4, 4, 4, 4]``) for a lean/flat
    variant.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_filters_per_level : list[int]
        Feature channels at each level (last entry = bottleneck).
    n_convs_per_block : int
        Number of Conv3d-ReLU-BN-Dropout units per encoder/decoder block.
    drop_prob : float
        Dropout probability applied after each conv unit.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_filters_per_level: list[int] | None = None,
        n_convs_per_block: int = 2,
        drop_prob: float = 0.005,
    ) -> None:
        super().__init__()
        if n_filters_per_level is None:
            n_filters_per_level = [4, 8, 16, 32, 64]

        self.n_levels = L = len(n_filters_per_level)
        F = n_filters_per_level

        # --- Encoder (column 0) ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(L):
            self.encoders.append(
                self._conv_block(ch, F[i], n_convs_per_block, drop_prob, f"enc{i}_")
            )
            if i < L - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = F[i]

        # --- Nested decoder nodes ---
        # Node (i, j) receives j same-level features + 1 upsampled feature.
        self.upsamples = nn.ModuleDict()
        self.nested_blocks = nn.ModuleDict()

        for j in range(1, L):
            for i in range(L - j):
                key = f"{i}_{j}"

                self.upsamples[key] = nn.ConvTranspose3d(
                    F[i + 1], F[i], kernel_size=2, stride=2,
                )

                in_ch = (j + 1) * F[i]

                # Final output node uses fewer convs (half-block)
                if i == 0 and j == L - 1:
                    n_convs = max(1, n_convs_per_block - 1)
                else:
                    n_convs = n_convs_per_block

                self.nested_blocks[key] = self._conv_block(
                    in_ch, F[i], n_convs, drop_prob, f"node_{key}_",
                )

        # --- Segmenter head (1 conv without dropout + 1×1 projection) ---
        self.segmenter = self._segmenter_block(F[0], F[0], "seg_")
        self.out_conv = nn.Conv3d(F[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.n_levels
        nodes: dict[tuple[int, int], torch.Tensor] = {}

        # Encoder backbone (column 0)
        for i in range(L):
            inp = x if i == 0 else self.pools[i - 1](nodes[(i - 1, 0)])
            nodes[(i, 0)] = self.encoders[i](inp)

        # Nested decoder (columns 1 … L-1)
        for j in range(1, L):
            for i in range(L - j):
                key = f"{i}_{j}"
                up = self.upsamples[key](nodes[(i + 1, j - 1)])
                dense = [nodes[(i, k)] for k in range(j)]
                dense.append(up)
                nodes[(i, j)] = self.nested_blocks[key](torch.cat(dense, dim=1))

        # Output head
        x = nodes[(0, L - 1)]
        x = self.segmenter(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)

    # ------------------------------------------------------------------
    # Block constructors
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        n_convs: int,
        drop_prob: float,
        name: str,
    ) -> nn.Sequential:
        """Stack of [Conv3d → ReLU → BatchNorm → Dropout] units."""
        layers: list[tuple[str, nn.Module]] = []
        ch = in_channels
        for i in range(n_convs):
            layers.extend([
                (f"{name}conv{i}", nn.Conv3d(ch, out_channels, kernel_size=3, padding=1, bias=True)),
                (f"{name}relu{i}", nn.ReLU(inplace=True)),
                (f"{name}bn{i}", nn.BatchNorm3d(out_channels)),
                (f"{name}drop{i}", nn.Dropout3d(p=drop_prob)),
            ])
            ch = out_channels
        return nn.Sequential(OrderedDict(layers))

    @staticmethod
    def _segmenter_block(
        in_channels: int,
        out_channels: int,
        name: str,
    ) -> nn.Sequential:
        """Single Conv3d → ReLU → BatchNorm (no dropout)."""
        return nn.Sequential(OrderedDict([
            (f"{name}conv", nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)),
            (f"{name}relu", nn.ReLU(inplace=True)),
            (f"{name}bn", nn.BatchNorm3d(out_channels)),
        ]))


# ======================================================================
# Preset factories
# ======================================================================

def _make_unetpp_small(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNetPlusPlus:
    """Small UNet++: 5 levels with doubling filters, 2 convs/block."""
    f = base_filters
    return UNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, 2*f, 4*f, 8*f, 16*f],
        n_convs_per_block=2,
        drop_prob=drop_prob,
    )


def _make_lunetpp_small(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNetPlusPlus:
    """Small LUNet++: 5 levels with constant filters, 2 convs/block."""
    f = base_filters
    return UNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, f, f, f, f],
        n_convs_per_block=2,
        drop_prob=drop_prob,
    )


def _make_unetpp_large(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNetPlusPlus:
    """Large UNet++: 4 levels with doubling filters, 3 convs/block."""
    f = base_filters
    return UNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, 2*f, 4*f, 8*f],
        n_convs_per_block=3,
        drop_prob=drop_prob,
    )


def _make_lunetpp_large(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNetPlusPlus:
    """Large LUNet++: 4 levels with constant filters, 3 convs/block."""
    f = base_filters
    return UNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, f, f, f],
        n_convs_per_block=3,
        drop_prob=drop_prob,
    )


# ======================================================================
# Named presets
# ======================================================================

# --- Small (5-level, 2 convs/block) ---

def UNetPP_small_4f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_small(4, in_channels, out_channels, drop_prob)

def UNetPP_small_3f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_small(3, in_channels, out_channels, drop_prob)

def UNetPP_small_2f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_small(2, in_channels, out_channels, drop_prob)

def UNetPP_small_1f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_small(1, in_channels, out_channels, drop_prob)

def LUNetPP_small_4f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_small(4, in_channels, out_channels, drop_prob)

def LUNetPP_small_3f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_small(3, in_channels, out_channels, drop_prob)

def LUNetPP_small_2f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_small(2, in_channels, out_channels, drop_prob)

def LUNetPP_small_1f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_small(1, in_channels, out_channels, drop_prob)

# --- Large (4-level, 3 convs/block) ---

def UNetPP_large_24f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_large(24, in_channels, out_channels, drop_prob)

def UNetPP_large_12f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_large(12, in_channels, out_channels, drop_prob)

def UNetPP_large_6f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_unetpp_large(6, in_channels, out_channels, drop_prob)

def LUNetPP_large_24f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_large(24, in_channels, out_channels, drop_prob)

def LUNetPP_large_12f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_large(12, in_channels, out_channels, drop_prob)

def LUNetPP_large_6f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNetPlusPlus:
    return _make_lunetpp_large(6, in_channels, out_channels, drop_prob)
