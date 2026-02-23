"""Flexible 3D U-Net with configurable depth, width, and convolutions per block.

Architecture
------------
Encoder:    [ConvBlock + MaxPool] × (num_levels - 1)
Bottleneck: ConvBlock
Decoder:    [ConvTranspose + Cat + ConvBlock] × (num_levels - 2)
Final:      ConvTranspose + Cat + HalfBlock + SegmenterBlock + 1×1 Conv + Sigmoid

Each ConvBlock contains ``n_convs_per_block`` units of Conv3d → ReLU → BN → Dropout.
The final decoder level splits the last block: (n_convs_per_block - 1) convs with
dropout ("half block") followed by 1 conv without dropout ("segmenter block"), so
every level has the same total number of 3×3 convolutions.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):
    """3D U-Net with configurable filters per level, convolutions per block,
    and dropout.

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

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder ---
        ch = in_channels
        for i, f in enumerate(n_filters_per_level[:-1]):
            self.encoders.append(
                self._conv_block(ch, f, n_convs_per_block, drop_prob, f"enc{i}_")
            )
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

        # --- Bottleneck ---
        self.bottleneck = self._conv_block(
            ch, n_filters_per_level[-1], n_convs_per_block, drop_prob, "bottleneck_"
        )

        # --- Decoder (all levels except the final one) ---
        for i in range(len(n_filters_per_level) - 2):
            in_f = n_filters_per_level[-1 - i]
            out_f = n_filters_per_level[-2 - i]
            self.upsamples.append(
                nn.ConvTranspose3d(in_f, out_f, kernel_size=2, stride=2)
            )
            self.decoders.append(
                self._conv_block(2 * out_f, out_f, n_convs_per_block, drop_prob, f"dec{i}_")
            )

        # --- Final decoder level: half-block + segmenter ---
        self.upsamples.append(
            nn.ConvTranspose3d(
                n_filters_per_level[1], n_filters_per_level[0],
                kernel_size=2, stride=2,
            )
        )
        n_final_convs = max(1, n_convs_per_block - 1)
        self.decoders.append(
            self._conv_block(
                2 * n_filters_per_level[0], n_filters_per_level[0],
                n_final_convs, drop_prob, "dec_final_",
            )
        )

        # --- Segmenter head (1 conv without dropout + 1×1 projection) ---
        self.segmenter = self._segmenter_block(
            n_filters_per_level[0], n_filters_per_level[0], "seg_"
        )
        self.out_conv = nn.Conv3d(n_filters_per_level[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips)
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

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

def _make_unet_small(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNet:
    """Small U-Net: 5 levels with doubling filters, 2 convs/block."""
    f = base_filters
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, 2*f, 4*f, 8*f, 16*f],
        n_convs_per_block=2,
        drop_prob=drop_prob,
    )


def _make_lunet_small(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNet:
    """Small LU-Net: 5 levels with constant filters, 2 convs/block."""
    f = base_filters
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, f, f, f, f],
        n_convs_per_block=2,
        drop_prob=drop_prob,
    )


def _make_unet_large(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNet:
    """Large U-Net: 4 levels with doubling filters, 3 convs/block."""
    f = base_filters
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters_per_level=[f, 2*f, 4*f, 8*f],
        n_convs_per_block=3,
        drop_prob=drop_prob,
    )


def _make_lunet_large(
    base_filters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    drop_prob: float = 0.0,
) -> UNet:
    """Large LU-Net: 4 levels with constant filters, 3 convs/block."""
    f = base_filters
    return UNet(
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

def UNet_small_4f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_small(4, in_channels, out_channels, drop_prob)

def UNet_small_3f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_small(3, in_channels, out_channels, drop_prob)

def UNet_small_2f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_small(2, in_channels, out_channels, drop_prob)

def UNet_small_1f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_small(1, in_channels, out_channels, drop_prob)

def LUNet_small_4f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_small(4, in_channels, out_channels, drop_prob)

def LUNet_small_3f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_small(3, in_channels, out_channels, drop_prob)

def LUNet_small_2f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_small(2, in_channels, out_channels, drop_prob)

def LUNet_small_1f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_small(1, in_channels, out_channels, drop_prob)

# --- Large (4-level, 3 convs/block) ---

def UNet_large_24f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_large(24, in_channels, out_channels, drop_prob)

def UNet_large_12f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_large(12, in_channels, out_channels, drop_prob)

def UNet_large_6f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_unet_large(6, in_channels, out_channels, drop_prob)

def LUNet_large_24f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_large(24, in_channels, out_channels, drop_prob)

def LUNet_large_12f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_large(12, in_channels, out_channels, drop_prob)

def LUNet_large_6f(in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0) -> UNet:
    return _make_lunet_large(6, in_channels, out_channels, drop_prob)
