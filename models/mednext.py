"""MedNeXt Small (V1) – 3D encoder-decoder with ConvNeXt-style blocks.

Based on the MedNeXt architecture by Roy et al.:

    S. Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for
    Medical Image Segmentation," MICCAI 2023.
    https://arxiv.org/abs/2303.09975

Reference implementation (Apache 2.0 License):
    https://github.com/MIC-DKFZ/MedNeXt

Architecture
------------
Encoder:    [MedNeXtBlock × N + DownBlock] × 4
Bottleneck: MedNeXtBlock × N
Decoder:    [UpBlock + Add skip + MedNeXtBlock × N] × 4
Final:      1×1 Conv → Sigmoid

Each MedNeXtBlock uses depthwise-separable convolutions with an inverted
bottleneck (expand → GELU → compress) and a residual connection.  Down/Up
blocks use strided (transposed) depthwise convolutions with matching residual
projections so spatial dimensions halve/double cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Building blocks
# ======================================================================

class MedNeXtBlock(nn.Module):
    """Inverted-bottleneck block with depthwise-separable 3D convolutions.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel dimensions (typically equal for identity-residual blocks).
    exp_r : int
        Expansion ratio for the hidden pointwise layer.
    kernel_size : int
        Spatial kernel size for the depthwise convolution.
    norm_type : ``"group"`` | ``"layer"``
        Normalisation strategy (group = one group per channel).
    grn : bool
        Enable Global Response Normalisation (V2 option, off by default).
    drop_prob : float
        Spatial dropout probability after the activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        norm_type: str = "group",
        grn: bool = False,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()

        hidden = exp_r * in_channels

        # 1) Depthwise spatial convolution
        self.conv_dw = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            groups=in_channels,
        )

        # 2) Normalisation
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        else:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        # 3) Pointwise expansion
        self.conv_expand = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(p=drop_prob) if drop_prob > 0 else nn.Identity()

        # 4) Pointwise compression
        self.conv_compress = nn.Conv3d(hidden, out_channels, kernel_size=1)

        # Optional Global Response Normalisation
        self.grn = grn
        if grn:
            self.grn_beta = nn.Parameter(torch.zeros(1, hidden, 1, 1, 1))
            self.grn_gamma = nn.Parameter(torch.zeros(1, hidden, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.conv_expand(x)
        x = self.act(x)

        if self.grn:
            gx = torch.norm(x, p=2, dim=(-3, -2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x = self.grn_gamma * (x * nx) + self.grn_beta + x

        x = self.drop(x)
        x = self.conv_compress(x)

        return residual + x


class MedNeXtDownBlock(MedNeXtBlock):
    """Strided MedNeXt block that halves spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        norm_type: str = "group",
    ) -> None:
        super().__init__(
            in_channels, out_channels, exp_r, kernel_size,
            norm_type=norm_type, grn=False, drop_prob=0.0,
        )
        # Strided depthwise conv (overrides parent)
        self.conv_dw = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=2, padding=kernel_size // 2,
            groups=in_channels,
        )
        # 1×1 strided projection for the residual path
        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.res_conv(x)

        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.conv_expand(x)
        x = self.act(x)
        x = self.conv_compress(x)

        return x_res + x


class MedNeXtUpBlock(MedNeXtBlock):
    """Transposed MedNeXt block that doubles spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        norm_type: str = "group",
    ) -> None:
        super().__init__(
            in_channels, out_channels, exp_r, kernel_size,
            norm_type=norm_type, grn=False, drop_prob=0.0,
        )
        # Transposed depthwise conv (overrides parent)
        self.conv_dw = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=2, padding=kernel_size // 2,
            groups=in_channels,
        )
        # 1×1 transposed projection for the residual path
        self.res_conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=1, stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.res_conv(x)

        x = self.conv_dw(x)
        # Pad to match encoder spatial dims (stride-2 transpose with k=3 yields 2L-1)
        x = F.pad(x, (1, 0, 1, 0, 1, 0))
        x_res = F.pad(x_res, (1, 0, 1, 0, 1, 0))

        x = self.norm(x)
        x = self.conv_expand(x)
        x = self.act(x)
        x = self.conv_compress(x)

        return x_res + x


# ======================================================================
# Full model
# ======================================================================

class MedNeXt(nn.Module):
    """MedNeXt Small (V1) 3D segmentation network.

    Mirrors the encoder–bottleneck–decoder layout of the reference UNet so it
    can be used as a drop-in replacement inside the training framework.

    Parameters
    ----------
    in_channels : int
        Number of input modalities.
    out_channels : int
        Number of output segmentation classes.
    init_features : int
        Base channel width (doubled at each encoder level).
    drop_prob : float
        Dropout probability applied inside ``MedNeXtBlock``.
    block_counts : list[int]
        Number of MedNeXt blocks per stage (9 entries: 4 encoder + 1
        bottleneck + 4 decoder).  Defaults to ``[2]*9`` (Small config).
    exp_r : int
        Expansion ratio for the inverted bottleneck.
    kernel_size : int
        Depthwise convolution kernel size.
    norm_type : ``"group"`` | ``"layer"``
    grn : bool
        Enable Global Response Normalisation (off for V1).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 4,
        drop_prob: float = 0.0,
        block_counts: list[int] | None = None,
        exp_r: int = 2,
        kernel_size: int = 3,
        norm_type: str = "group",
        grn: bool = False,
    ) -> None:
        super().__init__()

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        self.exp_r = exp_r
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.grn = grn

        # Build per-block dropout schedule (17 slots, uniform by default)
        self._drop_probs = {i: drop_prob for i in range(17)}

        n = init_features

        # ---- Stem ----
        self.stem = nn.Conv3d(in_channels, n, kernel_size=1)

        # ---- Encoder ----
        self.enc_block_0 = self._make_stage(n, n, block_counts[0], start_key=0)
        self.down_0 = MedNeXtDownBlock(n, 2 * n, exp_r, kernel_size, norm_type)
        self.enc_block_1 = self._make_stage(2 * n, 2 * n, block_counts[1], start_key=2)
        self.down_1 = MedNeXtDownBlock(2 * n, 4 * n, exp_r, kernel_size, norm_type)
        self.enc_block_2 = self._make_stage(4 * n, 4 * n, block_counts[2], start_key=4)
        self.down_2 = MedNeXtDownBlock(4 * n, 8 * n, exp_r, kernel_size, norm_type)
        self.enc_block_3 = self._make_stage(8 * n, 8 * n, block_counts[3], start_key=6)

        # ---- Bottleneck ----
        self.down_3 = MedNeXtDownBlock(8 * n, 16 * n, exp_r, kernel_size, norm_type)
        self.bottleneck = self._make_stage(16 * n, 16 * n, block_counts[4], start_key=8)

        # ---- Decoder ----
        self.up_3 = MedNeXtUpBlock(16 * n, 8 * n, exp_r, kernel_size, norm_type)
        self.dec_block_3 = self._make_stage(8 * n, 8 * n, block_counts[5], start_key=10)
        self.up_2 = MedNeXtUpBlock(8 * n, 4 * n, exp_r, kernel_size, norm_type)
        self.dec_block_2 = self._make_stage(4 * n, 4 * n, block_counts[6], start_key=12)
        self.up_1 = MedNeXtUpBlock(4 * n, 2 * n, exp_r, kernel_size, norm_type)
        self.dec_block_1 = self._make_stage(2 * n, 2 * n, block_counts[7], start_key=14)
        self.up_0 = MedNeXtUpBlock(2 * n, n, exp_r, kernel_size, norm_type)
        self.dec_block_0 = self._make_stage(n, n, block_counts[8], start_key=16)

        # ---- Output head ----
        self.out_conv = nn.Conv3d(n, out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_stage(
        self, in_c: int, out_c: int, count: int, start_key: int,
    ) -> nn.Sequential:
        """Create a sequence of ``count`` MedNeXtBlocks with cycling dropout keys."""
        return nn.Sequential(*[
            MedNeXtBlock(
                in_channels=in_c,
                out_channels=out_c,
                exp_r=self.exp_r,
                kernel_size=self.kernel_size,
                norm_type=self.norm_type,
                grn=self.grn,
                drop_prob=self._drop_probs[(start_key + i) % 17],
            )
            for i in range(count)
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Encoder
        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)

        # Bottleneck
        x = self.down_3(x_res_3)
        x = self.bottleneck(x)

        # Decoder (additive skip connections)
        x = x_res_3 + self.up_3(x)
        x = self.dec_block_3(x)
        x = x_res_2 + self.up_2(x)
        x = self.dec_block_2(x)
        x = x_res_1 + self.up_1(x)
        x = self.dec_block_1(x)
        x = x_res_0 + self.up_0(x)
        x = self.dec_block_0(x)

        # Output head
        x = self.out_conv(x)
        return torch.sigmoid(x)


# ======================================================================
# Presets
# ======================================================================

def MedNeXt_small(
    in_channels: int = 1, out_channels: int = 1, drop_prob: float = 0.0,
) -> MedNeXt:
    return MedNeXt(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=4,
        drop_prob=drop_prob,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        exp_r=2,
        kernel_size=3,
    )
