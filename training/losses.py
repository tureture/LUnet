"""Loss functions for 3-D segmentation.

All losses expect:
- **predictions**: (B, C, D, H, W) float — sigmoid probabilities from the model
- **targets**:     (B, C, D, H, W) float — ground-truth masks (0/1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Soft Dice loss.

    Computes a global Dice coefficient per class (summing over the batch and
    spatial dimensions together), then averages across classes.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type(probs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice = (2.0 * intersection / (cardinality + self.eps)).mean()
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss (weighted sum)."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCELoss()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_float = targets.type(probs.type())
        return self.dw * self.dice(probs, targets_float) + self.bw * self.bce(probs, targets_float)


def build_loss(cfg: dict) -> nn.Module:
    """Build a loss function from the config."""
    lcfg = cfg["training"]["loss"]
    name = lcfg["name"]

    if name == "dice":
        return SoftDiceLoss()
    elif name == "bce":
        return nn.BCELoss()
    elif name == "dice_bce":
        return DiceBCELoss(
            dice_weight=lcfg.get("dice_weight", 1.0),
            bce_weight=lcfg.get("bce_weight", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss: {name}")
