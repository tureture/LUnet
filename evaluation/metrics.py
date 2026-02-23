"""Segmentation metrics for 3-D volumes."""

from __future__ import annotations

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    exclude_bg: bool = True,
) -> float:
    """Mean Dice coefficient over foreground classes.

    Parameters
    ----------
    preds : (B, D, H, W)  long – predicted class indices.
    targets : (B, D, H, W) long – ground-truth class indices.
    num_classes : int or None – inferred from data if None.
    exclude_bg : bool – ignore class 0 (background).
    """
    if num_classes is None:
        num_classes = int(max(preds.max(), targets.max()) + 1)

    start_cls = 1 if exclude_bg else 0
    dices: list[float] = []
    for c in range(start_cls, num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item()
        if union == 0:
            dices.append(1.0 if inter == 0 else 0.0)
        else:
            dices.append(2.0 * inter / union)
    return float(np.mean(dices)) if dices else 0.0


def hausdorff_distance_95(
    preds: torch.Tensor,
    targets: torch.Tensor,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """95th-percentile Hausdorff distance (averaged over foreground classes).

    Works on a single sample (no batch dim) for simplicity; caller should
    iterate over the batch.
    """
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    num_classes = int(max(preds_np.max(), targets_np.max()) + 1)

    hd95s: list[float] = []
    for c in range(1, num_classes):  # skip background
        p_mask = (preds_np == c)
        t_mask = (targets_np == c)
        if not p_mask.any() and not t_mask.any():
            hd95s.append(0.0)
            continue
        if not p_mask.any() or not t_mask.any():
            # One is empty → maximum possible distance
            hd95s.append(np.inf)
            continue

        # Distance transforms
        dt_pred = distance_transform_edt(~p_mask, sampling=voxel_spacing)
        dt_target = distance_transform_edt(~t_mask, sampling=voxel_spacing)

        # Surface distances
        d_p2t = dt_target[p_mask]
        d_t2p = dt_pred[t_mask]
        all_distances = np.concatenate([d_p2t, d_t2p])
        hd95s.append(float(np.percentile(all_distances, 95)))

    return float(np.mean(hd95s)) if hd95s else 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute requested metrics and return as a dict."""
    results: dict[str, float] = {}
    for name in metric_names:
        if name == "dice":
            results["dice"] = dice_coefficient(preds, targets)
        elif name == "hausdorff95":
            # Process per-sample, then average
            vals = []
            for i in range(preds.shape[0]):
                vals.append(hausdorff_distance_95(preds[i], targets[i]))
            results["hausdorff95"] = float(np.mean(vals))
        else:
            raise ValueError(f"Unknown metric: {name}")
    return results
