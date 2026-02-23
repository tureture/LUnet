"""Visualization helpers – training curves and 3-D slice viewers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict[str, list[float]],
    save_dir: str | Path,
) -> None:
    """Plot loss, Dice, and learning-rate curves and save as PNG."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history["val_dice"], label="Val Dice", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history["lr"], label="LR", color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slice comparison
# ---------------------------------------------------------------------------

def plot_slice_comparison(
    image: np.ndarray | torch.Tensor,
    ground_truth: np.ndarray | torch.Tensor,
    prediction: np.ndarray | torch.Tensor,
    save_path: str | Path,
    slice_axis: int = 2,
    slice_idx: int | None = None,
    title: str = "",
) -> None:
    """Plot a single 2-D slice: image | ground truth | prediction.

    Parameters
    ----------
    image : 3-D array (D, H, W)
    ground_truth : 3-D int array (D, H, W)
    prediction : 3-D int array (D, H, W)
    save_path : output PNG path
    slice_axis : axis perpendicular to the slice plane (0/1/2)
    slice_idx : index along `slice_axis`; middle slice if None
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    # Squeeze leading singleton dims (e.g. channel dim) to get (D, H, W)
    image = np.squeeze(image)
    ground_truth = np.squeeze(ground_truth)
    prediction = np.squeeze(prediction)

    n_slices = image.shape[slice_axis]
    if slice_idx is None:
        slice_idx = n_slices // 2

    slicer = [slice(None)] * 3
    slicer[slice_axis] = slice_idx

    img_slice = image[tuple(slicer)]
    gt_slice = ground_truth[tuple(slicer)]
    pred_slice = prediction[tuple(slicer)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, cmap="tab10", interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap="tab10", interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_predictions(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str | Path,
    num_samples: int = 4,
    slice_axis: int = 2,
) -> None:
    """Run inference on a few validation samples and save slice comparisons."""
    save_dir = Path(save_dir) / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            image = batch["image"].to(device)
            label = batch["label"]
            logits = model(image)
            pred = logits.argmax(dim=1).cpu()

            # Take first item in batch
            plot_slice_comparison(
                image=image[0, 0].cpu(),
                ground_truth=label[0],
                prediction=pred[0],
                save_path=save_dir / f"sample_{i:03d}.png",
                slice_axis=slice_axis,
                title=f"Sample {i}",
            )
