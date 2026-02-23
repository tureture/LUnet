"""Training and validation loop.

Core logic:
- Loss averaged over batches (not individual samples)
- Thresholded (rounded) Dice computed as a separate metric
- Optional wandb logging
- Periodic prediction saving to disk
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from evaluation import compute_metrics
from utils import save_checkpoint

logger = logging.getLogger("lunet")


def _thresholded_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Dice on rounded (binarised) predictions — matches the reference metric.

    Parameters
    ----------
    pred : (B, 1, D, H, W) float — sigmoid probabilities
    target : (B, 1, D, H, W) float — ground-truth (0/1)
    """
    pred = torch.round(pred)
    target = torch.round(target)
    dims = (0,) + tuple(range(2, pred.ndimension()))
    intersection = torch.sum(pred * target, dims)
    cardinality = torch.sum(pred + target, dims)
    return (2.0 * intersection / (cardinality + eps)).mean()


class Trainer:
    """Training / validation loop.

    Parameters
    ----------
    model : nn.Module
    criterion : nn.Module
    train_loader, val_loader : DataLoader
    cfg : dict — full configuration dictionary.
    device : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        tcfg = cfg["training"]

        # ---- Optimizer ----
        opt_name = tcfg.get("optimizer", "adam").lower()
        opt_params = dict(lr=tcfg["learning_rate"])
        if opt_name == "adamw":
            opt_params["weight_decay"] = tcfg.get("weight_decay", 1e-5)
            self.optimizer = torch.optim.AdamW(model.parameters(), **opt_params)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), **opt_params)

        # ---- LR Scheduler ----
        sched = tcfg.get("scheduler", "none")
        if sched == "cosine":
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = CosineAnnealingLR(
                self.optimizer, T_max=tcfg["epochs"]
            )
        elif sched == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=tcfg["scheduler_step_size"],
                gamma=tcfg["scheduler_gamma"],
            )
        else:
            self.scheduler = None

        # ---- History (for plotting) ----
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_dice": [],
            "val_loss": [],
            "val_dice": [],
            "lr": [],
            "epoch_time": [],
        }
        # Pre-create history keys for every configured eval metric
        eval_metric_names: list[str] = cfg.get("evaluation", {}).get("metrics", [])
        for m in eval_metric_names:
            key = f"val_{m}"
            if key not in self.history:
                self.history[key] = []

        self.best_val_loss = float("inf")

        # ---- Evaluation metrics from config ----
        self.eval_metric_names: list[str] = cfg.get("evaluation", {}).get("metrics", [])

        # ---- Output paths ----
        self.output_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir = self.output_dir / "predictions"
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        # ---- Logging config ----
        self.log_interval: int = tcfg.get("log_interval", 100)

        # ---- Wandb (optional) ----
        self.wandb_run = None
        wcfg = cfg.get("wandb")
        if wcfg and wcfg.get("enabled", False):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wcfg.get("project", "lunet"),
                    name=cfg["experiment"]["name"],
                    config=cfg,
                    mode=wcfg.get("mode", "online"),
                    resume="allow",
                )
            except Exception as e:
                logger.warning("wandb init failed: %s", e)

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, global_step: int) -> tuple[float, float, int]:
        """Train for one epoch. Returns (avg_loss, avg_dice, updated global_step)."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)

            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

            dice_score = _thresholded_dice(preds.detach(), labels)

            total_loss += loss.item()
            total_dice += dice_score.item()
            n_batches += 1
            del loss

        global_step += 1
        avg_loss = total_loss / max(n_batches, 1)
        avg_dice = total_dice / max(n_batches, 1)
        return avg_loss, avg_dice, global_step

    # ------------------------------------------------------------------
    # Single validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_epoch(self, global_step: int) -> tuple[float, float, dict[str, float]]:
        """Validate. Returns (avg_loss, avg_dice, eval_metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        # Accumulators for configured evaluation metrics
        eval_totals: dict[str, float] = {m: 0.0 for m in self.eval_metric_names}

        for batch_idx, batch in enumerate(self.val_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            preds = self.model(images)

            loss = self.criterion(preds, labels)
            dice_score = _thresholded_dice(preds, labels)

            total_loss += loss.item()
            total_dice += dice_score.item()
            n_batches += 1

            # Compute configured evaluation metrics (e.g. hausdorff95)
            if self.eval_metric_names:
                # Binarise predictions to class indices (B, D, H, W)
                preds_bin = torch.round(preds).squeeze(1).long()
                targets_cls = torch.round(labels).squeeze(1).long()
                batch_metrics = compute_metrics(preds_bin, targets_cls, self.eval_metric_names)
                for m, v in batch_metrics.items():
                    eval_totals[m] += v

            # Save predictions at log_interval
            if global_step % self.log_interval == 1 and batch_idx % 10 == 0:
                self._save_predictions(images, labels, preds, global_step, batch_idx, prefix="val")

        avg_loss = total_loss / max(n_batches, 1)
        avg_dice = total_dice / max(n_batches, 1)
        avg_eval: dict[str, float] = {m: v / max(n_batches, 1) for m, v in eval_totals.items()}
        return avg_loss, avg_dice, avg_eval

    # ------------------------------------------------------------------
    # Prediction saving
    # ------------------------------------------------------------------

    def _save_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        step: int,
        batch_idx: int,
        prefix: str = "",
    ) -> None:
        """Save image / label / prediction volumes as .npy for inspection."""
        for i in range(images.shape[0]):
            tag = f"{prefix}_{step}_{batch_idx}_{i}"
            np.save(self.pred_dir / f"{tag}_image.npy", images[i, 0].cpu().numpy())
            np.save(self.pred_dir / f"{tag}_label.npy", labels[i].cpu().numpy())
            np.save(self.pred_dir / f"{tag}_pred.npy", preds[i, 0].cpu().numpy())

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, list[float]]:
        """Run the complete training loop. Returns the history dict."""
        epochs = self.cfg["training"]["epochs"]
        ckpt_every = self.cfg["training"]["checkpoint"]["save_every"]
        global_step = 0

        logger.info("Starting training for %d epochs on %s", epochs, self.device)

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_dice, global_step = self._train_epoch(epoch, global_step)
            val_loss, val_dice, eval_metrics = self._validate_epoch(global_step)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.perf_counter() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            # Build extra metrics string for logging
            eval_str = "".join(f"  {k}={v:.4f}" for k, v in eval_metrics.items())
            logger.info(
                "Epoch %03d/%03d  train_loss=%.4f  train_dice=%.4f  "
                "val_loss=%.4f  val_dice=%.4f%s  lr=%.2e  (%.1fs)",
                epoch, epochs, train_loss, train_dice, val_loss, val_dice, eval_str, lr, elapsed,
            )

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_dice)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["lr"].append(lr)
            self.history["epoch_time"].append(elapsed)
            for m, v in eval_metrics.items():
                self.history[f"val_{m}"].append(v)

            # Wandb logging
            if self.wandb_run is not None:
                import wandb
                log_dict = {
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "lr": lr,
                    "epoch_time": elapsed,
                }
                for m, v in eval_metrics.items():
                    log_dict[f"val_{m}"] = v
                wandb.log(log_dict, step=global_step)

            # ---- Save checkpoint every N epochs ----
            if epoch % ckpt_every == 0:
                save_checkpoint(
                    self.output_dir / f"checkpoint_epoch{epoch:03d}.pt",
                    self.model, self.optimizer, epoch, val_loss,
                )

            # ---- Track best model ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.cfg["training"]["checkpoint"]["keep_best"]:
                    save_checkpoint(
                        self.output_dir / "best_model.pt",
                        self.model, self.optimizer, epoch, val_loss,
                    )
                    logger.info("  New best model (val_loss=%.4f)", val_loss)

        logger.info("Training complete. Best val_loss: %.4f", self.best_val_loss)
        return self.history
