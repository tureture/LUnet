"""LUnet – entry point for training, evaluation, and benchmarking."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from data import build_dataloaders
from evaluation import run_benchmark
from models import build_model
from training import Trainer, build_loss
from utils import get_device, load_config, set_seed, setup_logging
from visualization import plot_training_curves
from visualization.visualize import visualize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LUnet Training Framework")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "benchmark", "visualize"],
        help="Run mode: train | benchmark | visualize",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a checkpoint to resume / load for eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["experiment"]["seed"])
    device = get_device()

    output_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    logger = setup_logging(output_dir)
    logger.info("Device: %s", device)
    logger.info("Config: %s", args.config)

    # ---- Build model ----
    model = build_model(cfg)
    logger.info(
        "Model: %s  |  params: %s",
        cfg["model"]["architecture"],
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # ================================================================
    # MODE: benchmark
    # ================================================================
    if args.mode == "benchmark":
        logger.info("Running benchmark …")
        results = run_benchmark(model, cfg, device)
        for k, v in results.items():
            logger.info("  %s: %s", k, f"{v:,.2f}" if isinstance(v, float) else v)
        with open(output_dir / "benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # ================================================================
    # MODE: train
    # ================================================================
    if args.mode == "train":
        train_loader, val_loader = build_dataloaders(cfg)
        logger.info("Train samples: %d | Val samples: %d",
                     len(train_loader.dataset), len(val_loader.dataset))

        criterion = build_loss(cfg)

        trainer = Trainer(model, criterion,
                          train_loader, val_loader, cfg, device)

        # Optionally resume from checkpoint
        if args.checkpoint:
            from utils import load_checkpoint
            ckpt = load_checkpoint(args.checkpoint, model, trainer.optimizer, device)
            logger.info("Resumed from epoch %d", ckpt["epoch"])

        history = trainer.fit()

        # Post-training plots
        if cfg["visualization"]["plot_training_curves"]:
            plot_training_curves(history, output_dir)
            logger.info("Training curves saved to %s", output_dir)

        # Post-training prediction visualisation
        if cfg["visualization"]["plot_slices"]:
            visualize_predictions(
                model, val_loader, device, output_dir,
                num_samples=cfg["visualization"]["num_samples"],
                slice_axis=cfg["visualization"]["slice_axis"],
            )
            logger.info("Prediction slices saved to %s/predictions", output_dir)
        return

    # ================================================================
    # MODE: visualize (from a saved checkpoint)
    # ================================================================
    if args.mode == "visualize":
        assert args.checkpoint, "--checkpoint is required for visualize mode"
        from utils import load_checkpoint
        load_checkpoint(args.checkpoint, model, device=device)
        model.to(device)

        _, val_loader = build_dataloaders(cfg)
        visualize_predictions(
            model, val_loader, device, output_dir,
            num_samples=cfg["visualization"]["num_samples"],
            slice_axis=cfg["visualization"]["slice_axis"],
        )
        logger.info("Visualisations saved to %s/predictions", output_dir)


if __name__ == "__main__":
    main()
