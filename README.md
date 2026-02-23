# Lean Unet: A Compact Model for Image Segmentation

Lean Unet (LUnet) is a lightweight variant of the 3D U-Net designed for efficient volumetric medical image segmentation. Unlike the standard U-Net, which doubles the number of feature channels at each encoder level, LUnet uses a **uniform filter width** across all levels — dramatically reducing parameter count while maintaining competitive segmentation performance.

This repository provides a configurable PyTorch training framework supporting several 3D segmentation architectures, enabling reproducible experiments and providing a flexible interface for evaluating the proposed methods on external datasets and previously unseen imaging domains.

### Model Presets

Preset names follow the pattern `{model}_{size}_{N}f`, where `N` is the number of base (first-level) convolutional filters.

| Family | Size | Variants | Levels | Convs/block |
|---|---|---|---|---|
| UNet | small | `unet_small_4f`, `3f`, `2f`, `1f` | 5 | 2 |
| LUNet | small | `lunet_small_4f`, `3f`, `2f`, `1f` | 5 | 2 |
| UNet | large | `unet_large_24f`, `12f`, `6f` | 4 | 3 |
| LUNet | large | `lunet_large_24f`, `12f`, `6f` | 4 | 3 |
| UNet++ | small | `unetpp_small_4f`, `3f`, `2f`, `1f` | 5 | 2 |
| LUNet++ | small | `lunetpp_small_4f`, `3f`, `2f`, `1f` | 5 | 2 |
| UNet++ | large | `unetpp_large_24f`, `12f`, `6f` | 4 | 3 |
| LUNet++ | large | `lunetpp_large_24f`, `12f`, `6f` | 4 | 3 |
| MedNeXt | small | `mednext_small` | — | — |

## Paper

If you use this code, please cite:

> **Lean Unet: A Compact Model for Image Segmentation**
> [arXiv:2512.03834](https://arxiv.org/abs/2512.03834)

## Installation

**Requirements:** Python ≥ 3.11, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/<your-username>/LUnet.git
cd LUnet
uv sync
```

This creates a virtual environment and installs all pinned dependencies from the lockfile.

To include development tools (pytest, ruff):

```bash
uv sync --extra dev
```

### Running commands

Activate the virtual environment once per terminal session:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Then run scripts with `python` as usual:

```bash
python main.py --config configs/default.yaml --mode train
```

## Usage

All experiments are driven by a YAML configuration file. Ready-made configs are provided for each model family with the hyperparameters used in the paper:

| Config file | Architecture | Batch size | Learning rate |
|---|---|---|---|
| `configs/unet_small.yaml` | U-Net small | 16 | 1e-2 |
| `configs/lunet_small.yaml` | LU-Net small | 16 | 1e-2 |
| `configs/unet_large.yaml` | U-Net large | 1 | 1e-3 |
| `configs/lunet_large.yaml` | LU-Net large | 1 | 1e-3 |
| `configs/default.yaml` | Any | 16 | 1e-2 |

To switch the base filter count, edit the `architecture` field in the config (e.g. `"lunet_small_2f"`).

### Training

```bash
python main.py --config configs/lunet_small.yaml --mode train
```

Key configuration options:

```yaml
model:
  architecture: "lunet_small_4f"   # Choose a preset from the table above
  in_channels: 1
  out_channels: 1

training:
  epochs: 10000
  batch_size: 16
  learning_rate: 1.0e-2
  loss:
    name: "dice"                   # "dice" | "bce" | "dice_bce"
```

Checkpoints and training curves are saved to the directory specified by `experiment.output_dir` / `experiment.name`.

### Benchmarking

Measure FLOPs, and some related performance metrics for a given architecture:

```bash
python main.py --config configs/default.yaml --mode benchmark
```

### Visualisation

Generate prediction slices from a saved checkpoint:

```bash
python main.py --config configs/default.yaml --mode visualize --checkpoint outputs/<experiment>/best_model.pt
```

### Resuming from a checkpoint

```bash
python main.py --config configs/default.yaml --mode train --checkpoint outputs/<experiment>/checkpoint_epoch100.pt
```

## Project Structure

```
├── main.py                 # Entry point (train / benchmark / visualize)
├── configs/
│   ├── default.yaml        # General-purpose configuration
│   ├── unet_small.yaml     # U-Net small (batch 16, lr 1e-2)
│   ├── lunet_small.yaml    # LU-Net small (batch 16, lr 1e-2)
│   ├── unet_large.yaml     # U-Net large (batch 1, lr 1e-3)
│   └── lunet_large.yaml    # LU-Net large (batch 1, lr 1e-3)
├── data/
│   └── dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── unet3d.py           # 3D U-Net & LUnet definitions
│   ├── unetplusplus.py     # UNet++ & LUnet++ definitions
│   └── mednext.py          # MedNeXt definition
├── training/
│   ├── trainer.py          # Training loop with checkpoint saving
│   └── losses.py           # Loss functions (Dice, BCE, Dice+BCE)
├── evaluation/
│   ├── benchmark.py        # Latency and FLOPs benchmarking
│   └── metrics.py          # Dice score, Hausdorff distance, etc.
├── visualization/
│   └── visualize.py        # Prediction slices and training curves
└── utils/
    └── utils.py            # Config loading, seeding, device helpers
```

## Reproduction of Pruning-Related Figures

To reproduce the pruning-related figures presented in the paper, please refer to the [STAMP repository](https://github.com/nkdinsdale/STAMP).