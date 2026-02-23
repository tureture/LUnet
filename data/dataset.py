"""Dataset base class and DataLoader factory."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class VolumeDataset(Dataset, ABC):
    """Abstract base class for paired 3-D image / label datasets.

    Subclasses must implement:
        ``_discover_samples`` — populate ``self.image_files`` and ``self.label_files``
        ``_load_sample``      — load one (image, label) pair from disk

    The base class handles conversion to tensors, channel dim, and the dict
    return format expected by the Trainer (``{"image": ..., "label": ...}``).
    """

    def __init__(self, root_dir: str | Path) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_files: list[str] = []
        self.label_files: list[str] = []
        self._discover_samples()

    @abstractmethod
    def _discover_samples(self) -> None:
        """Populate ``self.image_files`` and ``self.label_files`` (sorted, matched)."""

    @abstractmethod
    def _load_sample(self, idx: int) -> tuple:
        """Return ``(image, label)`` as numpy arrays.

        image : float32, shape (D, H, W)
        label : float32, shape (D, H, W)
        """

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image, label = self._load_sample(idx)
        image_t = torch.from_numpy(image).float().unsqueeze(0)   # (1, D, H, W)
        label_t = torch.from_numpy(label).float().unsqueeze(0)   # (1, D, H, W)
        return {"image": image_t, "label": label_t}


# ======================================================================
# HarP dataset
# ======================================================================

class HarPDataset(VolumeDataset):
    """HarP dataset — paired NRRD volumes with ``ct_*`` / ``label_*`` naming.

    Expected layout::

        root_dir/
            ct_001.nrrd
            label_001.nrrd
            ct_002.nrrd
            label_002.nrrd
            ...
    """

    def _discover_samples(self) -> None:
        files = os.listdir(self.root_dir)
        self.image_files = sorted(f for f in files if f.startswith("ct_"))
        self.label_files = sorted(f for f in files if f.startswith("label_"))

        if len(self.image_files) != len(self.label_files):
            raise ValueError(
                f"Mismatched file counts: {len(self.image_files)} images, "
                f"{len(self.label_files)} labels in {self.root_dir}"
            )
        for img, lbl in zip(self.image_files, self.label_files):
            img_id = img.split("_", 1)[1]
            lbl_id = lbl.split("_", 1)[1]
            if img_id != lbl_id:
                raise ValueError(f"Filename mismatch: {img} vs {lbl}")

    def _load_sample(self, idx: int) -> tuple:
        import nrrd

        image, _ = nrrd.read(str(self.root_dir / self.image_files[idx]))
        label, _ = nrrd.read(str(self.root_dir / self.label_files[idx]))
        return image.astype("float32"), label.astype("float32")


# ======================================================================
# Dataset registry & DataLoader factory
# ======================================================================

_DATASET_REGISTRY: dict[str, type[VolumeDataset]] = {
    "harp": HarPDataset,
}


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from the config.

    Config ``data`` section example::

        data:
          dataset: "harp"
          train_dir: "/path/to/your/train_data"
          val_dir:   "/path/to/your/val_data"
          num_workers: 0
          pin_memory: false
    """
    dcfg = cfg["data"]
    ds_name = dcfg["dataset"].lower()

    if ds_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{ds_name}'. Available: {list(_DATASET_REGISTRY.keys())}"
        )

    DatasetClass = _DATASET_REGISTRY[ds_name]
    train_ds = DatasetClass(dcfg["train_dir"])
    val_ds = DatasetClass(dcfg["val_dir"])

    num_workers = dcfg.get("num_workers", 0)
    loader_kw: dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=dcfg.get("pin_memory", False),
        persistent_workers=dcfg.get("persistent_workers", False) and num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        **loader_kw,
    )
    return train_loader, val_loader
