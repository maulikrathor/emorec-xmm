"""Class-weighted cross-entropy loss.

CONTRACT: `class_counts` must be computed from the TRAIN split ONLY, never
dev/test — computing weights from dev/test would leak split composition
into the loss and invalidate every comparison downstream.
"""

import torch
from omegaconf import DictConfig
from torch import nn


def build_loss(cfg: DictConfig, class_counts: list[int]) -> nn.Module:
    """Build the loss module named by `cfg.loss.name` (only `"weighted_ce"` implemented tonight).

    Weighted cross-entropy: `w_c = 1 / sqrt(count_c)`, then mean-normalized
    so `mean(w) == 1` (docs/ARCHITECTURE.md "Loss / optimization"). `class_counts`
    is a plain list ordered by class id (`class_counts[i]` = train-split count
    of class `i`) — the caller (not this function) is responsible for
    computing it from the TRAIN split only. Raises `ValueError` if any count
    is <= 0 (a class absent from the counts would produce an infinite
    weight — fail loud rather than silently poisoning training with `inf`/`nan`).
    """
    if cfg.loss.name != "weighted_ce":
        raise ValueError(f"Unknown loss name {cfg.loss.name!r}; only 'weighted_ce' is implemented")

    counts = torch.tensor(class_counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError(
            f"class_counts must be strictly positive for every class, got {class_counts}"
        )

    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.mean()

    return nn.CrossEntropyLoss(weight=weights)
