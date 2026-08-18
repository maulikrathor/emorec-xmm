"""Deterministic seeding.

CONTRACT: `seed_everything` is the ONLY sanctioned way to seed randomness in
this repo. Call it once at the top of every entrypoint (training, eval, XAI
scripts, tests that need determinism) before any random state is touched.
Two calls with the same seed followed by the same sequence of operations
must produce bitwise-identical results on CPU.
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed `random`, `numpy`, and `torch` (CPU + all CUDA devices) with `seed`.

    Also forces `torch.backends.cudnn.deterministic = True` and
    `torch.backends.cudnn.benchmark = False`, trading some GPU throughput for
    run-to-run reproducibility. Must be called before any model, dataset, or
    dataloader is constructed — seeding after workers/RNG state has already
    advanced does not retroactively fix anything.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
