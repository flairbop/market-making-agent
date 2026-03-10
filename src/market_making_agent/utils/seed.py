"""
seed.py
-------
Global seeding utilities for reproducibility.

Sets seeds on Python's random, numpy, and torch (CPU and CUDA)
so that experiments can be exactly reproduced with the same seed.
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set deterministic random seeds across the full stack.

    Parameters
    ----------
    seed:
        Integer seed value. Use the same seed across train/eval
        to guarantee reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
