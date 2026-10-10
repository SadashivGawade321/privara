"""Reproducibility utilities for Privara.

Provides deterministic random seed management across Python, NumPy, and PyTorch.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducible execution across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int) -> torch.Generator:
    """Return a deterministic PyTorch Generator."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
