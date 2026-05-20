# src/utils_seed.py
"""
Centralized seed management for reproducibility.

All MRSA NLP and prediction pipelines use the same seed (7) to ensure:
  - Deterministic cohort sampling
  - Reproducible feature extraction
  - Consistent model training and evaluation

Usage:
    from src.utils_seed import set_seed, GLOBAL_SEED

    set_seed(GLOBAL_SEED)  # or set_seed(custom_seed)
"""

import os
import random
import logging

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

LOG = logging.getLogger("mrsa.seed")

# Global seed used across prediction, rule-based, and NER pipelines
GLOBAL_SEED = 7


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """
    Set random seed for all libraries used in the pipeline.

    Ensures reproducibility across:
      - Python standard library (random)
      - NumPy (np.random)
      - PyTorch (torch, if available)
      - TensorFlow (tf, if available)
      - Environment (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG)

    Parameters
    ----------
    seed : int, optional
        Random seed (default: GLOBAL_SEED = 7).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic CUDA operations
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if tf is not None:
        tf.random.set_seed(seed)

    LOG.info(f"Global seed set to {seed}")
