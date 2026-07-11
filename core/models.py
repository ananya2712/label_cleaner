"""Shared dataclasses for label_cleaner service modules."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PreparedSplit:
    """Fixed train/test split plus indices for traceability."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class MethodCurves:
    """Accuracy-vs-cleaning curves for all cleaning methodologies."""
    datascope: List[float]
    random_mean: List[float]
    random_std: List[float]
    cleanlab: List[float]
    baseline: float
    proportions: np.ndarray
    datascope_removal: Optional[List[float]] = None
    baseline_dp: Optional[float] = None
    datascope_dp: Optional[List[float]] = None
    cleanlab_dp: Optional[List[float]] = None
    random_dp_mean: Optional[List[float]] = None
    random_dp_std: Optional[List[float]] = None
    datascope_removal_dp: Optional[List[float]] = None


@dataclass
class NoiseBundle:
    """
    Noisy data payload returned by noise injectors.

    For label noise, X_noisy is usually the same as X_train and y_noisy differs.
    For feature noise, X_noisy differs and y_noisy is usually the same as y_train.
    """
    X_noisy: np.ndarray
    y_noisy: np.ndarray
    noisy_positions: np.ndarray
    metadata: Dict[str, float]


@dataclass
class ExperimentArtifacts:
    """Detailed experiment outputs for reporting and cleaned-row inspection."""
    curves: MethodCurves
    split: PreparedSplit
    bundle: NoiseBundle
    datascope_ranked: np.ndarray
    cleanlab_ranked: np.ndarray
    random_rankings: List[np.ndarray]
