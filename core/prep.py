"""
Dataset preparation helpers for modular experiment execution.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .models import PreparedSplit


def prepare_fixed_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                        random_state: int = 0) -> PreparedSplit:
    """
    Create the single fixed split used across all methodologies.
    """
    all_idx = np.arange(len(X))
    _, _, _, _, train_idx, test_idx = train_test_split(
        X, y, all_idx, test_size=test_size, random_state=random_state
    )
    return PreparedSplit(
        X_train=X[train_idx].copy(),
        X_test=X[test_idx].copy(),
        y_train=y[train_idx].copy(),
        y_test=y[test_idx].copy(),
        train_idx=train_idx,
        test_idx=test_idx,
    )

