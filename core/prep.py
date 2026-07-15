"""
Dataset preparation helpers for modular experiment execution.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .models import PreparedSplit


def prepare_fixed_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                        val_size: float = 0.1, val_cap: int = 1000,
                        random_state: int = 0) -> PreparedSplit:
    """
    Create the fixed train/validation/test split used across all methodologies.

    The validation split is used ONLY to score the Shapley utility function in
    clean_datascope()/clean_datascope_fair(). Previously those functions scored
    importance directly against X_test/y_test, the same split later used to
    report final accuracy/DP -- a test-set leak that gave DataScope information
    (and a matching-metric advantage) that CleanLab, Entropy, and Random never
    had. Carving out a dedicated validation split removes that asymmetry.
    The test split's size and composition (test_size=0.2, random_state=0) are
    unchanged from the original single split, so final-metric comparability is
    preserved; only the training-set size shrinks slightly to make room for
    validation. Validation is capped at val_cap rows for tractability on the
    larger datasets -- this does not affect train or test split sizes.
    """
    all_idx = np.arange(len(X))
    X_trainval, X_test, y_trainval, y_test, trainval_idx, test_idx = train_test_split(
        X, y, all_idx, test_size=test_size, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_trainval, y_trainval, trainval_idx, test_size=val_frac, random_state=random_state
    )
    if val_cap is not None and len(val_idx) > val_cap:
        rng = np.random.RandomState(random_state)
        keep = rng.choice(len(val_idx), size=val_cap, replace=False)
        X_val, y_val, val_idx = X_val[keep], y_val[keep], val_idx[keep]
    return PreparedSplit(
        X_train=X_train.copy(),
        X_val=X_val.copy(),
        X_test=X_test.copy(),
        y_train=y_train.copy(),
        y_val=y_val.copy(),
        y_test=y_test.copy(),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

