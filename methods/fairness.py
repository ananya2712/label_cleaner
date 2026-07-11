"""
Fairness metrics and datascope utilities.

demographic_parity_gap — |P(ŷ=1 | protected) − P(ŷ=1 | unprotected)|
"""

import numpy as np


def demographic_parity_gap(y_pred: np.ndarray, protected_mask: np.ndarray) -> float:
    """
    Absolute difference in positive prediction rate between the protected
    and unprotected groups. Returns 0.0 if either group is empty.

    Parameters
    ----------
    y_pred         : predicted labels (0/1), shape (n,)
    protected_mask : bool array, True = protected group, shape (n,)
    """
    protected_mask = np.asarray(protected_mask, dtype=bool)
    y_pred = np.asarray(y_pred)
    if not protected_mask.any() or protected_mask.all():
        return 0.0
    rate_p = float(np.mean(y_pred[protected_mask] == 1))
    rate_u = float(np.mean(y_pred[~protected_mask] == 1))
    return abs(rate_p - rate_u)
