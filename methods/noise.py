"""
Noise injection functions — one per noise type.

Each function takes clean data and returns noisy data plus the positions
(row indices in the training set) that were corrupted.  The positions are
used by cleaning methods to know which samples to target.

Noise types
-----------
inject_outlier   — feature corruption, random rows           (MAR, feature-space)
inject_rnd_label — label corruption, random rows             (MAR, label-space)
inject_nnar      — label corruption, protected subgroup      (NNAR)
inject_mnar      — feature corruption (NaN), protected group (MNAR)
"""

import numpy as np


def inject_outlier(X: np.ndarray, col_idx: int, noise_level: float,
                   outlier_value: float = 100, seed: int = 42):
    """
    Inject a fixed extreme value into `col_idx` for a uniformly random fraction
    of rows — no demographic targeting (MAR).

    Parameters
    ----------
    X             : feature array (n_samples, n_features) — not modified in place
    col_idx       : column to corrupt
    noise_level   : fraction of ALL rows to corrupt
    outlier_value : value to inject (default 100)
    seed          : random seed

    Returns
    -------
    X_noisy        : corrupted feature array
    noisy_positions: row indices of corrupted samples
    cap_value      : 2-sigma cap computed from clean X (use for cleaning)
    """
    rng      = np.random.RandomState(seed)
    n_noisy  = int(noise_level * len(X))
    noisy_positions = rng.choice(len(X), n_noisy, replace=False)

    mean_x    = np.nanmean(X[:, col_idx])
    std_x     = np.nanstd(X[:, col_idx])
    cap_value = mean_x + 2 * std_x

    X_noisy = X.copy().astype(float)
    X_noisy[noisy_positions, col_idx] = outlier_value

    return X_noisy, noisy_positions, cap_value


def inject_rnd_label(y: np.ndarray, noise_level: float, seed: int = 42):
    """
    Flip labels for a uniformly random fraction of ALL rows — no demographic
    targeting (MAR, label-space).

    Parameters
    ----------
    y           : label array (n_samples,) — not modified in place
    noise_level : fraction of ALL samples to flip
    seed        : random seed

    Returns
    -------
    y_noisy        : label array with flipped entries
    noisy_positions: row indices of flipped samples
    """
    rng             = np.random.RandomState(seed)
    n_flip          = int(noise_level * len(y))
    noisy_positions = rng.choice(len(y), n_flip, replace=False)
    y_noisy         = y.copy()
    y_noisy[noisy_positions] = 1 - y_noisy[noisy_positions]
    return y_noisy, noisy_positions


def inject_nnar(y: np.ndarray, protected_mask: np.ndarray,
                noise_level: float, seed: int = 42):
    """
    Flip labels for `noise_level` fraction of the protected subgroup only
    (Noise Not At Random — NNAR), simulating systematic measurement bias
    correlated with a sensitive attribute.

    Parameters
    ----------
    y              : label array (n_samples,) — not modified in place
    protected_mask : bool array (n_samples,) — True = protected group
    noise_level    : fraction of protected-group samples to flip
    seed           : random seed

    Returns
    -------
    y_noisy        : label array with flipped entries
    noisy_positions: row indices (within y) of flipped samples
    """
    rng        = np.random.RandomState(seed)
    candidates = np.where(protected_mask)[0]
    n_flip     = int(noise_level * len(candidates))
    if n_flip == 0:
        return y.copy(), np.array([], dtype=int)
    noisy_positions = rng.choice(candidates, n_flip, replace=False)
    y_noisy = y.copy()
    y_noisy[noisy_positions] = 1 - y_noisy[noisy_positions]
    return y_noisy, noisy_positions


def inject_mnar(X: np.ndarray, protected_mask: np.ndarray,
                col_indices: list, noise_level: float, seed: int = 42):
    """
    Set `col_indices` features to NaN for `noise_level` fraction of the
    protected subgroup (Missing Not At Random — MNAR), simulating data
    omission that disproportionately affects a demographic group.

    Parameters
    ----------
    X              : feature array (n_samples, n_features) — not modified in place
    protected_mask : bool array (n_samples,) — True = protected group
    col_indices    : list[int] — columns to blank out
    noise_level    : fraction of protected-group samples to corrupt
    seed           : random seed

    Returns
    -------
    X_noisy        : feature array with NaN at corrupted positions
    noisy_positions: row indices of corrupted samples
    """
    rng        = np.random.RandomState(seed)
    candidates = np.where(protected_mask)[0]
    n_missing  = int(noise_level * len(candidates))
    if n_missing == 0:
        return X.copy().astype(float), np.array([], dtype=int)
    noisy_positions = rng.choice(candidates, n_missing, replace=False)
    X_noisy = X.copy().astype(float)
    for col_idx in col_indices:
        X_noisy[noisy_positions, col_idx] = np.nan
    return X_noisy, noisy_positions
