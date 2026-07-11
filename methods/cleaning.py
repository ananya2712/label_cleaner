"""
Cleaning methodology functions — one per method.

Each cleaner takes a ranked ordering of noisy positions and an action function,
then runs an incremental cleaning loop over a set of proportions.

The action function encapsulates *what* to do when cleaning a sample — it varies
by noise type:
  - outlier   : cap feature value at 2-sigma
  - rnd_label : restore ground-truth label
  - nnar      : restore ground-truth label
  - mnar      : remove detected rows

Cleaners
--------
clean_datascope  — rank true noisy rows by Shapley importance
clean_cleanlab   — rank by CleanLab self_confidence, clean most-suspicious first
clean_random     — shuffle noisy positions randomly (baseline)
"""

from typing import Callable, List, Tuple

import numpy as np
from cleanlab.filter import find_label_issues
from datascope.importance import SklearnModelAccuracy
from datascope.importance.shapley import ImportanceMethod, ShapleyImportance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


# ---------------------------------------------------------------------------
# DataScope
# ---------------------------------------------------------------------------

def _compute_importances(
    pipeline, X_train, y_train, X_test, y_test,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> np.ndarray:
    """Compute per-sample Shapley importance scores. Pipeline must be already fitted."""
    utility       = SklearnModelAccuracy(pipeline[-1])
    feature_pipes = pipeline[:-1]
    imp = ShapleyImportance(
        method=importance_method,
        pipeline=feature_pipes,
        utility=utility,
        mc_iterations=mc_iterations,
    )
    return imp.fit(X_train, y_train).score(X_test, y_test)


def _safe_accuracy(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Return NaN when aggressive filtering leaves no trainable dataset."""
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return float("nan")
    pipeline = pipeline_factory()
    try:
        pipeline.fit(X_train, y_train)
        return accuracy_score(y_test, pipeline.predict(X_test))
    except ValueError:
        return float("nan")


def clean_datascope(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noisy_positions: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> Tuple[List[float], np.ndarray]:
    """
    DataScope cleaning: rank noisy samples by Shapley importance (most harmful
    first), then incrementally apply `action_fn` to the top-k% and measure accuracy.

    Parameters
    ----------
    pipeline_factory : callable() → fresh sklearn Pipeline
    X_train, y_train : training data (with injected noise)
    X_test,  y_test  : clean held-out test set
    noisy_positions  : ground-truth noisy row indices (in train space)
    action_fn        : callable(X_tr, y_tr, positions) → (X_tr_clean, y_tr_clean)
                       encapsulates the noise-type-specific cleaning action
    proportions      : array of fractions in [0, 1] — cleaning proportions to evaluate

    Returns
    -------
    accs         : accuracy at each proportion
    ranked_noisy : noisy_positions sorted most-harmful first (DataScope order)
    """
    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)

    importances  = _compute_importances(pipeline, X_train, y_train, X_test, y_test, importance_method, mc_iterations)
    sorted_order = np.argsort(importances[noisy_positions])  # ascending: lowest (most harmful) Shapley importance first
    ranked_noisy = noisy_positions[sorted_order]

    accs = []
    for p in proportions:
        n_clean   = int(p * len(ranked_noisy))
        X_c, y_c  = action_fn(X_train, y_train, ranked_noisy[:n_clean])
        accs.append(_safe_accuracy(pipeline_factory, X_c, y_c, X_test, y_test))

    return accs, ranked_noisy



# ---------------------------------------------------------------------------
# clean_datascope_hybrid — DISABLED (blending DataScope + CleanLab signals)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CleanLab
# ---------------------------------------------------------------------------

def clean_cleanlab(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    n_jobs: int = 1,
) -> Tuple[List[float], np.ndarray]:
    """
    CleanLab cleaning: rank ALL training samples by self_confidence score
    (most suspicious first), then incrementally apply `action_fn` to top-k%.

    CleanLab operates unsupervised — it does not know the ground-truth noisy
    positions. Its top-k may include false positives (clean samples flagged)
    and miss some true noisy ones.

    Parameters
    ----------
    pipeline_factory : callable() → fresh sklearn Pipeline
    X_train, y_train : training data (with injected noise)
    X_test,  y_test  : clean held-out test set
    action_fn        : callable(X_tr, y_tr, positions) → (X_tr_clean, y_tr_clean)
    proportions      : array of fractions in [0, 1]
    n_jobs           : parallelism for cross_val_predict (default 1 for safety)

    Returns
    -------
    accs        : accuracy at each proportion
    cl_ranked   : all training indices ranked most-to-least suspicious
    """
    pipeline   = pipeline_factory()
    pred_probs = cross_val_predict(
        pipeline, X_train, y_train,
        cv=5, method="predict_proba", n_jobs=n_jobs,
    )
    cl_ranked = find_label_issues(
        labels=y_train,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        n_jobs=n_jobs,
    )

    accs = []
    for p in proportions:
        n_clean   = int(p * len(cl_ranked))
        X_c, y_c  = action_fn(X_train, y_train, cl_ranked[:n_clean])
        accs.append(_safe_accuracy(pipeline_factory, X_c, y_c, X_test, y_test))

    return accs, cl_ranked


# ---------------------------------------------------------------------------
# clean_cleanlab_adaptive — DISABLED (KDE-valley adaptive correction/removal)
# _kde_valley_threshold and _otsu_threshold also disabled
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def clean_random(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noisy_positions: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    n_seeds: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Random cleaning baseline: shuffle noisy_positions randomly and apply
    `action_fn` to top-k%.  Repeated over `n_seeds` shuffles; mean and std
    are returned.

    This isolates the *ordering* benefit of DataScope: both DataScope and
    random clean the same set of samples, just in different order.

    Parameters
    ----------
    pipeline_factory : callable() → fresh sklearn Pipeline
    X_train, y_train : training data (with injected noise)
    X_test,  y_test  : clean held-out test set
    noisy_positions  : ground-truth noisy row indices
    action_fn        : callable(X_tr, y_tr, positions) → (X_tr_clean, y_tr_clean)
    proportions      : array of fractions in [0, 1]
    n_seeds          : number of random shuffles to average over

    Returns
    -------
    mean_accs : mean accuracy at each proportion across seeds
    std_accs  : std  accuracy at each proportion across seeds
    """
    all_accs = []
    for seed in range(n_seeds):
        rng  = np.random.RandomState(seed + 100)
        perm = noisy_positions.copy()
        rng.shuffle(perm)

        seed_accs = []
        for p in proportions:
            n_clean  = int(p * len(perm))
            X_c, y_c = action_fn(X_train, y_train, perm[:n_clean])
            seed_accs.append(_safe_accuracy(pipeline_factory, X_c, y_c, X_test, y_test))
        all_accs.append(seed_accs)

    arr = np.array(all_accs)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


# ---------------------------------------------------------------------------
# clean_kairos — DISABLED (Kairos data values; _kairos_scores also disabled)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Action functions — noise-type-specific cleaning actions
# ---------------------------------------------------------------------------

def action_cap(col_idx: int, cap_value: float) -> Callable:
    """
    Returns an action_fn that caps `col_idx` at `cap_value` for given positions.
    Use with outlier noise.
    """
    def _action(X_tr, y_tr, positions):
        X_c = X_tr.copy()
        if len(positions) > 0:
            X_c[positions, col_idx] = cap_value
        return X_c, y_tr.copy()
    return _action


def action_remove() -> Callable:
    """
    Returns an action_fn that removes (drops) rows at given positions from
    training. Use with outlier noise as an alternative to capping.

    Note: the returned (X_c, y_c) will have fewer rows than (X_tr, y_tr).
    """
    def _action(X_tr, y_tr, positions):
        if len(positions) == 0:
            return X_tr.copy(), y_tr.copy()
        keep = np.ones(len(X_tr), dtype=bool)
        keep[positions] = False
        return X_tr[keep], y_tr[keep]
    return _action


def action_restore_labels(y_clean: np.ndarray) -> Callable:
    """
    Returns an action_fn that restores ground-truth labels at given positions.
    Use with rnd_label or nnar noise.

    Parameters
    ----------
    y_clean : the original (pre-noise) labels for the training set
    """
    def _action(X_tr, y_tr, positions):
        y_c = y_tr.copy()
        if len(positions) > 0:
            y_c[positions] = y_clean[positions]
        return X_tr.copy(), y_c
    return _action


def action_flip_labels() -> Callable:
    """
    Returns an action_fn that flips labels at given positions.
    Use with mnar noise (feature corruption that can't be undone directly).
    """
    def _action(X_tr, y_tr, positions):
        y_c = y_tr.copy()
        if len(positions) > 0:
            y_c[positions] = 1 - y_c[positions]
        return X_tr.copy(), y_c
    return _action


# ---------------------------------------------------------------------------
# clean_hybrid_auto — DISABLED (auto-routing hybrid; _detect_noise_type also disabled)
# ---------------------------------------------------------------------------
