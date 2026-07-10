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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
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
    sorted_order = np.argsort(-importances[noisy_positions])
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
# Kairos
# ---------------------------------------------------------------------------

def _kairos_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambda_weight: float = 0.97,
    sigma_feature: float = 3.0,
    max_kernel_samples: int = 2000,
) -> np.ndarray:
    """
    Compute per-sample Kairos data values (Lodino et al., NeurIPS 2025).

    Combines two signals:
      - Feature score: RBF kernel similarity of each training sample to the
        validation distribution minus its similarity to the training distribution.
        Higher = more representative of validation = more valuable.
      - Residual score: P(correct label | x_i) from a logistic regression
        trained on the validation set. Higher = label consistent with val = clean.

    Final value = lambda_weight * feature_score + (1 - lambda_weight) * residual_score.
    Lower value → more likely noisy.

    For large datasets the full n×n kernel matrix is infeasible. Reference sets
    are capped at max_kernel_samples / 500 rows respectively to keep memory
    and compute tractable while preserving a good distribution estimate.
    """
    gamma = 1.0 / (2.0 * sigma_feature ** 2)
    rng   = np.random.RandomState(42)

    # Subsample reference sets so kernel matrices stay manageable
    val_ref = X_val[rng.choice(len(X_val),   size=min(len(X_val),   500),              replace=False)]
    trn_ref = X_train[rng.choice(len(X_train), size=min(len(X_train), max_kernel_samples), replace=False)]

    # Feature scores: how similar each training sample is to val vs. train distribution
    K_tv = rbf_kernel(X_train, val_ref, gamma=gamma)  # (n_train, n_val_ref)
    K_tt = rbf_kernel(X_train, trn_ref, gamma=gamma)  # (n_train, n_trn_ref)
    feature_scores = K_tv.mean(axis=1) - K_tt.mean(axis=1)

    # Residual scores
    clf = LogisticRegression(max_iter=1000, random_state=0)
    try:
        clf.fit(X_val, y_val)
        proba = clf.predict_proba(X_train)             # (n_train, n_classes)
        class_idx = {c: i for i, c in enumerate(clf.classes_)}
        residual_scores = proba[
            np.arange(len(y_train)),
            [class_idx[y] for y in y_train],
        ]
    except Exception:
        residual_scores = np.full(len(y_train), 0.5)

    return lambda_weight * feature_scores + (1.0 - lambda_weight) * residual_scores


def clean_kairos(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    lambda_weight: float = 0.97,
    sigma_feature: float = 3.0,
) -> Tuple[List[float], np.ndarray]:
    """
    Kairos cleaning: rank all training samples by ascending data value
    (most harmful / lowest value first), then incrementally apply `action_fn`.

    X_test / y_test are used for both Kairos scoring and accuracy evaluation,
    consistent with how DataScope uses the test set for Shapley importance scoring.

    Returns
    -------
    accs         : accuracy at each cleaning proportion
    kairos_ranked: all training indices sorted most-harmful first
    """
    # Fit pipeline on full noisy training data, then transform through feature
    # steps only (pipeline[:-1]) so the kernel operates on clean numeric features.
    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)
    feature_pipe = pipeline[:-1]
    X_train_t = feature_pipe.transform(X_train)
    X_test_t  = feature_pipe.transform(X_test)

    scores = _kairos_scores(X_train_t, y_train, X_test_t, y_test, lambda_weight, sigma_feature)
    kairos_ranked = np.argsort(scores)   # ascending: lowest value = noisiest first

    accs = []
    for p in proportions:
        n_clean  = int(p * len(kairos_ranked))
        X_c, y_c = action_fn(X_train, y_train, kairos_ranked[:n_clean])
        accs.append(_safe_accuracy(pipeline_factory, X_c, y_c, X_test, y_test))

    return accs, kairos_ranked


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
# Auto-routing hybrid cleaner
# ---------------------------------------------------------------------------

def _detect_noise_type(X_train: np.ndarray, y_train: np.ndarray,
                        pred_probs: np.ndarray,
                        feature_anomaly_thresh: float = 0.25,
                        clustering_thresh: float = 0.30) -> str:
    """
    Heuristic two-level noise type detector operating on OOF probabilities.

    Level 1 — Feature vs label noise
    ---------------------------------
    Flags the bottom-20% self-confidence samples (most suspicious) and checks
    what fraction have at least one feature value with |z-score| > 3.  A high
    fraction suggests the corruption lives in feature space (outlier / MNAR).

    Level 2 — Clustered vs dispersed
    ----------------------------------
    Measures how tightly the flagged samples cluster in feature space using the
    ratio of within-flagged variance to overall variance.  Low within-flagged
    variance (high clustering) means the suspicious samples share a common
    region — characteristic of structured noise (NNAR / MNAR).

    Decision tree
    -------------
    - High feature anomaly + low clustering  → 'outlier'  → route to Kairos
    - High feature anomaly + high clustering → 'mnar'     → route to CleanLab
    - Low  feature anomaly + high clustering → 'nnar'     → route to DataScope
    - Low  feature anomaly + low clustering  → 'rnd_label'→ route to CleanLab

    Returns one of: 'outlier', 'mnar', 'nnar', 'rnd_label'
    """
    self_conf = pred_probs[np.arange(len(y_train)), y_train]
    threshold = np.percentile(self_conf, 20)
    flagged   = np.where(self_conf <= threshold)[0]

    if len(flagged) < 2:
        return "rnd_label"

    # Level 1: feature anomaly
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8
    z   = np.abs((X_train[flagged] - mu) / sig)
    feature_anomaly = float((z.max(axis=1) > 3).mean())

    # Level 2: spatial clustering of flagged samples
    within_var  = float(X_train[flagged].var(axis=0).mean())
    overall_var = float(X_train.var(axis=0).mean()) + 1e-8
    clustering  = 1.0 - within_var / overall_var   # high = tightly clustered

    is_feature_noise  = feature_anomaly >= feature_anomaly_thresh
    is_clustered      = clustering      >= clustering_thresh

    if is_feature_noise and not is_clustered:
        return "outlier"
    if is_feature_noise and is_clustered:
        return "mnar"
    if not is_feature_noise and is_clustered:
        return "nnar"
    return "rnd_label"


def clean_hybrid_auto(
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
    n_jobs: int = 1,
) -> Tuple[List[float], np.ndarray, str]:
    """
    Auto-routing hybrid cleaner.

    Detects the likely noise type from the data using OOF predicted probabilities,
    then delegates to the empirically best-performing method for that type:

      outlier   → Kairos      (RBF kernel feature score detects feature anomalies)
      mnar      → CleanLab    (Kairos collapses on MNAR; CL is most stable)
      nnar      → DataScope   (Shapley captures structured group-based label flips)
      rnd_label → CleanLab    (self_confidence reliably ranks random label noise)

    Returns
    -------
    accs         : accuracy at each cleaning proportion
    ranked       : training indices ranked most-to-least harmful (method-specific)
    noise_type   : detected noise type string for diagnostics
    """
    pipeline   = pipeline_factory()
    pred_probs = cross_val_predict(
        pipeline, X_train, y_train,
        cv=5, method="predict_proba", n_jobs=n_jobs,
    )

    # Transform through feature steps for Kairos (needs numeric features)
    pipeline.fit(X_train, y_train)
    feature_pipe = pipeline[:-1]
    X_train_t    = feature_pipe.transform(X_train)

    noise_type = _detect_noise_type(X_train_t, y_train, pred_probs)

    if noise_type == "outlier":
        accs, ranked = clean_kairos(
            pipeline_factory, X_train, y_train, X_test, y_test,
            action_fn, proportions,
        )
    elif noise_type == "mnar":
        accs, ranked = clean_cleanlab(
            pipeline_factory, X_train, y_train, X_test, y_test,
            action_fn, proportions, n_jobs=n_jobs,
        )
    elif noise_type == "nnar":
        accs, ranked = clean_datascope(
            pipeline_factory, X_train, y_train, X_test, y_test,
            noisy_positions, action_fn, proportions,
            importance_method=importance_method, mc_iterations=mc_iterations,
        )
    else:  # rnd_label
        accs, ranked = clean_cleanlab(
            pipeline_factory, X_train, y_train, X_test, y_test,
            action_fn, proportions, n_jobs=n_jobs,
        )

    return accs, ranked, noise_type
