"""
Cleaning methodology functions — one per method.

Each cleaner takes a ranked ordering of noisy positions and an action function,
then runs an incremental cleaning loop over a set of proportions.

The action function encapsulates *what* to do when cleaning a sample — it varies
by noise type:
  - outlier   : cap feature value at 2-sigma
  - rnd_label : restore ground-truth label
  - nnar      : restore ground-truth label
  - mnar      : flip label (feature corruption can't be undone)

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
from scipy.optimize import brentq
from scipy.stats import beta as beta_dist
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
        n_clean       = int(p * len(ranked_noisy))
        X_c, y_c     = action_fn(X_train, y_train, ranked_noisy[:n_clean])
        pipeline.fit(X_c, y_c)
        accs.append(accuracy_score(y_test, pipeline.predict(X_test)))

    return accs, ranked_noisy
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
        X_c, y_c = action_fn(X_train, y_train, cl_ranked[:n_clean])
        pipeline.fit(X_c, y_c)
        accs.append(accuracy_score(y_test, pipeline.predict(X_test)))

    return accs, cl_ranked


# ---------------------------------------------------------------------------
# Beta Mixture Model threshold helper
# ---------------------------------------------------------------------------

def _bmm_threshold(scores: np.ndarray, n_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Fit a 2-component Beta Mixture Model to `scores` via EM and return the
    crossing point of the two component PDFs as the correction threshold.

    The low-mean component captures noisy/unrecoverable samples (→ remove).
    The high-mean component captures moderately suspicious ones (→ correct).
    Brent's method finds the root of pdf_low(x) - pdf_high(x) on the interval
    between the two component means.

    Falls back to the median if the mixture degenerates or the root search fails.
    """
    scores = np.clip(scores, 1e-6, 1 - 1e-6)
    n = len(scores)

    # --- initialise two components by splitting at the median ---
    med = float(np.median(scores))
    params = []
    for subset in [scores[scores <= med], scores[scores > med]]:
        if len(subset) < 2:
            subset = scores
        mu  = float(np.clip(np.mean(subset), 0.01, 0.99))
        var = float(np.clip(np.var(subset),  1e-6, mu * (1 - mu) - 1e-6))
        fac = mu * (1 - mu) / var - 1
        params.append([max(fac * mu, 0.1), max(fac * (1 - mu), 0.1)])

    alphas  = np.array([p[0] for p in params])
    betas_  = np.array([p[1] for p in params])
    weights = np.array([0.5, 0.5])

    prev_ll = -np.inf
    for _ in range(n_iter):
        # E-step
        resp = np.column_stack([
            weights[k] * beta_dist.pdf(scores, alphas[k], betas_[k])
            for k in range(2)
        ])
        row_sum = resp.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum < 1e-300, 1e-300, row_sum)
        resp /= row_sum

        # M-step
        Nk = resp.sum(axis=0)
        weights = Nk / n
        for k in range(2):
            mu_k  = float(np.clip((resp[:, k] * scores).sum() / Nk[k], 0.01, 0.99))
            var_k = float(np.clip((resp[:, k] * (scores - mu_k) ** 2).sum() / Nk[k],
                                  1e-6, mu_k * (1 - mu_k) - 1e-6))
            fac = mu_k * (1 - mu_k) / var_k - 1
            alphas[k]  = max(fac * mu_k,        0.1)
            betas_[k]  = max(fac * (1 - mu_k),  0.1)

        ll = float(np.log(row_sum).sum())
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # component means
    means = alphas / (alphas + betas_)
    lo, hi = int(np.argmin(means)), int(np.argmax(means))

    if means[lo] >= means[hi]:          # degenerate — fall back
        return float(np.median(scores))

    def _diff(x: float) -> float:
        return (weights[lo] * float(beta_dist.pdf(x, alphas[lo], betas_[lo])) -
                weights[hi] * float(beta_dist.pdf(x, alphas[hi], betas_[hi])))

    # bracket must straddle zero; if it doesn't, the distributions overlap heavily
    lo_bound = float(means[lo]) + 0.01
    hi_bound = float(means[hi]) - 0.01
    if lo_bound >= hi_bound or _diff(lo_bound) * _diff(hi_bound) > 0:
        return float(np.median(scores))

    try:
        return float(brentq(_diff, lo_bound, hi_bound))
    except ValueError:
        return float(np.median(scores))


# ---------------------------------------------------------------------------
# Adaptive CleanLab (correction vs. filtering)
# ---------------------------------------------------------------------------

def clean_cleanlab_adaptive(
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
    Adaptive CleanLab: uses self_confidence scores to decide per-sample action.

    For each flagged sample in the top-k at cleaning proportion p:
      - self_confidence < correction_threshold  → apply action_fn (correct the label/feature)
      - self_confidence >= correction_threshold → remove the sample entirely

    Rationale: samples with higher self_confidence (upper half) are moderately
    suspicious — the model still has some belief in their label, so correction
    is the safer action. Samples with very low self_confidence (lower half) are
    highly uncertain and removing them avoids introducing wrong corrections.

    Parameters
    ----------
    action_fn            : noise-type-specific correction (cap / restore / flip)
    correction_threshold : self_confidence cutoff; samples below this get corrected,
                           samples at or above get removed (default 0.5)

    Returns
    -------
    accs      : accuracy at each proportion
    cl_ranked : training indices ranked most-to-least suspicious (same as clean_cleanlab)
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
    # self_confidence[i] = P(current label is correct | x_i)
    self_conf = pred_probs[np.arange(len(y_train)), y_train]

    # Fit a 2-component Beta Mixture Model to the self_confidence scores of all
    # CleanLab-flagged samples and use the crossing point of the two component
    # PDFs as the split threshold.
    #
    # Low-mean component  → highly suspicious samples  → remove
    # High-mean component → moderately suspicious ones → correct via action_fn
    #
    # Falls back to the median when the two components overlap heavily or the
    # mixture degenerates (e.g., very few flagged samples).
    adaptive_threshold = _bmm_threshold(self_conf[cl_ranked])

    accs = []
    for p in proportions:
        n_clean    = int(p * len(cl_ranked))
        top_k      = cl_ranked[:n_clean]
        to_correct = top_k[self_conf[top_k] >= adaptive_threshold]
        to_remove  = top_k[self_conf[top_k] <  adaptive_threshold]

        # Apply correction first (preserves array size), then filter rows
        X_c, y_c = action_fn(X_train, y_train, to_correct)
        if len(to_remove) > 0:
            keep       = np.ones(len(X_c), dtype=bool)
            keep[to_remove] = False
            X_c, y_c  = X_c[keep], y_c[keep]

        pipeline = pipeline_factory()
        pipeline.fit(X_c, y_c)
        accs.append(accuracy_score(y_test, pipeline.predict(X_test)))

    return accs, cl_ranked


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
            n_clean   = int(p * len(perm))
            pipeline  = pipeline_factory()
            X_c, y_c  = action_fn(X_train, y_train, perm[:n_clean])
            pipeline.fit(X_c, y_c)
            seed_accs.append(accuracy_score(y_test, pipeline.predict(X_test)))
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
        pipeline = pipeline_factory()
        pipeline.fit(X_c, y_c)
        accs.append(accuracy_score(y_test, pipeline.predict(X_test)))

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
