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


def clean_datascope_iterative(
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
    rerank_every: float = 0.05,
) -> Tuple[List[float], np.ndarray]:
    """
    Iterative DataScope with LOO reranking.

    Uses full Shapley for the initial ranking, then after every `rerank_every`
    fraction of noisy samples have been cleaned, refits the model on the current
    cleaned data and re-scores remaining noisy candidates via Leave-One-Out (LOO):

        LOO_importance[i] = acc(X without i, y without i) - acc(X with i, y with i)

    LOO is O(n_remaining × fit_time) per reranking step — cheap for linear models
    and much faster than a full Shapley recomputation.  The initial Shapley ranking
    anchors the ordering; LOO refines it as noise is progressively removed.

    Parameters
    ----------
    rerank_every : fraction of noisy samples between reranking steps (default 0.05)
    """
    # --- Initial Shapley ranking ---
    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)
    importances  = _compute_importances(
        pipeline, X_train, y_train, X_test, y_test, importance_method, mc_iterations
    )
    sorted_order = np.argsort(-importances[noisy_positions])
    ranked_noisy = noisy_positions[sorted_order].tolist()   # mutable list

    rerank_interval = max(1, int(rerank_every * len(noisy_positions)))
    since_last_rerank = 0

    # Working copies of training data that we update as samples are cleaned
    X_cur, y_cur = X_train.copy(), y_train.copy()

    accs = []
    cleaned_so_far: list = []

    for p in proportions:
        target = int(p * len(ranked_noisy) + len(cleaned_so_far))
        # Clean up to `target` total samples
        while len(cleaned_so_far) < target and len(ranked_noisy) > 0:
            next_idx = ranked_noisy.pop(0)
            cleaned_so_far.append(next_idx)
            since_last_rerank += 1

            # Apply action to the working copy
            X_cur, y_cur = action_fn(X_cur, y_cur, np.array([next_idx]))

            # Rerank remaining candidates every `rerank_interval` cleans
            if since_last_rerank >= rerank_interval and len(ranked_noisy) > 0:
                pipeline.fit(X_cur, y_cur)
                current_acc = accuracy_score(y_test, pipeline.predict(X_test))

                loo_scores = []
                for cand in ranked_noisy:
                    mask = np.ones(len(X_cur), dtype=bool)
                    mask[cand] = False
                    p_loo = pipeline_factory()
                    p_loo.fit(X_cur[mask], y_cur[mask])
                    loo_scores.append(
                        accuracy_score(y_test, p_loo.predict(X_test)) - current_acc
                    )
                ranked_noisy = [
                    ranked_noisy[i]
                    for i in np.argsort(-np.array(loo_scores))
                ]
                since_last_rerank = 0

        pipeline.fit(X_cur, y_cur)
        accs.append(accuracy_score(y_test, pipeline.predict(X_test)))

    # Return initial + cleaned order as the final ranking for traceability
    final_ranked = np.array(cleaned_so_far + ranked_noisy, dtype=int)
    return accs, final_ranked


def clean_datascope_hybrid(
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
    alpha: float = 0.5,
    n_jobs: int = 1,
) -> Tuple[List[float], np.ndarray]:
    """
    Hybrid DataScope + CleanLab ranking.

    Combines two complementary signals over the known noisy positions:
      - DataScope Shapley importance: how much each sample hurts accuracy
      - CleanLab (1 - self_confidence): how likely each label is wrong

    Both scores are min-max normalised to [0, 1] then blended:
        hybrid_score = alpha * shapley + (1 - alpha) * (1 - self_conf)

    Higher combined score → clean first.

    Parameters
    ----------
    alpha   : weight for the Shapley signal (default 0.5 = equal blend)
    n_jobs  : parallelism for CleanLab cross_val_predict
    """
    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)

    # --- Shapley importances (over noisy positions only) ---
    importances = _compute_importances(
        pipeline, X_train, y_train, X_test, y_test, importance_method, mc_iterations
    )
    shap_noisy = importances[noisy_positions].astype(float)

    # --- CleanLab self_confidence (over ALL training samples) ---
    pred_probs = cross_val_predict(
        pipeline_factory(), X_train, y_train,
        cv=5, method="predict_proba", n_jobs=n_jobs,
    )
    self_conf_all = pred_probs[np.arange(len(y_train)), y_train]
    conf_noisy = self_conf_all[noisy_positions].astype(float)

    # --- Normalise both signals to [0, 1] ---
    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-12)

    shap_norm = _minmax(shap_noisy)
    conf_norm = _minmax(1.0 - conf_noisy)   # high = label likely wrong

    hybrid = alpha * shap_norm + (1.0 - alpha) * conf_norm
    sorted_order = np.argsort(-hybrid)       # descending: most harmful first
    ranked_noisy = noisy_positions[sorted_order]

    accs = []
    for p in proportions:
        n_clean = int(p * len(ranked_noisy))
        X_c, y_c = action_fn(X_train, y_train, ranked_noisy[:n_clean])
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
# Otsu threshold helper
# ---------------------------------------------------------------------------
# KDE valley detection threshold helper
# ---------------------------------------------------------------------------

def _kde_valley_threshold(scores: np.ndarray, n_bins: int = 512) -> float:
    """
    Find the adaptive correction/removal threshold via KDE valley detection.

    Smooths a histogram of `scores` using Scott's bandwidth rule, finds the
    two most prominent peaks via `find_peaks`, then returns the position of
    the minimum density between them as the threshold:

      below threshold → highly suspicious (remove)
      above threshold → moderately suspicious (correct via action_fn)

    Falls back to the median when fewer than two prominent peaks are found —
    indicating a unimodal distribution (e.g. random label noise) where any
    split would be arbitrary.

    Parameters
    ----------
    scores : self_confidence scores for CleanLab-flagged samples, in [0, 1]
    n_bins : histogram resolution before smoothing (default 512)
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    scores = np.clip(scores, 1e-6, 1 - 1e-6)

    # Scott's bandwidth rule: h = 1.06 * std * n^(-1/5), converted to bin units
    sigma_bins = max(1.0, 1.06 * float(np.std(scores)) * len(scores) ** (-0.2) * n_bins)

    counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    density = gaussian_filter1d(counts.astype(float), sigma=sigma_bins)

    # Find peaks with minimum prominence = 10% of the density range,
    # and minimum distance of n_bins // 8 bins apart to avoid nearby spurious peaks
    prominence = 0.10 * (density.max() - density.min())
    peaks, props = find_peaks(density, prominence=prominence, distance=n_bins // 8)

    if len(peaks) < 2:
        return float(np.median(scores))

    # Take the two most prominent peaks
    top2 = peaks[np.argsort(props["prominences"])[-2:]]
    lo, hi = int(top2.min()), int(top2.max())

    # Valley = position of minimum density between the two peaks
    valley = lo + int(np.argmin(density[lo:hi + 1]))
    return float(bin_centres[valley])


def _otsu_threshold(scores: np.ndarray, n_bins: int = 256) -> float:
    """
    Find the adaptive correction/removal threshold via Otsu's method.

    Otsu sweeps every candidate threshold and picks the one that maximises
    between-class variance of the two resulting groups (below = remove,
    above = correct).  Non-parametric — makes no distributional assumptions.

    Before applying the Otsu split the bimodality coefficient (BC) is checked:
        BC = (skewness² + 1) / Pearson-kurtosis
    BC > 5/9 ≈ 0.556 indicates a bimodal or heavy-tailed distribution where
    Otsu is meaningful.  Falls back to the median when BC ≤ 5/9, which catches
    unimodal distributions (e.g. random label noise) where any split would
    be arbitrary.
    """
    from scipy.stats import skew, kurtosis as sp_kurtosis

    scores = np.clip(scores, 1e-6, 1 - 1e-6)

    # Bimodality check — skip Otsu if distribution appears unimodal
    g1 = float(skew(scores))
    g2 = float(sp_kurtosis(scores, fisher=False))   # Pearson kurtosis (normal=3)
    bc = (g1 ** 2 + 1) / g2 if g2 > 0 else 0.0
    if bc <= 5 / 9:
        return float(np.median(scores))

    counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return float(np.median(scores))

    prefix_w  = np.cumsum(counts)
    prefix_wm = np.cumsum(counts * bin_centres)

    best_var = -1.0
    best_t   = float(np.median(scores))

    for i in range(1, n_bins):
        w0 = prefix_w[i - 1]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = prefix_wm[i - 1] / w0
        mu1 = (prefix_wm[-1] - prefix_wm[i - 1]) / w1
        var_between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t   = float(bin_centres[i])

    return best_t


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

    # KDE valley detection: smooth the self_confidence density of all flagged
    # samples and split at the deepest valley between two peaks.
    #
    # Below threshold → highly suspicious (remove)
    # Above threshold → moderately suspicious (correct via action_fn)
    #
    # Falls back to the median when no valley exists (unimodal distribution).
    adaptive_threshold = _kde_valley_threshold(self_conf[cl_ranked])

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
