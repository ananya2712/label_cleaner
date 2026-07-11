"""Plain-python tests for methods/fairness.py. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py
"""
import numpy as np

from label_cleaner.methods.fairness import demographic_parity_gap


def test_gap_basic():
    # protected: 2 of 3 predicted positive (0.667); unprotected: 1 of 2 (0.5)
    y_pred = np.array([1, 1, 0, 1, 0])
    prot   = np.array([True, True, True, False, False])
    assert abs(demographic_parity_gap(y_pred, prot) - abs(2/3 - 1/2)) < 1e-12


def test_gap_zero_when_equal():
    y_pred = np.array([1, 0, 1, 0])
    prot   = np.array([True, True, False, False])
    assert demographic_parity_gap(y_pred, prot) == 0.0


def test_gap_degenerate_group_returns_zero():
    y_pred = np.array([1, 0, 1])
    assert demographic_parity_gap(y_pred, np.array([True, True, True])) == 0.0
    assert demographic_parity_gap(y_pred, np.array([False, False, False])) == 0.0


def _make_biased_data(seed=0, n=240):
    """Synthetic binary task with NNAR-style planted bias: some protected-group
    training labels are flipped to 0. Returns train/test arrays, the flipped
    train positions, and the test protected mask."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    prot = rng.rand(n) < 0.4
    X[:, 3] = prot.astype(float)
    y = (X[:, 0] + rng.randn(n) * 0.3 > 0).astype(int)
    n_train = 180
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train].copy(), y[n_train:]
    prot_tr, prot_te = prot[:n_train], prot[n_train:]
    flip_candidates = np.where(prot_tr & (ytr == 1))[0]
    flipped = rng.choice(flip_candidates, size=len(flip_candidates) // 2, replace=False)
    ytr[flipped] = 0
    return Xtr, ytr, Xte, yte, flipped, prot_te


def test_dp_utility_neighbor_sign():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
    from label_cleaner.methods.fairness import SklearnModelDemographicParityDifference

    Xtr, ytr, Xte, yte, flipped, prot_te = _make_biased_data()
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
    util = SklearnModelDemographicParityDifference(pipe[-1], groupings=prot_te.astype(int))
    imp = ShapleyImportance(method=ImportanceMethod.NEIGHBOR, pipeline=pipe[:-1], utility=util)
    scores = imp.fit(Xtr, ytr).score(Xte, yte)
    assert scores.shape == (len(Xtr),)
    assert np.isfinite(scores).all()
    # Sign check: bias-injected rows inflate the DP gap, so under the
    # negative-gap utility they must score BELOW the overall mean.
    assert scores[flipped].mean() < scores.mean(), (
        f"sign convention wrong: flipped mean {scores[flipped].mean():.6f} "
        f">= overall mean {scores.mean():.6f}"
    )


def test_dp_utility_montecarlo_runs():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
    from label_cleaner.methods.fairness import SklearnModelDemographicParityDifference

    Xtr, ytr, Xte, yte, _, prot_te = _make_biased_data(seed=1)
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
    util = SklearnModelDemographicParityDifference(pipe[-1], groupings=prot_te.astype(int))
    imp = ShapleyImportance(method=ImportanceMethod.MONTECARLO, pipeline=pipe[:-1],
                            utility=util, mc_iterations=5)
    scores = imp.fit(Xtr, ytr).score(Xte, yte)
    assert scores.shape == (len(Xtr),)
    assert np.isfinite(scores).all()


def test_clean_datascope_fair_reduces_gap():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from label_cleaner.methods.cleaning import clean_datascope_fair
    from label_cleaner.methods.cleaning import action_restore_labels

    Xtr, ytr_noisy, Xte, yte, flipped, prot_te = _make_biased_data(seed=2)
    ytr_clean = ytr_noisy.copy()
    ytr_clean[flipped] = 1  # ground truth: flips were 1 -> 0
    factory = lambda: Pipeline([("sc", StandardScaler()),
                                ("m", LogisticRegression(max_iter=1000))])
    accs, dps, ranked = clean_datascope_fair(
        factory, Xtr, ytr_noisy, Xte, yte,
        noisy_positions=flipped,
        action_fn=action_restore_labels(ytr_clean),
        proportions=np.array([0.0, 1.0]),
        protected_test=prot_te,
    )
    assert len(accs) == len(dps) == 2
    assert set(ranked.tolist()) == set(flipped.tolist())
    # Restoring all planted bias must not worsen the parity gap.
    assert dps[1] <= dps[0] + 0.02, dps


def test_fair_heuristic_ranking_order():
    from label_cleaner.methods.cleaning import _dp_heuristic_scores

    # protected: labels [0,0,1] (rate 1/3); unprotected: [1,1] (rate 1).
    y = np.array([0, 0, 1, 1, 1])
    prot = np.array([True, True, True, False, False])
    candidates = np.array([0, 2, 3])
    scores = _dp_heuristic_scores(y, prot, candidates)
    # Removing candidate 0 (protected, y=0) raises the protected rate to 1/2:
    # gap 2/3 -> 1/2, reduction > 0. Removing candidate 2 (protected, y=1)
    # lowers it to 0: gap -> 1, reduction < 0. Removing candidate 3
    # (unprotected, y=1) keeps unprotected rate 1: reduction = 0.
    assert scores[0] > scores[2]
    assert scores[0] > scores[1]
    base = abs(1/3 - 1.0)
    assert abs(scores[0] - (base - abs(1/2 - 1.0))) < 1e-12
    assert abs(scores[2] - 0.0) < 1e-12
    assert abs(scores[1] - (base - abs(0.0 - 1.0))) < 1e-12


if __name__ == "__main__":
    test_gap_basic()
    test_gap_zero_when_equal()
    test_gap_degenerate_group_returns_zero()
    test_dp_utility_neighbor_sign()
    test_dp_utility_montecarlo_runs()
    test_clean_datascope_fair_reduces_gap()
    test_fair_heuristic_ranking_order()
    print("test_fairness: OK")
