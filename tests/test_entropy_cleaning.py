"""Unit: clean_entropy wiring invariants on synthetic data. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py
(fast: ~seconds, synthetic data only)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from label_cleaner.methods.cleaning import action_restore_labels, clean_entropy


def _factory():
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=200))])


def test_clean_entropy_wiring():
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 4)
    y_clean = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_noisy = y_clean.copy()
    flip = rng.choice(n, size=20, replace=False)
    y_noisy[flip] = 1 - y_noisy[flip]

    X_test = rng.randn(40, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    protected_test = X_test[:, 2] > 0

    proportions = np.array([0.0, 0.5, 1.0])
    accs, dps, ranked = clean_entropy(
        _factory, X, y_noisy, X_test, y_test,
        action_restore_labels(y_clean), proportions,
        protected_test=protected_test,
    )

    assert len(accs) == 3 and len(dps) == 3
    assert all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in accs), accs
    assert all(np.isfinite(d) and 0.0 <= d <= 1.0 for d in dps), dps
    # Unsupervised: the ranking covers every training row exactly once.
    assert sorted(int(i) for i in ranked) == list(range(n))


if __name__ == "__main__":
    test_clean_entropy_wiring()
    print("test_entropy_cleaning: OK")
