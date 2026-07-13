"""Unit: inject_outlier wiring invariants on synthetic data. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_outlier_injection.py
(fast: ~1s, synthetic data only)
"""
import numpy as np

from label_cleaner.methods.noise import inject_outlier


def test_inject_outlier_scale_aware():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 3)
    X_orig = X.copy()
    col = 1
    mean_x, std_x = X[:, col].mean(), X[:, col].std()

    X_noisy, pos, cap = inject_outlier(X, col, noise_level=0.2, seed=7)

    # Count and uniqueness of corrupted rows.
    assert len(pos) == int(0.2 * len(X))
    assert len(np.unique(pos)) == len(pos)
    # Every corrupted value is a high-side extreme at least 3 sigma out.
    assert np.all(X_noisy[pos, col] >= mean_x + 3 * std_x - 1e-9)
    # ... and no more than 5 sigma out.
    assert np.all(X_noisy[pos, col] <= mean_x + 5 * std_x + 1e-9)
    # Per-row variation: the outliers are not one constant.
    assert len(np.unique(X_noisy[pos, col])) > 1
    # Untouched rows and other columns are unchanged; input not mutated.
    untouched = np.setdiff1d(np.arange(len(X)), pos)
    assert np.array_equal(X_noisy[np.ix_(untouched, [col])], X_orig[np.ix_(untouched, [col])])
    other_cols = [c for c in range(X.shape[1]) if c != col]
    assert np.array_equal(X_noisy[:, other_cols], X_orig[:, other_cols])
    assert np.array_equal(X, X_orig)
    # Cap is still the clean 2-sigma threshold.
    assert abs(cap - (mean_x + 2 * std_x)) < 1e-9
    # Same seed reproduces identical output.
    X_noisy2, pos2, cap2 = inject_outlier(X, col, noise_level=0.2, seed=7)
    assert np.array_equal(X_noisy, X_noisy2)
    assert np.array_equal(pos, pos2)
    assert cap == cap2


if __name__ == "__main__":
    test_inject_outlier_scale_aware()
    print("test_outlier_injection: OK")
