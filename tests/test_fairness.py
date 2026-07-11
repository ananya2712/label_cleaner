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


if __name__ == "__main__":
    test_gap_basic()
    test_gap_zero_when_equal()
    test_gap_degenerate_group_returns_zero()
    print("test_fairness: OK")
