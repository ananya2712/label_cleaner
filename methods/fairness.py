"""
Fairness metrics and datascope utilities.

demographic_parity_gap — |P(ŷ=1 | protected) − P(ŷ=1 | unprotected)|
"""

from typing import Hashable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from datascope.importance.utility import MetricCallable, SklearnModelUtility


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


class SklearnModelDemographicParityDifference(SklearnModelUtility):
    """
    Datascope utility that scores a model by the NEGATIVE demographic parity
    gap of its test-set predictions (higher = fairer), so that ascending
    Shapley importance = most fairness-harmful sample first — the same
    convention as accuracy-based DataScope cleaning.

    Group membership comes from a precomputed `groupings` int array aligned
    with the test set (0 = unprotected, 1 = protected). Passing groupings
    (rather than a sensitive-feature column index) is required because the
    p2b pipeline's PCA destroys column identity, and datascope's Monte Carlo
    method subsamples the test set internally (handled via `indices`).
    """

    def __init__(self, model, groupings: NDArray) -> None:
        super().__init__(model, None)
        self._groupings = np.asarray(groupings, dtype=int)

    def _metric_score(
        self,
        metric: Optional[MetricCallable],
        y_test: Union[NDArray, Series, DataFrame],
        y_pred: Union[NDArray, Series, DataFrame],
        y_pred_proba: Optional[Union[NDArray, Series, DataFrame]] = None,
        *,
        X_test: Optional[Union[NDArray, DataFrame]] = None,
        indices: Optional[NDArray] = None,
        metric_requires_probabilities: bool = False,
        classes: Optional[List[Hashable]] = None,
    ) -> float:
        _, y_pred_processed, _ = self._process_metric_score_inputs(y_test, y_pred)
        groupings = self._groupings
        if indices is not None:
            groupings = groupings[indices]
        return -demographic_parity_gap(y_pred_processed, groupings.astype(bool))

    def elementwise_score(
        self,
        X_train: Union[NDArray, DataFrame],
        y_train: Union[NDArray, Series],
        X_test: Union[NDArray, DataFrame],
        y_test: Union[NDArray, Series],
        metadata_train: Optional[Union[NDArray, DataFrame]] = None,
        metadata_test: Optional[Union[NDArray, DataFrame]] = None,
    ) -> NDArray:
        # Per-test-point decomposition of the negative DP gap, linearized at
        # the full-training-data operating point (the same trick datascope's
        # EqualizedOddsDifference utility uses for TPR/FPR).
        n_test = X_test.shape[0]
        classes = np.unique(y_train)
        utilities = np.zeros((len(classes), n_test), dtype=float)
        groupings = self._groupings
        try:
            model = self._model_fit(self.model, X_train, y_train, metadata=metadata_train)
            y_pred = self._model_predict(model, X_test, metadata=metadata_test)
            idx_p = groupings == 1
            idx_u = groupings == 0
            if not idx_p.any() or not idx_u.any():
                return utilities
            rate_p = float(np.mean(y_pred[idx_p] == 1))
            rate_u = float(np.mean(y_pred[idx_u] == 1))
            if rate_p == rate_u:
                return utilities
            idx_max = idx_p if rate_p > rate_u else idx_u
            idx_min = idx_u if rate_p > rate_u else idx_p
            pos_rows = np.where(classes == 1)[0]
            if len(pos_rows) == 1:  # class 1 present in y_train (np.unique gives at most one match)
                r = pos_rows[0]
                # Predicting 1 for a higher-rate-group point widens the gap
                # (negative utility); for a lower-rate-group point it narrows it.
                utilities[r, idx_max] = -1.0 / float(np.sum(idx_max))
                utilities[r, idx_min] = +1.0 / float(np.sum(idx_min))
        except ValueError:
            pass
        return utilities

    def elementwise_null_score(
        self,
        X_train: Union[NDArray, DataFrame],
        y_train: Union[NDArray, Series],
        X_test: Union[NDArray, DataFrame],
        y_test: Union[NDArray, Series],
        metadata_train: Optional[Union[NDArray, DataFrame]] = None,
        metadata_test: Optional[Union[NDArray, DataFrame]] = None,
    ) -> NDArray:
        return np.zeros(len(y_test), dtype=float)
