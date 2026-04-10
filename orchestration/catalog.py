"""
Service catalog for modular wiring.

Provides a single place to resolve:
  - dataset loaders
  - pipeline factories
  - experiment runners
"""

from functools import partial
from typing import Callable, Dict

from .experiments import (
    run_mnar_experiment,
    run_nnar_experiment,
    run_outlier_experiment,
    run_random_label_experiment,
)
from ..methods.pipelines import make_pipeline_a, make_pipeline_b


def pipeline_factory_a(num_col_indices, cat_col_indices):
    return partial(make_pipeline_a, num_col_indices=num_col_indices, cat_col_indices=cat_col_indices)


def pipeline_factory_b(num_col_indices=None, cat_col_indices=None, n_features: int = 8):
    n_pca = min(8, n_features)
    k_best = min(5, n_pca)
    return partial(make_pipeline_b, n_pca_components=n_pca, k_best=k_best)


EXPERIMENT_RUNNERS: Dict[str, Callable] = {
    "outlier": run_outlier_experiment,
    "rnd_label": run_random_label_experiment,
    "nnar": run_nnar_experiment,
    "mnar": run_mnar_experiment,
}
