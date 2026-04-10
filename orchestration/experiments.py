"""
Experiment service layer.

Each run_* function orchestrates:
  1) fixed split prep
  2) noise injection
  3) DataScope / Random / CleanLab cleaning methodologies
"""

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score

from ..methods.cleaning import (
    action_cap,
    action_flip_labels,
    action_remove,
    action_restore_labels,
    clean_cleanlab,
    clean_datascope,
    clean_random,
)
from ..data.datasets import DatasetInfo
from ..core.models import MethodCurves, NoiseBundle
from ..methods.noise import inject_mnar, inject_nnar, inject_outlier, inject_rnd_label
from ..core.prep import prepare_fixed_split


DEFAULT_PROPORTIONS = np.linspace(0, 1, num=21)


def _baseline_acc(pipeline_factory: Callable, X_train, y_train, X_test, y_test) -> float:
    p = pipeline_factory()
    p.fit(X_train, y_train)
    return accuracy_score(y_test, p.predict(X_test))


def _run_methods(pipeline_factory: Callable, X_train_noisy, y_train_noisy, X_test, y_test,
                 noisy_positions, action_fn, proportions, n_cleanlab_jobs: int = 1):
    accs_ds, ds_ranked = clean_datascope(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions
    )
    rnd_mean, rnd_std = clean_random(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions
    )
    accs_cl, cl_ranked = clean_cleanlab(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions, n_jobs=n_cleanlab_jobs
    )
    return accs_ds, rnd_mean, rnd_std, accs_cl, ds_ranked, cl_ranked


def build_noise_bundle_outlier(split, outlier_col_idx: int, noise_level: float,
                               seed: int = 42) -> NoiseBundle:
    """
    Inject outliers globally before split, then map noisy rows to train positions.
    """
    X_full = np.empty((len(split.train_idx) + len(split.test_idx), split.X_train.shape[1]))
    y_full = np.empty((len(split.train_idx) + len(split.test_idx),), dtype=int)
    X_full[split.train_idx] = split.X_train
    X_full[split.test_idx] = split.X_test
    y_full[split.train_idx] = split.y_train
    y_full[split.test_idx] = split.y_test

    X_noisy_full, global_noisy_positions, cap_value = inject_outlier(
        X_full, outlier_col_idx, noise_level=noise_level, seed=seed
    )
    train_mask = np.isin(split.train_idx, global_noisy_positions)
    noisy_positions = np.where(train_mask)[0]
    X_train_noisy = X_noisy_full[split.train_idx]
    return NoiseBundle(
        X_noisy=X_train_noisy,
        y_noisy=split.y_train.copy(),
        noisy_positions=noisy_positions,
        metadata={"cap_value": float(cap_value)},
    )


def run_outlier_experiment(ds: DatasetInfo, pipeline_factory: Callable, noise_level: float = 0.2,
                           proportions: np.ndarray = DEFAULT_PROPORTIONS) -> MethodCurves:
    split = prepare_fixed_split(ds.X, ds.y)
    bundle = build_noise_bundle_outlier(split, ds.outlier_col_idx, noise_level=noise_level)

    baseline = _baseline_acc(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test
    )
    cap_fn = action_cap(ds.outlier_col_idx, bundle.metadata["cap_value"])
    accs_ds, rnd_mean, rnd_std, accs_cl, ds_ranked, _ = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, cap_fn, proportions,
    )

    # Outlier-specific removal curve (drop top-k DataScope-ranked noisy rows)
    remove_fn = action_remove()
    accs_rm = []
    for p in proportions:
        n_rm = int(p * len(ds_ranked))
        X_c, y_c = remove_fn(bundle.X_noisy, bundle.y_noisy, ds_ranked[:n_rm])
        pipe = pipeline_factory()
        pipe.fit(X_c, y_c)
        accs_rm.append(accuracy_score(split.y_test, pipe.predict(split.X_test)))

    return MethodCurves(
        datascope=accs_ds,
        random_mean=rnd_mean,
        random_std=rnd_std,
        cleanlab=accs_cl,
        baseline=baseline,
        proportions=proportions,
        datascope_removal=accs_rm,
    )


def run_random_label_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                                noise_level: float = 0.2,
                                proportions: np.ndarray = DEFAULT_PROPORTIONS,
                                seed: int = 42) -> MethodCurves:
    split = prepare_fixed_split(ds.X, ds.y)
    y_noisy, noisy_positions = inject_rnd_label(split.y_train, noise_level=noise_level, seed=seed)
    bundle = NoiseBundle(
        X_noisy=split.X_train.copy(),
        y_noisy=y_noisy,
        noisy_positions=noisy_positions,
        metadata={},
    )
    baseline = _baseline_acc(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test
    )
    restore_fn = action_restore_labels(split.y_train)
    accs_ds, rnd_mean, rnd_std, accs_cl, _, _ = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, restore_fn, proportions,
    )
    return MethodCurves(
        datascope=accs_ds, random_mean=rnd_mean, random_std=rnd_std,
        cleanlab=accs_cl, baseline=baseline, proportions=proportions
    )


def run_nnar_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                        noise_level: float = 0.2,
                        proportions: np.ndarray = DEFAULT_PROPORTIONS,
                        seed: int = 42) -> MethodCurves:
    split = prepare_fixed_split(ds.X, ds.y)
    protected_train = ds.protected_group_mask[split.train_idx]
    y_noisy, noisy_positions = inject_nnar(
        split.y_train, protected_train, noise_level=noise_level, seed=seed
    )
    bundle = NoiseBundle(
        X_noisy=split.X_train.copy(),
        y_noisy=y_noisy,
        noisy_positions=noisy_positions,
        metadata={},
    )
    baseline = _baseline_acc(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test
    )
    restore_fn = action_restore_labels(split.y_train)
    accs_ds, rnd_mean, rnd_std, accs_cl, _, _ = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, restore_fn, proportions,
    )
    return MethodCurves(
        datascope=accs_ds, random_mean=rnd_mean, random_std=rnd_std,
        cleanlab=accs_cl, baseline=baseline, proportions=proportions
    )


def run_mnar_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                        noise_level: float = 0.2,
                        proportions: np.ndarray = DEFAULT_PROPORTIONS,
                        seed: int = 42) -> MethodCurves:
    split = prepare_fixed_split(ds.X, ds.y)
    protected_train = ds.protected_group_mask[split.train_idx]
    X_noisy, noisy_positions = inject_mnar(
        split.X_train, protected_train, [ds.outlier_col_idx], noise_level=noise_level, seed=seed
    )
    bundle = NoiseBundle(
        X_noisy=X_noisy,
        y_noisy=split.y_train.copy(),
        noisy_positions=noisy_positions,
        metadata={},
    )
    baseline = _baseline_acc(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test
    )
    # Feature corruption is not directly restorable; label flip is the comparison action.
    flip_fn = action_flip_labels()
    accs_ds, rnd_mean, rnd_std, accs_cl, _, _ = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, flip_fn, proportions,
    )
    return MethodCurves(
        datascope=accs_ds, random_mean=rnd_mean, random_std=rnd_std,
        cleanlab=accs_cl, baseline=baseline, proportions=proportions
    )
