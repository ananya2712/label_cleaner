"""
Experiment service layer.

Each run_* function orchestrates:
  1) fixed split prep
  2) noise injection
  3) DataScope / Random / CleanLab cleaning methodologies
"""

from typing import Callable

import numpy as np
from datascope.importance.shapley import ImportanceMethod
from sklearn.metrics import accuracy_score

from ..methods.cleaning import (
    action_cap,
    action_remove,
    action_restore_labels,
    clean_cleanlab,
    clean_datascope,
    clean_hybrid_auto,
    clean_kairos,
    clean_random,
)
from ..data.datasets import DatasetInfo
from ..core.models import ExperimentArtifacts, MethodCurves, NoiseBundle
from ..methods.noise import inject_mnar, inject_nnar, inject_outlier, inject_rnd_label
from ..core.prep import prepare_fixed_split


DEFAULT_PROPORTIONS = np.linspace(0, 1, num=21)


def _baseline_acc(pipeline_factory: Callable, X_train, y_train, X_test, y_test) -> float:
    p = pipeline_factory()
    p.fit(X_train, y_train)
    return accuracy_score(y_test, p.predict(X_test))


def _random_rankings(noisy_positions: np.ndarray, n_seeds: int = 3):
    rankings = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + 100)
        perm = noisy_positions.copy()
        rng.shuffle(perm)
        rankings.append(perm)
    return rankings


def _run_methods(pipeline_factory: Callable, X_train_noisy, y_train_noisy, X_test, y_test,
                 noisy_positions, action_fn, proportions, n_cleanlab_jobs: int = 1,
                 importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
                 mc_iterations: int = 50):
    accs_ds, ds_ranked = clean_datascope(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
    )
    rnd_mean, rnd_std = clean_random(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions
    )
    accs_cl, cl_ranked = clean_cleanlab(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions, n_jobs=n_cleanlab_jobs
    )
    accs_kr, kr_ranked = clean_kairos(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions,
    )
    accs_auto, _, detected_noise = clean_hybrid_auto(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
        n_jobs=n_cleanlab_jobs,
    )
    return accs_ds, rnd_mean, rnd_std, accs_cl, accs_kr, accs_auto, detected_noise, ds_ranked, cl_ranked, kr_ranked


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


def run_outlier_experiment_with_artifacts(
    ds: DatasetInfo,
    pipeline_factory: Callable,
    noise_level: float = 0.2,
    proportions: np.ndarray = DEFAULT_PROPORTIONS,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> ExperimentArtifacts:
    split = prepare_fixed_split(ds.X, ds.y)
    bundle = build_noise_bundle_outlier(split, ds.outlier_col_idx, noise_level=noise_level)

    baseline = _baseline_acc(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test
    )
    cap_fn = action_cap(ds.outlier_col_idx, bundle.metadata["cap_value"])
    accs_ds, rnd_mean, rnd_std, accs_cl, accs_kr, accs_auto, detected_noise, ds_ranked, cl_ranked, kr_ranked = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, cap_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
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

    curves = MethodCurves(
        datascope=accs_ds,
        random_mean=rnd_mean,
        random_std=rnd_std,
        cleanlab=accs_cl,
        baseline=baseline,
        proportions=proportions,
        datascope_removal=accs_rm,
        kairos=accs_kr,
        hybrid_auto=accs_auto,
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        kairos_ranked=kr_ranked,
    )


def run_outlier_experiment(ds: DatasetInfo, pipeline_factory: Callable, noise_level: float = 0.2,
                           proportions: np.ndarray = DEFAULT_PROPORTIONS) -> MethodCurves:
    return run_outlier_experiment_with_artifacts(
        ds, pipeline_factory, noise_level=noise_level, proportions=proportions
    ).curves


def run_random_label_experiment_with_artifacts(
    ds: DatasetInfo,
    pipeline_factory: Callable,
    noise_level: float = 0.2,
    proportions: np.ndarray = DEFAULT_PROPORTIONS,
    seed: int = 42,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> ExperimentArtifacts:
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
    accs_ds, rnd_mean, rnd_std, accs_cl, accs_kr, accs_auto, detected_noise, ds_ranked, cl_ranked, kr_ranked = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, restore_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
    )
    curves = MethodCurves(
        datascope=accs_ds, random_mean=rnd_mean, random_std=rnd_std,
        cleanlab=accs_cl, baseline=baseline, proportions=proportions,
        kairos=accs_kr, hybrid_auto=accs_auto,
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        kairos_ranked=kr_ranked,
    )


def run_random_label_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                                noise_level: float = 0.2,
                                proportions: np.ndarray = DEFAULT_PROPORTIONS,
                                seed: int = 42) -> MethodCurves:
    return run_random_label_experiment_with_artifacts(
        ds, pipeline_factory, noise_level=noise_level, proportions=proportions, seed=seed
    ).curves


def run_nnar_experiment_with_artifacts(
    ds: DatasetInfo,
    pipeline_factory: Callable,
    noise_level: float = 0.2,
    proportions: np.ndarray = DEFAULT_PROPORTIONS,
    seed: int = 42,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> ExperimentArtifacts:
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
    accs_ds, rnd_mean, rnd_std, accs_cl, accs_kr, accs_auto, detected_noise, ds_ranked, cl_ranked, kr_ranked = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, restore_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
    )
    curves = MethodCurves(
        datascope=accs_ds, random_mean=rnd_mean, random_std=rnd_std,
        cleanlab=accs_cl, baseline=baseline, proportions=proportions,
        kairos=accs_kr, hybrid_auto=accs_auto,
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        kairos_ranked=kr_ranked,
    )


def run_nnar_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                        noise_level: float = 0.2,
                        proportions: np.ndarray = DEFAULT_PROPORTIONS,
                        seed: int = 42) -> MethodCurves:
    return run_nnar_experiment_with_artifacts(
        ds, pipeline_factory, noise_level=noise_level, proportions=proportions, seed=seed
    ).curves


def run_mnar_experiment_with_artifacts(
    ds: DatasetInfo,
    pipeline_factory: Callable,
    noise_level: float = 0.2,
    proportions: np.ndarray = DEFAULT_PROPORTIONS,
    seed: int = 42,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> ExperimentArtifacts:
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
    # For MNAR feature corruption, remove the detected rows rather than altering labels.
    remove_fn = action_remove()
    accs_ds, rnd_mean, rnd_std, accs_cl, accs_kr, accs_auto, detected_noise, ds_ranked, cl_ranked, kr_ranked = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, remove_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
    )
    curves = MethodCurves(
        datascope=accs_ds, random_mean=rnd_mean, random_std=rnd_std,
        cleanlab=accs_cl, baseline=baseline, proportions=proportions,
        kairos=accs_kr, hybrid_auto=accs_auto,
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        kairos_ranked=kr_ranked,
    )


def run_mnar_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                        noise_level: float = 0.2,
                        proportions: np.ndarray = DEFAULT_PROPORTIONS,
                        seed: int = 42) -> MethodCurves:
    return run_mnar_experiment_with_artifacts(
        ds, pipeline_factory, noise_level=noise_level, proportions=proportions, seed=seed
    ).curves
