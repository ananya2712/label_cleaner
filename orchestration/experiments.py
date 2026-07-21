"""
Experiment service layer.

Each run_* function orchestrates:
  1) fixed split prep
  2) noise injection
  3) DataScope / Random / CleanLab / Entropy cleaning methodologies
"""

from typing import Callable, Dict, Tuple

import numpy as np
from datascope.importance.shapley import ImportanceMethod
from sklearn.metrics import accuracy_score

from ..methods.cleaning import (
    _safe_eval,
    action_cap,
    action_flip,
    action_remove,
    action_restore_labels,
    clean_cleanlab,
    clean_datascope,
    clean_entropy,
    clean_random,
)
from ..data.datasets import DatasetInfo
from ..core.models import ExperimentArtifacts, MethodCurves, NoiseBundle
from ..methods.fairness import demographic_parity_gap
from ..methods.noise import inject_mnar, inject_nnar, inject_outlier, inject_rnd_label
from ..core.prep import prepare_fixed_split


DEFAULT_PROPORTIONS = np.linspace(0, 1, num=21)
# No-noise sensitivity control (thesis Sec 4.1) only sweeps 0-50%: beyond that
# point every method is flipping a majority of already-correct labels, which
# isn't an informative regime for a false-positive-cost measurement.
DEFAULT_NO_NOISE_PROPORTIONS = np.linspace(0, 0.5, num=11)


def _baseline_eval(pipeline_factory: Callable, X_train, y_train, X_test, y_test,
                   protected_test) -> Tuple[float, float]:
    p = pipeline_factory()
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    return accuracy_score(y_test, y_pred), demographic_parity_gap(y_pred, protected_test)


def _random_rankings(noisy_positions: np.ndarray, n_seeds: int = 3):
    rankings = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed + 100)
        perm = noisy_positions.copy()
        rng.shuffle(perm)
        rankings.append(perm)
    return rankings


def _run_methods(pipeline_factory: Callable, X_train_noisy, y_train_noisy, X_test, y_test,
                 noisy_positions, action_fn, proportions, protected_test, protected_train,
                 n_cleanlab_jobs: int = 1,
                 importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
                 mc_iterations: int = 50,
                 X_val=None, y_val=None, protected_val=None,
                 entropy_max_proportion: float = 1.0) -> Dict:
    accs_ds, dps_ds, ds_ranked = clean_datascope(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
        protected_test=protected_test, X_val=X_val, y_val=y_val,
    )
    rnd_acc_mean, rnd_acc_std, rnd_dp_mean, rnd_dp_std = clean_random(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions, protected_test=protected_test,
    )
    accs_cl, dps_cl, cl_ranked = clean_cleanlab(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions, n_jobs=n_cleanlab_jobs, protected_test=protected_test,
    )
    accs_ent, dps_ent, ent_ranked = clean_entropy(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions, n_jobs=n_cleanlab_jobs, protected_test=protected_test,
        max_proportion=entropy_max_proportion,
    )
    return {
        "datascope": {"acc": accs_ds, "dp": dps_ds, "ranked": ds_ranked},
        "cleanlab": {"acc": accs_cl, "dp": dps_cl, "ranked": cl_ranked},
        "entropy": {"acc": accs_ent, "dp": dps_ent, "ranked": ent_ranked},
        "datascope_fair": {"acc": None, "dp": None, "ranked": None},
        "fair_heuristic": {"acc": None, "dp": None, "ranked": None},
        "random": {"acc_mean": rnd_acc_mean, "acc_std": rnd_acc_std,
                   "dp_mean": rnd_dp_mean, "dp_std": rnd_dp_std},
    }


def build_noise_bundle_outlier(split, outlier_col_idx: int, noise_level: float,
                               seed: int = 42) -> NoiseBundle:
    """
    Inject outliers globally over the train+val+test pool, then keep only the
    corrupted rows that fall in the training partition.

    Uses a dense concatenated pool (train rows first) rather than reconstructing
    by original row index, because val_idx may be a capped subsample (see
    prepare_fixed_split's val_cap) and therefore train_idx/val_idx/test_idx no
    longer necessarily partition {0, ..., N-1} densely.

    The mean/σ used for the extreme values and the cap are computed from the
    training partition only (`stats_X=split.X_train`) — the pool is used just
    to select which rows get corrupted, not to source feature statistics, so
    the corruption and cleaning threshold never see test-set data.
    """
    X_pool = np.concatenate([split.X_train, split.X_val, split.X_test], axis=0)
    n_train = len(split.X_train)

    X_noisy_pool, pool_noisy_positions, cap_value = inject_outlier(
        X_pool, outlier_col_idx, noise_level=noise_level, seed=seed,
        stats_X=split.X_train,
    )
    # Rows [0, n_train) of the pool are exactly the training rows, in order.
    noisy_positions = np.sort(pool_noisy_positions[pool_noisy_positions < n_train])
    X_train_noisy = X_noisy_pool[:n_train]
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

    protected_test = ds.protected_group_mask[split.test_idx]
    protected_train = ds.protected_group_mask[split.train_idx]
    protected_val = ds.protected_group_mask[split.val_idx]
    baseline, baseline_dp = _baseline_eval(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test,
        protected_test,
    )
    cap_fn = action_cap(ds.outlier_col_idx, bundle.metadata["cap_value"])
    results = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, cap_fn, proportions, protected_test, protected_train,
        importance_method=importance_method, mc_iterations=mc_iterations,
        X_val=split.X_val, y_val=split.y_val, protected_val=protected_val,
    )
    ds_ranked = results["datascope"]["ranked"]
    cl_ranked = results["cleanlab"]["ranked"]

    curves = MethodCurves(
        datascope=results["datascope"]["acc"],
        random_mean=results["random"]["acc_mean"],
        random_std=results["random"]["acc_std"],
        cleanlab=results["cleanlab"]["acc"],
        baseline=baseline,
        proportions=proportions,
        baseline_dp=baseline_dp,
        datascope_dp=results["datascope"]["dp"],
        cleanlab_dp=results["cleanlab"]["dp"],
        random_dp_mean=results["random"]["dp_mean"],
        random_dp_std=results["random"]["dp_std"],
        datascope_fair=results["datascope_fair"]["acc"],
        datascope_fair_dp=results["datascope_fair"]["dp"],
        fair_heuristic=results["fair_heuristic"]["acc"],
        fair_heuristic_dp=results["fair_heuristic"]["dp"],
        entropy=results["entropy"]["acc"],
        entropy_dp=results["entropy"]["dp"],
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        datascope_fair_ranked=results["datascope_fair"]["ranked"],
        fair_heuristic_ranked=results["fair_heuristic"]["ranked"],
        entropy_ranked=results["entropy"]["ranked"],
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
    protected_test = ds.protected_group_mask[split.test_idx]
    protected_train = ds.protected_group_mask[split.train_idx]
    protected_val = ds.protected_group_mask[split.val_idx]
    baseline, baseline_dp = _baseline_eval(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test,
        protected_test,
    )
    restore_fn = action_restore_labels(split.y_train)
    results = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, restore_fn, proportions, protected_test, protected_train,
        importance_method=importance_method, mc_iterations=mc_iterations,
        X_val=split.X_val, y_val=split.y_val, protected_val=protected_val,
    )
    ds_ranked = results["datascope"]["ranked"]
    cl_ranked = results["cleanlab"]["ranked"]
    curves = MethodCurves(
        datascope=results["datascope"]["acc"],
        random_mean=results["random"]["acc_mean"],
        random_std=results["random"]["acc_std"],
        cleanlab=results["cleanlab"]["acc"],
        baseline=baseline, proportions=proportions,
        baseline_dp=baseline_dp,
        datascope_dp=results["datascope"]["dp"],
        cleanlab_dp=results["cleanlab"]["dp"],
        random_dp_mean=results["random"]["dp_mean"],
        random_dp_std=results["random"]["dp_std"],
        datascope_fair=results["datascope_fair"]["acc"],
        datascope_fair_dp=results["datascope_fair"]["dp"],
        fair_heuristic=results["fair_heuristic"]["acc"],
        fair_heuristic_dp=results["fair_heuristic"]["dp"],
        entropy=results["entropy"]["acc"],
        entropy_dp=results["entropy"]["dp"],
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        datascope_fair_ranked=results["datascope_fair"]["ranked"],
        fair_heuristic_ranked=results["fair_heuristic"]["ranked"],
        entropy_ranked=results["entropy"]["ranked"],
    )


def run_random_label_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                                noise_level: float = 0.2,
                                proportions: np.ndarray = DEFAULT_PROPORTIONS,
                                seed: int = 42) -> MethodCurves:
    return run_random_label_experiment_with_artifacts(
        ds, pipeline_factory, noise_level=noise_level, proportions=proportions, seed=seed
    ).curves


def run_no_noise_control_experiment_with_artifacts(
    ds: DatasetInfo,
    pipeline_factory: Callable,
    proportions: np.ndarray = DEFAULT_NO_NOISE_PROPORTIONS,
    importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
    mc_iterations: int = 50,
) -> ExperimentArtifacts:
    """
    No-noise sensitivity control (thesis Sec 4.1).

    Runs every cleaning method against the fixed split with NO injected
    corruption, then deliberately flips the label of whichever rows each
    method ranks as most harmful, sweeping the flipped proportion 0-50%.
    Since the training data starts genuinely clean, every point here is a
    candidate (there is no ground-truth noisy subset to restrict to) and any
    accuracy drop is purely the false-positive cost of that method's ranking
    — a method that meaningfully distinguishes signal from noise should
    degrade no faster than the random-order baseline on this control.
    """
    split = prepare_fixed_split(ds.X, ds.y)
    candidates = np.arange(len(split.y_train))
    bundle = NoiseBundle(
        X_noisy=split.X_train.copy(),
        y_noisy=split.y_train.copy(),
        noisy_positions=candidates,
        metadata={"control": "no_noise"},
    )
    protected_test = ds.protected_group_mask[split.test_idx]
    protected_train = ds.protected_group_mask[split.train_idx]
    protected_val = ds.protected_group_mask[split.val_idx]
    baseline, baseline_dp = _baseline_eval(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test,
        protected_test,
    )
    flip_fn = action_flip()
    results = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        candidates, flip_fn, proportions, protected_test, protected_train,
        importance_method=importance_method, mc_iterations=mc_iterations,
        X_val=split.X_val, y_val=split.y_val, protected_val=protected_val,
    )
    ds_ranked = results["datascope"]["ranked"]
    cl_ranked = results["cleanlab"]["ranked"]
    curves = MethodCurves(
        datascope=results["datascope"]["acc"],
        random_mean=results["random"]["acc_mean"],
        random_std=results["random"]["acc_std"],
        cleanlab=results["cleanlab"]["acc"],
        baseline=baseline, proportions=proportions,
        baseline_dp=baseline_dp,
        datascope_dp=results["datascope"]["dp"],
        cleanlab_dp=results["cleanlab"]["dp"],
        random_dp_mean=results["random"]["dp_mean"],
        random_dp_std=results["random"]["dp_std"],
        datascope_fair=results["datascope_fair"]["acc"],
        datascope_fair_dp=results["datascope_fair"]["dp"],
        fair_heuristic=results["fair_heuristic"]["acc"],
        fair_heuristic_dp=results["fair_heuristic"]["dp"],
        entropy=results["entropy"]["acc"],
        entropy_dp=results["entropy"]["dp"],
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(candidates),
        datascope_fair_ranked=results["datascope_fair"]["ranked"],
        fair_heuristic_ranked=results["fair_heuristic"]["ranked"],
        entropy_ranked=results["entropy"]["ranked"],
    )


def run_no_noise_control_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                                    proportions: np.ndarray = DEFAULT_NO_NOISE_PROPORTIONS) -> MethodCurves:
    return run_no_noise_control_experiment_with_artifacts(
        ds, pipeline_factory, proportions=proportions
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
    protected_val = ds.protected_group_mask[split.val_idx]
    y_noisy, noisy_positions = inject_nnar(
        split.y_train, protected_train, noise_level=noise_level, seed=seed
    )
    bundle = NoiseBundle(
        X_noisy=split.X_train.copy(),
        y_noisy=y_noisy,
        noisy_positions=noisy_positions,
        metadata={},
    )
    protected_test = ds.protected_group_mask[split.test_idx]
    baseline, baseline_dp = _baseline_eval(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test,
        protected_test,
    )
    restore_fn = action_restore_labels(split.y_train)
    results = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, restore_fn, proportions, protected_test, protected_train,
        importance_method=importance_method, mc_iterations=mc_iterations,
        X_val=split.X_val, y_val=split.y_val, protected_val=protected_val,
    )
    ds_ranked = results["datascope"]["ranked"]
    cl_ranked = results["cleanlab"]["ranked"]
    curves = MethodCurves(
        datascope=results["datascope"]["acc"],
        random_mean=results["random"]["acc_mean"],
        random_std=results["random"]["acc_std"],
        cleanlab=results["cleanlab"]["acc"],
        baseline=baseline, proportions=proportions,
        baseline_dp=baseline_dp,
        datascope_dp=results["datascope"]["dp"],
        cleanlab_dp=results["cleanlab"]["dp"],
        random_dp_mean=results["random"]["dp_mean"],
        random_dp_std=results["random"]["dp_std"],
        datascope_fair=results["datascope_fair"]["acc"],
        datascope_fair_dp=results["datascope_fair"]["dp"],
        fair_heuristic=results["fair_heuristic"]["acc"],
        fair_heuristic_dp=results["fair_heuristic"]["dp"],
        entropy=results["entropy"]["acc"],
        entropy_dp=results["entropy"]["dp"],
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        datascope_fair_ranked=results["datascope_fair"]["ranked"],
        fair_heuristic_ranked=results["fair_heuristic"]["ranked"],
        entropy_ranked=results["entropy"]["ranked"],
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
    protected_val = ds.protected_group_mask[split.val_idx]
    X_noisy, noisy_positions = inject_mnar(
        split.X_train, protected_train, [ds.outlier_col_idx], noise_level=noise_level, seed=seed
    )
    bundle = NoiseBundle(
        X_noisy=X_noisy,
        y_noisy=split.y_train.copy(),
        noisy_positions=noisy_positions,
        metadata={},
    )
    protected_test = ds.protected_group_mask[split.test_idx]
    baseline, baseline_dp = _baseline_eval(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test,
        protected_test,
    )
    # For MNAR feature corruption, remove the detected rows rather than altering labels.
    remove_fn = action_remove()
    results = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, remove_fn, proportions, protected_test, protected_train,
        importance_method=importance_method, mc_iterations=mc_iterations,
        X_val=split.X_val, y_val=split.y_val, protected_val=protected_val,
        # Entropy ranks the ENTIRE training set (unlike DataScope/CleanLab,
        # which only ever touch a much smaller candidate subset), so under
        # MNAR's removal action a proportions[-1]=1.0 point would empty the
        # whole training set and _safe_eval would report NaN. Cap at 95% so
        # that boundary point stays a meaningful (if aggressive) removal.
        entropy_max_proportion=0.95,
    )
    ds_ranked = results["datascope"]["ranked"]
    cl_ranked = results["cleanlab"]["ranked"]
    curves = MethodCurves(
        datascope=results["datascope"]["acc"],
        random_mean=results["random"]["acc_mean"],
        random_std=results["random"]["acc_std"],
        cleanlab=results["cleanlab"]["acc"],
        baseline=baseline, proportions=proportions,
        baseline_dp=baseline_dp,
        datascope_dp=results["datascope"]["dp"],
        cleanlab_dp=results["cleanlab"]["dp"],
        random_dp_mean=results["random"]["dp_mean"],
        random_dp_std=results["random"]["dp_std"],
        datascope_fair=results["datascope_fair"]["acc"],
        datascope_fair_dp=results["datascope_fair"]["dp"],
        fair_heuristic=results["fair_heuristic"]["acc"],
        fair_heuristic_dp=results["fair_heuristic"]["dp"],
        entropy=results["entropy"]["acc"],
        entropy_dp=results["entropy"]["dp"],
    )
    return ExperimentArtifacts(
        curves=curves,
        split=split,
        bundle=bundle,
        datascope_ranked=ds_ranked,
        cleanlab_ranked=cl_ranked,
        random_rankings=_random_rankings(bundle.noisy_positions),
        datascope_fair_ranked=results["datascope_fair"]["ranked"],
        fair_heuristic_ranked=results["fair_heuristic"]["ranked"],
        entropy_ranked=results["entropy"]["ranked"],
    )


def run_mnar_experiment(ds: DatasetInfo, pipeline_factory: Callable,
                        noise_level: float = 0.2,
                        proportions: np.ndarray = DEFAULT_PROPORTIONS,
                        seed: int = 42) -> MethodCurves:
    return run_mnar_experiment_with_artifacts(
        ds, pipeline_factory, noise_level=noise_level, proportions=proportions, seed=seed
    ).curves
