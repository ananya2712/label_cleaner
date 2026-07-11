# Demographic Parity Measurement + Fairness-Aware Cleaning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the demographic parity (DP) gap for every cleaning method at every cleaning proportion, and add two fairness-aware cleaners — Shapley ranking under a DP utility (`clean_datascope_fair`) and a model-free heuristic baseline (`clean_fair_heuristic`).

**Architecture:** A new `methods/fairness.py` module provides the DP gap metric and a datascope utility class (mirroring the library's built-in `SklearnModelEqualizedOddsDifference`, including the hand-derived `elementwise_score` needed by the fast NEIGHBOR method). `_safe_accuracy` in `methods/cleaning.py` generalizes to `_safe_eval` returning `(accuracy, dp_gap)`; every cleaner returns a DP curve alongside its accuracy curve; `MethodCurves` gains flat optional DP fields; the two report scripts plot DP in **separate figures** (never combined with accuracy panels).

**Tech Stack:** numpy, scikit-learn, datascope (`ShapleyImportance`, `SklearnModelUtility`), cleanlab, matplotlib. No pytest in the environment — tests are plain-python scripts with `assert`, run directly.

**Spec:** `docs/superpowers/specs/2026-07-11-demographic-parity-design.md`

## Global Constraints

- All commands run from `/Users/ananyauppal/Desktop/label_cleaner` with `PYTHONPATH=/Users/ananyauppal/Desktop` (the package is imported as `label_cleaner.*`).
- DP gap definition: `|P(ŷ=1 | protected) − P(ŷ=1 | unprotected)|`, computed on test-set predictions. If either group is empty in the rows scored, the gap is `0.0`.
- Utility sign convention: the datascope utility returns the **negative** gap (higher = fairer), so ascending Shapley importance = most fairness-harmful first — same "ascending = clean first" convention as `clean_datascope`.
- DP is plotted in separate figures only. Existing accuracy figures must not change.
- Tests are plain scripts under `tests/`, run as `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/<name>.py`; success = prints ending in `OK`, failure = AssertionError/traceback.
- Follow existing code style: flat `Optional` dataclass fields, numpydoc-ish docstrings, section-divider comments.

---

### Task 1: DP gap metric (`methods/fairness.py`)

**Files:**
- Create: `methods/fairness.py`
- Modify: `methods/__init__.py`
- Test: `tests/test_fairness.py`

**Interfaces:**
- Produces: `demographic_parity_gap(y_pred: np.ndarray, protected_mask: np.ndarray) -> float` — used by Tasks 2, 3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fairness.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `ModuleNotFoundError: No module named 'label_cleaner.methods.fairness'`

- [ ] **Step 3: Write minimal implementation**

Create `methods/fairness.py`:

```python
"""
Fairness metrics and datascope utilities.

demographic_parity_gap — |P(ŷ=1 | protected) − P(ŷ=1 | unprotected)|
"""

import numpy as np


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `test_fairness: OK`

- [ ] **Step 5: Export from `methods/__init__.py`**

Add after the existing cleaning imports:

```python
from .fairness import demographic_parity_gap
```

and append `"demographic_parity_gap",` to `__all__`.

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 -c "from label_cleaner.methods import demographic_parity_gap; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add methods/fairness.py methods/__init__.py tests/test_fairness.py
git commit -m "feat: add demographic parity gap metric"
```

---

### Task 2: DataScope DP utility class

**Files:**
- Modify: `methods/fairness.py`
- Test: `tests/test_fairness.py` (extend)

**Interfaces:**
- Consumes: `demographic_parity_gap` (Task 1).
- Produces: `SklearnModelDemographicParityDifference(model, groupings: np.ndarray)` — datascope `Utility`; `groupings` is the **test-set** protected mask as an int array (0 = unprotected, 1 = protected). Used by Task 4.

**Background for the implementer:** datascope's NEIGHBOR (KNN-Shapley) method requires the utility to implement `elementwise_score` — a per-test-point decomposition of the metric. The library's `SklearnModelEqualizedOddsDifference` shows the pattern (read it: `python3 -c "import inspect; from datascope.importance.utility import SklearnModelEqualizedOddsDifference as C; print(inspect.getsource(C))"`). For DP the decomposition is: fit the model on the full training set, find which group currently has the higher positive-prediction rate (`g_max`) and lower (`g_min`); predicting class 1 for a `g_max` test point widens the gap (utility −1/n_max), predicting 1 for a `g_min` point narrows it (utility +1/n_min); predicting 0 contributes 0. Monte Carlo instead calls `_metric_score`, which must subset `groupings` by the `indices` argument (datascope subsamples internally).

- [ ] **Step 1: Extend the test file with failing tests**

Append to `tests/test_fairness.py` (before the `__main__` block) and add the new calls to `__main__`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `ImportError: cannot import name 'SklearnModelDemographicParityDifference'`

- [ ] **Step 3: Implement the utility class**

Append to `methods/fairness.py`:

```python
from typing import Hashable, List, Optional, Union

from numpy.typing import NDArray
from pandas import DataFrame, Series
from datascope.importance.utility import MetricCallable, SklearnModelUtility


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
            if len(pos_rows) == 1:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `test_fairness: OK` (the sign-check assertion is the critical one — if it fails, the utility's sign is inverted; do NOT silently flip the ranking order in a later task, fix the utility here).

- [ ] **Step 5: Commit**

```bash
git add methods/fairness.py tests/test_fairness.py
git commit -m "feat: add datascope demographic-parity utility with NEIGHBOR support"
```

---

### Task 3: DP measurement through the existing pipeline

**Files:**
- Modify: `methods/cleaning.py`
- Modify: `core/models.py`
- Modify: `orchestration/experiments.py`
- Test: `tests/test_dp_pipeline.py`

**Interfaces:**
- Consumes: `demographic_parity_gap` (Task 1).
- Produces (Tasks 4–6 rely on these exact signatures):
  - `_safe_eval(pipeline_factory, X_train, y_train, X_test, y_test, protected_test) -> Tuple[float, float]` — `(accuracy, dp_gap)`, both NaN when untrainable.
  - `clean_datascope(...) -> (accs, dps, ranked_noisy)`; `clean_cleanlab(...) -> (accs, dps, cl_ranked)`; `clean_random(...) -> (acc_mean, acc_std, dp_mean, dp_std)` — each gains a keyword-only `protected_test: np.ndarray` parameter (bool mask over test rows).
  - `MethodCurves` new fields (all default `None`): `baseline_dp: Optional[float]`, `datascope_dp`, `cleanlab_dp`, `random_dp_mean`, `random_dp_std`, `datascope_removal_dp` (each `Optional[List[float]]`).
  - `_run_methods(...)` gains `protected_test` parameter and returns the dict `{"datascope": {"acc", "dp", "ranked"}, "cleanlab": {"acc", "dp", "ranked"}, "random": {"acc_mean", "acc_std", "dp_mean", "dp_std"}}`.
  - `_baseline_eval(pipeline_factory, X_train, y_train, X_test, y_test, protected_test) -> Tuple[float, float]` replaces `_baseline_acc`.

- [ ] **Step 1: Replace `_safe_accuracy` with `_safe_eval` in `methods/cleaning.py`**

Replace the whole `_safe_accuracy` function with:

```python
def _safe_eval(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    protected_test: np.ndarray,
) -> Tuple[float, float]:
    """Fit and score one cleaned dataset.

    Returns (accuracy, dp_gap) on the test set; (NaN, NaN) when aggressive
    filtering leaves no trainable dataset."""
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return float("nan"), float("nan")
    pipeline = pipeline_factory()
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        return (
            accuracy_score(y_test, y_pred),
            demographic_parity_gap(y_pred, protected_test),
        )
    except ValueError:
        return float("nan"), float("nan")
```

Add the import near the top with the other package imports:

```python
from .fairness import demographic_parity_gap
```

- [ ] **Step 2: Thread DP through the three existing cleaners**

In `clean_datascope`, add keyword parameter `protected_test: np.ndarray = None` after `mc_iterations`, change the loop and return to:

```python
    accs, dps = [], []
    for p in proportions:
        n_clean   = int(p * len(ranked_noisy))
        X_c, y_c  = action_fn(X_train, y_train, ranked_noisy[:n_clean])
        acc, dp = _safe_eval(pipeline_factory, X_c, y_c, X_test, y_test, protected_test)
        accs.append(acc)
        dps.append(dp)

    return accs, dps, ranked_noisy
```

Update the docstring Returns section accordingly. Apply the identical loop change to `clean_cleanlab` (returns `accs, dps, cl_ranked`).

In `clean_random`, collect both metrics per seed and return four lists:

```python
    all_accs, all_dps = [], []
    for seed in range(n_seeds):
        rng  = np.random.RandomState(seed + 100)
        perm = noisy_positions.copy()
        rng.shuffle(perm)

        seed_accs, seed_dps = [], []
        for p in proportions:
            n_clean  = int(p * len(perm))
            X_c, y_c = action_fn(X_train, y_train, perm[:n_clean])
            acc, dp = _safe_eval(pipeline_factory, X_c, y_c, X_test, y_test, protected_test)
            seed_accs.append(acc)
            seed_dps.append(dp)
        all_accs.append(seed_accs)
        all_dps.append(seed_dps)

    acc_arr, dp_arr = np.array(all_accs), np.array(all_dps)
    return (acc_arr.mean(axis=0).tolist(), acc_arr.std(axis=0).tolist(),
            dp_arr.mean(axis=0).tolist(), dp_arr.std(axis=0).tolist())
```

- [ ] **Step 3: Add DP fields to `core/models.py`**

In `MethodCurves`, after `datascope_removal`, add:

```python
    baseline_dp: Optional[float] = None
    datascope_dp: Optional[List[float]] = None
    cleanlab_dp: Optional[List[float]] = None
    random_dp_mean: Optional[List[float]] = None
    random_dp_std: Optional[List[float]] = None
    datascope_removal_dp: Optional[List[float]] = None
```

- [ ] **Step 4: Update `orchestration/experiments.py`**

Replace `_baseline_acc` with:

```python
def _baseline_eval(pipeline_factory: Callable, X_train, y_train, X_test, y_test,
                   protected_test) -> Tuple[float, float]:
    p = pipeline_factory()
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    return accuracy_score(y_test, y_pred), demographic_parity_gap(y_pred, protected_test)
```

(add `from typing import Callable, Tuple` and `from ..methods.fairness import demographic_parity_gap` to imports).

Replace `_run_methods` with:

```python
def _run_methods(pipeline_factory: Callable, X_train_noisy, y_train_noisy, X_test, y_test,
                 noisy_positions, action_fn, proportions, protected_test,
                 n_cleanlab_jobs: int = 1,
                 importance_method: ImportanceMethod = ImportanceMethod.NEIGHBOR,
                 mc_iterations: int = 50) -> Dict:
    accs_ds, dps_ds, ds_ranked = clean_datascope(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions,
        importance_method=importance_method, mc_iterations=mc_iterations,
        protected_test=protected_test,
    )
    rnd_acc_mean, rnd_acc_std, rnd_dp_mean, rnd_dp_std = clean_random(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions, protected_test=protected_test,
    )
    accs_cl, dps_cl, cl_ranked = clean_cleanlab(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions, n_jobs=n_cleanlab_jobs, protected_test=protected_test,
    )
    return {
        "datascope": {"acc": accs_ds, "dp": dps_ds, "ranked": ds_ranked},
        "cleanlab": {"acc": accs_cl, "dp": dps_cl, "ranked": cl_ranked},
        "random": {"acc_mean": rnd_acc_mean, "acc_std": rnd_acc_std,
                   "dp_mean": rnd_dp_mean, "dp_std": rnd_dp_std},
    }
```

(add `Dict` to the `typing` import).

In **each of the four runners**, compute the mask and rewire. Pattern for `run_outlier_experiment_with_artifacts` (the other three are identical except they have no removal loop):

```python
    protected_test = ds.protected_group_mask[split.test_idx]
    baseline, baseline_dp = _baseline_eval(
        pipeline_factory, bundle.X_noisy, bundle.y_noisy, split.X_test, split.y_test,
        protected_test,
    )
    cap_fn = action_cap(ds.outlier_col_idx, bundle.metadata["cap_value"])
    results = _run_methods(
        pipeline_factory,
        bundle.X_noisy, bundle.y_noisy,
        split.X_test, split.y_test,
        bundle.noisy_positions, cap_fn, proportions, protected_test,
        importance_method=importance_method, mc_iterations=mc_iterations,
    )
    ds_ranked = results["datascope"]["ranked"]
    cl_ranked = results["cleanlab"]["ranked"]

    # Outlier-specific removal curve (drop top-k DataScope-ranked noisy rows)
    remove_fn = action_remove()
    accs_rm, dps_rm = [], []
    for p in proportions:
        n_rm = int(p * len(ds_ranked))
        X_c, y_c = remove_fn(bundle.X_noisy, bundle.y_noisy, ds_ranked[:n_rm])
        acc, dp = _safe_eval(pipeline_factory, X_c, y_c, split.X_test, split.y_test,
                             protected_test)
        accs_rm.append(acc)
        dps_rm.append(dp)

    curves = MethodCurves(
        datascope=results["datascope"]["acc"],
        random_mean=results["random"]["acc_mean"],
        random_std=results["random"]["acc_std"],
        cleanlab=results["cleanlab"]["acc"],
        baseline=baseline,
        proportions=proportions,
        datascope_removal=accs_rm,
        baseline_dp=baseline_dp,
        datascope_dp=results["datascope"]["dp"],
        cleanlab_dp=results["cleanlab"]["dp"],
        random_dp_mean=results["random"]["dp_mean"],
        random_dp_std=results["random"]["dp_std"],
        datascope_removal_dp=dps_rm,
    )
```

The removal loop's direct `pipe.fit(...)` block is replaced by `_safe_eval` as shown — import `_safe_eval` in `orchestration/experiments.py` via `from ..methods.cleaning import _safe_eval` (added to the existing import list; also remove the now-unused `accuracy_score` usage if nothing else uses it — `_baseline_eval` still does, so keep the import).

The three label-noise runners (`rnd_label`, `nnar`, `mnar`) get the same `protected_test` / `_baseline_eval` / `results` rewiring and pass `baseline_dp`, `datascope_dp`, `cleanlab_dp`, `random_dp_mean`, `random_dp_std` to `MethodCurves` (no removal fields).

- [ ] **Step 5: Write the integration test**

Create `tests/test_dp_pipeline.py`:

```python
"""Integration: DP curves flow through a real (small) experiment. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py
(takes ~1-2 minutes: titanic NNAR p1a with 3 proportions)
"""
from pathlib import Path

import numpy as np

from label_cleaner.data.datasets import load_dataset
from label_cleaner.orchestration.catalog import pipeline_factory_a
from label_cleaner.orchestration.experiments import run_nnar_experiment_with_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nnar_titanic_dp_curves():
    ds = load_dataset("titanic", REPO_ROOT / "datasets")
    factory = pipeline_factory_a(ds.num_col_indices, ds.cat_col_indices)
    artifacts = run_nnar_experiment_with_artifacts(
        ds, factory, noise_level=0.2, proportions=np.array([0.0, 0.5, 1.0])
    )
    c = artifacts.curves
    for name, curve in [("datascope_dp", c.datascope_dp), ("cleanlab_dp", c.cleanlab_dp),
                        ("random_dp_mean", c.random_dp_mean)]:
        assert curve is not None and len(curve) == 3, name
        assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in curve), (name, curve)
    assert c.baseline_dp is not None and 0.0 <= c.baseline_dp <= 1.0
    # At 0% cleaned, every method's DP equals the baseline DP.
    assert abs(c.datascope_dp[0] - c.baseline_dp) < 1e-9
    # NNAR noise targets the protected group: cleaning 100% of it must not
    # leave the model less fair than the noisy baseline (allow tiny jitter).
    assert c.datascope_dp[-1] <= c.baseline_dp + 0.02, (c.datascope_dp, c.baseline_dp)


if __name__ == "__main__":
    test_nnar_titanic_dp_curves()
    print("test_dp_pipeline: OK")
```

- [ ] **Step 6: Run both test scripts**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py && PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py`
Expected: `test_fairness: OK` then `test_dp_pipeline: OK`

- [ ] **Step 7: Commit**

```bash
git add methods/cleaning.py core/models.py orchestration/experiments.py tests/test_dp_pipeline.py
git commit -m "feat: measure demographic parity for all existing cleaning methods"
```

---

### Task 4: `clean_datascope_fair` (Shapley ranking under the DP utility)

**Files:**
- Modify: `methods/cleaning.py`
- Modify: `methods/__init__.py`
- Modify: `core/models.py`
- Modify: `orchestration/experiments.py`
- Test: `tests/test_fairness.py` (extend)

**Interfaces:**
- Consumes: `SklearnModelDemographicParityDifference` (Task 2), `_safe_eval` (Task 3).
- Produces:
  - `clean_datascope_fair(pipeline_factory, X_train, y_train, X_test, y_test, noisy_positions, action_fn, proportions, protected_test) -> (accs, dps, ranked_noisy)`
  - `MethodCurves.datascope_fair`, `MethodCurves.datascope_fair_dp` (both `Optional[List[float]] = None`)
  - `ExperimentArtifacts.datascope_fair_ranked: np.ndarray = None`
  - `_run_methods` result dict gains key `"datascope_fair": {"acc", "dp", "ranked"}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fairness.py` (and register in `__main__`):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `ImportError: cannot import name 'clean_datascope_fair'`

- [ ] **Step 3: Implement the cleaner**

In `methods/cleaning.py`, add after `clean_datascope` (new section divider `# DataScope-Fair (demographic parity Shapley)`):

```python
def clean_datascope_fair(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noisy_positions: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    protected_test: np.ndarray,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Fairness-aware DataScope cleaning (λ=1, pure demographic parity).

    Ranks noisy samples by their Shapley contribution to the demographic
    parity gap (via SklearnModelDemographicParityDifference, which returns
    the negative gap so higher importance = fairer). Ascending order = most
    fairness-harmful first — the same convention as accuracy DataScope.
    Uses ImportanceMethod.NEIGHBOR (the utility implements elementwise_score).

    Returns
    -------
    accs   : accuracy at each proportion
    dps    : demographic parity gap at each proportion
    ranked : noisy_positions sorted most fairness-harmful first
    """
    pipeline = pipeline_factory()
    pipeline.fit(X_train, y_train)

    utility = SklearnModelDemographicParityDifference(
        pipeline[-1], groupings=np.asarray(protected_test, dtype=int)
    )
    imp = ShapleyImportance(
        method=ImportanceMethod.NEIGHBOR,
        pipeline=pipeline[:-1],
        utility=utility,
    )
    importances  = imp.fit(X_train, y_train).score(X_test, y_test)
    sorted_order = np.argsort(importances[noisy_positions])  # ascending: most gap-inflating first
    ranked_noisy = noisy_positions[sorted_order]

    accs, dps = [], []
    for p in proportions:
        n_clean   = int(p * len(ranked_noisy))
        X_c, y_c  = action_fn(X_train, y_train, ranked_noisy[:n_clean])
        acc, dp = _safe_eval(pipeline_factory, X_c, y_c, X_test, y_test, protected_test)
        accs.append(acc)
        dps.append(dp)

    return accs, dps, ranked_noisy
```

Add to the imports at the top of `methods/cleaning.py`:

```python
from .fairness import SklearnModelDemographicParityDifference, demographic_parity_gap
```

(replacing the Task 3 import line). Export `clean_datascope_fair` from `methods/__init__.py` (import + `__all__`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `test_fairness: OK`

- [ ] **Step 5: Wire into models and orchestration**

`core/models.py` — `MethodCurves` gains (after `datascope_removal_dp`):

```python
    datascope_fair: Optional[List[float]] = None
    datascope_fair_dp: Optional[List[float]] = None
```

`ExperimentArtifacts` gains (after `random_rankings`):

```python
    datascope_fair_ranked: np.ndarray = None
```

`orchestration/experiments.py` — in `_run_methods`, after the `clean_cleanlab` call:

```python
    accs_dsf, dps_dsf, dsf_ranked = clean_datascope_fair(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions, protected_test,
    )
```

and add to the returned dict: `"datascope_fair": {"acc": accs_dsf, "dp": dps_dsf, "ranked": dsf_ranked},`. Import `clean_datascope_fair` in the module's import block. In all four runners add to `MethodCurves(...)`: `datascope_fair=results["datascope_fair"]["acc"], datascope_fair_dp=results["datascope_fair"]["dp"],` and to `ExperimentArtifacts(...)`: `datascope_fair_ranked=results["datascope_fair"]["ranked"],`.

- [ ] **Step 6: Run the integration test**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py`
Expected: `test_dp_pipeline: OK` (existing assertions still pass; the new method runs inside `_run_methods`).

- [ ] **Step 7: Commit**

```bash
git add methods/cleaning.py methods/__init__.py core/models.py orchestration/experiments.py tests/test_fairness.py
git commit -m "feat: add fairness-aware DataScope cleaning (demographic parity Shapley)"
```

---

### Task 5: `clean_fair_heuristic` (model-free DP-impact baseline)

**Files:**
- Modify: `methods/cleaning.py`
- Modify: `methods/__init__.py`
- Modify: `core/models.py`
- Modify: `orchestration/experiments.py`
- Test: `tests/test_fairness.py` (extend)

**Interfaces:**
- Consumes: `_safe_eval` (Task 3).
- Produces:
  - `clean_fair_heuristic(pipeline_factory, X_train, y_train, X_test, y_test, noisy_positions, action_fn, proportions, protected_test, protected_train) -> (accs, dps, ranked_noisy)`
  - `MethodCurves.fair_heuristic`, `MethodCurves.fair_heuristic_dp`; `ExperimentArtifacts.fair_heuristic_ranked`.
  - `_run_methods` gains parameter `protected_train` and result key `"fair_heuristic": {"acc", "dp", "ranked"}`; all four runners pass `ds.protected_group_mask[split.train_idx]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fairness.py` (and register in `__main__`):

```python
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
```

(Note the index mapping: `scores` is aligned with `candidates`, so `scores[1]` is candidate 2 and `scores[2]` is candidate 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `ImportError: cannot import name '_dp_heuristic_scores'`

- [ ] **Step 3: Implement scorer and cleaner**

In `methods/cleaning.py`, add a new section `# Fair heuristic (model-free DP-impact ranking)`:

```python
def _dp_heuristic_scores(y_train: np.ndarray, protected_train: np.ndarray,
                         candidates: np.ndarray) -> np.ndarray:
    """
    Model-free DP-impact score per candidate: the reduction in the
    training-label selection-rate gap if that sample alone were removed
    from its (group, label) cell. Positive = removal shrinks the gap.
    Each candidate is scored independently against the ORIGINAL counts.
    """
    prot = np.asarray(protected_train, dtype=bool)
    n_p, n_u = int(prot.sum()), int((~prot).sum())
    pos_p = int(y_train[prot].sum())
    pos_u = int(y_train[~prot].sum())
    rate_p = pos_p / n_p if n_p else 0.0
    rate_u = pos_u / n_u if n_u else 0.0
    base_gap = abs(rate_p - rate_u)

    scores = np.empty(len(candidates), dtype=float)
    for k, i in enumerate(candidates):
        if prot[i]:
            n_, pos_ = n_p - 1, pos_p - int(y_train[i])
            gap = abs((pos_ / n_ if n_ else 0.0) - rate_u)
        else:
            n_, pos_ = n_u - 1, pos_u - int(y_train[i])
            gap = abs(rate_p - (pos_ / n_ if n_ else 0.0))
        scores[k] = base_gap - gap
    return scores


def clean_fair_heuristic(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noisy_positions: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    protected_test: np.ndarray,
    protected_train: np.ndarray,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Heuristic fairness baseline: rank noisy candidates by the model-free
    DP-impact score (largest training-label gap reduction first, stable
    sort for ties), then run the standard incremental cleaning loop.
    No model fits are used for ranking.

    Returns (accs, dps, ranked) — same shape as clean_datascope_fair.
    """
    scores = _dp_heuristic_scores(y_train, protected_train, noisy_positions)
    order = np.argsort(-scores, kind="stable")
    ranked_noisy = noisy_positions[order]

    accs, dps = [], []
    for p in proportions:
        n_clean   = int(p * len(ranked_noisy))
        X_c, y_c  = action_fn(X_train, y_train, ranked_noisy[:n_clean])
        acc, dp = _safe_eval(pipeline_factory, X_c, y_c, X_test, y_test, protected_test)
        accs.append(acc)
        dps.append(dp)

    return accs, dps, ranked_noisy
```

Export `clean_fair_heuristic` from `methods/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py`
Expected: `test_fairness: OK`

- [ ] **Step 5: Wire into models and orchestration**

`core/models.py`: `MethodCurves` gains `fair_heuristic: Optional[List[float]] = None` and `fair_heuristic_dp: Optional[List[float]] = None`; `ExperimentArtifacts` gains `fair_heuristic_ranked: np.ndarray = None`.

`orchestration/experiments.py`: `_run_methods` signature gains `protected_train` (positional, right after `protected_test`); body gains:

```python
    accs_fh, dps_fh, fh_ranked = clean_fair_heuristic(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        noisy_positions, action_fn, proportions, protected_test, protected_train,
    )
```

with result key `"fair_heuristic": {"acc": accs_fh, "dp": dps_fh, "ranked": fh_ranked},`. Each runner computes `protected_train = ds.protected_group_mask[split.train_idx]`, passes it to `_run_methods`, and adds `fair_heuristic=...`, `fair_heuristic_dp=...` to `MethodCurves` and `fair_heuristic_ranked=...` to `ExperimentArtifacts` (same pattern as Task 4 Step 5).

- [ ] **Step 6: Run the integration test**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py`
Expected: `test_dp_pipeline: OK`

- [ ] **Step 7: Commit**

```bash
git add methods/cleaning.py methods/__init__.py core/models.py orchestration/experiments.py tests/test_fairness.py
git commit -m "feat: add model-free DP-impact heuristic cleaning baseline"
```

---

### Task 6: DP reporting in `scripts/run_all_experiments.py`

**Files:**
- Modify: `scripts/run_all_experiments.py`

**Interfaces:**
- Consumes: the `MethodCurves` DP fields (Tasks 3–5). `asdict(artifacts.curves)` already serializes them into `summary.json` — no cache-writing change needed.
- Produces: `figures/{slug}__dp.png` per experiment, `figures/{dataset}__all_noise_types_dp.png` per dataset; DP columns in `report.md` and `results_summary.csv` rows; `_curves_from_cache` restores DP fields.

- [ ] **Step 1: Extend `_curves_from_cache`**

Add to the `MethodCurves(...)` construction (after `datascope_removal=...`):

```python
        baseline_dp=c.get("baseline_dp"),
        datascope_dp=c.get("datascope_dp"),
        cleanlab_dp=c.get("cleanlab_dp"),
        random_dp_mean=c.get("random_dp_mean"),
        random_dp_std=c.get("random_dp_std"),
        datascope_removal_dp=c.get("datascope_removal_dp"),
        datascope_fair=c.get("datascope_fair"),
        datascope_fair_dp=c.get("datascope_fair_dp"),
        fair_heuristic=c.get("fair_heuristic"),
        fair_heuristic_dp=c.get("fair_heuristic_dp"),
```

- [ ] **Step 2: Add `_plot_dp_curves` (separate figure — do NOT touch `_plot_curves`)**

Add after `_plot_curves`:

```python
def _plot_dp_curves(path: Path, dataset: str, noise_type: str, pipeline_key: str, curves) -> None:
    proportions_pct = np.array(curves.proportions) * 100.0
    dp_rnd_mean = np.array(curves.random_dp_mean)
    dp_rnd_std = np.array(curves.random_dp_std)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(proportions_pct, curves.datascope_dp,
            color="#1f77b4", linestyle="-",  linewidth=1.8, label="DataScope")
    ax.plot(proportions_pct, curves.cleanlab_dp,
            color="#d62728", linestyle="--", linewidth=1.8, label="CleanLab")
    ax.plot(proportions_pct, dp_rnd_mean,
            color="#ff7f0e", linestyle="--", linewidth=1.4, label="Random")
    ax.fill_between(proportions_pct, dp_rnd_mean - dp_rnd_std, dp_rnd_mean + dp_rnd_std,
                    color="#ff7f0e", alpha=0.25, label="±1σ Random")
    ax.plot(proportions_pct, curves.datascope_fair_dp,
            color="#9467bd", linestyle="-", linewidth=2.0, label="DataScope-Fair")
    ax.plot(proportions_pct, curves.fair_heuristic_dp,
            color="#8c564b", linestyle="-.", linewidth=1.8, label="Fair heuristic")
    if curves.datascope_removal_dp is not None:
        ax.plot(proportions_pct, curves.datascope_removal_dp,
                color="#2ca02c", linestyle="-", linewidth=1.4, label="DS removal")
    ax.axhline(curves.baseline_dp, color="gray", linestyle=":", linewidth=1.0,
               label=f"Baseline DP={curves.baseline_dp:.3f}")
    ax.set_xlabel("% of training set cleaned", fontsize=10)
    ax.set_ylabel("Demographic parity gap (lower = fairer)", fontsize=10)
    ax.set_title(f"{dataset} | {noise_type} | {pipeline_key} — demographic parity", fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Add `_plot_dp_grid` (separate figure — do NOT touch `_plot_grid`)**

Add after `_plot_grid`, mirroring its structure with DP series. The per-cell body plots `curves.datascope_dp`, `curves.cleanlab_dp`, `curves.random_dp_mean` (±1σ fill), `curves.datascope_fair_dp`, `curves.fair_heuristic_dp`, `curves.datascope_removal_dp` (if not None) and an `axhline` at `curves.baseline_dp`, with the same colors/styles as Step 2, y-label `f"{pipeline_key}\nDP gap"`, final-value legend labels (`f"DataScope: {curves.datascope_dp[-1]:.3f}"` etc.), and suptitle:

```python
    fig.suptitle(
        f"Demographic Parity — {dataset.upper()} (noise_level={noise_pct}%)\n"
        "Blue=DataScope, Red dashed=CleanLab, Orange dashed=Random (±1σ shaded), "
        "Purple=DataScope-Fair, Brown dash-dot=Fair heuristic",
        fontsize=10, y=1.01,
    )
```

Signature: `def _plot_dp_grid(path, dataset, noise_level, noise_types, pipelines, curves_grid, proportions) -> None` — identical to `_plot_grid`.

- [ ] **Step 4: Call the new plotters and extend rows**

In the main experiment loop, after the `_plot_curves(figure_path, ...)` call add:

```python
                dp_figure_path = figures_dir / f"{slug}__dp.png"
                _plot_dp_curves(dp_figure_path, dataset, noise_type, pipeline_key, artifacts.curves)
```

After the `_plot_grid(...)` call (dataset grid) add:

```python
        _plot_dp_grid(
            figures_dir / f"{dataset}__all_noise_types_dp.png",
            dataset, args.noise_level, noise_types_run, pipelines_run,
            dataset_curves, proportions,
        )
```

(match the actual local variable names used at the existing `_plot_grid` call site — read them before editing).

In `summary_rows.append({...})` add:

```python
                        "baseline_dp": round(float(artifacts.curves.baseline_dp), 4),
                        "datascope_dp_final": round(float(artifacts.curves.datascope_dp[final_idx]), 4),
                        "datascope_fair_final": round(float(artifacts.curves.datascope_fair[final_idx]), 4),
                        "datascope_fair_dp_final": round(float(artifacts.curves.datascope_fair_dp[final_idx]), 4),
                        "fair_heuristic_dp_final": round(float(artifacts.curves.fair_heuristic_dp[final_idx]), 4),
                        "dp_figure": f"figures/{slug}__dp.png",
```

In `report_sections` (per-experiment block) add after the accuracy lines:

```python
                        f"![{slug} dp](figures/{slug}__dp.png)",
                        "",
                        f"- Baseline DP gap: `{artifacts.curves.baseline_dp:.4f}`",
                        f"- Final DataScope-Fair DP gap: `{artifacts.curves.datascope_fair_dp[-1]:.4f}`",
                        f"- Final Fair-heuristic DP gap: `{artifacts.curves.fair_heuristic_dp[-1]:.4f}`",
                        f"- Final DataScope DP gap: `{artifacts.curves.datascope_dp[-1]:.4f}`",
```

(insert into the existing list literal; keep accuracy items unchanged).

- [ ] **Step 5: Smoke-run one experiment**

Run:
```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 scripts/run_all_experiments.py \
  --datasets titanic --noise-types nnar --pipelines p1a --noise-level 0.2 \
  --output-dir /private/tmp/claude-501/-Users-ananyauppal-Desktop-label-cleaner/479175c6-f937-4a8a-a0d3-88dc497e17f6/scratchpad/dp_smoke
```
Expected: exits 0; the output dir contains `figures/titanic__nnar__p1a.png`, `figures/titanic__nnar__p1a__dp.png`, `figures/titanic__all_noise_types_dp.png`, and `caches/titanic__nnar__p1a/summary.json` whose `curves` object contains non-null `datascope_fair_dp`. Verify with:
```bash
python3 -c "
import json; d = json.load(open('<output-dir>/caches/titanic__nnar__p1a/summary.json'))
c = d['curves']; assert c['datascope_fair_dp'] and c['baseline_dp'] is not None; print('cache ok')"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_all_experiments.py
git commit -m "feat: separate demographic-parity figures and report columns in experiment runner"
```

---

### Task 7: DP section in `scripts/generate_combined_report.py`

**Files:**
- Modify: `scripts/generate_combined_report.py`

**Interfaces:**
- Consumes: DP keys in cached `summary.json` `curves` dicts (Task 6). Note this script loads curves as **plain dicts** (`c["datascope"]`, `c.get("kairos")` style), not `MethodCurves`.
- Produces: `combined_report/{dataset}_grid_dp.png` per dataset and a "Demographic Parity" section in `combined_report.md` with a DP summary table.

- [ ] **Step 1: Add a DP grid plotter**

Add `_plot_dataset_grid_dp(dataset, noise_level, run_dir, all_curves, output_path)` next to `_plot_dataset_grid`, mirroring its structure but plotting `c["datascope_dp"]`, `c["cleanlab_dp"]`, `c["random_dp_mean"]` (±1σ via `c["random_dp_std"]`), `c["datascope_fair_dp"]`, `c["fair_heuristic_dp"]`, `c.get("datascope_removal_dp")` (if present), and `axhline(c["baseline_dp"])`; colors: DataScope `#1f77b4` solid, CleanLab `#d62728` dashed, Random `#ff7f0e` dashed, DataScope-Fair `#9467bd` solid, Fair heuristic `#8c564b` dash-dot, DS removal `#2ca02c` solid; y-label `DP gap`; suptitle `f"Demographic Parity — {dataset.upper()} (noise_level={noise_pct}%)"`. Skip cells where `c.get("datascope_dp")` is missing (old caches) via `ax.set_visible(False)`.

- [ ] **Step 2: Generate the DP grids and report section in `main()`**

After the existing per-dataset grid loop add:

```python
    dataset_dp_grid_figs = {}
    for run in RUNS:
        ds        = run["dataset"]
        fig_path  = OUTPUT_DIR / f"{ds}_grid_dp.png"
        _plot_dataset_grid_dp(ds, run["noise_level"], run["run_dir"], all_curves, fig_path)
        dataset_dp_grid_figs[ds] = fig_path
        print(f"Saved: {fig_path}")
```

After the accuracy summary-table section (`sections += [...]` for "## Accuracy Summary Table"), add a DP section:

```python
    dp_rows = []
    for run in RUNS:
        ds = run["dataset"]
        for nt in NOISE_TYPES:
            for pipeline in PIPELINES:
                c = all_curves.get((ds, nt, pipeline))
                if c is None or not c.get("datascope_dp"):
                    continue
                dp_rows.append({
                    "dataset":       ds,
                    "noise":         nt,
                    "pipeline":      pipeline,
                    "baseline_dp":   f"{c['baseline_dp']:.4f}",
                    "DataScope":     f"{c['datascope_dp'][-1]:.4f}",
                    "CleanLab":      f"{c['cleanlab_dp'][-1]:.4f}",
                    "DS-Fair":       f"{c['datascope_fair_dp'][-1]:.4f}",
                    "Fair-heuristic": f"{c['fair_heuristic_dp'][-1]:.4f}",
                    "Random":        f"{c['random_dp_mean'][-1]:.4f}",
                })

    if dp_rows:
        sections += [
            "## Demographic Parity Summary Table",
            "",
            "Final DP gap at 100% cleaning (lower = fairer). `baseline_dp` is the gap with no cleaning.",
            "",
            _md_table(dp_rows,
                      ["dataset", "noise", "pipeline", "baseline_dp",
                       "DataScope", "CleanLab", "DS-Fair", "Fair-heuristic", "Random"]),
            "",
            "### DP curves per dataset",
            "",
        ]
        for run in RUNS:
            ds = run["dataset"]
            sections += [f"![{ds} dp grid]({ds}_grid_dp.png)", ""]
        sections += ["---", ""]
```

- [ ] **Step 3: Smoke-run against the Task 6 smoke cache**

Point `RUNS` cannot be edited for a smoke test; instead run the real script only after a full rerun (Task 8). For now verify syntax and dict handling:

```bash
python3 -m py_compile scripts/generate_combined_report.py && echo compiles
```
Expected: `compiles`

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_combined_report.py
git commit -m "feat: demographic parity section in combined report"
```

---

### Task 8: Full benchmark rerun (run_v6) + end-to-end validation

**Files:**
- Modify: `scripts/generate_combined_report.py` (RUNS → run_v6 paths)

- [ ] **Step 1: Run both test scripts one final time**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py && PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py`
Expected: both `OK`.

- [ ] **Step 2: Update `RUNS` in `generate_combined_report.py`**

Change the three `run_dir` values from `run_v5` to `run_v6` (same `*_20pct` names, noise levels stay 0.20) and `OUTPUT_DIR` to `... / "run_v6" / "combined_report"`.

- [ ] **Step 3: Launch the full rerun in the background**

```bash
mkdir -p artifacts/run_v6/logs && PYTHONPATH=/Users/ananyauppal/Desktop nohup sh -c '
python3 scripts/run_all_experiments.py --datasets adult   --noise-level 0.2 --output-dir artifacts/run_v6/adult_20pct   > artifacts/run_v6/logs/adult.log   2>&1
python3 scripts/run_all_experiments.py --datasets german  --noise-level 0.2 --output-dir artifacts/run_v6/german_20pct  > artifacts/run_v6/logs/german.log  2>&1
python3 scripts/run_all_experiments.py --datasets titanic --noise-level 0.2 --output-dir artifacts/run_v6/titanic_20pct > artifacts/run_v6/logs/titanic.log 2>&1
'
```

(run in background; runtime is roughly the run_v5 duration plus the two new methods — DataScope-Fair adds one NEIGHBOR Shapley per experiment, the heuristic is negligible).

- [ ] **Step 4: Generate the combined report and validate spec expectations**

```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 scripts/generate_combined_report.py
```

Then check the spec's end-to-end expectations on titanic NNAR p1a:

```bash
python3 -c "
import json
c = json.load(open('artifacts/run_v6/titanic_20pct/caches/titanic__nnar__p1a/summary.json'))['curves']
import math
for k in ('datascope_fair_dp', 'fair_heuristic_dp', 'datascope_dp'):
    assert c[k] and all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in c[k]), k
print('baseline_dp:', c['baseline_dp'])
print('datascope_dp final:', c['datascope_dp'][-1])
print('datascope_fair_dp final:', c['datascope_fair_dp'][-1])
print('fair_heuristic_dp final:', c['fair_heuristic_dp'][-1])
print('wiring invariants: ok — report these numbers to the user')"
```

Expected: wiring invariants pass; the printed DP numbers are reported to the user as empirical results (direction of DP change is a finding, not an invariant).

- [ ] **Step 5: Commit and report results**

```bash
git add scripts/generate_combined_report.py
git commit -m "chore: point combined report at run_v6 (demographic parity results)"
```

Summarize for the user: where the DP figures/tables live, and the headline comparison (DS-Fair vs heuristic vs accuracy-driven methods on NNAR/MNAR).
