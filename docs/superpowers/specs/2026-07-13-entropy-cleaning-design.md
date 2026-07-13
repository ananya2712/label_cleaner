# Entropy-Based Cleaning Method — Design

**Date:** 2026-07-13
**Status:** Approved

## Background

The F25 thesis experiment (`thesis work - F25/datascope_replication.py`) compared
DataScope, random, and entropy-based cleaning on the Adult dataset under random
label flips. The entropy method was never carried over when the codebase was
split into `label_cleaner`. This change reintroduces it as a first-class,
suite-wide cleaning method so it runs across all datasets, noise types, and
pipelines, and receives accuracy + demographic-parity measurement like every
other method.

## Decisions

- **Integration:** suite-wide method (not a standalone replication script).
- **Probability source:** out-of-fold probabilities via 5-fold
  `cross_val_predict(..., method="predict_proba")` — statistically sounder than
  the original in-sample fit, and consistent with how CleanLab is run here.
- **Scoring:** Shannon entropy `-Σ p·log2(p + 1e-9)` per training row (same
  formula as the original script), ranked descending — most uncertain first.
- **Re-running the full experiment suite (run_v7) is out of scope** for this
  change; implement + test first, run later.

## Components

### 1. `methods/cleaning.py` — `clean_entropy`

Modeled on `clean_cleanlab`:

```
clean_entropy(pipeline_factory, X_train, y_train, X_test, y_test,
              action_fn, proportions, n_jobs=1, protected_test=None)
  -> (accs: List[float], dps: List[float], ranked: np.ndarray)
```

- Unsupervised: ranks **all** training rows; does not see ground-truth noisy
  positions.
- At each proportion, applies the noise-type-specific `action_fn` to the
  top-k% ranked rows and evaluates via `_safe_eval` (accuracy + DP gap).

### 2. Wiring

- `orchestration/experiments.py`: `_run_methods` calls `clean_entropy` and adds
  an `"entropy"` key (`acc`, `dp`, `ranked`); all four
  `run_*_experiment_with_artifacts` pass results into `MethodCurves` /
  artifacts.
- `core/models.py`: new `entropy`, `entropy_dp` fields on `MethodCurves` and
  `entropy_ranked` on the artifacts dataclass — Optional-defaulted so cached
  older results still load.
- `scripts/run_all_experiments.py`: entropy in accuracy plots, DP plots, split
  grids, summary tables, and the cleaned-sample cache.
- `scripts/generate_combined_report.py`: entropy in METHODS, labels, colors,
  styles, accuracy/DP grids, and value tables.
- Style: purple (`#9467bd`), dotted line, label "Entropy".
- Cache compatibility: `from-cache` runs missing entropy render "—" / skip the
  curve rather than crash (same guard pattern as the DP columns).

### 3. Testing

Wiring-invariant tests in the style of `tests/test_dp_pipeline.py`:

- `clean_entropy` returns acc/dp lists matching `len(proportions)` and a
  ranking that is a duplicate-free subset of training indices.
- `_run_methods` result dict carries the `entropy` key with the expected
  shape.
- No assertions on empirical direction (matches existing test philosophy).

## Error handling

- `predict_proba` availability: both suite pipelines (LR, RF) support it; no
  special casing.
- Degenerate probabilities guarded by the `+ 1e-9` epsilon inside the log.
- Evaluation failures already handled by `_safe_eval`.

## Out of scope

- Faithful standalone replication of the original Adult/GBC figure.
- PPT generator updates (can follow once run_v7 exists).
- Running the full suite.
