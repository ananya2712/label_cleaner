# Demographic Parity Measurement + Fairness-Aware Cleaning — Design

**Date:** 2026-07-11
**Status:** Approved pending user review

## Goal

Extend the label_cleaner benchmark so that, alongside accuracy:

1. **Demographic parity (DP) is measured** for every cleaning method at every
   cleaning proportion.
2. **A fairness-aware cleaning method** ranks noisy samples purely by their
   Shapley contribution to the DP gap (λ = 1, no accuracy blending).
3. **A heuristic fairness baseline** provides the comparison methodology: a
   model-free group/label bookkeeping ranking, to show whether Shapley's
   model-aware ranking adds value.

## Metric definition

Demographic parity gap, computed on the clean held-out test set:

```
dp_gap = | P(ŷ = 1 | protected) − P(ŷ = 1 | unprotected) |
```

- Protected membership comes from the existing `DatasetInfo.protected_group_mask`
  restricted to test rows: `ds.protected_group_mask[split.test_idx]`.
- Degenerate case: if either group is empty in the rows being scored, return `0.0`.
- Lower is better. `baseline_dp` is the gap at 0% cleaning.

## Components

### 1. `methods/fairness.py` (new module)

- `demographic_parity_gap(y_pred, protected_mask) -> float` — the metric above.
- `SklearnModelDemographicParityDifference(SklearnModelUtility)` — datascope
  utility class mirroring the built-in `SklearnModelEqualizedOddsDifference`:
  overrides `_metric_score`, computes the DP gap of `y_pred`, derives group
  membership from a **precomputed `groupings` array** (the test-set protected
  mask) indexed via the `indices` argument datascope passes when it subsamples.
  This is required because:
  - a plain metric closure breaks under Monte Carlo's internal subsampling
    (verified: IndexError), and
  - a sensitive-feature *column index* is meaningless after p2b's PCA.
  Verified compatible with `ImportanceMethod.NEIGHBOR` (fast KNN-Shapley).

### 2. New cleaners in `methods/cleaning.py`

- `clean_datascope_fair(pipeline_factory, X_train, y_train, X_test, y_test,
  noisy_positions, action_fn, proportions, protected_test, ...)` — identical
  loop structure to `clean_datascope`, but Shapley importances are computed
  under the DP utility. Ranking order: samples whose presence most inflates
  the parity gap are cleaned first. The sign convention must be validated
  empirically during implementation (print mean importance of known-noisy rows
  vs. overall mean, as done for the accuracy-Shapley fix).
- `clean_fair_heuristic(...)` — comparison baseline. Ranks noisy candidates by
  a deterministic, model-free score: for each candidate i, compute the change
  in the training-label selection-rate gap if i were removed from its
  (group, label) cell —
  `score_i = gap(counts) − gap(counts without i)` where
  `gap = |pos_rate(protected) − pos_rate(unprotected)|` over training labels.
  Candidates are cleaned in descending score order (largest gap reduction
  first); ties broken by stable sort. No model fits are used for ranking; the
  same incremental action/eval loop as the other cleaners produces its curves.

### 3. DP measurement for all methods

- `_safe_accuracy` generalizes to `_safe_eval(pipeline_factory, X_tr, y_tr,
  X_te, y_te, protected_test) -> (accuracy, dp_gap)`; NaN-for-degenerate-data
  behavior is preserved for both values.
- All five cleaners (datascope, cleanlab, random, datascope_fair,
  fair_heuristic) and the outlier-only removal loop return a DP curve
  alongside the accuracy curve.

### 4. Data model (`core/models.py`)

`MethodCurves` gains flat optional fields, matching existing style:

- Accuracy curves: `datascope_fair`, `fair_heuristic`
- DP curves: `datascope_dp`, `cleanlab_dp`, `random_dp_mean`, `random_dp_std`,
  `datascope_fair_dp`, `fair_heuristic_dp`, `datascope_removal_dp`
- Scalar: `baseline_dp`

`ExperimentArtifacts` gains `datascope_fair_ranked` and `fair_heuristic_ranked`.

### 5. Orchestration (`orchestration/experiments.py`)

- `_run_methods` takes `protected_test` and runs all five methods; all four
  noise-type runners pass `ds.protected_group_mask[split.test_idx]`.
- Runs for all 4 noise types × 2 pipelines × 3 datasets. NNAR/MNAR are the
  headline cases (noise targets the protected group); outlier/rnd_label show
  what pure-fairness ranking costs when the noise is group-neutral.

### 6. Reporting — DP plotted separately, never clubbed with accuracy

- Existing accuracy figures are unchanged.
- Per experiment: new separate figure `figures/{slug}__dp.png` — DP gap vs.
  cleaning proportion, all methods plus `baseline_dp` reference line.
- Per dataset: new separate grid `figures/{dataset}__all_noise_types_dp.png`.
- `summary.json` caches include all DP curves (existing keys unchanged).
- `report.md` / `results_summary.csv`: final-DP columns added.
- `generate_combined_report.py`: separate DP section — DP summary table +
  per-dataset DP grid images (`{dataset}_grid_dp.png` in combined_report/).

### 7. Validation

- Unit-style check (committed as a small test script or verify step): DP gap
  function on hand-computed cases; utility class returns finite scores under
  both NEIGHBOR and MONTECARLO on synthetic data with a known planted bias.
- End-to-end: titanic NNAR p1a — assert wiring invariants only (finite DP
  curves in [0,1]; DP at 0% cleaning equals baseline DP). Empirical finding
  during implementation: cleaning NNAR noise on titanic INCREASES the DP gap
  (noisy ≈ 0.68 → clean ≈ 0.81) because random protected-group label flips
  pull the group's observed positive rate toward the middle; DP measures
  outcome-rate parity, not correctness. Direction of DP change is reported
  from benchmark runs, not asserted.
- Full benchmark rerun (run_v6) after implementation.

## Error handling

- Empty group in test subset → gap 0.0 (documented in docstring).
- `_safe_eval` returns `(nan, nan)` when filtering leaves no trainable data.
- Non-binary label sets raise `ValueError` in the utility (same guard as the
  EOD utility).

## Out of scope

- Blended accuracy/fairness objectives (λ < 1), fairness-aware stopping rules,
  reweighing/post-processing comparisons, equalized odds — all deliberately
  excluded per user decisions during brainstorming.
- `generate_ppt.py` deck changes (can follow once run_v6 numbers exist).
