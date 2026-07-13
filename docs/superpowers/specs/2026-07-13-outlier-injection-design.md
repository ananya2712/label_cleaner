# Scale-Aware Outlier Injection — Design

**Date:** 2026-07-13
**Status:** Approved

## Background

`inject_outlier` currently writes a constant value `100` into a single
designated column for a random fraction of rows. The constant is arbitrary:
depending on the column's scale it may not even be extreme, and in run_v7 the
outlier experiment barely dents baseline accuracy, so cleaning curves show
little separation. Every outlier being the same constant also makes detection
trivially easy.

## Decision

Replace the constant with per-row, scale-aware, high-side extremes:
`mean + k·σ` with `k ~ Uniform(3, 5)` drawn per corrupted row, where mean and
σ are the clean column's `nanmean` / `nanstd`. High-side only, so the existing
repair (cap at `mean + 2σ`) stays coherent.

Scope is injection realism only. Explicitly unchanged:
- single corrupted column (`ds.outlier_col_idx`)
- cleaning action (`action_cap` at 2σ) and the DS-removal alternative curve
- noise level (20%) and all experiment/report wiring

## Components

### 1. `inject_outlier` (`methods/noise.py`)

Signature change: `outlier_value: float = 100` → `k_range: tuple = (3.0, 5.0)`.

- One seeded `RandomState` drives both row selection and the per-row `k`
  draws (reproducible for a given seed).
- Inject `X_noisy[pos, col_idx] = mean_x + k[i] * std_x` per corrupted row.
- Return signature unchanged: `(X_noisy, noisy_positions, cap_value)` with
  `cap_value = mean_x + 2 * std_x` computed from clean data.
- Only call site is `build_noise_bundle_outlier`
  (`orchestration/experiments.py:110`), which never passed `outlier_value` —
  no caller changes needed.

### 2. Tests (`tests/test_outlier_injection.py`, new)

Fast synthetic-data wiring invariants:
- corrupted rows' values all exceed `mean + 3σ` of the clean column
- corrupted values are not all identical
- untouched rows and other columns are bit-identical to the input
- corrupted-row count is `int(noise_level * n)`
- same seed → identical output; input array not modified in place

### 3. Fresh experiment (run_v8, outlier only)

- `scripts/run_all_experiments.py --noise-types outlier` per dataset into
  `artifacts/run_v8/<dataset>_20pct/` (3 datasets × 2 pipelines = 6 configs).
- run_v7 is preserved as the constant-100 record.
- The combined report stays pointed at run_v7 (it needs all four noise types
  from one run dir); comparison of old-vs-new outlier curves uses run_v8's
  own `report.md` and figures.
- If run_v8 is later promoted to the full record, rerun the other noise
  types into it (unaffected by this change) and repoint the combined report.

## Error handling

- Zero-variance column (`std_x == 0`): injected values degenerate to the
  mean and the 2σ cap equals it; benchmark datasets don't hit this, and the
  existing constant-100 code had no guard either — no new guard added.
- NaN in the column is already handled via `nanmean`/`nanstd`.

## Out of scope

- Multi-column or low-side outliers, demographic targeting
- Changing the cleaning/repair actions
- Rerunning non-outlier noise types or repointing the combined report
