# Reduced Outlier Figures (outlier_focus) — Design

**Date:** 2026-07-13
**Status:** Approved

## Background

The run_v8 outlier figures plot six series (DataScope, CleanLab, Entropy,
Random ±1σ, baseline, DS removal). For the thesis write-up a reduced view is
wanted: only the DataScope cap-repair curve, the DataScope removal curve, and
the noisy baseline. No new computation — all values come from existing run_v8
caches. The cap repair is labeled honestly as **2σ** capping (that is what
`action_cap` does), per user confirmation.

## Components

### `scripts/generate_outlier_focus.py` (new, standalone)

Reads `artifacts/run_v8/<dataset>_20pct/caches/<dataset>__outlier__<pipeline>/summary.json`
for datasets `adult, german, titanic` × pipelines `p1a, p2b` (6 configs),
following `generate_combined_report.py`'s read-from-cache pattern (same
`REPO_ROOT`/`sys.path`/matplotlib-Agg preamble).

Series per axes — exactly three:
- **DataScope (2σ capping)** — `curves["datascope"]`, color `#1f77b4`, solid
- **DataScope (removal)** — `curves["datascope_removal"]`, color `#2ca02c`, solid
- **Baseline** — `curves["baseline"]` as a gray dashed `axhline`

X axis: `proportions × 100` ("% of training set cleaned"); y axis: accuracy.
Y-limits from finite values only, padded, clamped to [0, 1].

### Outputs — `artifacts/run_v8/outlier_focus/` (new folder)

- `<dataset>__outlier__<pipeline>.png` — one per config (6)
- `outlier_focus_grid.png` — one 2×3 grid (rows = pipelines, cols = datasets)
- `README.md` — table of final values per config: baseline,
  DataScope (2σ capping) final, DataScope (removal) final

run_v8's existing figures and caches are untouched.

## Error handling

- Missing cache file or missing/None `datascope_removal`: skip that config
  with a printed message; the grid hides that axes.
- Only finite values enter y-limit computation.

## Verification (no permanent test file)

Run the script; confirm exit 0, the 7 PNGs and README exist, and each
single-config figure's legend has exactly 3 entries (checked programmatically
in the verification step). Report generators in this repo carry no unit
tests; this follows that convention.

## Out of scope

- DP variants of these figures
- Any new cleaning computation (e.g., a real 1σ-cap curve)
- Changes to existing run/report scripts
