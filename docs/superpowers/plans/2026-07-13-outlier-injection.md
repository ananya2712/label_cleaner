# Scale-Aware Outlier Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the constant-100 outlier injection with per-row, scale-aware, high-side extremes (`mean + k·σ`, `k ~ Uniform(3, 5)`) and produce a fresh outlier-only benchmark run (run_v8), per `docs/superpowers/specs/2026-07-13-outlier-injection-design.md`.

**Architecture:** `inject_outlier` in `methods/noise.py` changes internally; its return contract `(X_noisy, noisy_positions, cap_value)` is unchanged and its single caller (`build_noise_bundle_outlier`, `orchestration/experiments.py:110`) needs no edits. A new fast unit test pins the wiring invariants. The fresh experiment reuses `scripts/run_all_experiments.py` unmodified with `--noise-types outlier`.

**Tech Stack:** Python 3, numpy. Tests are plain scripts with `__main__` blocks run via `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/<file>.py` (the package imports as `label_cleaner` from the Desktop parent dir).

## Global Constraints

- Injected value per corrupted row: `mean_x + k * std_x` with `k ~ Uniform(3.0, 5.0)` per row; `mean_x`/`std_x` from the clean column via `np.nanmean`/`np.nanstd`. High-side only.
- Signature change: `outlier_value: float = 100` → `k_range: tuple = (3.0, 5.0)`; all other parameters and the return triple `(X_noisy, noisy_positions, cap_value)` unchanged; `cap_value` stays `mean_x + 2 * std_x`.
- One seeded `np.random.RandomState(seed)` drives both row selection and the `k` draws.
- No new zero-variance guard (matches the old code's stance; benchmark datasets don't hit it).
- Scope: `methods/noise.py` + new test only. No changes to cleaning actions, runners, or report scripts.
- Fresh run goes to `artifacts/run_v8/<dataset>_20pct/`, outlier noise type only; run_v7 and the combined report are left untouched.
- Work on branch `outlier-v2` (created from `main` in Task 1); merge is handled after the final review, not by a task.

---

### Task 1: Rewrite `inject_outlier` with scale-aware extremes

**Files:**
- Modify: `methods/noise.py:19-50` (the `inject_outlier` function)
- Test: `tests/test_outlier_injection.py` (create)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `inject_outlier(X: np.ndarray, col_idx: int, noise_level: float, k_range: tuple = (3.0, 5.0), seed: int = 42) -> (X_noisy, noisy_positions, cap_value)` — same return contract the existing caller `build_noise_bundle_outlier` (`orchestration/experiments.py:110`) already uses positionally/by keyword `noise_level=`; Task 2 runs experiments against this.

- [ ] **Step 1: Create the working branch**

```bash
git checkout -b outlier-v2 main
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_outlier_injection.py`:

```python
"""Unit: inject_outlier wiring invariants on synthetic data. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_outlier_injection.py
(fast: ~1s, synthetic data only)
"""
import numpy as np

from label_cleaner.methods.noise import inject_outlier


def test_inject_outlier_scale_aware():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 3)
    X_orig = X.copy()
    col = 1
    mean_x, std_x = X[:, col].mean(), X[:, col].std()

    X_noisy, pos, cap = inject_outlier(X, col, noise_level=0.2, seed=7)

    # Count and uniqueness of corrupted rows.
    assert len(pos) == int(0.2 * len(X))
    assert len(np.unique(pos)) == len(pos)
    # Every corrupted value is a high-side extreme at least 3 sigma out.
    assert np.all(X_noisy[pos, col] >= mean_x + 3 * std_x - 1e-9)
    # ... and no more than 5 sigma out.
    assert np.all(X_noisy[pos, col] <= mean_x + 5 * std_x + 1e-9)
    # Per-row variation: the outliers are not one constant.
    assert len(np.unique(X_noisy[pos, col])) > 1
    # Untouched rows and other columns are unchanged; input not mutated.
    untouched = np.setdiff1d(np.arange(len(X)), pos)
    assert np.array_equal(X_noisy[np.ix_(untouched, [col])], X_orig[np.ix_(untouched, [col])])
    other_cols = [c for c in range(X.shape[1]) if c != col]
    assert np.array_equal(X_noisy[:, other_cols], X_orig[:, other_cols])
    assert np.array_equal(X, X_orig)
    # Cap is still the clean 2-sigma threshold.
    assert abs(cap - (mean_x + 2 * std_x)) < 1e-9
    # Same seed reproduces identical output.
    X_noisy2, pos2, cap2 = inject_outlier(X, col, noise_level=0.2, seed=7)
    assert np.array_equal(X_noisy, X_noisy2)
    assert np.array_equal(pos, pos2)
    assert cap == cap2


if __name__ == "__main__":
    test_inject_outlier_scale_aware()
    print("test_outlier_injection: OK")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_outlier_injection.py`
Expected: FAIL — the current implementation injects the constant `100`, which violates the `<= mean + 5σ` assertion (and the not-one-constant assertion) on this synthetic data.

- [ ] **Step 4: Rewrite the implementation**

Replace the whole `inject_outlier` function in `methods/noise.py` (lines 19–50) with:

```python
def inject_outlier(X: np.ndarray, col_idx: int, noise_level: float,
                   k_range: tuple = (3.0, 5.0), seed: int = 42):
    """
    Inject scale-aware high-side extremes into `col_idx` for a uniformly
    random fraction of rows — no demographic targeting (MAR).

    Each corrupted row gets its own extreme value `mean + k·σ` with
    k ~ Uniform(k_range), where mean and σ come from the clean column.

    Parameters
    ----------
    X           : feature array (n_samples, n_features) — not modified in place
    col_idx     : column to corrupt
    noise_level : fraction of ALL rows to corrupt
    k_range     : (low, high) bounds for the per-row sigma multiplier
    seed        : random seed

    Returns
    -------
    X_noisy        : corrupted feature array
    noisy_positions: row indices of corrupted samples
    cap_value      : 2-sigma cap computed from clean X (use for cleaning)
    """
    rng      = np.random.RandomState(seed)
    n_noisy  = int(noise_level * len(X))
    noisy_positions = rng.choice(len(X), n_noisy, replace=False)

    mean_x    = np.nanmean(X[:, col_idx])
    std_x     = np.nanstd(X[:, col_idx])
    cap_value = mean_x + 2 * std_x

    k = rng.uniform(k_range[0], k_range[1], size=n_noisy)
    X_noisy = X.copy().astype(float)
    X_noisy[noisy_positions, col_idx] = mean_x + k * std_x

    return X_noisy, noisy_positions, cap_value
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_outlier_injection.py
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py
```
Expected: `test_outlier_injection: OK`, `test_entropy_cleaning: OK`, `test_fairness: OK`. (Skip `tests/test_dp_pipeline.py` here — it exercises NNAR, not outlier, and takes ~2 min; it runs before merge instead.)

- [ ] **Step 6: Commit**

```bash
git add methods/noise.py tests/test_outlier_injection.py
git commit -m "feat: scale-aware per-row outlier injection (mean + k·σ, k~U(3,5))"
```

---

### Task 2: Fresh outlier-only benchmark (run_v8)

**Files:**
- Create: `artifacts/run_v8/<dataset>_20pct/` run outputs (git-ignored; nothing committed by this task)
- No source changes.

**Interfaces:**
- Consumes: the rewritten `inject_outlier` from Task 1 (via `scripts/run_all_experiments.py --noise-types outlier`, no script changes).
- Produces: `artifacts/run_v8/{adult,german,titanic}_20pct/` each containing `caches/<ds>__outlier__{p1a,p2b}/summary.json`, `figures/`, and `report.md` — the fresh-experiment deliverable the spec names.

- [ ] **Step 1: Run the outlier-only benchmark for all three datasets**

```bash
mkdir -p artifacts/run_v8
for ds in adult german titanic; do
  echo "=== $ds start $(date) ===" >> artifacts/run_v8/run_all.log
  PYTHONPATH=/Users/ananyauppal/Desktop python3 scripts/run_all_experiments.py \
    --datasets "$ds" --noise-types outlier \
    --output-dir "artifacts/run_v8/${ds}_20pct" \
    >> artifacts/run_v8/run_all.log 2>&1 || { echo "=== $ds FAILED ===" >> artifacts/run_v8/run_all.log; exit 1; }
  echo "=== $ds done $(date) ===" >> artifacts/run_v8/run_all.log
done
echo "ALL DONE $(date)" >> artifacts/run_v8/run_all.log
```

Expected: exits 0; log shows all three datasets done (adult ~10 min, german/titanic well under a minute each).

- [ ] **Step 2: Verify the run and compare against run_v7**

```bash
python3 - <<'EOF'
import json, glob
new_paths = sorted(glob.glob("artifacts/run_v8/*/caches/*__outlier__*/summary.json"))
assert len(new_paths) == 6, f"expected 6 outlier configs, found {len(new_paths)}"
print(f"{'config':34} {'v7 base':>8} {'v8 base':>8} {'v7 DS':>8} {'v8 DS':>8}")
for p in new_paths:
    c_new = json.load(open(p))["curves"]
    old = p.replace("run_v8", "run_v7")
    c_old = json.load(open(old))["curves"]
    slug = p.split("/")[-2]
    print(f"{slug:34} {c_old['baseline']:8.4f} {c_new['baseline']:8.4f} "
          f"{c_old['datascope'][-1]:8.4f} {c_new['datascope'][-1]:8.4f}")
    for key in ("entropy", "entropy_dp", "datascope_removal"):
        assert c_new.get(key), (slug, key)
print("run_v8 verification: OK")
EOF
```

Expected: prints the six-config comparison table and `run_v8 verification: OK`. Record the table in the task report — it is the old-vs-new evidence the spec asks for (no required direction: whether baselines drop further under the new injection is an empirical finding, not an invariant).

- [ ] **Step 3: Commit the run log reference (docs only if needed)**

Nothing to commit — `artifacts/` is git-ignored. Confirm with:

```bash
git status --short
```

Expected: no modified tracked files (untracked `label_cleaner_review.md` at repo root may appear; leave it alone).
