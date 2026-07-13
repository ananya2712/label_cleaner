# Entropy-Based Cleaning Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `clean_entropy` (out-of-fold prediction-entropy ranking) as a suite-wide cleaning method alongside DataScope/CleanLab/Random, per `docs/superpowers/specs/2026-07-13-entropy-cleaning-design.md`.

**Architecture:** `clean_entropy` mirrors `clean_cleanlab`: unsupervised ranking of ALL training rows (descending Shannon entropy of 5-fold out-of-fold `predict_proba`), incremental `action_fn` application, `_safe_eval` scoring. Results flow through `_run_methods` → `MethodCurves`/`ExperimentArtifacts` → run script figures/tables/caches → combined report.

**Tech Stack:** Python 3, numpy, scikit-learn (`cross_val_predict`), matplotlib. Tests are plain scripts with `__main__` blocks run via `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/<file>.py` (repo convention — the package is imported as `label_cleaner` from the Desktop parent dir).

## Global Constraints

- Entropy formula exactly as the original F25 script: `-np.sum(p * np.log2(p + 1e-9), axis=1)`.
- Probabilities from `cross_val_predict(pipeline, X, y, cv=5, method="predict_proba", n_jobs=n_jobs)` — same call shape as `clean_cleanlab`.
- All new `MethodCurves`/`ExperimentArtifacts` fields are Optional-defaulted (`None`) so cached pre-entropy results still load; every consumer of entropy curves guards for `None` / missing key.
- Plot style for entropy everywhere: color `#17becf` (teal), linestyle `":"` (dotted), label `"Entropy"`. (Teal, not the spec's purple `#9467bd` — purple is already taken by DataScope-Fair in the DP figures; Task 1 updates the spec line.)
- No assertions on empirical direction in tests — wiring invariants only (existing test philosophy).
- Entropy ranks ALL rows, so with the MNAR remove-action at proportion 1.0 the training set empties and `_safe_eval` returns NaN. That is expected; y-limit computations must filter non-finite values.

---

### Task 1: `clean_entropy` in `methods/cleaning.py`

**Files:**
- Modify: `methods/cleaning.py` (new function after `clean_cleanlab`, which ends ~line 323; also module docstring lines 14–19)
- Modify: `docs/superpowers/specs/2026-07-13-entropy-cleaning-design.md` (color line)
- Test: `tests/test_entropy_cleaning.py` (create)

**Interfaces:**
- Consumes: existing `_safe_eval(pipeline_factory, X_train, y_train, X_test, y_test, protected_test) -> Tuple[float, float]`, `action_restore_labels(y_clean)` from the same module, `cross_val_predict` (already imported at `methods/cleaning.py:28`).
- Produces: `clean_entropy(pipeline_factory, X_train, y_train, X_test, y_test, action_fn, proportions, n_jobs=1, protected_test=None) -> Tuple[List[float], List[float], np.ndarray]` — `(accs, dps, ent_ranked)` where `ent_ranked` is a permutation of `np.arange(len(X_train))`, most-suspicious first. Task 2 calls this from `_run_methods`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_entropy_cleaning.py`:

```python
"""Unit: clean_entropy wiring invariants on synthetic data. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py
(fast: ~seconds, synthetic data only)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from label_cleaner.methods.cleaning import action_restore_labels, clean_entropy


def _factory():
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=200))])


def test_clean_entropy_wiring():
    rng = np.random.RandomState(0)
    n = 100
    X = rng.randn(n, 4)
    y_clean = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_noisy = y_clean.copy()
    flip = rng.choice(n, size=20, replace=False)
    y_noisy[flip] = 1 - y_noisy[flip]

    X_test = rng.randn(40, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    protected_test = X_test[:, 2] > 0

    proportions = np.array([0.0, 0.5, 1.0])
    accs, dps, ranked = clean_entropy(
        _factory, X, y_noisy, X_test, y_test,
        action_restore_labels(y_clean), proportions,
        protected_test=protected_test,
    )

    assert len(accs) == 3 and len(dps) == 3
    assert all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in accs), accs
    assert all(np.isfinite(d) and 0.0 <= d <= 1.0 for d in dps), dps
    # Unsupervised: the ranking covers every training row exactly once.
    assert sorted(int(i) for i in ranked) == list(range(n))


if __name__ == "__main__":
    test_clean_entropy_wiring()
    print("test_entropy_cleaning: OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py`
Expected: FAIL with `ImportError: cannot import name 'clean_entropy'`

- [ ] **Step 3: Write the implementation**

In `methods/cleaning.py`, insert after `clean_cleanlab` (after its closing `return accs, dps, cl_ranked` and before the `clean_cleanlab_adaptive — DISABLED` comment block):

```python
# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def clean_entropy(
    pipeline_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    action_fn: Callable,
    proportions: np.ndarray,
    n_jobs: int = 1,
    protected_test: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Entropy-based cleaning: rank ALL training samples by the Shannon entropy
    of their out-of-fold predicted class probabilities (most uncertain first),
    then incrementally apply `action_fn` to the top-k%.

    Reintroduces the entropy method from the original thesis experiment
    (F25 datascope_replication.py), with out-of-fold probabilities from
    5-fold cross_val_predict instead of the original in-sample fit.

    Like CleanLab, this method is unsupervised — it does not know the
    ground-truth noisy positions.

    Parameters
    ----------
    pipeline_factory : callable() → fresh sklearn Pipeline
    X_train, y_train : training data (with injected noise)
    X_test,  y_test  : clean held-out test set
    action_fn        : callable(X_tr, y_tr, positions) → (X_tr_clean, y_tr_clean)
    proportions      : array of fractions in [0, 1]
    n_jobs           : parallelism for cross_val_predict (default 1 for safety)
    protected_test   : bool mask over test rows for demographic parity measurement

    Returns
    -------
    accs       : accuracy at each proportion
    dps        : demographic parity gap at each proportion
    ent_ranked : all training indices ranked most-to-least uncertain
    """
    pipeline   = pipeline_factory()
    pred_probs = cross_val_predict(
        pipeline, X_train, y_train,
        cv=5, method="predict_proba", n_jobs=n_jobs,
    )
    entropy    = -np.sum(pred_probs * np.log2(pred_probs + 1e-9), axis=1)
    ent_ranked = np.argsort(entropy)[::-1]

    accs, dps = [], []
    for p in proportions:
        n_clean   = int(p * len(ent_ranked))
        X_c, y_c  = action_fn(X_train, y_train, ent_ranked[:n_clean])
        acc, dp = _safe_eval(pipeline_factory, X_c, y_c, X_test, y_test, protected_test)
        accs.append(acc)
        dps.append(dp)

    return accs, dps, ent_ranked
```

Also update the module docstring cleaner list (lines 14–19) — add one line after the `clean_cleanlab` entry:

```
clean_entropy    — rank ALL rows by out-of-fold prediction entropy, most-uncertain first
```

And in the spec `docs/superpowers/specs/2026-07-13-entropy-cleaning-design.md`, replace the line

```
- Style: purple (`#9467bd`), dotted line, label "Entropy".
```

with

```
- Style: teal (`#17becf`), dotted line, label "Entropy" (purple was taken by
  DataScope-Fair in the DP figures).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py`
Expected: `test_entropy_cleaning: OK`

- [ ] **Step 5: Commit**

```bash
git add methods/cleaning.py tests/test_entropy_cleaning.py docs/superpowers/specs/2026-07-13-entropy-cleaning-design.md
git commit -m "feat: add entropy-based cleaning method (OOF prediction entropy)"
```

---

### Task 2: Wire entropy through models and experiment runners

**Files:**
- Modify: `core/models.py` (`MethodCurves` ~line 20–39, `ExperimentArtifacts` ~line 56–66)
- Modify: `orchestration/experiments.py` (imports ~line 1–35, `_run_methods` lines 55–89, and all four `run_*_experiment_with_artifacts`: outlier ~118–185, rnd_label ~195–253, nnar ~265–327, mnar ~337–402)
- Test: `tests/test_dp_pipeline.py` (extend)

**Interfaces:**
- Consumes: `clean_entropy` from Task 1 (exact signature above).
- Produces: `_run_methods` result dict gains key `"entropy": {"acc": List[float], "dp": List[float], "ranked": np.ndarray}`. `MethodCurves` gains fields `entropy: Optional[List[float]] = None`, `entropy_dp: Optional[List[float]] = None`. `ExperimentArtifacts` gains `entropy_ranked: np.ndarray = None`. Tasks 3–4 consume these names verbatim (including as JSON keys via `asdict`).

- [ ] **Step 1: Extend the integration test (failing first)**

In `tests/test_dp_pipeline.py`, inside `test_nnar_titanic_dp_curves`, change the DP-curve loop list to include entropy and add ranking assertions before the `baseline_dp` assertion:

```python
    for name, curve in [("datascope_dp", c.datascope_dp), ("cleanlab_dp", c.cleanlab_dp),
                        ("random_dp_mean", c.random_dp_mean), ("entropy_dp", c.entropy_dp)]:
        assert curve is not None and len(curve) == 3, name
        assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in curve), (name, curve)
    assert c.entropy is not None and len(c.entropy) == 3
    ranked = artifacts.entropy_ranked
    assert ranked is not None and len(ranked) == len(artifacts.split.y_train)
    assert len(np.unique(ranked)) == len(ranked)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py`
Expected: FAIL with `AttributeError: 'MethodCurves' object has no attribute 'entropy_dp'` (after the ~1–2 min experiment completes).

- [ ] **Step 3: Add fields to `core/models.py`**

Append to `MethodCurves` (after `fair_heuristic_dp: Optional[List[float]] = None`):

```python
    entropy: Optional[List[float]] = None
    entropy_dp: Optional[List[float]] = None
```

Append to `ExperimentArtifacts` (after `fair_heuristic_ranked: np.ndarray = None`):

```python
    entropy_ranked: np.ndarray = None
```

- [ ] **Step 4: Call `clean_entropy` in `_run_methods`**

In `orchestration/experiments.py`, add `clean_entropy` to the existing import block at the top of the file — alphabetically after `clean_datascope_fair`:

```python
from ..methods.cleaning import (
    _safe_eval,
    action_cap,
    action_remove,
    action_restore_labels,
    clean_cleanlab,
    clean_datascope,
    clean_datascope_fair,
    clean_entropy,
    clean_fair_heuristic,
    clean_random,
)
```

Also update the module docstring's step 3 line to `3) DataScope / Random / CleanLab / Entropy cleaning methodologies`.

In `_run_methods`, after the `clean_cleanlab` call (lines 70–73), insert:

```python
    accs_ent, dps_ent, ent_ranked = clean_entropy(
        pipeline_factory, X_train_noisy, y_train_noisy, X_test, y_test,
        action_fn, proportions, n_jobs=n_cleanlab_jobs, protected_test=protected_test,
    )
```

And add to the returned dict (after the `"cleanlab"` entry):

```python
        "entropy": {"acc": accs_ent, "dp": dps_ent, "ranked": ent_ranked},
```

- [ ] **Step 5: Pass entropy results through all four runners**

In each of `run_outlier_experiment_with_artifacts`, `run_random_label_experiment_with_artifacts`, `run_nnar_experiment_with_artifacts`, `run_mnar_experiment_with_artifacts` (they share the same pattern), add to the `MethodCurves(...)` construction (after `fair_heuristic_dp=...`):

```python
        entropy=results["entropy"]["acc"],
        entropy_dp=results["entropy"]["dp"],
```

and to the `ExperimentArtifacts(...)` construction (after `fair_heuristic_ranked=...`):

```python
        entropy_ranked=results["entropy"]["ranked"],
```

All four call sites, no exceptions.

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py
```
Expected: all three print their OK lines (dp_pipeline takes ~1–2 min).

- [ ] **Step 7: Commit**

```bash
git add core/models.py orchestration/experiments.py tests/test_dp_pipeline.py
git commit -m "feat: wire entropy cleaning through experiment runners and curves"
```

---

### Task 3: Entropy in `scripts/run_all_experiments.py`

**Files:**
- Modify: `scripts/run_all_experiments.py` (`_cleaned_cache` ~169–195, `_curves_from_cache` ~224–245, `_plot_curves` ~248–277, `_plot_dp_curves` ~296–326, `_plot_grid` ~329–397, `_plot_dp_grid` ~400–475, `main` summary payload/rows/report ~546–652 and Run Summary columns ~675–699)

**Interfaces:**
- Consumes: `MethodCurves.entropy` / `.entropy_dp` and `ExperimentArtifacts.entropy_ranked` from Task 2 (serialized to `summary.json` automatically via the existing `asdict(artifacts.curves)`).
- Produces: `summary.json` curves now contain `entropy` / `entropy_dp` keys; per-slug cache contains `entropy_ranked_*` and `cleaned_cache["entropy"]`; Task 4 reads the `entropy` / `entropy_dp` JSON keys.

All entropy consumption in this file must tolerate `None` (old caches loaded with `--from-cache`). Fresh-run paths (summary payload, cleaned cache, report bullets) may assume entropy is present since Task 2 always produces it.

- [ ] **Step 1: `_curves_from_cache` — load entropy fields**

Add after `fair_heuristic_dp=c.get("fair_heuristic_dp"),`:

```python
        entropy=c.get("entropy"),
        entropy_dp=c.get("entropy_dp"),
```

- [ ] **Step 2: `_cleaned_cache` — record entropy-cleaned rows**

Change the initial dict to:

```python
    cache = {
        "datascope": {},
        "cleanlab": {},
        "entropy": {},
        "random": {},
    }
```

Inside the loop, after the `cl_positions` line add:

```python
        ent_positions = _cleaned_prefix(artifacts.entropy_ranked, proportion)
```

and after the `cache["cleanlab"][key] = {...}` block add:

```python
        cache["entropy"][key] = {
            "train_positions": [int(x) for x in ent_positions],
            "dataset_indices": _to_dataset_indices(artifacts.split.train_idx, ent_positions),
        }
```

- [ ] **Step 3: `_plot_curves` and `_plot_dp_curves` — entropy line**

In `_plot_curves`, after the CleanLab `ax.plot` (line ~256–257) add:

```python
    if curves.entropy is not None:
        ax.plot(proportions_pct, curves.entropy,
                color="#17becf", linestyle=":", linewidth=1.8, label="Entropy")
```

In `_plot_dp_curves`, after the CleanLab DP `ax.plot` (line ~304–305) add:

```python
    if curves.entropy_dp is not None:
        ax.plot(proportions_pct, curves.entropy_dp,
                color="#17becf", linestyle=":", linewidth=1.8, label="Entropy")
```

Do NOT add entropy to `_has_dp_curves` — old caches without entropy must still produce DP figures.

- [ ] **Step 4: `_plot_grid` — entropy line, y-limits, legend**

These grid legends map `ax.get_lines()[:n_legend]` onto `final_labels` positionally, so the plot-call order and label order must stay in sync. In the per-cell block:

After the CleanLab `ax.plot(x, curves.cleanlab, ...)` line add:

```python
            if curves.entropy is not None:
                ax.plot(x, curves.entropy, color="#17becf", linestyle=":", linewidth=1.8, label="Entropy")
```

After the `all_y = [...]` line (which currently includes `curves.datascope`, `curves.cleanlab`, `rnd_mean`, `curves.baseline`) add:

```python
            if curves.entropy is not None:
                all_y.extend(v for v in curves.entropy if np.isfinite(v))
```

(The finite filter matters: with the MNAR remove-action, entropy at 100% removes every row and yields NaN, which would poison `min`/`max`.)

Replace the `final_labels = [...]` literal with:

```python
            final_labels = [
                f"DataScope: {curves.datascope[-1]:.3f}",
                f"CleanLab: {curves.cleanlab[-1]:.3f}",
            ]
            if curves.entropy is not None:
                final_labels.append(f"Entropy: {curves.entropy[-1]:.3f}")
            final_labels += [
                f"Random: {rnd_mean[-1]:.3f}",
                f"Baseline: {curves.baseline:.3f}",
            ]
```

(the existing `if curves.datascope_removal is not None: final_labels.append(...)` stays after this).

- [ ] **Step 5: `_plot_dp_grid` — same treatment for DP**

After the CleanLab DP `ax.plot(x, curves.cleanlab_dp, ...)` line add:

```python
            if curves.entropy_dp is not None:
                ax.plot(x, curves.entropy_dp, color="#17becf", linestyle=":", linewidth=1.8, label="Entropy")
```

After the `all_y = [...]` line add:

```python
            if curves.entropy_dp is not None:
                all_y.extend(v for v in curves.entropy_dp if np.isfinite(v))
```

Replace the `final_labels = [...]` literal with:

```python
            final_labels = [
                f"DataScope: {curves.datascope_dp[-1]:.3f}",
                f"CleanLab: {curves.cleanlab_dp[-1]:.3f}",
            ]
            if curves.entropy_dp is not None:
                final_labels.append(f"Entropy: {curves.entropy_dp[-1]:.3f}")
            final_labels += [
                f"Random: {dp_rnd_mean[-1]:.3f}",
                f"DataScope-Fair: {curves.datascope_fair_dp[-1]:.3f}",
                f"Fair heuristic: {curves.fair_heuristic_dp[-1]:.3f}",
                f"Baseline: {curves.baseline_dp:.3f}",
            ]
```

(the existing DS-removal append stays after this).

- [ ] **Step 6: `main` — summary payload, summary rows, report bullets**

In `summary_payload`, after the `cleanlab_ranked_dataset_indices` entry add:

```python
                    "entropy_ranked_train_positions": [
                        int(x) for x in artifacts.entropy_ranked.tolist()
                    ],
                    "entropy_ranked_dataset_indices": _to_dataset_indices(
                        artifacts.split.train_idx, artifacts.entropy_ranked
                    ),
```

In `summary_rows.append({...})`, after `"cleanlab_final": ...` add:

```python
                        "entropy_final": round(float(artifacts.curves.entropy[final_idx]), 4),
```

In the Run Summary `_markdown_table` column list, add `"entropy_final"` immediately after `"cleanlab_final"`.

In the per-experiment `report_sections.extend([...])`, after the `Final CleanLab accuracy` bullet add:

```python
                        f"- Final Entropy accuracy: `{artifacts.curves.entropy[-1]:.4f}`",
```

- [ ] **Step 7: Verify against old caches (no-crash regression)**

Run the from-cache path against the existing run_v6 output into a scratch directory (copy first so nothing in `artifacts/` is touched):

```bash
SCRATCH=/private/tmp/claude-501/-Users-ananyauppal-Desktop-label-cleaner/50777e64-0132-42ee-ac4e-5bcd813b745f/scratchpad/entropy_task3
mkdir -p "$SCRATCH" && cp -r artifacts/run_v6/titanic_20pct "$SCRATCH/out"
PYTHONPATH=/Users/ananyauppal/Desktop python3 scripts/run_all_experiments.py \
  --datasets titanic --output-dir "$SCRATCH/out" --from-cache
```

Expected: exits 0, regenerates figures without entropy curves (old caches have no `entropy` key), no traceback.

- [ ] **Step 8: Verify a fresh mini-run produces entropy output**

```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 scripts/run_all_experiments.py \
  --datasets titanic --noise-types nnar mnar --pipelines p1a \
  --proportions 0.0 0.5 1.0 --output-dir "$SCRATCH/fresh"
python3 - <<'EOF'
import json
c = json.load(open("/private/tmp/claude-501/-Users-ananyauppal-Desktop-label-cleaner/50777e64-0132-42ee-ac4e-5bcd813b745f/scratchpad/entropy_task3/fresh/caches/titanic__nnar__p1a/summary.json"))
assert c["curves"]["entropy"] is not None and len(c["curves"]["entropy"]) == 3
assert c["curves"]["entropy_dp"] is not None
assert "entropy" in c["cleaned_cache"]
print("fresh-run entropy cache: OK")
EOF
```

Expected: run exits 0 (mnar included deliberately to exercise the NaN-at-100% path), assertion script prints `fresh-run entropy cache: OK`, and `report.md` in `$SCRATCH/fresh` shows the `entropy_final` column and `Final Entropy accuracy` bullets.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_all_experiments.py
git commit -m "feat: entropy curves in run figures, summary tables, and caches"
```

---

### Task 4: Entropy in `scripts/generate_combined_report.py`

**Files:**
- Modify: `scripts/generate_combined_report.py` (config ~43–58, `_plot_dataset_grid` ~147–218, `_plot_dataset_grid_dp` ~241–316, accuracy tables ~389–418 and ~474–495, DP table ~420–448)

**Interfaces:**
- Consumes: `entropy` / `entropy_dp` keys in cached `summary.json` curves (Task 3). Old run_v6 caches lack these keys — every use must go through `c.get(...)` / `_final(...)` with a `"—"` fallback.
- Produces: combined report markdown + grid figures with an Entropy series.

- [ ] **Step 1: Config — method registry**

Change the four config dicts:

```python
METHODS     = ["datascope", "cleanlab", "entropy", "random_mean"]
METHOD_LABELS = {
    "datascope":   "DataScope",
    "cleanlab":    "CleanLab",
    "entropy":     "Entropy",
    "random_mean": "Random",
}
COLORS = {
    "datascope":   "#1f77b4",
    "cleanlab":    "#d62728",
    "entropy":     "#17becf",
    "random_mean": "#ff7f0e",
}
STYLES = {
    "datascope":   "-",
    "cleanlab":    "--",
    "entropy":     ":",
    "random_mean": "--",
}
```

This automatically adds an Entropy column to the comparison heatmap (`method_list` derives from `METHODS`; missing keys become NaN cells, already handled) and to `_best_method`.

- [ ] **Step 2: `_plot_dataset_grid` — entropy line**

After the CleanLab `ax.plot` line (~170) add:

```python
            if c.get("entropy"):
                ax.plot(props, c["entropy"], color=COLORS["entropy"], linestyle=STYLES["entropy"], linewidth=1.8, label="Entropy")
```

After the `all_y = [...]` line add:

```python
            if c.get("entropy"):
                all_y.extend(v for v in c["entropy"] if np.isfinite(v))
```

Replace the `final_labels = [...]` literal with:

```python
            final_labels = [
                f"DataScope: {c['datascope'][-1]:.3f}",
                f"CleanLab: {c['cleanlab'][-1]:.3f}",
            ]
            if c.get("entropy"):
                final_labels.append(f"Entropy: {c['entropy'][-1]:.3f}")
            final_labels += [
                f"Random: {rnd_mean[-1]:.3f}",
                f"Baseline: {c['baseline']:.3f}",
            ]
```

(the existing `datascope_removal` append stays after this; the legend maps `ax.get_lines()` positionally, which this ordering preserves).

- [ ] **Step 3: `_plot_dataset_grid_dp` — entropy DP line**

After the CleanLab DP `ax.plot` line (~264) add:

```python
            if c.get("entropy_dp"):
                ax.plot(props, c["entropy_dp"], color="#17becf", linestyle=":", linewidth=1.8, label="Entropy")
```

After the `all_y = [...]` line add:

```python
            if c.get("entropy_dp"):
                all_y.extend(v for v in c["entropy_dp"] if np.isfinite(v))
```

Replace the `final_labels = [...]` literal with:

```python
            final_labels = [
                f"DataScope: {c['datascope_dp'][-1]:.3f}",
                f"CleanLab: {c['cleanlab_dp'][-1]:.3f}",
            ]
            if c.get("entropy_dp"):
                final_labels.append(f"Entropy: {c['entropy_dp'][-1]:.3f}")
            final_labels += [
                f"Random: {rnd_mean[-1]:.3f}",
                f"DS-Fair: {c['datascope_fair_dp'][-1]:.3f}",
                f"Fair heuristic: {c['fair_heuristic_dp'][-1]:.3f}",
            ]
```

(the existing `datascope_removal_dp` append and trailing `Baseline` append stay after this). Do NOT add entropy keys to `_has_dp`.

- [ ] **Step 4: Tables**

Global accuracy summary rows (~397–407): after the `"CleanLab"` entry add

```python
                    "Entropy":   f"{_final(c, 'entropy'):.4f}"    if _final(c, 'entropy')    else "—",
```

and add `"Entropy"` to the `_md_table` column list after `"CleanLab"` (~413–414).

Per-noise-type accuracy rows (~479–486): same new entry after `"CleanLab"`, and `"Entropy"` added to that `_md_table` column list (~492–493).

DP summary rows (~428–438): after the `"CleanLab"` entry add

```python
                    "Entropy":       f"{c['entropy_dp'][-1]:.4f}" if c.get("entropy_dp") else "—",
```

and add `"Entropy"` to the DP `_md_table` column list after `"CleanLab"` (~447–448).

Methods-compared prose list (~372–376): after the CleanLab bullet add

```python
        "- **Entropy** — out-of-fold prediction-entropy ranking (most uncertain first), correction action",
```

- [ ] **Step 5: Verify against run_v6 (entropy-less caches)**

```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 scripts/generate_combined_report.py
```

Expected: exits 0, prints the saved-figure and report lines. `artifacts/run_v6/combined_report/combined_report.md` shows Entropy columns filled with `—` and the heatmap gains an Entropy column of empty (NaN) cells. Skim the report to confirm nothing else changed. (This regenerates the run_v6 combined report in place; the inputs are unchanged so only the new empty Entropy columns differ. If the artifacts are git-tracked and the diff is unwanted, restore with `git checkout -- artifacts/run_v6/combined_report`.)

- [ ] **Step 6: Full test sweep and commit**

```bash
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_entropy_cleaning.py
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_fairness.py
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py
git add scripts/generate_combined_report.py
git commit -m "feat: entropy method in combined report figures and tables"
```

Expected: all three tests print OK, commit succeeds.
