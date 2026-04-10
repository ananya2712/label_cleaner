#!/usr/bin/env python3
"""
Generate a combined markdown report from multiple per-dataset experiment runs.

Usage:
    python3 scripts/generate_combined_report.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

cache_root = REPO_ROOT / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────

RUNS = [
    {"dataset": "adult",   "noise_level": 0.10, "run_dir": REPO_ROOT / "artifacts" / "adult_10pct"},
    {"dataset": "german",  "noise_level": 0.20, "run_dir": REPO_ROOT / "artifacts" / "german_20pct"},
    {"dataset": "titanic", "noise_level": 0.20, "run_dir": REPO_ROOT / "artifacts" / "titanic_20pct"},
]

NOISE_TYPES = ["outlier", "rnd_label", "nnar", "mnar"]
PIPELINES   = ["p1a", "p2b"]
METHODS     = ["datascope", "cleanlab", "cleanlab_adaptive", "kairos", "random_mean"]
METHOD_LABELS = {
    "datascope":         "DataScope",
    "cleanlab":          "CleanLab",
    "cleanlab_adaptive": "CL-Adaptive",
    "kairos":            "Kairos",
    "random_mean":       "Random",
}
COLORS = {
    "datascope":         "#1f77b4",
    "cleanlab":          "#d62728",
    "cleanlab_adaptive": "#8c1717",
    "kairos":            "#9467bd",
    "random_mean":       "#ff7f0e",
}
STYLES = {
    "datascope":         "-",
    "cleanlab":          "--",
    "cleanlab_adaptive": "-.",
    "kairos":            "-",
    "random_mean":       "--",
}

OUTPUT_DIR = REPO_ROOT / "artifacts" / "combined_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_curves(run_dir: Path, dataset: str, noise_type: str, pipeline: str) -> dict | None:
    path = run_dir / "caches" / f"{dataset}__{noise_type}__{pipeline}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["curves"]


def _final(curves: dict, method: str) -> float | None:
    v = curves.get(method)
    if v is None:
        return None
    return float(v[-1]) if isinstance(v, list) else float(v)


def _md_table(rows: list[dict], cols: list[str]) -> str:
    header  = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    body    = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([header, divider, *body])


def _best_method(curves: dict) -> str:
    scores = {m: _final(curves, m) for m in METHODS if _final(curves, m) is not None}
    if not scores:
        return "—"
    best = max(scores, key=scores.get)
    return f"{METHOD_LABELS[best]} ({scores[best]:.4f})"


# ── Cross-dataset comparison figure ──────────────────────────────────────────

def _plot_comparison(all_curves: dict, output_path: Path) -> None:
    """
    Heatmap: rows = dataset × noise_type, cols = methods, values = mean final
    accuracy across pipelines, normalised relative to baseline.
    """
    datasets    = [r["dataset"] for r in RUNS]
    row_labels  = [f"{ds} | {nt}" for ds in datasets for nt in NOISE_TYPES]
    method_list = [m for m in METHODS if m != "random_mean"]

    data = np.full((len(row_labels), len(method_list)), np.nan)

    for ri, (ds, nt) in enumerate([(ds, nt) for ds in datasets for nt in NOISE_TYPES]):
        for mi, method in enumerate(method_list):
            vals = []
            for pipeline in PIPELINES:
                c = all_curves.get((ds, nt, pipeline))
                if c is None:
                    continue
                v = _final(c, method)
                b = _final(c, "baseline")
                if v is not None and b is not None and b > 0:
                    vals.append(v - b)          # improvement over baseline
            if vals:
                data[ri, mi] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(len(method_list) * 1.6 + 1.5, len(row_labels) * 0.55 + 1.5))
    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy − Baseline")

    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in method_list], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for ri in range(len(row_labels)):
        for mi in range(len(method_list)):
            v = data[ri, mi]
            if not np.isnan(v):
                ax.text(mi, ri, f"{v:+.3f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < vmax * 0.6 else "white")

    ax.set_title("Method improvement over baseline (mean across pipelines)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── Per-dataset summary figure ────────────────────────────────────────────────

def _plot_dataset_grid(dataset: str, noise_level: float, run_dir: Path,
                       all_curves: dict, output_path: Path) -> None:
    x_vals = None
    n_rows, n_cols = len(PIPELINES), len(NOISE_TYPES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    for row, pipeline in enumerate(PIPELINES):
        for col, noise_type in enumerate(NOISE_TYPES):
            ax   = axes[row][col]
            c    = all_curves.get((dataset, noise_type, pipeline))
            if c is None:
                ax.set_visible(False)
                continue

            props = np.array(c["proportions"]) * 100.0
            if x_vals is None:
                x_vals = props

            rnd_mean = np.array(c["random_mean"])
            rnd_std  = np.array(c["random_std"])

            ax.plot(props, c["datascope"],         color=COLORS["datascope"],         linestyle=STYLES["datascope"],         linewidth=1.8, label="DataScope")
            ax.plot(props, c["cleanlab"],           color=COLORS["cleanlab"],           linestyle=STYLES["cleanlab"],           linewidth=1.8, label="CleanLab")
            ax.plot(props, rnd_mean,                color=COLORS["random_mean"],        linestyle=STYLES["random_mean"],        linewidth=1.4, label="Random")
            ax.fill_between(props, rnd_mean - rnd_std, rnd_mean + rnd_std, color=COLORS["random_mean"], alpha=0.25)
            ax.axhline(c["baseline"],              color=COLORS["random_mean"],        linestyle="--", linewidth=1.0, label="Baseline")
            if c.get("kairos"):
                ax.plot(props, c["kairos"],         color=COLORS["kairos"],             linestyle=STYLES["kairos"],             linewidth=1.8, label="Kairos")
            if c.get("cleanlab_adaptive"):
                ax.plot(props, c["cleanlab_adaptive"], color=COLORS["cleanlab_adaptive"], linestyle=STYLES["cleanlab_adaptive"], linewidth=1.8, label="CL-Adaptive")
            if c.get("datascope_removal"):
                ax.plot(props, c["datascope_removal"], color="#2ca02c", linestyle="-", linewidth=1.4, label="DS removal")

            all_y = [*c["datascope"], *c["cleanlab"], *rnd_mean, c["baseline"]]
            for k in ("kairos", "cleanlab_adaptive", "datascope_removal"):
                if c.get(k):
                    all_y.extend(c[k])
            y_min, y_max = min(all_y), max(all_y)
            pad = max(0.005, (y_max - y_min) * 0.15)
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
            ax.tick_params(labelsize=7)
            ax.text(0.97, 0.97, f"Noise: {noise_type}", transform=ax.transAxes,
                    fontsize=6.5, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

            final_labels = [
                f"DataScope: {c['datascope'][-1]:.3f}",
                f"CleanLab: {c['cleanlab'][-1]:.3f}",
                f"Random: {rnd_mean[-1]:.3f}",
                f"Baseline: {c['baseline']:.3f}",
            ]
            if c.get("kairos"):
                final_labels.append(f"Kairos: {c['kairos'][-1]:.3f}")
            if c.get("cleanlab_adaptive"):
                final_labels.append(f"CL-Adaptive: {c['cleanlab_adaptive'][-1]:.3f}")
            if c.get("datascope_removal"):
                final_labels.append(f"DS removal: {c['datascope_removal'][-1]:.3f}")
            n_legend = len(final_labels)
            ax.legend(ax.get_lines()[:n_legend], final_labels,
                      fontsize=5.5, loc="upper left", framealpha=0.8,
                      handlelength=1.4, handletextpad=0.4)

            if row == n_rows - 1:
                ax.set_xlabel("% of training set cleaned", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{pipeline}\nAccuracy", fontsize=8)
            if row == 0:
                ax.set_title(noise_type, fontsize=9, fontweight="bold")

    noise_pct = int(noise_level * 100)
    fig.suptitle(
        f"All Noise Types — {dataset.upper()} (noise_level={noise_pct}%)\n"
        "Blue=DataScope, Red dashed=CleanLab, Orange dashed=Random (±1σ shaded), "
        "Purple=Kairos, Dark Red=CL-Adaptive",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Report assembly ───────────────────────────────────────────────────────────

def main() -> int:
    # Load all curves
    all_curves: dict = {}
    for run in RUNS:
        ds  = run["dataset"]
        rdir = run["run_dir"]
        for nt in NOISE_TYPES:
            for pipeline in PIPELINES:
                c = _load_curves(rdir, ds, nt, pipeline)
                if c is not None:
                    all_curves[(ds, nt, pipeline)] = c

    if not all_curves:
        print("No cached results found. Run experiments first.")
        return 1

    # Generate figures
    comparison_fig = OUTPUT_DIR / "comparison_heatmap.png"
    _plot_comparison(all_curves, comparison_fig)
    print(f"Saved: {comparison_fig}")

    dataset_grid_figs = {}
    for run in RUNS:
        ds        = run["dataset"]
        fig_path  = OUTPUT_DIR / f"{ds}_grid.png"
        _plot_dataset_grid(ds, run["noise_level"], run["run_dir"], all_curves, fig_path)
        dataset_grid_figs[ds] = fig_path
        print(f"Saved: {fig_path}")

    # ── Build report ────────────────────────────────────────────────────────
    sections = [
        "# Combined Experiment Report",
        "",
        "## Setup",
        "",
        "| Dataset | Noise Level | Train rows (approx) |",
        "| --- | --- | --- |",
        "| Adult   | 10% | ~26 000 |",
        "| German  | 20% | ~800    |",
        "| Titanic | 20% | ~712    |",
        "",
        "**Pipelines:** p1a (KNN-preprocessor + LogReg), p2b (PCA + SelectKBest + LogReg)",
        "",
        "**Methods compared:**",
        "- **DataScope** — Shapley-importance ranking (NEIGHBOR method), correction action",
        "- **CleanLab** — self_confidence ranking, correction action",
        "- **CL-Adaptive** — CleanLab ranking; top-half by confidence → correct, bottom-half → remove",
        "- **Kairos** — RBF kernel + logistic residual data valuation, correction action",
        "- **Random** — random ordering baseline (±1σ over 3 seeds)",
        "- **DS removal** *(outlier only)* — DataScope ranking, remove instead of cap",
        "",
        "---",
        "",
        "## Cross-Dataset Comparison",
        "",
        "The heatmap shows each method's mean accuracy improvement over the baseline, "
        "averaged across both pipelines. Green = better than baseline, red = worse.",
        "",
        f"![Comparison heatmap](comparison_heatmap.png)",
        "",
    ]

    # Global summary table
    summary_rows = []
    for run in RUNS:
        ds = run["dataset"]
        for nt in NOISE_TYPES:
            for pipeline in PIPELINES:
                c = all_curves.get((ds, nt, pipeline))
                if c is None:
                    continue
                row = {
                    "dataset":   ds,
                    "noise":     nt,
                    "pipeline":  pipeline,
                    "baseline":  f"{c['baseline']:.4f}",
                    "DataScope": f"{_final(c, 'datascope'):.4f}" if _final(c, 'datascope') else "—",
                    "CleanLab":  f"{_final(c, 'cleanlab'):.4f}" if _final(c, 'cleanlab') else "—",
                    "CL-Adapt":  f"{_final(c, 'cleanlab_adaptive'):.4f}" if _final(c, 'cleanlab_adaptive') else "—",
                    "Kairos":    f"{_final(c, 'kairos'):.4f}" if _final(c, 'kairos') else "—",
                    "Random":    f"{_final(c, 'random_mean'):.4f}" if _final(c, 'random_mean') else "—",
                    "best":      _best_method(c),
                }
                summary_rows.append(row)

    sections += [
        "## Summary Table",
        "",
        _md_table(summary_rows,
                  ["dataset", "noise", "pipeline", "baseline",
                   "DataScope", "CleanLab", "CL-Adapt", "Kairos", "Random", "best"]),
        "",
        "---",
        "",
    ]

    # Per-dataset sections
    for run in RUNS:
        ds         = run["dataset"]
        noise_pct  = int(run["noise_level"] * 100)
        grid_rel   = f"{ds}_grid.png"

        sections += [
            f"## {ds.capitalize()} — {noise_pct}% noise",
            "",
            f"![{ds} grid]({grid_rel})",
            "",
        ]

        for nt in NOISE_TYPES:
            best_rows = []
            for pipeline in PIPELINES:
                c = all_curves.get((ds, nt, pipeline))
                if c is None:
                    continue
                best_rows.append({
                    "pipeline":  pipeline,
                    "baseline":  f"{c['baseline']:.4f}",
                    "DataScope": f"{_final(c, 'datascope'):.4f}" if _final(c, 'datascope') else "—",
                    "CleanLab":  f"{_final(c, 'cleanlab'):.4f}" if _final(c, 'cleanlab') else "—",
                    "CL-Adapt":  f"{_final(c, 'cleanlab_adaptive'):.4f}" if _final(c, 'cleanlab_adaptive') else "—",
                    "Kairos":    f"{_final(c, 'kairos'):.4f}" if _final(c, 'kairos') else "—",
                    "Random":    f"{_final(c, 'random_mean'):.4f}" if _final(c, 'random_mean') else "—",
                    "best":      _best_method(c),
                })
            if best_rows:
                sections += [
                    f"### {nt}",
                    "",
                    _md_table(best_rows,
                              ["pipeline", "baseline", "DataScope", "CleanLab",
                               "CL-Adapt", "Kairos", "Random", "best"]),
                    "",
                ]

        sections.append("---")
        sections.append("")

    (OUTPUT_DIR / "combined_report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to: {OUTPUT_DIR / 'combined_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
