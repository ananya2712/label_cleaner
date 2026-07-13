#!/usr/bin/env python3
"""
Generate reduced outlier figures from existing run_v8 caches.

Plots only DataScope (2σ capping), DataScope (removal), and the noisy
baseline — no other methods. No experiments are run; all values come from
cached summary.json files.

Usage:
    python3 scripts/generate_outlier_focus.py
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

RUN_DIR   = REPO_ROOT / "artifacts" / "run_v8"
DATASETS  = ["adult", "german", "titanic"]
PIPELINES = ["p1a", "p2b"]

OUTPUT_DIR = RUN_DIR / "outlier_focus"

SERIES = [
    ("datascope",         "DataScope (2σ capping)", "#1f77b4"),
    ("datascope_removal", "DataScope (removal)",    "#2ca02c"),
]


def _load_curves(dataset: str, pipeline: str) -> dict | None:
    path = (RUN_DIR / f"{dataset}_20pct" / "caches"
            / f"{dataset}__outlier__{pipeline}" / "summary.json")
    if not path.exists():
        print(f"skipping {dataset}/{pipeline}: {path} missing")
        return None
    c = json.loads(path.read_text())["curves"]
    if not c.get("datascope_removal"):
        print(f"skipping {dataset}/{pipeline}: no datascope_removal curve")
        return None
    return c


def _draw(ax, c: dict) -> None:
    props = np.array(c["proportions"]) * 100.0
    all_y = [c["baseline"]]
    for key, label, color in SERIES:
        ax.plot(props, c[key], color=color, linestyle="-", linewidth=1.8, label=label)
        all_y.extend(v for v in c[key] if np.isfinite(v))
    ax.axhline(c["baseline"], color="#7f7f7f", linestyle="--", linewidth=1.2,
               label=f"Baseline ({c['baseline']:.3f})")
    y_min, y_max = min(all_y), max(all_y)
    pad = max(0.005, (y_max - y_min) * 0.15)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.8)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readme_rows = []

    all_curves = {}
    for ds in DATASETS:
        for pipeline in PIPELINES:
            c = _load_curves(ds, pipeline)
            if c is not None:
                all_curves[(ds, pipeline)] = c

    # Per-config figures
    for (ds, pipeline), c in all_curves.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        _draw(ax, c)
        ax.set_xlabel("% of training set cleaned", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{ds} | outlier | {pipeline}", fontsize=10)
        fig.tight_layout()
        out = OUTPUT_DIR / f"{ds}__outlier__{pipeline}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
        readme_rows.append(
            f"| {ds} | {pipeline} | {c['baseline']:.4f} "
            f"| {c['datascope'][-1]:.4f} | {c['datascope_removal'][-1]:.4f} |"
        )

    # Grid figure: rows = pipelines, cols = datasets
    fig, axes = plt.subplots(len(PIPELINES), len(DATASETS),
                             figsize=(4.2 * len(DATASETS), 3.6 * len(PIPELINES)),
                             squeeze=False)
    for row, pipeline in enumerate(PIPELINES):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]
            c = all_curves.get((ds, pipeline))
            if c is None:
                ax.set_visible(False)
                continue
            _draw(ax, c)
            ax.tick_params(labelsize=7)
            if row == len(PIPELINES) - 1:
                ax.set_xlabel("% of training set cleaned", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{pipeline}\nAccuracy", fontsize=8)
            if row == 0:
                ax.set_title(ds, fontsize=9, fontweight="bold")
    fig.suptitle("Outlier noise — DataScope capping vs removal (run_v8)", fontsize=10, y=1.01)
    fig.tight_layout()
    grid_out = OUTPUT_DIR / "outlier_focus_grid.png"
    fig.savefig(grid_out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {grid_out}")

    readme = [
        "# Outlier Focus Figures (run_v8)",
        "",
        "Reduced view: DataScope (2σ capping), DataScope (removal), baseline.",
        "Generated from cached run_v8 results — no experiments were re-run.",
        "",
        "| dataset | pipeline | baseline | DataScope (2σ capping) | DataScope (removal) |",
        "| --- | --- | --- | --- | --- |",
        *readme_rows,
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote: {OUTPUT_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
