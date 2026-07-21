#!/usr/bin/env python3
"""
No-noise sensitivity control (thesis Sec 4.1).

Runs every cleaning method against the fixed split with NO injected
corruption, deliberately flipping the labels each method ranks most harmful,
and sweeps the flipped proportion 0-50%. Since the data starts genuinely
clean, any accuracy drop below the random-order baseline's own drop is a
false-positive cost intrinsic to that method's ranking, not evidence of it
finding real noise. Writes one figure + one JSON summary per dataset/pipeline.
"""

from __future__ import annotations

import argparse
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

from label_cleaner.data.datasets import load_dataset
from label_cleaner.orchestration.catalog import pipeline_factory_a, pipeline_factory_b
from label_cleaner.orchestration.experiments import (
    DEFAULT_NO_NOISE_PROPORTIONS,
    run_no_noise_control_experiment_with_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "no_noise_control")
    parser.add_argument("--datasets", nargs="+", default=["adult", "german", "titanic"])
    parser.add_argument("--pipelines", nargs="+", default=["p1a", "p2b"])
    parser.add_argument(
        "--proportions", nargs="+", type=float, default=list(DEFAULT_NO_NOISE_PROPORTIONS),
        help="Fraction of training rows to flip, swept low to high (default 0-50%%).",
    )
    return parser.parse_args()


def _make_pipeline_factory(pipeline_key: str, ds):
    if pipeline_key == "p1a":
        return pipeline_factory_a(ds.num_col_indices, ds.cat_col_indices)
    if pipeline_key == "p2b":
        return pipeline_factory_b(n_features=len(ds.feature_names))
    raise ValueError(f"Unknown pipeline key: {pipeline_key}")


def _plot(path: Path, dataset: str, pipeline_key: str, curves) -> None:
    proportions_pct = np.array(curves.proportions) * 100.0
    rnd_mean = np.array(curves.random_mean)
    rnd_std = np.array(curves.random_std)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(proportions_pct, curves.datascope, color="#1f77b4", linewidth=1.8, label="DataScope")
    ax.plot(proportions_pct, curves.cleanlab, color="#d62728", linestyle="--", linewidth=1.8, label="CleanLab")
    if curves.entropy is not None:
        ax.plot(proportions_pct, curves.entropy, color="#17becf", linestyle=":", linewidth=1.8, label="Entropy")
    ax.plot(proportions_pct, rnd_mean, color="#ff7f0e", linestyle="--", linewidth=1.4, label="Random")
    ax.fill_between(proportions_pct, rnd_mean - rnd_std, rnd_mean + rnd_std,
                    color="#ff7f0e", alpha=0.25, label="±1σ Random")
    ax.axhline(curves.baseline, color="#2ca02c", linestyle="--", linewidth=1.0, label="Baseline (no flips)")
    ax.set_xlabel("% of clean training set flipped", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(
        f"{dataset} | {pipeline_key} | no-noise sensitivity control\n"
        "Every method ranking its own “most harmful” rows on already-clean data",
        fontsize=9,
    )
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    proportions = np.array(sorted(args.proportions))
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in args.datasets:
        ds = load_dataset(dataset, args.datasets_dir)
        summary[dataset] = {}
        for pipeline_key in args.pipelines:
            pipeline_factory = _make_pipeline_factory(pipeline_key, ds)
            artifacts = run_no_noise_control_experiment_with_artifacts(
                ds, pipeline_factory, proportions=proportions,
            )
            curves = artifacts.curves
            slug = f"{dataset}__{pipeline_key}__no_noise_control"
            _plot(figures_dir / f"{slug}.png", dataset, pipeline_key, curves)
            summary[dataset][pipeline_key] = {
                "proportions": list(curves.proportions),
                "baseline": curves.baseline,
                "datascope": curves.datascope,
                "cleanlab": curves.cleanlab,
                "entropy": curves.entropy,
                "random_mean": curves.random_mean,
                "random_std": curves.random_std,
            }
            print(f"{slug}: baseline={curves.baseline:.4f}  "
                  f"datascope@50%={curves.datascope[-1]:.4f}  "
                  f"cleanlab@50%={curves.cleanlab[-1]:.4f}  "
                  f"random@50%={curves.random_mean[-1]:.4f}")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
