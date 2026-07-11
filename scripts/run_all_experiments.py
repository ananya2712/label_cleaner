#!/usr/bin/env python3
"""Run the full experiment matrix, save figures, and write a markdown report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

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

from datascope.importance.shapley import ImportanceMethod
from label_cleaner.core.models import MethodCurves
from label_cleaner.data.datasets import load_dataset
from label_cleaner.orchestration.catalog import pipeline_factory_a, pipeline_factory_b
from label_cleaner.orchestration.experiments import (
    DEFAULT_PROPORTIONS,
    run_mnar_experiment_with_artifacts,
    run_nnar_experiment_with_artifacts,
    run_outlier_experiment_with_artifacts,
    run_random_label_experiment_with_artifacts,
)


DETAILED_RUNNERS = {
    "outlier": run_outlier_experiment_with_artifacts,
    "rnd_label": run_random_label_experiment_with_artifacts,
    "nnar": run_nnar_experiment_with_artifacts,
    "mnar": run_mnar_experiment_with_artifacts,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Directory containing input dataset files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "full_run",
        help="Directory where figures, caches, and report.md will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "german", "titanic"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=["outlier", "rnd_label", "nnar", "mnar"],
        help="Noise types to run.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["p1a", "p2b"],
        help="Pipeline keys to run.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Noise level passed into each experiment.",
    )
    parser.add_argument(
        "--proportions",
        nargs="+",
        type=float,
        default=list(DEFAULT_PROPORTIONS),
        help="Cleaning proportions to evaluate.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Skip experiments; load curves from existing cache and regenerate figures only.",
    )
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=50,
        help="Monte Carlo iterations for MONTECARLO ImportanceMethod (p2b). Default: 50.",
    )
    return parser.parse_args()


def _slug(dataset: str, noise_type: str, pipeline_key: str) -> str:
    return f"{dataset}__{noise_type}__{pipeline_key}"


def _pipeline_factory(ds, pipeline_key: str):
    if pipeline_key == "p1a":
        return pipeline_factory_a(ds.num_col_indices, ds.cat_col_indices)
    if pipeline_key == "p2b":
        return pipeline_factory_b(n_features=ds.X.shape[1])
    raise ValueError(f"Unsupported pipeline key: {pipeline_key!r}")


def _importance_method(pipeline_key: str) -> ImportanceMethod:
    # NEIGHBOR is used for both pipelines: TMC Shapley was evaluated for p2b
    # (RandomForest classifier) but rejected for runtime cost — see implementation.md.
    return ImportanceMethod.NEIGHBOR


def _cleaned_prefix(ranked: np.ndarray, proportion: float) -> np.ndarray:
    n_clean = int(proportion * len(ranked))
    return ranked[:n_clean]


def _to_dataset_indices(train_idx: np.ndarray, positions: np.ndarray) -> List[int]:
    return [int(train_idx[pos]) for pos in positions]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _round_key(value: float) -> str:
    return f"{value:.2f}"


def _record_rows(ds, artifacts) -> List[Dict]:
    noisy_set = set(int(pos) for pos in artifacts.bundle.noisy_positions.tolist())
    rows = []
    for train_pos, dataset_idx in enumerate(artifacts.split.train_idx):
        rows.append(
            {
                "train_position": int(train_pos),
                "dataset_index": int(dataset_idx),
                "is_true_noisy": train_pos in noisy_set,
                "is_protected": bool(ds.protected_group_mask[dataset_idx]),
                "clean_label": int(artifacts.split.y_train[train_pos]),
                "noisy_label": int(artifacts.bundle.y_noisy[train_pos]),
                "clean_features": artifacts.split.X_train[train_pos].tolist(),
                "noisy_features": artifacts.bundle.X_noisy[train_pos].tolist(),
            }
        )
    return rows


def _cleaned_cache(artifacts, proportions: Iterable[float]) -> Dict:
    cache = {
        "datascope": {},
        "cleanlab": {},
        "random": {},
    }
    for proportion in proportions:
        key = _round_key(float(proportion))
        ds_positions = _cleaned_prefix(artifacts.datascope_ranked, proportion)
        cl_positions = _cleaned_prefix(artifacts.cleanlab_ranked, proportion)
        cache["datascope"][key] = {
            "train_positions": [int(x) for x in ds_positions],
            "dataset_indices": _to_dataset_indices(artifacts.split.train_idx, ds_positions),
        }
        cache["cleanlab"][key] = {
            "train_positions": [int(x) for x in cl_positions],
            "dataset_indices": _to_dataset_indices(artifacts.split.train_idx, cl_positions),
        }
        random_entries = {}
        for seed_idx, ranking in enumerate(artifacts.random_rankings):
            rnd_positions = _cleaned_prefix(ranking, proportion)
            random_entries[f"seed_{seed_idx}"] = {
                "train_positions": [int(x) for x in rnd_positions],
                "dataset_indices": _to_dataset_indices(artifacts.split.train_idx, rnd_positions),
            }
        cache["random"][key] = random_entries
    return cache


def _overlap_summary(cleaned_cache: Dict, proportions: Iterable[float]) -> List[Dict]:
    rows = []
    for proportion in proportions:
        key = _round_key(float(proportion))
        ds_set = set(cleaned_cache["datascope"][key]["dataset_indices"])
        cl_set = set(cleaned_cache["cleanlab"][key]["dataset_indices"])
        union = ds_set | cl_set
        rows.append(
            {
                "proportion": key,
                "datascope_count": len(ds_set),
                "cleanlab_count": len(cl_set),
                "intersection_count": len(ds_set & cl_set),
                "jaccard": 0.0 if not union else len(ds_set & cl_set) / len(union),
            }
        )
    return rows


def _save_train_records(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default))
            handle.write("\n")


def _curves_from_cache(cache_dir: Path) -> MethodCurves:
    payload = json.loads((cache_dir / "summary.json").read_text(encoding="utf-8"))
    c = payload["curves"]
    return MethodCurves(
        datascope=c["datascope"],
        random_mean=c["random_mean"],
        random_std=c["random_std"],
        cleanlab=c["cleanlab"],
        baseline=c["baseline"],
        proportions=np.array(c["proportions"]),
        datascope_removal=c.get("datascope_removal"),
    )


def _plot_curves(path: Path, dataset: str, noise_type: str, pipeline_key: str, curves) -> None:
    proportions_pct = np.array(curves.proportions) * 100.0
    rnd_mean = np.array(curves.random_mean)
    rnd_std = np.array(curves.random_std)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(proportions_pct, curves.datascope,
            color="#1f77b4", linestyle="-",  linewidth=1.8, label="DataScope")
    ax.plot(proportions_pct, curves.cleanlab,
            color="#d62728", linestyle="--", linewidth=1.8, label="CleanLab")
    ax.plot(proportions_pct, rnd_mean,
            color="#ff7f0e", linestyle="--", linewidth=1.4, label="Random")
    ax.fill_between(proportions_pct, rnd_mean - rnd_std, rnd_mean + rnd_std,
                    color="#ff7f0e", alpha=0.25, label="±1σ Random")
    ax.axhline(curves.baseline, color="#ff7f0e", linestyle="--", linewidth=1.0, label="Baseline")
    if curves.datascope_removal is not None:
        ax.plot(proportions_pct, curves.datascope_removal,
                color="#2ca02c", linestyle="-", linewidth=1.4, label="DS removal")
    ax.set_xlabel("% of training set cleaned", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(
        f"{dataset} | {noise_type} | {pipeline_key}\n"
        "Blue=DataScope, Red dashed=CleanLab, Orange dashed=Random baseline (±1σ shaded)",
        fontsize=9,
    )
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_grid(
    path: Path,
    dataset: str,
    noise_level: float,
    noise_types: List[str],
    pipelines: List[str],
    curves_grid: Dict,  # {pipeline_key: {noise_type: curves}}
    proportions: np.ndarray,
) -> None:
    x = proportions * 100.0
    n_rows, n_cols = len(pipelines), len(noise_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    for row, pipeline_key in enumerate(pipelines):
        for col, noise_type in enumerate(noise_types):
            ax = axes[row][col]
            curves = curves_grid[pipeline_key][noise_type]
            rnd_mean = np.array(curves.random_mean)
            rnd_std = np.array(curves.random_std)

            ax.plot(x, curves.datascope, color="#1f77b4", linestyle="-",  linewidth=1.8, label="DataScope")
            ax.plot(x, curves.cleanlab,  color="#d62728", linestyle="--", linewidth=1.8, label="CleanLab")
            ax.plot(x, rnd_mean,         color="#ff7f0e", linestyle="--", linewidth=1.4, label="Random")
            ax.fill_between(x, rnd_mean - rnd_std, rnd_mean + rnd_std, color="#ff7f0e", alpha=0.25)
            ax.axhline(curves.baseline,  color="#ff7f0e", linestyle="--", linewidth=1.0, label="Baseline")
            if curves.datascope_removal is not None:
                ax.plot(x, curves.datascope_removal, color="#2ca02c", linestyle="-", linewidth=1.4, label="DS removal")
            all_y = [*curves.datascope, *curves.cleanlab, *rnd_mean, curves.baseline]
            if curves.datascope_removal is not None:
                all_y.extend(curves.datascope_removal)
            y_min, y_max = min(all_y), max(all_y)
            pad = max(0.005, (y_max - y_min) * 0.15)
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
            ax.tick_params(labelsize=7)
            ax.text(0.97, 0.97, f"Noise: {noise_type}", transform=ax.transAxes,
                    fontsize=6.5, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

            final_labels = [
                f"DataScope: {curves.datascope[-1]:.3f}",
                f"CleanLab: {curves.cleanlab[-1]:.3f}",
                f"Random: {rnd_mean[-1]:.3f}",
                f"Baseline: {curves.baseline:.3f}",
            ]
            if curves.datascope_removal is not None:
                final_labels.append(f"DS removal: {curves.datascope_removal[-1]:.3f}")
            n_legend = len(final_labels)
            ax.legend(ax.get_lines()[:n_legend], final_labels,
                      fontsize=5.5, loc="upper left", framealpha=0.8,
                      handlelength=1.4, handletextpad=0.4)

            if row == n_rows - 1:
                ax.set_xlabel("% of training set cleaned", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{pipeline_key}\nAccuracy", fontsize=8)
            if row == 0:
                ax.set_title(noise_type, fontsize=9, fontweight="bold")

    noise_pct = int(noise_level * 100)
    fig.suptitle(
        f"All Noise Types — {dataset.upper()} (noise_level={noise_pct}%)\n"
        "Blue=DataScope, Red dashed=CleanLab, Orange dashed=Random (\u00b11\u03c3 shaded)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(rows: List[Dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def main() -> int:
    args = parse_args()
    proportions = np.array(args.proportions, dtype=float)
    figures_dir = args.output_dir / "figures"
    caches_dir = args.output_dir / "caches"
    figures_dir.mkdir(parents=True, exist_ok=True)
    caches_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    report_sections = [
        "# Experiment Report",
        "",
        f"- Datasets dir: `{args.datasets_dir}`",
        f"- Noise level: `{args.noise_level}`",
        f"- Proportions: `{', '.join(_round_key(x) for x in proportions)}`",
        "",
    ]

    for dataset in args.datasets:
        ds = load_dataset(dataset, args.datasets_dir)
        dataset_curves: Dict[str, Dict] = {}
        for noise_type in args.noise_types:
            runner = DETAILED_RUNNERS[noise_type]
            for pipeline_key in args.pipelines:
                slug = _slug(dataset, noise_type, pipeline_key)
                cache_dir = caches_dir / slug

                if args.from_cache:
                    curves = _curves_from_cache(cache_dir)
                    figure_path = figures_dir / f"{slug}.png"
                    _plot_curves(figure_path, dataset, noise_type, pipeline_key, curves)
                    dataset_curves.setdefault(pipeline_key, {})[noise_type] = curves
                    continue

                pipeline_factory = _pipeline_factory(ds, pipeline_key)
                artifacts = runner(
                    ds=ds,
                    pipeline_factory=pipeline_factory,
                    noise_level=args.noise_level,
                    proportions=proportions,
                    importance_method=_importance_method(pipeline_key),
                    mc_iterations=args.mc_iterations,
                )
                cleaned_cache = _cleaned_cache(artifacts, proportions)
                train_rows = _record_rows(ds, artifacts)
                overlap_rows = _overlap_summary(cleaned_cache, proportions)

                figure_path = figures_dir / f"{slug}.png"
                cache_dir.mkdir(parents=True, exist_ok=True)
                _plot_curves(figure_path, dataset, noise_type, pipeline_key, artifacts.curves)
                dataset_curves.setdefault(pipeline_key, {})[noise_type] = artifacts.curves
                _save_train_records(cache_dir / "train_records.jsonl", train_rows)

                summary_payload = {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "pipeline_key": pipeline_key,
                    "noise_level": args.noise_level,
                    "feature_names": ds.feature_names,
                    "protected_col_name": ds.protected_col_name,
                    "outlier_col_name": ds.feature_names[ds.outlier_col_idx],
                    "train_size": int(len(artifacts.split.train_idx)),
                    "test_size": int(len(artifacts.split.test_idx)),
                    "true_noisy_train_positions": [
                        int(x) for x in artifacts.bundle.noisy_positions.tolist()
                    ],
                    "true_noisy_dataset_indices": _to_dataset_indices(
                        artifacts.split.train_idx, artifacts.bundle.noisy_positions
                    ),
                    "datascope_ranked_train_positions": [
                        int(x) for x in artifacts.datascope_ranked.tolist()
                    ],
                    "datascope_ranked_dataset_indices": _to_dataset_indices(
                        artifacts.split.train_idx, artifacts.datascope_ranked
                    ),
                    "cleanlab_ranked_train_positions": [
                        int(x) for x in artifacts.cleanlab_ranked.tolist()
                    ],
                    "cleanlab_ranked_dataset_indices": _to_dataset_indices(
                        artifacts.split.train_idx, artifacts.cleanlab_ranked
                    ),
                    "random_ranked_train_positions": {
                        f"seed_{idx}": [int(x) for x in ranking.tolist()]
                        for idx, ranking in enumerate(artifacts.random_rankings)
                    },
                    "random_ranked_dataset_indices": {
                        f"seed_{idx}": _to_dataset_indices(artifacts.split.train_idx, ranking)
                        for idx, ranking in enumerate(artifacts.random_rankings)
                    },
                    "cleaned_cache": cleaned_cache,
                    "overlap_summary": overlap_rows,
                    "curves": asdict(artifacts.curves),
                }
                with (cache_dir / "summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(summary_payload, handle, indent=2, default=_json_default)

                final_idx = -1
                summary_rows.append(
                    {
                        "slug": slug,
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "pipeline": pipeline_key,
                        "baseline": round(float(artifacts.curves.baseline), 4),
                        "datascope_final": round(float(artifacts.curves.datascope[final_idx]), 4),
                        "cleanlab_final": round(float(artifacts.curves.cleanlab[final_idx]), 4),
                        "random_final": round(float(artifacts.curves.random_mean[final_idx]), 4),
                        "figure": f"figures/{slug}.png",
                        "cache": f"caches/{slug}/summary.json",
                    }
                )

                report_sections.extend(
                    [
                        f"## {dataset} | {noise_type} | {pipeline_key}",
                        "",
                        f"![{slug}](figures/{slug}.png)",
                        "",
                        f"- Baseline accuracy: `{artifacts.curves.baseline:.4f}`",
                        f"- Final DataScope accuracy: `{artifacts.curves.datascope[-1]:.4f}`",
                        f"- Final CleanLab accuracy: `{artifacts.curves.cleanlab[-1]:.4f}`",
                        f"- Final Random mean accuracy: `{artifacts.curves.random_mean[-1]:.4f}`",
                        f"- True noisy training rows: `{len(artifacts.bundle.noisy_positions)}`",
                        f"- Cache files: `caches/{slug}/summary.json`, `caches/{slug}/train_records.jsonl`",
                        "",
                        _markdown_table(
                            [
                                {
                                    "proportion": row["proportion"],
                                    "datascope_count": row["datascope_count"],
                                    "cleanlab_count": row["cleanlab_count"],
                                    "intersection_count": row["intersection_count"],
                                    "jaccard": f"{row['jaccard']:.3f}",
                                }
                                for row in overlap_rows
                            ],
                            [
                                "proportion",
                                "datascope_count",
                                "cleanlab_count",
                                "intersection_count",
                                "jaccard",
                            ],
                        ),
                        "",
                    ]
                )

        _plot_grid(
            figures_dir / f"{dataset}__all_noise_types.png",
            dataset, args.noise_level, args.noise_types, args.pipelines,
            dataset_curves, proportions,
        )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, default=_json_default)

    report_sections.extend(
        [
            "## Run Summary",
            "",
            _markdown_table(
                summary_rows,
                [
                    "slug",
                    "baseline",
                    "datascope_final",
                    "cleanlab_final",
                    "random_final",
                    "figure",
                    "cache",
                ],
            ),
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_sections), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
