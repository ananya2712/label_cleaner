"""
Service-layer API for the modular label_cleaner architecture.

This class is the boundary you can later expose via HTTP/gRPC without
changing experiment internals.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..orchestration.catalog import EXPERIMENT_RUNNERS, pipeline_factory_a, pipeline_factory_b
from ..data.datasets import load_dataset


class LabelCleanerService:
    """
    Orchestration facade for experiment execution.
    """

    def __init__(self, datasets_dir: Path):
        self.datasets_dir = Path(datasets_dir)

    def run(self, dataset: str, noise_type: str, pipeline_key: str,
            noise_level: float = 0.2,
            proportions: Optional[np.ndarray] = None) -> Dict:
        """
        Execute one experiment and return a JSON-serializable dictionary.

        pipeline_key:
          - "p1a" -> Pipeline A
          - "p2b" -> Pipeline B
        """
        if proportions is None:
            proportions = np.linspace(0, 1, 21)

        ds = load_dataset(dataset, self.datasets_dir)
        if pipeline_key == "p1a":
            p_factory = pipeline_factory_a(ds.num_col_indices, ds.cat_col_indices)
        elif pipeline_key == "p2b":
            p_factory = pipeline_factory_b(n_features=ds.X.shape[1])
        else:
            raise ValueError(f"Unknown pipeline_key={pipeline_key!r}, use p1a or p2b.")

        if noise_type not in EXPERIMENT_RUNNERS:
            raise ValueError(
                f"Unknown noise_type={noise_type!r}. "
                f"Supported: {sorted(EXPERIMENT_RUNNERS)}"
            )

        result = EXPERIMENT_RUNNERS[noise_type](
            ds=ds,
            pipeline_factory=p_factory,
            noise_level=noise_level,
            proportions=proportions,
        )
        out = asdict(result)
        out["proportions"] = [float(x) for x in result.proportions]
        return out
