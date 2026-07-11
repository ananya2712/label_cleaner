"""Integration: DP curves flow through a real (small) experiment. Run:
PYTHONPATH=/Users/ananyauppal/Desktop python3 tests/test_dp_pipeline.py
(takes ~1-2 minutes: titanic NNAR p1a with 3 proportions)
"""
from pathlib import Path

import numpy as np

from label_cleaner.data.datasets import load_dataset
from label_cleaner.orchestration.catalog import pipeline_factory_a
from label_cleaner.orchestration.experiments import run_nnar_experiment_with_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nnar_titanic_dp_curves():
    ds = load_dataset("titanic", REPO_ROOT / "datasets")
    factory = pipeline_factory_a(ds.num_col_indices, ds.cat_col_indices)
    artifacts = run_nnar_experiment_with_artifacts(
        ds, factory, noise_level=0.2, proportions=np.array([0.0, 0.5, 1.0])
    )
    c = artifacts.curves
    for name, curve in [("datascope_dp", c.datascope_dp), ("cleanlab_dp", c.cleanlab_dp),
                        ("random_dp_mean", c.random_dp_mean)]:
        assert curve is not None and len(curve) == 3, name
        assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in curve), (name, curve)
    assert c.baseline_dp is not None and 0.0 <= c.baseline_dp <= 1.0
    # At 0% cleaned, every method's DP equals the baseline DP.
    assert abs(c.datascope_dp[0] - c.baseline_dp) < 1e-9
    # NOTE: no directional assertion on the DP gap. On titanic, NNAR cleaning
    # RESTORES the true (larger) group disparity — the noisy model scores
    # fairer by DP than the clean one. Direction of DP change is an empirical
    # result reported from benchmark runs, not a wiring invariant.


if __name__ == "__main__":
    test_nnar_titanic_dp_curves()
    print("test_dp_pipeline: OK")
