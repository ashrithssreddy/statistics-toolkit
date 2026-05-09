import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ab_utils_05_ab_testing import run_ab_test


def test_null_rejection_rate_is_reasonable_for_independent_t_test():
    rng = np.random.default_rng(2026)
    runs = 120
    alpha = 0.05
    pvals = []

    for _ in range(runs):
        n = 120
        c = rng.normal(0, 1, n)
        t = rng.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "group": ["control"] * n + ["treatment"] * n,
                "metric": np.concatenate([c, t]),
            }
        )
        result = run_ab_test(
            df=df,
            group_col="group",
            metric_col="metric",
            group_labels=("control", "treatment"),
            test_family="t_test",
            variant="independent",
            alpha=alpha,
        )
        pvals.append(result["p_value"])

    reject_rate = np.mean(np.array(pvals) < alpha)
    assert 0.0 <= reject_rate <= 0.15


def test_detects_large_effect_with_very_small_pvalue():
    rng = np.random.default_rng(404)
    n = 160
    control = rng.normal(50, 10, n)
    treatment = rng.normal(58, 10, n)
    df = pd.DataFrame(
        {
            "group": ["control"] * n + ["treatment"] * n,
            "metric": np.concatenate([control, treatment]),
        }
    )
    result = run_ab_test(
        df=df,
        group_col="group",
        metric_col="metric",
        group_labels=("control", "treatment"),
        test_family="t_test",
        variant="independent",
        alpha=0.05,
    )
    assert result["p_value"] < 1e-3


def test_multiple_testing_adjustments_have_expected_shape_and_bounds():
    raw = np.array([0.001, 0.01, 0.02, 0.07, 0.2, 0.9])
    bonf = multipletests(raw, alpha=0.05, method="bonferroni")[1]
    bh = multipletests(raw, alpha=0.05, method="fdr_bh")[1]

    assert bonf.shape == raw.shape
    assert bh.shape == raw.shape
    assert np.all((bonf >= 0) & (bonf <= 1))
    assert np.all((bh >= 0) & (bh <= 1))
    assert np.all(bonf >= raw)
    assert np.all(bh >= raw)

