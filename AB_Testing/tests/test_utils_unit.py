import numpy as np

from ab_utils_01_data_setup import add_outcome_metrics, create_dummy_ab_data
from ab_utils_03_randomization import (
    apply_block_randomization,
    apply_simple_randomization,
    apply_stratified_randomization,
)
from ab_utils_05_ab_testing import run_ab_test
from ab_utils_06_post_hoc import apply_cuped


def test_create_dummy_ab_data_column_order_and_placeholders():
    df = create_dummy_ab_data(
        observations_count=25,
        seed=7,
        outcome_metric_col="engagement_score",
        guardrail_metric_col="bounce_rate",
    )
    assert list(df.columns[:4]) == [
        "user_id",
        "engagement_score",
        "bounce_rate",
        "past_purchase_count",
    ]
    assert df["engagement_score"].isna().all()
    assert df["bounce_rate"].isna().all()


def test_randomization_methods_assign_only_declared_groups():
    base = create_dummy_ab_data(60, seed=11, outcome_metric_col="engagement_score")
    labels = ("control", "treatment")

    s = apply_simple_randomization(base.copy(), group_labels=labels, group_col="group", seed=11)
    st = apply_stratified_randomization(
        base.copy(), stratify_col="platform", group_labels=labels, group_col="group", seed=11
    )
    b = apply_block_randomization(
        base.copy(),
        observation_id_col="user_id",
        block_size=8,
        group_labels=labels,
        group_col="group",
        seed=11,
    )

    for df in (s, st, b):
        assert set(df["group"].unique()).issubset(set(labels))
        assert len(df) == len(base)
        assert df.columns[0] == "group"


def test_apply_cuped_adds_adjusted_column_and_keeps_group_first():
    df = create_dummy_ab_data(100, seed=13, outcome_metric_col="engagement_score")
    df = apply_simple_randomization(df, group_labels=("control", "treatment"), group_col="group", seed=13)
    df = add_outcome_metrics(
        df,
        group_col="group",
        group_labels=("control", "treatment"),
        outcome_metric_col="engagement_score",
        seed=13,
    )

    adjusted = apply_cuped(
        df=df,
        pre_metric="past_purchase_count",
        outcome_metric_col="engagement_score",
        group_col="group",
        group_labels=("control", "treatment"),
        verbose=False,
    )

    col = "engagement_score_cuped_adjusted"
    assert col in adjusted.columns
    assert adjusted.columns[0] == "group"
    assert np.isfinite(adjusted[col]).all()


def test_run_ab_test_supports_paired_t_test_family():
    rng = np.random.default_rng(99)
    n = 120
    control = rng.normal(50, 8, n)
    treatment = control + rng.normal(1.5, 2, n)
    df = {
        "group": ["control"] * n + ["treatment"] * n,
        "metric": np.concatenate([control, treatment]),
    }

    import pandas as pd

    df = pd.DataFrame(df)
    result = run_ab_test(
        df=df,
        group_col="group",
        metric_col="metric",
        group_labels=("control", "treatment"),
        test_family="paired_t_test",
        alpha=0.05,
    )
    assert result["test"] == "paired t-test"
    assert 0 <= result["p_value"] <= 1
    assert "t_stat" in result

