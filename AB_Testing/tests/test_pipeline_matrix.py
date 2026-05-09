import pytest

from ab_utils_01_data_setup import add_outcome_metrics, create_dummy_ab_data
from ab_utils_02_power_analysis import determine_test_family
from ab_utils_03_randomization import (
    apply_matched_pair_randomization,
    apply_simple_randomization,
)
from ab_utils_05_ab_testing import run_ab_test


def normalize_family_for_runner(family: str) -> str:
    mapping = {
        "two_proportion_z_test": "z_test",
        "chi_square_test": "chi_square",
        "two_sample_t_test": "t_test",
        "welch_t_test": "t_test",
        "paired_t_test": "paired_t_test",
        "mann_whitney_u_test": "mann_whitney_u_test",
    }
    return mapping.get(family, family)


@pytest.mark.parametrize(
    "scenario",
    [
        {
            "name": "continuous_independent_normal",
            "datatype": "continuous",
            "variant": "independent",
            "normality": True,
            "variance_equal": True,
            "n": 300,
            "randomizer": "simple",
            "metric_col": "engagement_score",
        },
        {
            "name": "continuous_paired_normal",
            "datatype": "continuous",
            "variant": "paired",
            "normality": True,
            "variance_equal": True,
            "n": 300,
            "randomizer": "matched",
            "metric_col": "engagement_score",
        },
        {
            "name": "continuous_independent_non_normal",
            "datatype": "continuous",
            "variant": "independent",
            "normality": False,
            "variance_equal": False,
            "n": 300,
            "randomizer": "simple",
            "metric_col": "engagement_score",
        },
    ],
)
def test_pipeline_scenarios_run_end_to_end(scenario):
    labels = ("control", "treatment")
    df = create_dummy_ab_data(
        observations_count=scenario["n"],
        seed=23,
        outcome_metric_col=scenario["metric_col"],
    )

    if scenario["randomizer"] == "matched":
        df = apply_matched_pair_randomization(
            df,
            sort_col="past_purchase_count",
            group_col="group",
            group_labels=labels,
        )
    else:
        df = apply_simple_randomization(df, group_labels=labels, group_col="group", seed=23)

    df = add_outcome_metrics(
        df,
        group_col="group",
        group_labels=labels,
        outcome_metric_col=scenario["metric_col"],
        treatment_effect=True,
        seed=23,
    )

    cfg = {
        "outcome_metric_datatype": scenario["datatype"],
        "group_count": 2,
        "variant": scenario["variant"],
        "normality": scenario["normality"],
        "variance_equal": scenario["variance_equal"],
    }
    selected_family = determine_test_family(cfg)
    run_family = normalize_family_for_runner(selected_family)

    result = run_ab_test(
        df=df,
        group_col="group",
        metric_col=scenario["metric_col"],
        group_labels=labels,
        test_family=run_family,
        variant=scenario["variant"],
        alpha=0.05,
    )

    assert result["test_family"] == run_family
    assert "summary" in result
    assert all(k in result["summary"] for k in labels)
    assert 0 <= result["p_value"] <= 1

