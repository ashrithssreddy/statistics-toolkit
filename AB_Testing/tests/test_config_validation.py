import pytest

from ab_utils_02_power_analysis import calculate_power_sample_size, determine_test_family


@pytest.mark.parametrize(
    "cfg, expected_family",
    [
        (
            {
                "outcome_metric_datatype": "continuous",
                "group_count": 2,
                "variant": "independent",
                "normality": True,
                "variance_equal": True,
            },
            "two_sample_t_test",
        ),
        (
            {
                "outcome_metric_datatype": "continuous",
                "group_count": 2,
                "variant": "paired",
                "normality": True,
                "variance_equal": True,
            },
            "paired_t_test",
        ),
        (
            {
                "outcome_metric_datatype": "continuous",
                "group_count": 2,
                "variant": "independent",
                "normality": False,
                "variance_equal": False,
            },
            "mann_whitney_u_test",
        ),
        (
            {
                "outcome_metric_datatype": "binary",
                "group_count": 2,
                "variant": "independent",
                "normality": True,
                "variance_equal": True,
            },
            "two_proportion_z_test",
        ),
    ],
)
def test_determine_test_family_matrix(cfg, expected_family):
    assert determine_test_family(cfg) == expected_family


def test_power_sample_size_requires_binary_inputs():
    with pytest.raises(ValueError, match="baseline_rate and mde required"):
        calculate_power_sample_size(
            test_family="two_proportion_z_test",
            alpha=0.05,
            power=0.80,
            baseline_rate=None,
            mde=0.02,
        )


def test_power_sample_size_requires_continuous_inputs():
    with pytest.raises(ValueError, match="effect_size OR \\(std_dev \\+ mde\\)"):
        calculate_power_sample_size(
            test_family="paired_t_test",
            variant="paired",
            alpha=0.05,
            power=0.80,
            effect_size=None,
            std_dev=None,
            mde=None,
        )

