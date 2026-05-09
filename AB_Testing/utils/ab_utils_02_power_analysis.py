# 02 Power Analysis — EDA (normality, variance, test family) + baseline, sample size, summary
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene
from statsmodels.stats.power import TTestIndPower, TTestPower


def test_normality(df, outcome_metric_col, group_col, group_labels):
    results = {}
    for group in group_labels:
        group_data = df[df[group_col] == group][outcome_metric_col]
        stat, p = shapiro(group_data)
        results[group] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
    return results


def test_equal_variance(df, outcome_metric_col, group_col, group_labels):
    group_data = [df[df[group_col] == label][outcome_metric_col] for label in group_labels]
    stat, p = levene(*group_data)
    return {'statistic': stat, 'p_value': p, 'equal_variance': p > 0.05}


def determine_test_family(test_config):
    """
    Determine the appropriate statistical test for an experiment.

    Decision factors:
    - outcome_metric_datatype: binary / continuous / categorical / count
    - group_count: number of variants
    - variant: independent or paired
    - normality: whether normality assumption holds
    - variance_equal: whether group variances are assumed equal
    """

    data_type = test_config.get("outcome_metric_datatype")
    group_count = test_config.get("group_count")
    variant = test_config.get("variant", "independent")
    normality = test_config.get("normality", True)
    variance_equal = test_config.get("variance_equal", True)

    # -------------------------
    # BINARY METRICS
    # -------------------------
    if data_type == "binary":

        if variant == "paired":
            return "mcnemar_test"

        if group_count == 2:
            return "two_proportion_z_test"

        return "chi_square_test"


    # -------------------------
    # CONTINUOUS METRICS
    # -------------------------
    elif data_type == "continuous":

        if variant == "paired":

            if normality:
                return "paired_t_test"
            else:
                return "wilcoxon_signed_rank_test"

        else:  # independent groups

            if group_count == 2:

                if normality:
                    if variance_equal:
                        return "two_sample_t_test"
                    else:
                        return "welch_t_test"

                else:
                    return "mann_whitney_u_test"

            else:  # 3+ groups

                if normality:
                    if variance_equal:
                        return "anova"
                    else:
                        return "welch_anova"
                else:
                    return "kruskal_wallis_test"


    # -------------------------
    # CATEGORICAL METRICS
    # -------------------------
    elif data_type == "categorical":

        return "chi_square_test"


    # -------------------------
    # COUNT DATA
    # -------------------------
    elif data_type == "count":

        if group_count == 2:
            return "poisson_rate_test"
        else:
            return "poisson_regression"


    else:
        raise ValueError(
            f"Unsupported outcome_metric_datatype: {data_type}"
        )


def compute_baseline_from_data(df, test_config, verbose=True):
    """
    Compute baseline rate/mean and std_dev from the whole dataset for power analysis.
    No group splitting: uses full sample so baselines are pre-experiment / design inputs.
    Uses test_config['family'] and test_config['outcome_metric_datatype'] to decide logic.
    Returns dict with keys: baseline_rate, baseline_mean, std_dev (None where not applicable).
    """
    metric_col = test_config['outcome_metric_col']
    family = test_config.get('family')
    data_type = test_config.get('outcome_metric_datatype')

    result = {'baseline_rate': None, 'baseline_mean': None, 'std_dev': None}

    if family == 'z_test':
        result['baseline_rate'] = df[metric_col].mean()
        if verbose:
            print(f"📊 Baseline conversion rate (full sample): {result['baseline_rate']:.2%}")
        return result

    if family in ['t_test', 'anova', 'non_parametric'] or data_type == 'continuous':
        col = df[metric_col].dropna()
        result['baseline_mean'] = col.mean()
        result['std_dev'] = col.std()
        if result['std_dev'] == 0 or np.isnan(result['std_dev']):
            result['std_dev'] = 1.0
        if verbose:
            print(f"📊 Baseline mean (historical): {result['baseline_mean']:.2f}")
            print(f"📏 Baseline std dev (historical): {result['std_dev']:.2f}")
        return result

    if verbose:
        print("📊 No baseline computed for this metric type.")
    return result


def calculate_power_sample_size(
    test_family,
    variant=None,
    alpha=0.05,
    power=0.80,
    baseline_rate=None,
    mde=None,
    std_dev=None,
    effect_size=None,
    num_groups=2
):
    """
    Calculate required sample size per group based on test type and assumptions.

    Supported families:
    - 'z_test'              : Binary outcomes (proportions)
    - 't_test'              : Continuous outcomes (independent or paired)
    - 'non_parametric'      : Mann-Whitney (approximated as t-test)
    - 'anova'               : Not implemented (default to t-test)
    - 'chi_square'          : Categorical outcomes (not used in this version)
    """

    # -------------------------
    # Binary tests
    # -------------------------
    if test_family in [
        "two_proportion_z_test",
        "z_test",
        "chi_square_test"
    ]:

        if baseline_rate is None or mde is None:
            raise ValueError("baseline_rate and mde required for binary tests")

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = p1 + mde

        pooled_std = np.sqrt(2 * p1 * (1 - p1))

        n = ((z_alpha + z_beta) ** 2 * pooled_std ** 2) / (mde ** 2)

        return int(np.ceil(n))


    # -------------------------
    # Continuous tests
    # -------------------------
    elif test_family in [
        "two_sample_t_test",
        "welch_t_test",
        "paired_t_test",
        "mann_whitney_u_test",
        "wilcoxon_signed_rank_test",
        "anova",
        "kruskal_wallis_test"
    ]:

        if effect_size is None:
            if std_dev is None or mde is None:
                raise ValueError(
                    "For continuous metrics provide effect_size OR (std_dev + mde)"
                )

            effect_size = mde / std_dev  # Cohen's d


        if variant == "paired":
            analysis = TTestPower()
        else:
            analysis = TTestIndPower()


        n = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha
        )

        return int(np.ceil(n))


    else:
        raise ValueError(f"Unsupported test: {test_family}")


def print_power_summary(
    test_family,
    variant,
    alpha,
    power,
    baseline_rate=None,
    mde=None,
    std_dev=None,
    required_sample_size=None
):

    print("📈 Power Analysis Summary")
    print(f"- Test: {test_family.upper()}{' (' + variant + ')' if variant else ''}")
    print(f"- Significance level (α): {alpha}")
    print(f"- Statistical power (1 - β): {power}")

    binary_tests = [
        "z_test",
        "two_proportion_z_test",
        "chi_square_test"
    ]

    continuous_tests = [
        "t_test",
        "two_sample_t_test",
        "welch_t_test",
        "paired_t_test",
        "mann_whitney_u_test",
        "wilcoxon_signed_rank_test",
        "anova",
        "welch_anova",
        "kruskal_wallis_test"
    ]

    if test_family in binary_tests:

        print(f"- Baseline conversion rate: {baseline_rate:.2%}")
        print(f"- MDE: {mde:.2%}")

        print(f"\n✅ To detect a lift from {baseline_rate:.2%} to {(baseline_rate + mde):.2%},")
        print(f"you need {required_sample_size} users per group → total {required_sample_size * 2} users.")

    elif test_family in continuous_tests:

        print(f"- Std Dev (baseline): {std_dev:.2f}")
        print(f"- MDE (mean difference): {mde}")
        print(f"- Cohen's d: {mde / std_dev:.2f}")

        print(f"\n✅ To detect a {mde}-unit lift in mean outcome,")
        print(f"you need {required_sample_size} users per group → total {required_sample_size * 2} users.")

    else:

        print("\n⚠️ Summary not specialized for this test.")
        print(f"Required sample size per group: {required_sample_size}")
        print(f"Total sample size: {required_sample_size * 2}")
