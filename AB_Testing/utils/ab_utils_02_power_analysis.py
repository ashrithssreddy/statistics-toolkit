# 02 Power Analysis — EDA (normality, variance, test family) + baseline, sample size, summary
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene
from statsmodels.stats.power import TTestIndPower, TTestPower


def test_normality(
    df,
    outcome_metric_col=None,
    group_col=None,
    group_labels=None,
    test_config=None,
    update_config=False
):
    """
    Run Shapiro normality test by group.

    Modes:
    - Raw mode: provide outcome_metric_col, group_col, group_labels -> returns per-group normality dict
    - Config mode: provide test_config + group_col (+df) and set update_config=True
      -> updates and returns test_config with 'normality' key
    """
    if test_config is not None:
        if test_config.get("outcome_metric_datatype") != "continuous":
            if update_config:
                test_config["normality"] = None
                return test_config
            return {}
        outcome_metric_col = test_config["outcome_metric_col"]
        group_labels = test_config["group_labels"]

    # Historical/pre-experiment data may not have group assignment yet.
    if group_col is None or group_col not in df.columns or group_labels is None:
        series = df[outcome_metric_col].dropna()
        stat, p = shapiro(series)
        results = {"overall": {"statistic": stat, "p_value": p, "normal": p > 0.05}}
    else:
        results = {}
        for group in group_labels:
            group_data = df[df[group_col] == group][outcome_metric_col].dropna()
            stat, p = shapiro(group_data)
            results[group] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}

    if update_config and test_config is not None:
        test_config["normality"] = all(v["normal"] for v in results.values())
        return test_config

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
    - group_relationship: independent or paired
    - normality: whether normality assumption holds
    """

    data_type = test_config.get("outcome_metric_datatype")
    group_count = test_config.get("group_count")
    group_relationship = test_config.get("group_relationship", "independent")
    normality = test_config.get("normality", True)

    # -------------------------
    # BINARY METRICS
    # -------------------------
    if data_type == "binary":

        if group_relationship == "paired":
            return "mcnemar_test"

        if group_count == 2:
            return "two_proportion_z_test"

        return "chi_square_test"


    # -------------------------
    # CONTINUOUS METRICS
    # -------------------------
    elif data_type == "continuous":

        if group_relationship == "paired":

            if normality:
                return "paired_t_test"
            else:
                return "wilcoxon_signed_rank_test"

        else:  # independent groups

            if group_count == 2:

                if normality:
                    return "two_sample_t_test"

                else:
                    return "mann_whitney_u_test"

            else:  # 3+ groups

                if normality:
                    return "anova_test"
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
    Returns dict with keys: baseline_rate, baseline_mean, baseline_std_dev (None where not applicable).
    """
    metric_col = test_config['outcome_metric_col']
    family = test_config.get('family')
    data_type = test_config.get('outcome_metric_datatype')

    result = {'baseline_rate': None, 'baseline_mean': None, 'baseline_std_dev': None}

    if family in ['one_proportion_z_test', 'two_proportion_z_test']:
        result['baseline_rate'] = df[metric_col].mean()
        if verbose:
            print(f"📊 Baseline conversion rate (full sample): {result['baseline_rate']:.2%}")
        return result

    if family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test', 'anova_test', 'mann_whitney_u_test', 'wilcoxon_signed_rank_test', 'kruskal_wallis_test'] or data_type == 'continuous':
        col = df[metric_col].dropna()
        result['baseline_mean'] = col.mean()
        result['baseline_std_dev'] = col.std()
        if result['baseline_std_dev'] == 0 or np.isnan(result['baseline_std_dev']):
            result['baseline_std_dev'] = 1.0
        if verbose:
            print(f"📊 Baseline mean (historical): {result['baseline_mean']:.2f}")
            print(f"📏 Baseline std dev (historical): {result['baseline_std_dev']:.2f}")
        return result

    if verbose:
        print("📊 No baseline computed for this metric type.")
    return result


def calculate_power_sample_size(
    test_family,
    group_relationship=None,
    alpha=0.05,
    power=0.80,
    baseline_rate=None,
    mde=None,
    std_dev=None,
    effect_size=None,
    num_groups=2,
    verbose=True
):
    """
    Calculate required sample size per group based on test type and assumptions.

    Supported families:
    - 'one_proportion_z_test'
    - 'two_proportion_z_test'
    - 'two_sample_t_test'
    - 'welch_two_sample_t_test'
    - 'paired_t_test'
    - 'mann_whitney_u_test'
    - 'wilcoxon_signed_rank_test'
    - 'anova_test'
    - 'welch_anova_test'
    - 'kruskal_wallis_test'
    - 'chi_square_test'
    """

    # -------------------------
    # Binary tests
    # -------------------------
    if test_family in [
        "one_proportion_z_test",
        "two_proportion_z_test",
        "chi_square_test"
    ]:

        if baseline_rate is None or mde is None:
            raise ValueError("baseline_rate and mde required for binary tests")

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = p1 + mde

        pooled_std = np.sqrt(2 * p1 * (1 - p1))

        n = int(np.ceil(((z_alpha + z_beta) ** 2 * pooled_std ** 2) / (mde ** 2)))
        if verbose:
            print("📈 Power Analysis Summary")
            print(f"- Test: {test_family.upper()}{' (' + group_relationship + ')' if group_relationship else ''}")
            print(f"- Significance level (α): {alpha}")
            print(f"- Statistical power (1 - β): {power}")
            print(f"- Baseline conversion rate: {baseline_rate:.2%}")
            print(f"- MDE: {mde:.2%}")
            print(f"\n✅ To detect a lift from {baseline_rate:.2%} to {(baseline_rate + mde):.2%},")
            print(f"you need {n} users per group → total {n * num_groups} users.")
        return n


    # -------------------------
    # Continuous tests
    # -------------------------
    elif test_family in [
        "one_sample_t_test",
        "two_sample_t_test",
        "welch_two_sample_t_test",
        "paired_t_test",
        "mann_whitney_u_test",
        "wilcoxon_signed_rank_test",
        "anova_test",
        "welch_anova_test",
        "kruskal_wallis_test"
    ]:

        if effect_size is None:
            if std_dev is None or mde is None:
                raise ValueError(
                    "For continuous metrics provide effect_size OR (std_dev + mde)"
                )

            effect_size = mde / std_dev  # Cohen's d


        if group_relationship == "paired":
            analysis = TTestPower()
        else:
            analysis = TTestIndPower()


        n = int(np.ceil(analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha
        )))
        if verbose:
            print("📈 Power Analysis Summary")
            print(f"- Test: {test_family.upper()}{' (' + group_relationship + ')' if group_relationship else ''}")
            print(f"- Significance level (α): {alpha}")
            print(f"- Statistical power (1 - β): {power}")
            if std_dev is not None:
                print(f"- Std Dev (baseline): {std_dev:.2f}")
            if mde is not None:
                print(f"- MDE (mean difference): {mde}")
            if effect_size is not None:
                print(f"- Cohen's d: {effect_size:.2f}")
            print(f"\n✅ To detect a {mde}-unit lift in mean outcome,")
            print(f"you need {n} users per group → total {n * num_groups} users.")
        return n


    else:
        raise ValueError(f"Unsupported test: {test_family}")

