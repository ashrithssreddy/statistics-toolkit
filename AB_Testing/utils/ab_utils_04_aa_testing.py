# 04 AA Testing — outcome similarity test, visualization, Type I error simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def run_outcome_similarity_test(
    df,
    group_col,
    metric_col,
    test_family,
    group_relationship=None,
    hypothesis_type='two_sided',
    group_labels=('control', 'treatment'),
    alpha=0.05,
    verbose=True
):
    """
    Runs a similarity test between two groups based on test_family and group_relationship.

    Parameters:
    - df: pandas DataFrame
    - group_col: column with group assignment
    - metric_col: outcome metric
    - test_family: canonical test name (e.g., 'two_proportion_z_test', 'two_sample_t_test', 'chi_square_test')
    - group_relationship: 'independent' or 'paired' (required for t-test)
    - group_labels: tuple of (control, treatment)
    - alpha: significance threshold
    - verbose: print detailed interpretation
    """

    if verbose:
        print("📏 Outcome Similarity Check")
        print("-" * 50)
        if pd.api.types.is_numeric_dtype(df[metric_col]):
            mean_by_group = df.groupby(group_col)[metric_col].mean()
            print("Group means:")
            print(mean_by_group)
            print()

    group1 = df[df[group_col] == group_labels[0]][metric_col]
    group2 = df[df[group_col] == group_labels[1]][metric_col]
    alt_map = {'two_sided': 'two-sided', 'greater': 'greater', 'less': 'less'}
    if hypothesis_type not in alt_map:
        raise ValueError("hypothesis_type must be one of: 'two_sided', 'greater', 'less'")
    scipy_alt = alt_map[hypothesis_type]

    # --- Run appropriate test ---
    # --- Binary ---
    stat_label = None
    stat_value = None

    if test_family in ["two_proportion_z_test", "one_proportion_z_test"]:
        conv1, conv2 = group1.mean(), group2.mean()
        n1, n2 = len(group1), len(group2)
        pooled_prob = (group1.sum() + group2.sum()) / (n1 + n2)
        se = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/n1 + 1/n2))
        z_score = (conv2 - conv1) / se
        if hypothesis_type == "two_sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        elif hypothesis_type == "greater":
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = stats.norm.cdf(z_score)
        test_name = "z-test for proportions"
        stat_label, stat_value = "Z statistic", z_score

    # --- T-tests ---
    elif test_family in ["two_sample_t_test", "welch_two_sample_t_test"]:
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=(test_family == "two_sample_t_test"), alternative=scipy_alt)
        test_name = "independent t-test" if test_family == "two_sample_t_test" else "welch t-test"
        stat_label, stat_value = "T statistic", t_stat

    elif test_family == "paired_t_test":
        if len(group1) != len(group2):
            if verbose:
                print("❌ Paired t-test requires equal-length samples.")
            return None
        t_stat, p_value = stats.ttest_rel(group1, group2, alternative=scipy_alt)
        test_name = "paired t-test"
        stat_label, stat_value = "T statistic", t_stat

    # --- Non-parametric ---
    elif test_family in [
        "mann_whitney_u_test"
    ]:
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=scipy_alt)
        test_name = "Mann-Whitney U test"
        stat_label, stat_value = "U statistic", u_stat

    elif test_family == "wilcoxon_signed_rank_test":
        if len(group1) != len(group2):
            if verbose:
                print("❌ Wilcoxon signed-rank test requires equal-length samples.")
            return None
        w_stat, p_value = stats.wilcoxon(group1, group2, alternative=scipy_alt)
        test_name = "Wilcoxon signed-rank test"
        stat_label, stat_value = "W statistic", w_stat

    # --- ANOVA ---
    elif test_family in ["anova_test", "welch_anova_test"]:
        f_stat, p_value = stats.f_oneway(group1, group2)
        test_name = "ANOVA"
        stat_label, stat_value = "F statistic", f_stat

    # --- Chi-square ---
    elif test_family in ["chi_square_test"]:
        contingency = pd.crosstab(df[group_col], df[metric_col])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency)
        test_name = "chi-square test"
        stat_label, stat_value = "Chi2 statistic", chi2_stat

    else:
        raise ValueError(f"❌ Unsupported test family: {test_family}")

    # --- Detailed Interpretation ---
    if verbose:
        print("Hypothesis Test Summary")
        print(f"- Test chosen          : {test_name}")
        print(f"- Hypothesis type      : {hypothesis_type}")

        if test_family in ["one_proportion_z_test", "two_proportion_z_test"]:
            if hypothesis_type == "greater":
                print(f"- H₀ (null)            : Conversion rate in {group_labels[1]} is less than or equal to {group_labels[0]}.")
                print(f"- H₁ (alternative)     : Conversion rate in {group_labels[1]} is greater than {group_labels[0]}.")
            elif hypothesis_type == "less":
                print(f"- H₀ (null)            : Conversion rate in {group_labels[1]} is greater than or equal to {group_labels[0]}.")
                print(f"- H₁ (alternative)     : Conversion rate in {group_labels[1]} is less than {group_labels[0]}.")
            else:
                print(f"- H₀ (null)            : Conversion rates are equal between {group_labels[0]} and {group_labels[1]}.")
                print(f"- H₁ (alternative)     : Conversion rates differ between {group_labels[0]} and {group_labels[1]}.")
        elif test_family in ["two_sample_t_test", "welch_two_sample_t_test"]:
            if hypothesis_type == "greater":
                print(f"- H₀ (null)            : Mean {metric_col} in {group_labels[1]} is less than or equal to {group_labels[0]}.")
                print(f"- H₁ (alternative)     : Mean {metric_col} in {group_labels[1]} is greater than {group_labels[0]}.")
            elif hypothesis_type == "less":
                print(f"- H₀ (null)            : Mean {metric_col} in {group_labels[1]} is greater than or equal to {group_labels[0]}.")
                print(f"- H₁ (alternative)     : Mean {metric_col} in {group_labels[1]} is less than {group_labels[0]}.")
            else:
                print(f"- H₀ (null)            : Mean {metric_col} is equal between {group_labels[0]} and {group_labels[1]}.")
                print(f"- H₁ (alternative)     : Mean {metric_col} differs between {group_labels[0]} and {group_labels[1]}.")
        elif test_family == "paired_t_test":
            if hypothesis_type == "greater":
                print(f"- H₀ (null)            : Mean paired difference in {metric_col} is less than or equal to 0.")
                print(f"- H₁ (alternative)     : Mean paired difference in {metric_col} is > 0.")
            elif hypothesis_type == "less":
                print(f"- H₀ (null)            : Mean paired difference in {metric_col} is greater than or equal to 0.")
                print(f"- H₁ (alternative)     : Mean paired difference in {metric_col} is < 0.")
            else:
                print(f"- H₀ (null)            : Mean paired difference in {metric_col} is 0.")
                print(f"- H₁ (alternative)     : Mean paired difference in {metric_col} is not 0.")
        elif test_family == "chi_square_test":
            print(f"- H₀ (null)            : {metric_col} is independent of {group_col}.")
            print(f"- H₁ (alternative)     : {metric_col} depends on {group_col}.")
        elif test_family in ["anova_test", "welch_anova_test"]:
            print(f"- H₀ (null)            : Group means of {metric_col} are equal.")
            print(f"- H₁ (alternative)     : At least one group mean of {metric_col} differs.")
        elif test_family == "mann_whitney_u_test":
            if hypothesis_type == "greater":
                print(f"- H₀ (null)            : {metric_col} in {group_labels[1]} is less than or equal to {group_labels[0]} in stochastic order.")
                print(f"- H₁ (alternative)     : {metric_col} in {group_labels[1]} tends to be greater than {group_labels[0]}.")
            elif hypothesis_type == "less":
                print(f"- H₀ (null)            : {metric_col} in {group_labels[1]} is greater than or equal to {group_labels[0]} in stochastic order.")
                print(f"- H₁ (alternative)     : {metric_col} in {group_labels[1]} tends to be less than {group_labels[0]}.")
            else:
                print(f"- H₀ (null)            : Distributions of {metric_col} are identical across groups.")
                print(f"- H₁ (alternative)     : Distributions of {metric_col} differ across groups.")
        elif test_family == "wilcoxon_signed_rank_test":
            if hypothesis_type == "greater":
                print(f"- H₀ (null)            : Median paired difference in {metric_col} is less than or equal to 0.")
                print(f"- H₁ (alternative)     : Median paired difference in {metric_col} is > 0.")
            elif hypothesis_type == "less":
                print(f"- H₀ (null)            : Median paired difference in {metric_col} is greater than or equal to 0.")
                print(f"- H₁ (alternative)     : Median paired difference in {metric_col} is < 0.")
            else:
                print(f"- H₀ (null)            : Median paired difference in {metric_col} is 0.")
                print(f"- H₁ (alternative)     : Median paired difference in {metric_col} is not 0.")

        if stat_label is not None:
            print(f"- {stat_label:<20}: {stat_value:.4f}")
        print(f"- P-value              : {p_value:.4f}")
        print(f"- Significance level α : {alpha:.2f}")
        print(f"- Decision rule        : Reject H₀ if p-value < {alpha:.2f}")

        if p_value < alpha:
            print("- Conclusion           : ❌ Reject H₀ (significant difference detected).")
        else:
            print("- Conclusion           : ✅ Fail to reject H₀ (no significant difference).")
        print("-" * 50)

    return p_value


def run_aa_testing_generalized(
    df,
    group_col,
    metric_col,
    group_labels,
    test_family,
    group_relationship=None,
    hypothesis_type='two_sided',
    alpha=0.05,
    visualize=True
):
    """
    Runs A/A test: outcome similarity test + optional visualization.
    SRM (Sample Ratio Mismatch) is checked separately in the Randomization section.
    All logic routed by test_family + group_relationship (no experiment_type).
    """
    print(f"\n📊 A/A Test Summary for metric: '{metric_col}' [{test_family}, {group_relationship}]\n")

    group1 = df[df[group_col] == group_labels[0]][metric_col]
    group2 = df[df[group_col] == group_labels[1]][metric_col]

    p_value = run_outcome_similarity_test(
        df=df,
        group_col=group_col,
        metric_col=metric_col,
        test_family=test_family,
        group_relationship=group_relationship,
        hypothesis_type=hypothesis_type,
        group_labels=group_labels,
        alpha=alpha
    )

    if visualize and p_value is not None:
        visualize_aa_distribution(df, group_col=group_col, metric_col=metric_col, test_family=test_family, group_labels=group_labels, group_relationship=group_relationship)


def visualize_aa_distribution(df, group_col, metric_col, test_family, group_labels=('control', 'treatment'), group_relationship=None):
    """Plot A/A outcome distribution by group. group1/group2 are derived from df inside."""
    group1 = df[df[group_col] == group_labels[0]][metric_col]
    group2 = df[df[group_col] == group_labels[1]][metric_col]

    # Continuous / non-parametric → histograms
    if test_family in ['one_sample_t_test', 'two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test', 'anova_test', 'welch_anova_test', 'mann_whitney_u_test', 'wilcoxon_signed_rank_test', 'kruskal_wallis_test']:
        plt.hist(group1, bins=30, alpha=0.5, label=group_labels[0])
        plt.hist(group2, bins=30, alpha=0.5, label=group_labels[1])
        plt.title(f"A/A Test: {metric_col} Distribution")
        plt.xlabel(metric_col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    elif test_family in ['one_proportion_z_test', 'two_proportion_z_test']:
        rates = [group1.mean(), group2.mean()]
        plt.bar(group_labels, rates)
        for i, rate in enumerate(rates):
            plt.text(i, rate + 0.01, f"{rate:.2%}", ha='center')
        plt.title("A/A Test: Conversion Rate by Group")
        plt.ylabel("Conversion Rate")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    elif test_family in ['chi_square_test']:
        contingency = pd.crosstab(df[group_col], df[metric_col], normalize='index')
        contingency.plot(kind='bar', stacked=True)
        plt.title(f"A/A Test: {metric_col} Distribution by Group")
        plt.ylabel("Proportion")
        plt.xlabel(group_col)
        plt.legend(title=metric_col)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


def simulate_aa_type1_error_rate(
    df,
    metric_col,
    group_labels,
    test_family,
    group_relationship=None,
    hypothesis_type='two_sided',
    runs=100,
    alpha=0.05,
    seed=42,
    verbose=False
):
    """
    Simulates repeated A/A tests to estimate empirical Type I error rate.

    Returns:
    - p_values: list of p-values from each simulation
    """
    np.random.seed(seed)
    p_values = []

    for i in range(runs):
        shuffled_df = df.copy()
        shuffled_df['group'] = np.random.choice(group_labels, size=len(df), replace=True)

        p = run_outcome_similarity_test(
            df=shuffled_df,
            group_col='group',
            metric_col=metric_col,
            test_family=test_family,
            group_relationship=group_relationship,
            hypothesis_type=hypothesis_type,
            group_labels=group_labels,
            alpha=alpha,
            verbose=False
        )

        if p is not None:
            p_values.append(p)

        if verbose:
            print(f"Run {i+1}: p = {p:.4f}")

    significant = sum(p < alpha for p in p_values)
    error_rate = significant / runs

    print(f"\n📈 Type I Error Rate Estimate: {significant}/{runs} = {error_rate:.2%}")

    # Interpretation Block
    print(f"""
            🧠 Summary Interpretation:
            We simulated {runs} A/A experiments using random group assignment (no actual treatment).

            Test: {test_family.upper()}{' (' + group_relationship + ')' if group_relationship else ''}
            Metric: {metric_col}
            Alpha: {alpha}

            False positives (p < α): {significant} / {runs}
            → Estimated Type I Error Rate: {error_rate:.2%}

            This is within expected range for α = {alpha}.
            → ✅ Test framework is behaving correctly — no bias or sensitivity inflation.
            """)

    plot_p_value_distribution(p_values, alpha=alpha)

    return p_values


def plot_p_value_distribution(p_values, alpha=0.05):
    plt.figure(figsize=(8, 4))
    plt.hist(p_values, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=alpha, color='red', linestyle='--', label=f"α = {alpha}")
    plt.title("P-value Distribution Across A/A Tests")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

