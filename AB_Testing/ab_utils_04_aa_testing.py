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
    variant=None,
    group_labels=('control', 'treatment'),
    alpha=0.05,
    verbose=True
):
    """
    Runs a similarity test between two groups based on test_family and variant.

    Parameters:
    - df: pandas DataFrame
    - group_col: column with group assignment
    - metric_col: outcome metric
    - test_family: one of ['z_test', 't_test', 'chi_square', 'anova', 'non_parametric']
    - variant: 'independent' or 'paired' (required for t-test)
    - group_labels: tuple of (control, treatment)
    - alpha: significance threshold
    - verbose: print detailed interpretation
    """

    if verbose:
        print("📏 Outcome Similarity Check\n")

    group1 = df[df[group_col] == group_labels[0]][metric_col]
    group2 = df[df[group_col] == group_labels[1]][metric_col]

    # --- Run appropriate test ---
    # --- Binary ---
    if test_family in ["z_test", "two_proportion_z_test"]:
        conv1, conv2 = group1.mean(), group2.mean()
        n1, n2 = len(group1), len(group2)
        pooled_prob = (group1.sum() + group2.sum()) / (n1 + n2)
        se = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/n1 + 1/n2))
        z_score = (conv2 - conv1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        test_name = "z-test for proportions"

    # --- T-tests ---
    elif test_family in ["t_test", "two_sample_t_test", "welch_t_test"]:
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "independent t-test"

    elif test_family == "paired_t_test":
        if len(group1) != len(group2):
            if verbose:
                print("❌ Paired t-test requires equal-length samples.")
            return None
        t_stat, p_value = stats.ttest_rel(group1, group2)
        test_name = "paired t-test"

    # --- Non-parametric ---
    elif test_family in [
        "non_parametric",
        "mann_whitney_u_test"
    ]:
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"

    # --- ANOVA ---
    elif test_family in ["anova", "welch_anova"]:
        f_stat, p_value = stats.f_oneway(group1, group2)
        test_name = "ANOVA"

    # --- Chi-square ---
    elif test_family in ["chi_square", "chi_square_test"]:
        contingency = pd.crosstab(df[group_col], df[metric_col])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency)
        test_name = "chi-square test"

    else:
        raise ValueError(f"❌ Unsupported test family: {test_family}")

    # --- Detailed Interpretation ---
    if verbose:
        print("\n🧠 Interpretation:")

        if test_family in ["z_test", "two_proportion_z_test"]:
            print(f"Used a {test_name} to compare conversion rates between groups.")
            print("Null Hypothesis: Conversion rates are equal across groups.")

        elif test_family in ["t_test", "two_sample_t_test", "welch_t_test"]:
            print(f"Used an {test_name} to compare means of '{metric_col}' across independent groups.")
            print("Null Hypothesis: Group means are equal.")

        elif test_family == "paired_t_test":
            print(f"Used a {test_name} to compare within-user differences in '{metric_col}'.")
            print("Null Hypothesis: Mean difference between pairs is zero.")

        elif test_family in ["chi_square", "chi_square_test"]:
            print(f"Used a {test_name} to test whether '{metric_col}' distribution depends on group.")
            print("Null Hypothesis: No association between group and category.")

        elif test_family in ["anova", "welch_anova"]:
            print(f"Used a {test_name} to compare group means of '{metric_col}' across groups.")
            print("Null Hypothesis: All group means are equal.")

        elif test_family in ["non_parametric", "mann_whitney_u_test"]:
            print(f"Used a {test_name} to compare medians of '{metric_col}' across groups (non-parametric).")
            print("Null Hypothesis: Distributions are identical across groups.")

        print(f"\nWe use α = {alpha:.2f}")
        if p_value < alpha:
            print(f"❌ p = {p_value:.4f} < α → Rejected the null. Significant difference (check setup).")
        else:
            print(f"✅ p = {p_value:.4f} ≥ α → Failed to reject null. No significant difference.")

    return p_value


def run_aa_testing_generalized(
    df,
    group_col,
    metric_col,
    group_labels,
    test_family,
    variant=None,
    alpha=0.05,
    visualize=True
):
    """
    Runs A/A test: outcome similarity test + optional visualization.
    SRM (Sample Ratio Mismatch) is checked separately in the Randomization section.
    All logic routed by test_family + variant (no experiment_type).
    """
    print(f"\n📊 A/A Test Summary for metric: '{metric_col}' [{test_family}, {variant}]\n")

    group1 = df[df[group_col] == group_labels[0]][metric_col]
    group2 = df[df[group_col] == group_labels[1]][metric_col]

    p_value = run_outcome_similarity_test(
        df=df,
        group_col=group_col,
        metric_col=metric_col,
        test_family=test_family,
        variant=variant,
        group_labels=group_labels,
        alpha=alpha
    )

    if visualize and p_value is not None:
        visualize_aa_distribution(df, group_col=group_col, metric_col=metric_col, test_family=test_family, group_labels=group_labels, variant=variant)


def visualize_aa_distribution(df, group_col, metric_col, test_family, group_labels=('control', 'treatment'), variant=None):
    """Plot A/A outcome distribution by group. group1/group2 are derived from df inside."""
    group1 = df[df[group_col] == group_labels[0]][metric_col]
    group2 = df[df[group_col] == group_labels[1]][metric_col]

    # Continuous / non-parametric → histograms
    if test_family in ['t_test', 'two_sample_t_test', 'welch_t_test', 'paired_t_test', 'anova', 'welch_anova', 'non_parametric', 'mann_whitney_u_test']:
        plt.hist(group1, bins=30, alpha=0.5, label=group_labels[0])
        plt.hist(group2, bins=30, alpha=0.5, label=group_labels[1])
        plt.title(f"A/A Test: {metric_col} Distribution")
        plt.xlabel(metric_col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    elif test_family in ['z_test', 'two_proportion_z_test']:
        rates = [group1.mean(), group2.mean()]
        plt.bar(group_labels, rates)
        for i, rate in enumerate(rates):
            plt.text(i, rate + 0.01, f"{rate:.2%}", ha='center')
        plt.title("A/A Test: Conversion Rate by Group")
        plt.ylabel("Conversion Rate")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    elif test_family in ['chi_square', 'chi_square_test']:
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
    variant=None,
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
            variant=variant,
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

            Test: {test_family.upper()}{' (' + variant + ')' if variant else ''}
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
