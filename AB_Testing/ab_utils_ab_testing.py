# A/B Testing — run test, summarize, plot, CIs, lift, final summary, estimate duration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display


def run_ab_test(
    df,
    group_col,
    metric_col,
    group_labels,
    test_family,
    variant=None,
    alpha=0.05
):
    """
    Runs the correct statistical test based on test_family + variant combo.

    Returns:
    - result dict with summary stats, test used, p-value, and test-specific values
    """
    group1, group2 = group_labels
    data1 = df[df[group_col] == group1][metric_col]
    data2 = df[df[group_col] == group2][metric_col]

    result = {
        'test_family': test_family,
        'variant': variant,
        'group_labels': group_labels,
        'alpha': alpha,
        'summary': {}
    }

    # --- Summary Stats ---
    result['summary'][group1] = {
        'n': len(data1),
        'mean': data1.mean(),
        'std': data1.std() if test_family in ['t_test', 'non_parametric', 'mann_whitney_u_test'] else None,
        'sum': data1.sum() if test_family == 'z_test' else None
    }
    result['summary'][group2] = {
        'n': len(data2),
        'mean': data2.mean(),
        'std': data2.std() if test_family in ['t_test', 'non_parametric', 'mann_whitney_u_test'] else None,
        'sum': data2.sum() if test_family == 'z_test' else None
    }

    # --- Binary Proportions (Z-Test) ---
    if test_family == 'z_test':
        x1, n1 = data1.sum(), len(data1)
        x2, n2 = data2.sum(), len(data2)
        p_pooled = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_stat = (x2/n2 - x1/n1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        result.update({'test': 'z-test for proportions', 'z_stat': z_stat, 'p_value': p_value})

    # --- Continuous (T-Test) ---
    elif test_family == 't_test':
        if variant == 'independent':
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            result.update({'test': 'independent t-test', 't_stat': t_stat, 'p_value': p_value})
        elif variant == 'paired':
            if len(data1) != len(data2):
                raise ValueError("Paired test requires equal-length matching samples.")
            t_stat, p_value = stats.ttest_rel(data1, data2)
            result.update({'test': 'paired t-test', 't_stat': t_stat, 'p_value': p_value})
        else:
            raise ValueError("Missing or invalid variant for t-test.")

    # --- Continuous (Non-parametric) ---
    elif test_family == 'non_parametric':
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        result.update({'test': 'Mann-Whitney U Test', 'u_stat': u_stat, 'p_value': p_value})

    # --- Mann-Whitney U (explicit test family) ---
    elif test_family == 'mann_whitney_u_test':
        alternative = variant if variant in ('two-sided', 'less', 'greater') else 'two-sided'
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
        result.update({'test': 'Mann-Whitney U test', 'u_stat': u_stat, 'p_value': p_value})

    # --- Categorical (Chi-square) ---
    elif test_family == 'chi_square':
        contingency = pd.crosstab(df[group_col], df[metric_col])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        result.update({'test': 'chi-square test', 'chi2_stat': chi2, 'p_value': p_value})

    else:
        raise ValueError(f"❌ Unsupported test_family: {test_family}")

    return result


def summarize_ab_test_result(result):
    """
    Prints A/B test results summary with statistical test outputs and lift analysis.
    """
    test_family = result['test_family']
    variant = result.get('variant')
    group1, group2 = result['group_labels']
    p_value = result.get('p_value')
    alpha = result.get('alpha', 0.05)

    print("\n" + "="*45)
    print(f"🧪 A/B Test Result Summary [{test_family.upper()}]")
    print("="*45)

    # ---- Hypothesis Test Output ----
    print("\n📊 Hypothesis Test Result")
    print(f"Test used: {result.get('test', 'N/A')}")
    if 'z_stat' in result:
        print(f"Z-statistic: {result['z_stat']:.4f}")
    elif 't_stat' in result:
        print(f"T-statistic: {result['t_stat']:.4f}")
    elif 'chi2_stat' in result:
        print(f"Chi2-statistic: {result['chi2_stat']:.4f}")
    elif 'u_stat' in result:
        print(f"U-statistic: {result['u_stat']:.4f}")

    if p_value is not None:
        print(f"P-value    : {p_value:.4f}")
        print("✅ Statistically significant difference detected." if p_value < alpha else "🚫 No significant difference detected.")
    else:
        print("⚠️ P-value not found.")

    # ---- Summary Table ----
    print("\n📋 Group Summary:\n")
    display(pd.DataFrame(result['summary']).T)

    # ---- Lift Analysis (for Z-test, T-test independent, or non-parametric / Mann-Whitney) ----
    if test_family in ['z_test', 't_test', 'non_parametric', 'mann_whitney_u_test'] and (variant == 'independent' or test_family in ['z_test', 'non_parametric', 'mann_whitney_u_test']):
        group1_mean = result['summary'][group1]['mean']
        group2_mean = result['summary'][group2]['mean']
        lift = group2_mean - group1_mean
        pct_lift = lift / group1_mean if group1_mean else np.nan

        print("\n📈 Lift Analysis")
        print(f"- Absolute Lift   : {lift:.4f}")
        print(f"- Percentage Lift : {pct_lift:.2%}")

        try:
            n1 = result['summary'][group1]['n']
            n2 = result['summary'][group2]['n']

            if test_family == 'z_test':
                se = np.sqrt(group1_mean * (1 - group1_mean) / n1 + group2_mean * (1 - group2_mean) / n2)
            else:
                sd1 = result['summary'][group1].get('std')
                sd2 = result['summary'][group2].get('std')
                se = np.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)

            z = 1.96
            ci_low = lift - z * se
            ci_high = lift + z * se
            print(f"- 95% CI for Lift : [{ci_low:.4f}, {ci_high:.4f}]")
        except Exception as e:
            print(f"⚠️ Could not compute confidence interval: {e}")

    print("="*45 + "\n")


def plot_ab_test_results(result):
    """
    Plots A/B test results by group mean or distribution depending on test family.
    """
    test_family = result['test_family']
    variant = result.get('variant')
    group1, group2 = result['group_labels']

    print("\n📊 Visualization:")

    if test_family in ['z_test', 't_test', 'non_parametric', 'mann_whitney_u_test']:
        labels = [group1, group2]
        values = [result['summary'][group1]['mean'], result['summary'][group2]['mean']]
        plt.bar(labels, values, color=['gray', 'skyblue'])

        for i, val in enumerate(values):
            label = f"{val:.2%}" if test_family == 'z_test' else f"{val:.2f}"
            plt.text(i, val + 0.01, label, ha='center')

        ylabel = "Conversion Rate" if test_family == 'z_test' else "Average Value"
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Group")
        plt.ylim(0, max(values) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    elif test_family == 'chi_square':
        dist = pd.DataFrame(result['summary'])
        dist.T.plot(kind='bar', stacked=True)
        plt.title(f"Categorical Distribution by Group")
        plt.ylabel("Proportion")
        plt.xlabel("Group")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()


def plot_confidence_intervals(result, z=1.96):
    """
    Plot 95% confidence intervals for group means (conversion rate or continuous).
    """
    test_family = result['test_family']
    variant = result.get('variant')
    group1, group2 = result['group_labels']
    summary = result['summary']

    if test_family not in ['z_test', 't_test', 'non_parametric', 'mann_whitney_u_test']:
        print(f"⚠️ CI plotting not supported for test family: {test_family}")
        return
    if test_family == 't_test' and variant != 'independent':
        print(f"⚠️ CI plotting only supported for independent t-tests.")
        return

    p1, p2 = summary[group1]['mean'], summary[group2]['mean']
    n1, n2 = summary[group1]['n'], summary[group2]['n']

    if test_family == 'z_test':
        se1 = np.sqrt(p1 * (1 - p1) / n1)
        se2 = np.sqrt(p2 * (1 - p2) / n2)
        ylabel = "Conversion Rate"
    else:
        # t_test (independent), non_parametric, mann_whitney_u_test: CI for mean
        sd1 = summary[group1]['std']
        sd2 = summary[group2]['std']
        se1 = sd1 / np.sqrt(n1)
        se2 = sd2 / np.sqrt(n2)
        ylabel = "Mean Outcome"

    ci1 = (p1 - z * se1, p1 + z * se1)
    ci2 = (p2 - z * se2, p2 + z * se2)

    plt.errorbar([group1, group2],
                 [p1, p2],
                 yerr=[[p1 - ci1[0], p2 - ci2[0]], [ci1[1] - p1, ci2[1] - p2]],
                 fmt='o', capsize=10, color='black')
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} with 95% Confidence Intervals")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def compute_lift_confidence_interval(result):
    """
    Compute CI for lift in binary or continuous-independent tests.
    """
    test_family = result['test_family']
    variant = result.get('variant')
    group1, group2 = result['group_labels']
    alpha = result.get('alpha', 0.05)
    z = 1.96

    print("\n" + "="*45)
    print(f"📈 95% CI for Difference in Outcome [{test_family}]")
    print("="*45)

    if test_family == 'z_test' or (test_family == 't_test' and variant == 'independent'):
        m1 = result['summary'][group1]['mean']
        m2 = result['summary'][group2]['mean']
        lift = m2 - m1
        n1 = result['summary'][group1]['n']
        n2 = result['summary'][group2]['n']

        if test_family == 'z_test':
            se = np.sqrt(m1 * (1 - m1) / n1 + m2 * (1 - m2) / n2)
        else:
            sd1 = result['summary'][group1]['std']
            sd2 = result['summary'][group2]['std']
            se = np.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)

        ci_low = lift - z * se
        ci_high = lift + z * se

        print(f"- Absolute Lift         : {lift:.4f}")
        print(f"- 95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")

        if ci_low > 0:
            print("✅ Likely positive impact (CI > 0)")
        elif ci_high < 0:
            print("🚫 Likely negative impact (CI < 0)")
        else:
            print("🤷 CI includes 0 — not statistically significant.")

    elif test_family in ['non_parametric', 'mann_whitney_u_test']:
        m1 = result['summary'][group1]['mean']
        m2 = result['summary'][group2]['mean']
        lift = m2 - m1
        n1 = result['summary'][group1]['n']
        n2 = result['summary'][group2]['n']
        sd1 = result['summary'][group1].get('std')
        sd2 = result['summary'][group2].get('std')
        if sd1 is not None and sd2 is not None:
            se = np.sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)
            ci_low = lift - z * se
            ci_high = lift + z * se
            print(f"- Absolute Lift (diff in means): {lift:.4f}")
            print(f"- 95% CI for difference        : [{ci_low:.4f}, {ci_high:.4f}]")
            if ci_low > 0:
                print("✅ Likely positive impact (CI > 0)")
            elif ci_high < 0:
                print("🚫 Likely negative impact (CI < 0)")
            else:
                print("🤷 CI includes 0 — not statistically significant.")
        else:
            print("- Mann-Whitney U: CI for difference in means (summary std used).")

    elif test_family == 't_test' and variant == 'paired':
        print("- Paired test: CI already accounted for in test logic.")

    elif test_family == 'chi_square':
        print("- Categorical test: per-category lift analysis required (not implemented).")

    print("="*45 + "\n")


def print_final_ab_test_summary(result):
    """
    Final wrap-up of results with summary stats and verdict.
    """
    test_family = result['test_family']
    variant = result.get('variant')
    group1, group2 = result['group_labels']
    p_value = result.get('p_value')
    alpha = result.get('alpha', 0.05)

    print("="*40)
    print("          📊 FINAL A/B TEST SUMMARY")
    print("="*40)

    if test_family == 'z_test' or (test_family == 't_test' and variant == 'independent'):
        mean1 = result['summary'][group1]['mean']
        mean2 = result['summary'][group2]['mean']
        lift = mean2 - mean1
        pct_lift = lift / mean1 if mean1 else np.nan

        label = "Conversion rate" if test_family == 'z_test' else "Avg outcome"
        test_name = result.get("test", "A/B test")

        print(f"👥  {group1.capitalize()} {label:<20}:  {mean1:.4f}")
        print(f"🧪  {group2.capitalize()} {label:<20}:  {mean2:.4f}")
        print(f"📈  Absolute lift              :  {lift:.4f}")
        print(f"📊  Percentage lift            :  {pct_lift:.2%}")
        print(f"🧪  P-value (from {test_name}) :  {p_value:.4f}")

    elif test_family in ['non_parametric', 'mann_whitney_u_test']:
        mean1 = result['summary'][group1]['mean']
        mean2 = result['summary'][group2]['mean']
        lift = mean2 - mean1
        pct_lift = lift / mean1 if mean1 else np.nan
        test_name = result.get("test", "Mann-Whitney U test")
        print(f"👥  {group1.capitalize()} Avg outcome         :  {mean1:.4f}")
        print(f"🧪  {group2.capitalize()} Avg outcome         :  {mean2:.4f}")
        print(f"📈  Absolute lift              :  {lift:.4f}")
        print(f"📊  Percentage lift            :  {pct_lift:.2%}")
        print(f"🧪  P-value (from {test_name}):  {p_value:.4f}")

    elif test_family == 't_test' and variant == 'paired':
        print("🧪 Paired T-Test was used to compare within-user outcomes.")
        print(f"🧪 P-value: {p_value:.4f}")

    elif test_family == 'chi_square':
        print("🧪 Chi-square test was used to compare categorical distributions.")
        print(f"🧪 P-value: {p_value:.4f}")

    else:
        print("⚠️ Unsupported test type.")

    print("-" * 40)

    if p_value is not None:
        if p_value < alpha:
            print("✅ RESULT: Statistically significant difference detected.")
        else:
            print("❌ RESULT: No statistically significant difference detected.")
    else:
        print("⚠️ No p-value available.")

    print("="*40 + "\n")


def estimate_test_duration(
    required_sample_size_per_group,
    daily_eligible_users,
    allocation_ratios=(0.5, 0.5),
    buffer_days=2,
    test_family=None  # renamed from experiment_type
):
    """
    Estimate test duration based on sample size, traffic, and allocation.

    Parameters:
    - required_sample_size_per_group: int
    - daily_eligible_users: int — total incoming traffic per day
    - allocation_ratios: tuple — traffic share per group (e.g., 50/50)
    - buffer_days: int — extra time for ramp-up or anomalies
    - test_family: str — optional metadata for clarity

    Returns:
    - dict with group durations and total estimated runtime
    """
    group_durations = []
    for alloc in allocation_ratios:
        users_per_day = daily_eligible_users * alloc
        days = required_sample_size_per_group / users_per_day if users_per_day else float('inf')
        group_durations.append(np.ceil(days))

    longest_group_runtime = int(max(group_durations))
    total_with_buffer = longest_group_runtime + buffer_days

    print("\n🧮 Estimated Test Duration")
    if test_family:
        print(f"- Test family               : {test_family}")
    print(f"- Required sample per group : {required_sample_size_per_group}")
    print(f"- Daily eligible traffic    : {daily_eligible_users}")
    print(f"- Allocation ratio          : {allocation_ratios}")
    print(f"- Longest group runtime     : {longest_group_runtime} days")
    print(f"- Buffer days               : {buffer_days}")
    print(f"✅ Total estimated duration : {total_with_buffer} days\n")

    return {
        'test_family': test_family,
        'per_group_days': group_durations,
        'longest_group_runtime': longest_group_runtime,
        'recommended_total_duration': total_with_buffer
    }
