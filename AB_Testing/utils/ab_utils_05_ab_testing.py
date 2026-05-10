# 05 A/B Testing — run test, summarize, plot, CIs, lift, final summary, estimate duration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from statsmodels.stats.contingency_tables import mcnemar


def print_hypothesis(test_config):
    """
    Print null and alternative hypotheses based on experiment setup.
    """
    metric = test_config['outcome_metric_col']
    datatype = test_config['outcome_metric_datatype']
    group1, group2 = test_config['group_labels']
    relationship = test_config['group_relationship']
    hypothesis_type = test_config.get('hypothesis_type', 'two_sided')

    print("\n🧠 Hypothesis Definition")
    print("-" * 40)

    if datatype == 'continuous':
        if relationship == 'paired':
            print(f"H₀: Mean paired difference in {metric} is 0.")
            if hypothesis_type == 'greater':
                print(f"H₁: Mean paired difference in {metric} is > 0.")
            elif hypothesis_type == 'less':
                print(f"H₁: Mean paired difference in {metric} is < 0.")
            else:
                print(f"H₁: Mean paired difference in {metric} is not 0.")
        else:
            print(f"H₀: Mean {metric} is equal between {group1} and {group2}.")
            if hypothesis_type == 'greater':
                print(f"H₁: Mean {metric} in {group2} is greater than {group1}.")
            elif hypothesis_type == 'less':
                print(f"H₁: Mean {metric} in {group2} is less than {group1}.")
            else:
                print(f"H₁: Mean {metric} differs between {group1} and {group2}.")

    elif datatype == 'binary':
        print(f"H₀: Conversion rate is equal between {group1} and {group2}.")
        if hypothesis_type == 'greater':
            print(f"H₁: Conversion rate in {group2} is greater than {group1}.")
        elif hypothesis_type == 'less':
            print(f"H₁: Conversion rate in {group2} is less than {group1}.")
        else:
            print(f"H₁: Conversion rate differs between {group1} and {group2}.")

    elif datatype == 'categorical':
        print(f"H₀: Distribution of {metric} is independent of group assignment.")
        print(f"H₁: Distribution of {metric} depends on group assignment.")

    else:
        print("⚠️ Unsupported outcome metric datatype.")

    print("-" * 40)


def run_ab_test(
    df,
    group_col,
    metric_col,
    group_labels,
    test_family,
    group_relationship=None,
    hypothesis_type='two_sided',
    alpha=0.05
):
    """
    Runs the correct statistical test based on test_family + group_relationship combo.

    Returns:
    - result dict with summary stats, test used, p-value, and test-specific values
    """
    test_family = test_family
    group1, group2 = group_labels
    data1 = df[df[group_col] == group1][metric_col]
    data2 = df[df[group_col] == group2][metric_col]
    alt_map = {'two_sided': 'two-sided', 'greater': 'greater', 'less': 'less'}
    if hypothesis_type not in alt_map:
        raise ValueError("hypothesis_type must be one of: 'two_sided', 'greater', 'less'")
    scipy_alt = alt_map[hypothesis_type]

    result = {
        'test_family': test_family,
        'group_relationship': group_relationship,
        'hypothesis_type': hypothesis_type,
        'group_labels': group_labels,
        'alpha': alpha,
        'summary': {}
    }

    # --- Summary Stats ---
    def _safe_mean(x):
        return x.mean() if pd.api.types.is_numeric_dtype(x) else None

    def _safe_std(x):
        return x.std() if pd.api.types.is_numeric_dtype(x) else None

    result['summary'][group1] = {
        'n': len(data1),
        'mean': _safe_mean(data1),
        'std': _safe_std(data1) if test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test', 'mann_whitney_u_test', 'wilcoxon_signed_rank_test'] else None,
        'sum': data1.sum() if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] else None
    }
    result['summary'][group2] = {
        'n': len(data2),
        'mean': _safe_mean(data2),
        'std': _safe_std(data2) if test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test', 'mann_whitney_u_test', 'wilcoxon_signed_rank_test'] else None,
        'sum': data2.sum() if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] else None
    }

    # --- Binary Proportions (Z-Test) ---
    if test_family in ['one_proportion_z_test', 'two_proportion_z_test']:
        x1, n1 = data1.sum(), len(data1)
        x2, n2 = data2.sum(), len(data2)
        p_pooled = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_stat = (x2/n2 - x1/n1) / se
        if hypothesis_type == 'two_sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif hypothesis_type == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)
        result.update({'test': 'two-proportion z-test', 'z_stat': z_stat, 'p_value': p_value})

    # --- Continuous (T-Test) ---
    elif test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test']:
        if test_family == 'paired_t_test':
            group_relationship = 'paired'
            result['group_relationship'] = 'paired'

        if group_relationship == 'independent':
            # Welch for explicit welch_two_sample_t_test, classic pooled for two_sample_t_test
            equal_var = test_family == 'two_sample_t_test'
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=scipy_alt)
            result.update(
                {
                    'test': 'welch t-test' if not equal_var else 'independent t-test',
                    't_stat': t_stat,
                    'p_value': p_value
                }
            )
        elif group_relationship == 'paired':
            if len(data1) != len(data2):
                min_n = min(len(data1), len(data2))
                data1 = data1.iloc[:min_n]
                data2 = data2.iloc[:min_n]
                result['pairing_warning'] = (
                    f"Unequal group sizes for paired test; aligned to first {min_n} observations per group."
                )
            t_stat, p_value = stats.ttest_rel(data1, data2, alternative=scipy_alt)
            result.update({'test': 'paired t-test', 't_stat': t_stat, 'p_value': p_value})
        else:
            raise ValueError("Missing or invalid group_relationship for t-test.")

    # --- Continuous (Non-parametric) ---
    elif test_family in ['mann_whitney_u_test']:
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative=scipy_alt)
        result.update({'test': 'Mann-Whitney U Test', 'u_stat': u_stat, 'p_value': p_value})
    elif test_family == 'wilcoxon_signed_rank_test':
        if len(data1) != len(data2):
            min_n = min(len(data1), len(data2))
            data1 = data1.iloc[:min_n]
            data2 = data2.iloc[:min_n]
            result['pairing_warning'] = (
                f"Unequal group sizes for paired test; aligned to first {min_n} observations per group."
            )
        w_stat, p_value = stats.wilcoxon(data1, data2, alternative=scipy_alt)
        result.update({'test': 'Wilcoxon signed-rank test', 'w_stat': w_stat, 'p_value': p_value})

    # --- Categorical (Chi-square) ---
    elif test_family in ['chi_square_test']:
        contingency = pd.crosstab(df[group_col], df[metric_col])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        result.update({'test': 'chi-square test', 'chi2_stat': chi2, 'p_value': p_value})

    # --- Paired Binary (McNemar) ---
    elif test_family == 'mcnemar_test':
        group_relationship = 'paired'
        result['group_relationship'] = 'paired'

        min_n = min(len(data1), len(data2))
        x = data1.iloc[:min_n].astype(int).to_numpy()
        y = data2.iloc[:min_n].astype(int).to_numpy()
        x = np.where(x > 0, 1, 0)
        y = np.where(y > 0, 1, 0)

        table = np.zeros((2, 2), dtype=int)
        table[0, 0] = int(np.sum((x == 0) & (y == 0)))
        table[0, 1] = int(np.sum((x == 0) & (y == 1)))
        table[1, 0] = int(np.sum((x == 1) & (y == 0)))
        table[1, 1] = int(np.sum((x == 1) & (y == 1)))

        mc = mcnemar(table, exact=True, correction=False)
        result.update(
            {
                'test': 'McNemar test',
                'mcnemar_stat': float(mc.statistic),
                'p_value': float(mc.pvalue),
                'contingency_table': table.tolist(),
                'paired_n': int(min_n),
            }
        )

    else:
        raise ValueError(f"❌ Unsupported test_family: {test_family}")

    return result


def summarize_ab_test_result(result):
    """
    Prints A/B test results summary with statistical test outputs and final verdict.
    """
    test_family = result['test_family']
    group_relationship = result.get('group_relationship')
    hypothesis_type = result.get('hypothesis_type', 'two_sided')
    group1, group2 = result['group_labels']
    p_value = result.get('p_value')
    alpha = result.get('alpha', 0.05)

    print("\n" + "="*45)
    print(f"🧪 A/B Test Result Summary [{test_family.upper()}]")
    print("="*45)

    # ---- Hypothesis Test Output ----
    print("\n📊 Hypothesis Test Result")
    print(f"Test used: {result.get('test', 'N/A')}")
    print(f"Hypothesis type: {hypothesis_type}")
    if 'z_stat' in result:
        print(f"Z-statistic: {result['z_stat']:.4f}")
    elif 't_stat' in result:
        print(f"T-statistic: {result['t_stat']:.4f}")
    elif 'chi2_stat' in result:
        print(f"Chi2-statistic: {result['chi2_stat']:.4f}")
    elif 'u_stat' in result:
        print(f"U-statistic: {result['u_stat']:.4f}")
    elif 'mcnemar_stat' in result:
        print(f"McNemar statistic: {result['mcnemar_stat']:.4f}")

    if p_value is not None:
        print(f"P-value    : {p_value:.4f}")
        print("✅ Statistically significant difference detected." if p_value < alpha else "🚫 No significant difference detected.")
    else:
        print("⚠️ P-value not found.")

    # ---- Summary Table ----
    print("\n📋 Group Summary:\n")
    display(pd.DataFrame(result['summary']).T)

    print("="*45 + "\n")

    # ---- Final Summary Block (moved from print_final_ab_test_summary) ----
    print("="*40)
    print("          📊 FINAL A/B TEST SUMMARY")
    print("="*40)

    if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] or (test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test'] and group_relationship == 'independent'):
        mean1 = result['summary'][group1]['mean']
        mean2 = result['summary'][group2]['mean']
        lift = mean2 - mean1
        pct_lift = lift / mean1 if mean1 else np.nan

        label = "Conversion rate" if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] else "Avg outcome"
        test_name = result.get("test", "A/B test")

        print(f"👥  {group1.capitalize()} {label:<20}:  {mean1:.4f}")
        print(f"🧪  {group2.capitalize()} {label:<20}:  {mean2:.4f}")
        print(f"📈  Absolute lift              :  {lift:.4f}")
        print(f"📊  Percentage lift            :  {pct_lift:.2%}")
        print(f"🧪  P-value (from {test_name}) :  {p_value:.4f}")

    elif test_family == 'mann_whitney_u_test':
        mean1 = result['summary'][group1]['mean']
        mean2 = result['summary'][group2]['mean']
        lift = mean2 - mean1
        pct_lift = lift / mean1 if mean1 else np.nan
        test_name = result.get("test", "Mann-Whitney U test")
        print(f"👥  {group1.capitalize()} Avg outcome         :  {mean1:.4f}")
        print(f"🧪  {group2.capitalize()} Avg outcome         :  {mean2:.4f}")
        print(f"📈  Absolute lift              :  {lift:.4f}")
        print(f"📊  Percentage lift            :  {pct_lift:.2%}")
        print(f"🧪  P-value (from {test_name}) :  {p_value:.4f}")

    elif test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test'] and group_relationship == 'paired':
        print("🧪 Paired T-Test was used to compare within-user outcomes.")
        print(f"🧪 P-value: {p_value:.4f}")

    elif test_family == 'mcnemar_test':
        print("🧪 McNemar test was used for paired binary outcomes.")
        if "paired_n" in result:
            print(f"👥 Paired observations analyzed: {result['paired_n']}")
        print(f"🧪 P-value: {p_value:.4f}")

    elif test_family == 'chi_square_test':
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


def plot_ab_test_results(result):
    """
    Plots A/B test results by group mean or distribution depending on test family.
    """
    test_family = result['test_family']
    group_relationship = result.get('group_relationship')
    group1, group2 = result['group_labels']

    print("\n📊 Visualization:")

    if test_family in ['one_proportion_z_test', 'two_proportion_z_test', 'two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test', 'mann_whitney_u_test', 'mcnemar_test']:
        labels = [group1, group2]
        values = [result['summary'][group1]['mean'], result['summary'][group2]['mean']]
        plt.bar(labels, values, color=['gray', 'skyblue'])

        for i, val in enumerate(values):
            label = f"{val:.2%}" if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] else f"{val:.2f}"
            plt.text(i, val + 0.01, label, ha='center')

        ylabel = "Conversion Rate" if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] else "Average Value"
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Group")
        plt.ylim(0, max(values) * 1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    elif test_family == 'chi_square_test':
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
    group_relationship = result.get('group_relationship')
    group1, group2 = result['group_labels']
    summary = result['summary']

    if test_family not in ['one_proportion_z_test', 'two_proportion_z_test', 'two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test', 'mann_whitney_u_test']:
        print(f"⚠️ CI plotting not supported for test family: {test_family}")
        return
    if test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test'] and group_relationship != 'independent':
        print(f"⚠️ CI plotting only supported for independent t-tests.")
        return

    p1, p2 = summary[group1]['mean'], summary[group2]['mean']
    n1, n2 = summary[group1]['n'], summary[group2]['n']

    if test_family in ['one_proportion_z_test', 'two_proportion_z_test']:
        se1 = np.sqrt(p1 * (1 - p1) / n1)
        se2 = np.sqrt(p2 * (1 - p2) / n2)
        ylabel = "Conversion Rate"
    else:
        # two-sample t / Mann-Whitney: CI for mean
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
    group_relationship = result.get('group_relationship')
    group1, group2 = result['group_labels']
    alpha = result.get('alpha', 0.05)
    z = 1.96

    print("\n" + "="*45)
    print(f"📈 95% CI for Difference in Outcome [{test_family}]")
    print("="*45)

    if test_family in ['one_proportion_z_test', 'two_proportion_z_test'] or (test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test'] and group_relationship == 'independent'):
        m1 = result['summary'][group1]['mean']
        m2 = result['summary'][group2]['mean']
        lift = m2 - m1
        n1 = result['summary'][group1]['n']
        n2 = result['summary'][group2]['n']

        if test_family in ['one_proportion_z_test', 'two_proportion_z_test']:
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

    elif test_family == 'mann_whitney_u_test':
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

    elif test_family in ['two_sample_t_test', 'welch_two_sample_t_test', 'paired_t_test'] and group_relationship == 'paired':
        print("- Paired test: CI already accounted for in test logic.")

    elif test_family == 'mcnemar_test':
        print("- McNemar test: CI for paired binary shift is not implemented.")

    elif test_family == 'chi_square_test':
        print("- Categorical test: per-category lift analysis required (not implemented).")

    print("="*45 + "\n")


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

