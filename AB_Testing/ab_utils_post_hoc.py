# Post Hoc Analysis — CUPED, segment lift, guardrail, rollout simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
import statsmodels.api as sm
from IPython.display import display
from ab_utils_common import my_seed


def apply_cuped(
    df,
    pre_metric,
    outcome_metric_col,  # observed outcome column (e.g., engagement_score)
    outcome_col=None,
    group_col=None,
    group_labels=None,
    seed=my_seed
):
    """
    Applies CUPED (Controlled Pre-Experiment Data) adjustment to reduce variance
    in the outcome metric using a pre-experiment covariate.

    CUPED is a post-randomization technique that reduces variance by adjusting the
    observed outcome using a baseline (pre-metric) variable that is correlated
    with the outcome.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing experiment data.
    pre_metric : str
        Column name of the pre-experiment covariate (e.g., 'past_purchase_count').
        This is the variable used to compute the adjustment factor (theta).
    outcome_metric_col : str
        Column name of the original observed outcome (e.g., 'engagement_score')
        that you are comparing across groups.
    outcome_col : str, default=None
        Name of the new column where the adjusted outcome will be stored.
    group_col : str
        Column indicating the experiment group assignment (e.g., 'control' vs 'treatment').
    group_labels : tuple
        Tuple containing the names of the experiment groups.
    seed : int
        Random seed for reproducibility (used only if randomness is introduced later).

    Returns:
    -------
    df : pandas.DataFrame
        DataFrame with an additional column [outcome_col] containing the CUPED-adjusted outcome.
    """
    np.random.seed(seed)

    # Step 1: Use actual observed experiment outcome
    y = df[outcome_metric_col].values

    # Step 2: Regress outcome on pre-metric to estimate correction factor (theta)
    X = sm.add_constant(df[[pre_metric]])
    theta = sm.OLS(y, X).fit().params[pre_metric]

    # Step 3: Apply CUPED adjustment and save in new column
    if outcome_col is None:
        outcome_col = f'{outcome_metric_col}_cuped_adjusted'
    df[outcome_col] = y - theta * df[pre_metric]
    if group_col in df.columns:
        df = df[[group_col] + [c for c in df.columns if c != group_col]]

    return df


def visualize_segment_lift(df_segment, segment_col):
    """
    Plots horizontal bar chart of mean lift per segment (Treatment - Control).
    """
    df_viz = df_segment.dropna(subset=['lift']).sort_values(by='lift', ascending=False)
    if df_viz.empty:
        print(f"⚠️ No lift data to visualize for '{segment_col}'\n")
        return

    plt.figure(figsize=(8, 0.4 * len(df_viz) + 2))
    bars = plt.barh(df_viz[segment_col], df_viz['lift'], color='skyblue')
    for bar, val in zip(bars, df_viz['lift']):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{val:.2f}", va='center', ha='left', fontsize=9)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"Lift from Control to Treatment by {segment_col}")
    plt.xlabel("Mean Difference (Treatment – Control)")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def analyze_segment_lift(
    df,
    test_config,
    segment_cols=['platform', 'device_type', 'user_tier', 'region'],
    min_count_per_group=30,
    visualize=True
):
    """
    Post-hoc lift analysis per segment (e.g., by platform or region).
    """

    group_col = 'group'
    group1, group2 = test_config['group_labels']
    metric_col = test_config['outcome_metric_col']
    outcome_type = test_config['outcome_metric_datatype']
    variant = test_config['variant']
    test_family = test_config['family']

    for segment in segment_cols:
        print(f"\n🔎 Segmenting by: {segment}")
        seg_data = []

        for val in df[segment].dropna().unique():
            subset = df[df[segment] == val]
            g1 = subset[subset[group_col] == group1][metric_col]
            g2 = subset[subset[group_col] == group2][metric_col]

            if len(g1) < min_count_per_group or len(g2) < min_count_per_group:
                print(f"⚠️ Skipping '{val}' under '{segment}' — too few users.")
                continue

            lift = g2.mean() - g1.mean()
            p_value = None

            if test_family == 'z_test':
                # Binary: z-test on proportions
                p1, n1 = g1.mean(), len(g1)
                p2, n2 = g2.mean(), len(g2)
                pooled_p = (g1.sum() + g2.sum()) / (n1 + n2)
                se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
                p_value = 2 * (1 - stats.norm.cdf(abs((p2 - p1) / se)))

            elif test_family == 't_test':
                if variant == 'independent':
                    _, p_value = stats.ttest_ind(g1, g2)
                elif variant == 'paired':
                    print(f"⚠️ Paired test not supported in segmented lift — skipped '{val}' under '{segment}'.")
                    lift, p_value = np.nan, None

            elif test_family == 'chi_square':
                print(f"⚠️ Categorical data — lift not defined for '{val}' in '{segment}'.")
                lift, p_value = np.nan, None

            seg_data.append({
                segment: val,
                'count_control': len(g1),
                'count_treatment': len(g2),
                'mean_control': g1.mean(),
                'mean_treatment': g2.mean(),
                'std_control': g1.std(),
                'std_treatment': g2.std(),
                'lift': lift,
                'p_value_lift': p_value
            })

        df_segment = pd.DataFrame(seg_data)
        display(df_segment)

        if visualize:
            visualize_segment_lift(df_segment, segment)


def run_guardrail_analysis(df, test_config, group_col='group', alpha=0.05):
    """
    Run guardrail analysis: evaluate guardrail metric (means, difference, p-value).
    If no guardrail is configured, prints a short message.
    """
    guardrail_col = test_config.get('guardrail_metric_col')
    if not guardrail_col or guardrail_col not in df.columns:
        print("🚦 No guardrail metric configured (set guardrail_metric_col in Experiment Setup to evaluate one).")
        return
    evaluate_guardrail_metric(df=df, test_config=test_config, guardrail_metric_col=guardrail_col, alpha=alpha)


def evaluate_guardrail_metric(
    df,
    test_config,
    guardrail_metric_col='bounce_rate',
    alpha=0.05
):
    """
    Checks for statistically significant changes in guardrail metric (e.g., bounce rate).

    Parameters:
    - df : pd.DataFrame — experiment dataset
    - test_config : dict — contains group info, variant, etc.
    - guardrail_metric_col : str — column name of guardrail metric
    - alpha : float — significance level (default 0.05)

    Returns:
    - None (prints result)
    """

    group_col = 'group'
    control, treatment = test_config['group_labels']

    control_vals = df[df[group_col] == control][guardrail_metric_col]
    treatment_vals = df[df[group_col] == treatment][guardrail_metric_col]

    mean_control = control_vals.mean()
    mean_treatment = treatment_vals.mean()
    diff = mean_treatment - mean_control

    t_stat, p_val = ttest_ind(treatment_vals, control_vals)

    print(f"🚦 Guardrail Metric Check → '{guardrail_metric_col}'")
    print("Hypothesis (two-sided t-test): H₀ — no difference in mean vs H₁ — means differ.")
    print(f"- {control:10}: {mean_control:.4f}")
    print(f"- {treatment:10}: {mean_treatment:.4f}")
    print(f"- Difference   : {diff:+.4f}")
    print(f"- P-value (t-test): {p_val:.4f}")

    if p_val < alpha:
        if diff > 0:
            print("❌ Significant *increase* — potential negative impact on guardrail.")
        else:
            print("✅ Significant *decrease* — potential positive impact.")
    else:
        print("🟡 No statistically significant change — guardrail looks stable.")


def simulate_rollout_impact(
    experiment_result,
    daily_eligible_observations,
    metric_unit='conversions'
):
    """
    Estimate potential impact of rolling out the treatment to all eligible traffic.

    Parameters:
    - experiment_result: dict
        Output of `run_ab_test()` — must contain summary + group_labels
    - daily_eligible_observations: int
        Number of eligible units per day (users, sessions, transactions, etc.)
    - metric_unit: str
        What the metric represents (e.g., 'conversions', 'revenue', 'clicks')

    Prints daily and monthly lift estimates.
    """

    group1, group2 = experiment_result['group_labels']
    summary = experiment_result['summary']

    # Extract means
    mean_control = summary[group1]['mean']
    mean_treatment = summary[group2]['mean']
    observed_lift = mean_treatment - mean_control

    # Impact calculation
    daily_impact = observed_lift * daily_eligible_observations
    monthly_impact = daily_impact * 30

    # Output
    print("\n📦 Rollout Simulation")
    print(f"- Outcome Metric      : {metric_unit}")
    print(f"- Observed Lift       : {observed_lift:.4f} per unit")
    print(f"- Daily Eligible Units: {daily_eligible_observations}")
    print(f"- Estimated Daily Impact   : {daily_impact:,.0f} {metric_unit}/day")
    print(f"- Estimated Monthly Impact : {monthly_impact:,.0f} {metric_unit}/month\n")
