# Data Setup — create_dummy_ab_data, create_historical_df, add_outcome_metrics
import numpy as np
import pandas as pd
from ab_utils_common import my_seed


def create_dummy_ab_data(observations_count=1000, seed=1995, outcome_metric_col=None, guardrail_metric_col=None):
    """Generate user population with attributes and pre-experiment variables only.
    Outcome and guardrail metrics are not generated here; they are created after randomization.
    If outcome_metric_col or guardrail_metric_col is provided, a placeholder column (NaN) is added so column order
    puts must-haves (user_id, outcome_metric_col, guardrail_metric_col, past_purchase_count) on the left."""
    np.random.seed(seed)
    users = pd.DataFrame({
        # required (from experiment setup / central control panel): identifier, pre-experiment metric, placeholders
        'user_id': range(1, observations_count + 1),
        'past_purchase_count': np.random.normal(loc=50, scale=10, size=observations_count).clip(0),
        # optional: segmentation (retained for stratified / segment analysis)
        'platform': np.random.choice(['iOS', 'Android'], size=observations_count, p=[0.6, 0.4]),
        'device_type': np.random.choice(['mobile', 'desktop'], size=observations_count, p=[0.7, 0.3]),
        # optional: uncomment if needed for cluster or segment analysis
        # 'user_tier': np.random.choice(['new', 'returning'], size=observations_count, p=[0.4, 0.6]),
        # 'region': np.random.choice(['North', 'South', 'East', 'West'], size=observations_count, p=[0.25, 0.25, 0.25, 0.25]),
        # 'plan_type': np.random.choice(['basic', 'premium', 'pro'], size=observations_count, p=[0.6, 0.3, 0.1]),
        # 'city': np.random.choice(['ny', 'sf', 'chicago', 'austin'], size=observations_count),
    })
    # Placeholder columns (filled after randomization / outcome collection)
    if outcome_metric_col:
        users[outcome_metric_col] = np.nan
    if guardrail_metric_col:
        users[guardrail_metric_col] = np.nan
    # Order: must-haves left, extras right
    must_have = ['user_id']
    if outcome_metric_col:
        must_have.append(outcome_metric_col)
    if guardrail_metric_col:
        must_have.append(guardrail_metric_col)
    must_have.append('past_purchase_count')
    extras = [c for c in users.columns if c not in must_have]
    users = users[must_have + extras]
    return users


def create_historical_df(df, outcome_metric_col, guardrail_metric_col=None, seed=my_seed):
    """
    Create a historical view of the population: same columns as df, but outcome and guardrail
    columns (which are NaN in df at creation) are filled with baseline-only values — no experiment,
    no group. Use this for power-analysis baseline so baselines come from historical data, not from df.
    """
    hist = df.copy()
    n = len(hist)
    np.random.seed(seed)
    if outcome_metric_col and outcome_metric_col in hist.columns:
        hist[outcome_metric_col] = np.random.normal(50, 15, n).clip(0, 100)
    if guardrail_metric_col and guardrail_metric_col in hist.columns:
        hist[guardrail_metric_col] = np.random.normal(0.5, 0.1, n).clip(0, 1)
    return hist


def add_outcome_metrics(df, group_col='group', group_labels=('control', 'treatment'), outcome_metric_col='engagement_score', guardrail_metric_col=None, treatment_effect=True, seed=my_seed):
    """
    Add outcome and optional guardrail metric to a dataframe that already has group assignment.
    Call this after randomization so outcomes are generated post-assignment.

    - outcome_metric_col: primary outcome (always filled).
    - guardrail_metric_col: optional guardrail metric column name (e.g. 'bounce_rate'); None to omit.
    - treatment_effect: if True, treatment group gets a lift (A/B simulation). If False, both groups
      from same distribution (A/A simulation). Guardrail also avoids treatment signal when False.
    """
    np.random.seed(seed)
    n = len(df)
    treatment_mask = df[group_col] == group_labels[1]
    # Primary outcome: baseline + optional treatment effect
    base_engagement = np.random.normal(50, 15, n)
    if treatment_effect:
        treatment_lift = np.where(treatment_mask, np.random.normal(5, 2, n), 0)
    else:
        treatment_lift = 0
    df[outcome_metric_col] = (base_engagement + treatment_lift).clip(0, 100)
    # Optional guardrail: no treatment signal when treatment_effect=False (A/A)
    if guardrail_metric_col:
        if treatment_effect:
            p_convert = 0.1 + 0.02 * treatment_mask.astype(float)
        else:
            p_convert = 0.12  # same for everyone
        _converted = np.random.binomial(n=1, p=p_convert, size=n)
        df[guardrail_metric_col] = np.where(
            _converted == 1,
            np.random.normal(loc=0.2, scale=0.05, size=n),
            np.random.normal(loc=0.6, scale=0.10, size=n)
        )
        df[guardrail_metric_col] = df[guardrail_metric_col].clip(0, 1)
    return df
