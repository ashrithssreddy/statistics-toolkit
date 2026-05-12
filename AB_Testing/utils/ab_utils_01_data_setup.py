# 01 Data Setup — create_dummy_ab_data, create_historical_df, add_outcome_metrics
import numpy as np
import pandas as pd

my_seed = 1995


def _is_missing(value):
    return value is None or pd.isna(value)


def validate_ab_test_config(
    test_config,
    randomization_method=None,
    historical_normality=None,
    mde=None,
    alpha=None,
    power=None,
    observations_count=None,
    cluster_col='city',
    stratify_col='platform',
    block_size=10,
    cuped_enabled=True,
    available_columns=None,
    raise_on_error=False,
):
    """
    Validate experiment configuration before running the notebook pipeline.

    Returns a structured dict with:
    - status: valid / invalid_config
    - valid: bool
    - errors: blocking configuration problems
    - warnings: non-blocking caveats
    - metadata: normalized/debug values
    """
    errors = []
    warnings = []

    valid_outcome_types = {'continuous', 'binary', 'categorical'}
    valid_group_relationships = {'independent', 'paired'}
    valid_hypothesis_types = {'two_sided', 'greater', 'less'}
    valid_randomization_methods = {'simple', 'stratified', 'block', 'matched_pair', 'cluster'}
    valid_historical_normality = {'normal', 'non_normal'}

    outcome_type = test_config.get('outcome_metric_datatype')
    group_relationship = test_config.get('group_relationship')
    hypothesis_type = test_config.get('hypothesis_type')
    group_labels = test_config.get('group_labels')
    group_count = test_config.get('group_count')
    outcome_metric_col = test_config.get('outcome_metric_col')
    observation_id_col = test_config.get('observation_id_col')
    pre_experiment_metric_col = test_config.get('pre_experiment_metric_col')
    guardrail_metric_col = test_config.get('guardrail_metric_col')

    if outcome_type not in valid_outcome_types:
        errors.append(f"outcome_metric_datatype must be one of {sorted(valid_outcome_types)}; got {outcome_type!r}")

    if group_relationship not in valid_group_relationships:
        errors.append(f"group_relationship must be one of {sorted(valid_group_relationships)}; got {group_relationship!r}")

    if hypothesis_type not in valid_hypothesis_types:
        errors.append(f"hypothesis_type must be one of {sorted(valid_hypothesis_types)}; got {hypothesis_type!r}")

    if randomization_method not in valid_randomization_methods:
        errors.append(f"randomization_method must be one of {sorted(valid_randomization_methods)}; got {randomization_method!r}")

    if historical_normality is not None and historical_normality not in valid_historical_normality:
        errors.append(f"historical_normality must be one of {sorted(valid_historical_normality)}; got {historical_normality!r}")

    if not isinstance(group_labels, (tuple, list)) or len(group_labels) < 2:
        errors.append("group_labels must contain at least two labels")
    elif len(set(group_labels)) != len(group_labels):
        errors.append("group_labels must be unique")

    if isinstance(group_labels, (tuple, list)) and group_count != len(group_labels):
        errors.append("group_count must match len(group_labels)")

    if group_count != 2:
        warnings.append("Current notebook analysis path is primarily implemented for two-arm tests")

    if not outcome_metric_col:
        errors.append("outcome_metric_col is required")

    if not observation_id_col:
        errors.append("observation_id_col is required")

    if randomization_method == 'matched_pair' and _is_missing(pre_experiment_metric_col):
        errors.append("matched_pair randomization requires pre_experiment_metric_col as the matching/sort column")

    if randomization_method == 'cluster' and not cluster_col:
        errors.append("cluster randomization requires cluster_col")

    if randomization_method == 'block' and (not isinstance(block_size, int) or block_size < 2):
        errors.append("block randomization requires integer block_size >= 2")

    if alpha is not None and not (0 < float(alpha) < 1):
        errors.append(f"alpha must be between 0 and 1; got {alpha!r}")

    if power is not None and not (0 < float(power) < 1):
        errors.append(f"power must be between 0 and 1; got {power!r}")

    if mde is not None:
        if outcome_type in {'binary', 'categorical'} and not (0 < float(mde) < 1):
            errors.append(f"mde must be between 0 and 1 for {outcome_type} metrics; got {mde!r}")
        elif outcome_type == 'continuous' and not (float(mde) > 0):
            errors.append(f"mde must be positive for continuous metrics; got {mde!r}")

    if observations_count is not None and int(observations_count) <= 0:
        errors.append("observations_count must be positive")

    if cuped_enabled and outcome_type != 'continuous':
        warnings.append("CUPED should be skipped for non-continuous outcome metrics")

    if cuped_enabled and outcome_type == 'continuous' and _is_missing(pre_experiment_metric_col):
        warnings.append("CUPED needs a pre-experiment covariate; notebook currently uses past_purchase_revenue")

    if available_columns is not None:
        required_columns = {col for col in [outcome_metric_col, observation_id_col] if col}
        if guardrail_metric_col:
            required_columns.add(guardrail_metric_col)
        if randomization_method == 'stratified':
            required_columns.add(stratify_col)
        if randomization_method == 'matched_pair' and not _is_missing(pre_experiment_metric_col):
            required_columns.add(pre_experiment_metric_col)
        if randomization_method == 'cluster':
            required_columns.add(cluster_col)

        missing_columns = sorted(col for col in required_columns if col not in available_columns)
        if missing_columns:
            errors.append(f"required columns are missing from dataframe: {missing_columns}")

    result = {
        'status': 'invalid_config' if errors else 'valid',
        'valid': not errors,
        'errors': errors,
        'warnings': warnings,
        'metadata': {
            'outcome_metric_datatype': outcome_type,
            'group_relationship': group_relationship,
            'hypothesis_type': hypothesis_type,
            'randomization_method': randomization_method,
            'historical_normality': historical_normality,
            'group_count': group_count,
            'group_labels': group_labels,
        },
    }

    if raise_on_error and errors:
        raise ValueError("Invalid AB test configuration: " + "; ".join(errors))

    return result


def print_validation_result(validation_result):
    """Print a compact validation report for the notebook."""
    print("Configuration Validation")
    print("-" * 40)
    print(f"Status: {validation_result['status']}")

    if validation_result['errors']:
        print("\nErrors")
        for error in validation_result['errors']:
            print(f"- {error}")

    if validation_result['warnings']:
        print("\nWarnings")
        for warning in validation_result['warnings']:
            print(f"- {warning}")

    if not validation_result['errors'] and not validation_result['warnings']:
        print("No validation issues detected.")


def create_dummy_ab_data(
    observations_count=1000,
    seed=1995,
    outcome_metric_col=None,
    guardrail_metric_col=None,
    randomization_method=None,
    cluster_col='city',
):
    """Generate user population with attributes and pre-experiment variables only.
    Outcome and guardrail metrics are not generated here; they are created after randomization.
    If outcome_metric_col or guardrail_metric_col is provided, a placeholder column (NaN) is added so column order
    puts must-haves (user_id, outcome_metric_col, guardrail_metric_col, past_purchase_revenue) on the left."""
    np.random.seed(seed)
    users = pd.DataFrame({
        # required (from experiment setup / central control panel): identifier, pre-experiment metric, placeholders
        'user_id': range(1, observations_count + 1),
        'past_purchase_revenue': np.random.normal(loc=50, scale=10, size=observations_count).clip(0),
        # optional: segmentation columns used later in the notebook
        'platform': np.random.choice(['iOS', 'Android'], size=observations_count, p=[0.6, 0.4]),
        'device_type': np.random.choice(['mobile', 'desktop'], size=observations_count, p=[0.7, 0.3]),
        # optional: uncomment if needed for cluster or segment analysis
        # 'user_tier': np.random.choice(['new', 'returning'], size=observations_count, p=[0.4, 0.6]),
        # 'region': np.random.choice(['North', 'South', 'East', 'West'], size=observations_count, p=[0.25, 0.25, 0.25, 0.25]),
        # 'plan_type': np.random.choice(['basic', 'premium', 'pro'], size=observations_count, p=[0.6, 0.3, 0.1]),
    })
    if randomization_method == 'cluster':
        if cluster_col == 'city':
            users[cluster_col] = np.random.choice(['ny', 'sf', 'chicago', 'austin'], size=observations_count)
        else:
            users[cluster_col] = np.random.choice([f'{cluster_col}_1', f'{cluster_col}_2', f'{cluster_col}_3', f'{cluster_col}_4'], size=observations_count)
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
    must_have.append('past_purchase_revenue')
    extras = [c for c in users.columns if c not in must_have]
    users = users[must_have + extras]
    return users


def create_historical_df(
    df,
    outcome_metric_col,
    guardrail_metric_col=None,
    seed=my_seed,
    historical_normality="normal",
    non_normal_distribution="random",
    outcome_metric_datatype="continuous",
):
    """
    Create a historical view of the population: same columns as df, but outcome and guardrail
    columns (which are NaN in df at creation) are filled with baseline-only values — no experiment,
    no group. Use this for power-analysis baseline so baselines come from historical data, not from df.
    """
    hist = df.copy()
    n = len(hist)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if outcome_metric_col and outcome_metric_col in hist.columns:
        if outcome_metric_datatype == "binary":
            hist[outcome_metric_col] = rng.binomial(n=1, p=0.12, size=n)
        elif outcome_metric_datatype == "categorical":
            hist[outcome_metric_col] = rng.choice(["low", "mid", "high"], size=n, p=[0.33, 0.34, 0.33])
        elif historical_normality == "normal":
            hist[outcome_metric_col] = rng.normal(50, 15, n).clip(0, 100)
        elif historical_normality == "non_normal":
            # Right-skewed historical behavior (common in spend/time metrics).
            # Choose one family deterministically by seed unless caller specifies one.
            if non_normal_distribution == "random":
                chosen = rng.choice(["lognormal", "gamma", "weibull"])
            else:
                chosen = non_normal_distribution

            if chosen == "lognormal":
                vals = rng.lognormal(mean=3.8, sigma=0.45, size=n)
            elif chosen == "gamma":
                vals = rng.gamma(shape=5.0, scale=10.0, size=n)
            elif chosen == "weibull":
                vals = rng.weibull(a=2.0, size=n) * 60.0
            else:
                raise ValueError("non_normal_distribution must be 'random', 'lognormal', 'gamma', or 'weibull'")

            hist[outcome_metric_col] = np.asarray(vals).clip(0, 100)
        elif outcome_metric_datatype == "continuous":
            raise ValueError("historical_normality must be 'normal' or 'non_normal'")
        else:
            raise ValueError("outcome_metric_datatype must be 'continuous', 'binary', or 'categorical'")
    if guardrail_metric_col and guardrail_metric_col in hist.columns:
        hist[guardrail_metric_col] = rng.normal(0.5, 0.1, n).clip(0, 1)
    return hist


def add_outcome_metrics(
    df,
    group_col='group',
    group_labels=('control', 'treatment'),
    outcome_metric_col='engagement_score',
    guardrail_metric_col=None,
    treatment_effect=True,
    seed=my_seed,
    outcome_metric_datatype='continuous',
    historical_normality='normal',
    non_normal_distribution='random',
):
    """
    Add outcome and optional guardrail metric to a dataframe that already has group assignment.
    Call this after randomization so outcomes are generated post-assignment.

    - outcome_metric_col: primary outcome (always filled).
    - guardrail_metric_col: optional guardrail metric column name (e.g. 'bounce_rate'); None to omit.
    - treatment_effect: if True, treatment group gets a lift (A/B simulation). If False, both groups
      from same distribution (A/A simulation). Guardrail also avoids treatment signal when False.
    - continuous outcomes use create_historical_df (same distribution as historical baselines), then lift.
    """
    np.random.seed(seed)
    n = len(df)
    treatment_mask = df[group_col] == group_labels[1]
    # Primary outcome: generated according to metric datatype.
    if outcome_metric_datatype == 'binary':
        p_outcome = 0.12 + (0.03 * treatment_mask.astype(float) if treatment_effect else 0)
        df[outcome_metric_col] = np.random.binomial(n=1, p=p_outcome, size=n)
    elif outcome_metric_datatype == 'categorical':
        control_probs = [0.50, 0.30, 0.20]
        treatment_probs = [0.42, 0.33, 0.25] if treatment_effect else control_probs
        categories = np.array(['low', 'mid', 'high'])
        control_values = np.random.choice(categories, size=n, p=control_probs)
        treatment_values = np.random.choice(categories, size=n, p=treatment_probs)
        df[outcome_metric_col] = np.where(treatment_mask, treatment_values, control_values)
    elif outcome_metric_datatype == 'continuous':
        work = df.copy()
        work[outcome_metric_col] = np.nan
        baseline = create_historical_df(
            work,
            outcome_metric_col,
            guardrail_metric_col=None,
            seed=seed,
            historical_normality=historical_normality,
            non_normal_distribution=non_normal_distribution,
            outcome_metric_datatype='continuous',
        )
        base = baseline[outcome_metric_col].to_numpy(dtype=float)
        lift = np.random.default_rng(seed + 1)
        treatment_lift = np.where(treatment_mask, lift.normal(5, 2, n), 0.0) if treatment_effect else 0.0
        df[outcome_metric_col] = (base + treatment_lift).clip(0, 100)
    else:
        raise ValueError("outcome_metric_datatype must be 'continuous', 'binary', or 'categorical'")
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

