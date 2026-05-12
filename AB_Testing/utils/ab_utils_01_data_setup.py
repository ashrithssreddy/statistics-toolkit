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
    issues = []

    def add_issue(severity, category, message):
        issue = {
            'severity': severity,
            'category': category,
            'message': message,
        }
        issues.append(issue)
        formatted = f"[{category}] {message}"
        if severity == 'error':
            errors.append(formatted)
        else:
            warnings.append(formatted)

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
    family = test_config.get('family')
    normality = test_config.get('normality')
    equal_variance = test_config.get('equal_variance')

    if family is None:
        if outcome_type == 'binary':
            expected_family = 'mcnemar_test' if group_relationship == 'paired' else (
                'two_proportion_z_test' if group_count == 2 else 'chi_square_test'
            )
        elif outcome_type == 'categorical':
            expected_family = 'chi_square_test'
        elif outcome_type == 'continuous':
            if group_relationship == 'paired':
                expected_family = 'paired_t_test' if normality else 'wilcoxon_signed_rank_test'
            elif group_count == 2:
                if normality:
                    expected_family = 'two_sample_t_test' if equal_variance else 'welch_two_sample_t_test'
                else:
                    expected_family = 'mann_whitney_u_test'
            else:
                if normality:
                    expected_family = 'anova_test' if equal_variance else 'welch_anova_test'
                else:
                    expected_family = 'kruskal_wallis_test'
        else:
            expected_family = None
    else:
        expected_family = family

    if outcome_type not in valid_outcome_types:
        add_issue('error', 'statistically_invalid', f"outcome_metric_datatype must be one of {sorted(valid_outcome_types)}; got {outcome_type!r}")

    if group_relationship not in valid_group_relationships:
        add_issue('error', 'statistically_invalid', f"group_relationship must be one of {sorted(valid_group_relationships)}; got {group_relationship!r}")

    if hypothesis_type not in valid_hypothesis_types:
        add_issue('error', 'statistically_invalid', f"hypothesis_type must be one of {sorted(valid_hypothesis_types)}; got {hypothesis_type!r}")

    if randomization_method not in valid_randomization_methods:
        add_issue('error', 'unsupported_by_current_notebook', f"randomization_method must be one of {sorted(valid_randomization_methods)}; got {randomization_method!r}")

    if historical_normality is not None and historical_normality not in valid_historical_normality:
        add_issue('error', 'statistically_invalid', f"historical_normality must be one of {sorted(valid_historical_normality)}; got {historical_normality!r}")

    if not isinstance(group_labels, (tuple, list)) or len(group_labels) < 2:
        add_issue('error', 'statistically_invalid', "group_labels must contain at least two labels")
    elif len(set(group_labels)) != len(group_labels):
        add_issue('error', 'statistically_invalid', "group_labels must be unique")

    if isinstance(group_labels, (tuple, list)) and group_count != len(group_labels):
        add_issue('error', 'statistically_invalid', "group_count must match len(group_labels)")

    if group_count != 2 and outcome_type == 'binary':
        add_issue('error', 'unsupported_by_current_notebook', "binary k-group tests route to chi-square, but downstream binary summaries/power are two-arm oriented")
    elif group_count != 2 and outcome_type == 'continuous' and group_relationship == 'paired':
        add_issue('error', 'unsupported_by_current_notebook', "paired continuous k-group tests are not implemented in the notebook path")
    elif group_count != 2 and outcome_type == 'categorical':
        add_issue('warning', 'supported_with_caveats', "categorical k-group chi-square is supported, but post-hoc pairwise interpretation is not implemented")
    elif group_count != 2 and outcome_type == 'continuous':
        add_issue('warning', 'supported_with_caveats', "continuous independent k-group omnibus tests are partially supported")

    if not outcome_metric_col:
        add_issue('error', 'statistically_invalid', "outcome_metric_col is required")

    if not observation_id_col:
        add_issue('error', 'statistically_invalid', "observation_id_col is required")

    if randomization_method == 'matched_pair' and _is_missing(pre_experiment_metric_col):
        add_issue('error', 'statistically_invalid', "matched_pair randomization requires pre_experiment_metric_col as the matching/sort column")

    if group_relationship == 'paired' and randomization_method != 'matched_pair':
        add_issue(
            'error',
            'unsupported_by_current_notebook',
            f"paired design requires paired construction; randomization_method={randomization_method!r} does not create explicit pairs"
        )

    if randomization_method == 'cluster' and not cluster_col:
        add_issue('error', 'statistically_invalid', "cluster randomization requires cluster_col")

    if randomization_method == 'block' and (not isinstance(block_size, int) or block_size < 2):
        add_issue('error', 'statistically_invalid', "block randomization requires integer block_size >= 2")

    if alpha is not None and not (0 < float(alpha) < 1):
        add_issue('error', 'statistically_invalid', f"alpha must be between 0 and 1; got {alpha!r}")

    if power is not None and not (0 < float(power) < 1):
        add_issue('error', 'statistically_invalid', f"power must be between 0 and 1; got {power!r}")

    if mde is not None:
        if outcome_type in {'binary', 'categorical'} and not (0 < float(mde) < 1):
            add_issue('error', 'statistically_invalid', f"mde must be between 0 and 1 for {outcome_type} metrics; got {mde!r}")
        elif outcome_type == 'continuous' and not (float(mde) > 0):
            add_issue('error', 'statistically_invalid', f"mde must be positive for continuous metrics; got {mde!r}")

    if observations_count is not None and int(observations_count) <= 0:
        add_issue('error', 'statistically_invalid', "observations_count must be positive")

    if cuped_enabled and outcome_type != 'continuous':
        add_issue('warning', 'unsupported_by_current_notebook', "CUPED should be skipped for non-continuous outcome metrics")

    if cuped_enabled and outcome_type == 'continuous' and _is_missing(pre_experiment_metric_col):
        add_issue('warning', 'unsupported_by_current_notebook', "CUPED needs a pre-experiment covariate; notebook currently uses past_purchase_revenue")

    omnibus_families = {'chi_square_test', 'anova_test', 'welch_anova_test', 'kruskal_wallis_test'}
    if hypothesis_type in {'greater', 'less'}:
        if outcome_type == 'categorical':
            add_issue('error', 'statistically_invalid', "directional hypotheses are not defined for categorical chi-square omnibus tests")
        elif group_count and group_count > 2:
            add_issue('error', 'statistically_invalid', "directional hypotheses are not defined for omnibus k-group tests")
        elif expected_family in omnibus_families:
            add_issue('error', 'statistically_invalid', f"directional hypotheses are not supported for {expected_family}")

    supported_families = {
        'one_proportion_z_test',
        'two_proportion_z_test',
        'mcnemar_test',
        'two_sample_t_test',
        'welch_two_sample_t_test',
        'paired_t_test',
        'mann_whitney_u_test',
        'wilcoxon_signed_rank_test',
        'anova_test',
        'welch_anova_test',
        'kruskal_wallis_test',
        'chi_square_test',
    }
    if expected_family and expected_family not in supported_families:
        add_issue('error', 'unsupported_by_current_notebook', f"selected test family is not supported by notebook utilities: {expected_family}")

    if family is not None and expected_family != family:
        add_issue('error', 'statistically_invalid', f"test_config family {family!r} is incompatible with selected design; expected {expected_family!r}")

    if outcome_type == 'binary' and expected_family not in {None, 'one_proportion_z_test', 'two_proportion_z_test', 'mcnemar_test', 'chi_square_test'}:
        add_issue('error', 'statistically_invalid', f"binary outcome is incompatible with test family {expected_family!r}")
    if outcome_type == 'categorical' and expected_family != 'chi_square_test':
        add_issue('error', 'statistically_invalid', f"categorical outcome is incompatible with test family {expected_family!r}")
    if outcome_type == 'continuous' and expected_family in {'one_proportion_z_test', 'two_proportion_z_test', 'mcnemar_test', 'chi_square_test'}:
        add_issue('error', 'statistically_invalid', f"continuous outcome is incompatible with test family {expected_family!r}")

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
            add_issue('error', 'unsupported_by_current_notebook', f"required columns are missing from dataframe: {missing_columns}")

    error_categories = sorted({issue['category'] for issue in issues if issue['severity'] == 'error'})
    warning_categories = sorted({issue['category'] for issue in issues if issue['severity'] == 'warning'})

    result = {
        'status': 'invalid_config' if errors else 'valid',
        'valid': not errors,
        'errors': errors,
        'warnings': warnings,
        'issues': issues,
        'error_categories': error_categories,
        'warning_categories': warning_categories,
        'metadata': {
            'outcome_metric_datatype': outcome_type,
            'group_relationship': group_relationship,
            'hypothesis_type': hypothesis_type,
            'randomization_method': randomization_method,
            'historical_normality': historical_normality,
            'group_count': group_count,
            'group_labels': group_labels,
            'expected_family': expected_family,
            'configured_family': family,
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
    metadata = validation_result.get('metadata', {})
    if metadata:
        print(f"Expected test family: {metadata.get('expected_family')}")
        if metadata.get('configured_family') is not None:
            print(f"Configured test family: {metadata.get('configured_family')}")

    if validation_result.get('error_categories'):
        print(f"Error categories: {', '.join(validation_result['error_categories'])}")
    if validation_result.get('warning_categories'):
        print(f"Warning categories: {', '.join(validation_result['warning_categories'])}")

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

