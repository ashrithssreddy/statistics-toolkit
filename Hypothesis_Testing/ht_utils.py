# ==========================================================
# region Setup
# ==========================================================

# Display Settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display, HTML
import warnings
import json
# %load_ext autoreload
# %autoreload 2

# Stats Libraries
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats
from scipy.stats import (
    t, norm, f, chi2, kstest,
    ttest_1samp, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu,
    shapiro, chi2_contingency, f_oneway, kruskal, fisher_exact, levene
)

# Data Transformation Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
my_seed = 1995

# ==========================================================
# region Config Dictionary Functions
# ==========================================================
pretty_json = lambda d: display(HTML(f"""
<pre style='font-size:14px; font-family:monospace;'>
{json.dumps(d, indent=4)
   .replace(": null", ': <span style="color:crimson;"><b>null</b></span>')}
</pre>
"""))
# <hr style='border: none; height: 1px; background-color: #ddd;' />

def print_config_summary(config):
    """
    Displays a structured summary of the test configuration with visual cues for missing or inferred values.

    This function:
    - Prints each key in the config with aligned formatting
    - Highlights `None` values in red (terminal only)
    - Provides a short inference summary based on group count and relationship

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing test settings like outcome type, group relationship,
        distribution, variance assumption, parametric flag, and alpha level

    Returns:
    --------
    None
        Prints the formatted configuration summary directly to output
    """
    def highlight(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "\033[91mNone\033[0m"
        return value

    print("📋 Hypothesis Test Configuration Summary\n")

    # Pretty key formatting
    max_key_length = max(len(k) for k in config.keys())

    for key, value in config.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"🔸 {formatted_key:<{max_key_length+2}} : {highlight(value)}")


def validate_config(config):
    """
    Validates the hypothesis test configuration dictionary for completeness and logical consistency.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary to validate.
    
    Returns:
    --------
    None
        Raises ValueError if issues are found.
    """
    
    required_keys = [
        'outcome_type', 'group_relationship', 'group_count', 'distribution',
        'variance_equal', 'tail_type', 'parametric', 'alpha', 'sample_size', 'effect_size'
    ]

    valid_outcome_types = ['continuous', 'binary', 'categorical', 'count']
    valid_group_relationships = ['independent', 'paired']
    valid_group_counts = ['one-sample', 'two-sample', 'multi-sample']
    valid_distributions = ['normal', 'non-normal', None]
    valid_variance_flags = ['equal', 'unequal', None]
    valid_tail_types = ['one-tailed', 'two-tailed']
    valid_parametric_flags = [True, False, None]

    # 1. Missing keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"❌ Missing key in config: '{key}'")

    # 2. Check values are within known sets
    if config['outcome_type'] not in valid_outcome_types:
        raise ValueError(f"❌ Invalid outcome_type: {config['outcome_type']}")

    if config['group_relationship'] not in valid_group_relationships and config['group_count'] != 'one-sample':
        raise ValueError(f"❌ Invalid group_relationship: {config['group_relationship']}")

    if config['group_count'] not in valid_group_counts:
        raise ValueError(f"❌ Invalid group_count: {config['group_count']}")

    if config['distribution'] not in valid_distributions:
        raise ValueError(f"❌ Invalid distribution: {config['distribution']}")

    if config['variance_equal'] not in valid_variance_flags:
        raise ValueError(f"❌ Invalid variance_equal: {config['variance_equal']}")

    if config['tail_type'] not in valid_tail_types:
        raise ValueError(f"❌ Invalid tail_type: {config['tail_type']}")

    if config['parametric'] not in valid_parametric_flags:
        raise ValueError(f"❌ Invalid parametric flag: {config['parametric']}")

    if not (0 < config['alpha'] < 1):
        raise ValueError("❌ Alpha level should be between 0 and 1.")

    if config['sample_size'] <= 0:
        raise ValueError("❌ Sample size must be positive.")

    # 3. Logical combination checks
    # One-sample + non-independent → override to independent
    if config['group_count'] == 'one-sample' and config['group_relationship'] != 'independent':
        print("⚠️ Overriding group_relationship to 'independent' for one-sample test.")
        config['group_relationship'] = 'independent'

    # Multi-sample + paired → not supported by this module
    if config['group_count'] == 'multi-sample' and config['group_relationship'] == 'paired':
        raise ValueError("❌ Paired relationship not supported for multi-sample tests.")

    # One-sample + missing population_mean → invalid config
    if config['group_count'] == 'one-sample' and 'population_mean' not in config:
        raise ValueError("❌ One-sample tests require `population_mean` to be specified.")
    
    # Paired + categorical (not supported by this module)
    if config['outcome_type'] == 'categorical' and config['group_relationship'] == 'paired':
        raise ValueError("❌ Paired tests are not supported for categorical outcomes in this module.")

    # Binary outcome + parametric + small n → warn about z-test validity
    if config['outcome_type'] == 'binary' and config['parametric'] is True:
        if config['sample_size'] < 30:
            print("⚠️ Sample size < 30 → z-test assumptions (np > 5) may be violated. Consider Fisher’s Exact.")

    # Parametric test selected, but distribution is missing
    if config['parametric'] is True and config['distribution'] is None:
        raise ValueError("❌ Parametric test requested, but distribution is not confirmed as normal.")

    # Count outcome + one-sample → not supported
    if config['outcome_type'] == 'count' and config['group_count'] == 'one-sample':
        raise ValueError("🔒 One-sample tests for count data are not supported by this module.")

    # One-sample + categorical → not supported (no goodness-of-fit in this module)
    if config['group_count'] == 'one-sample' and config['outcome_type'] == 'categorical':
        raise ValueError("❌ One-sample categorical (goodness-of-fit) tests are not supported by this module.")

    # Paired + count → not supported
    if config['group_relationship'] == 'paired' and config['outcome_type'] == 'count':
        raise ValueError("❌ Paired tests for count outcomes are not supported by this module.")

    # Distribution only applicable for continuous outcome
    if config['outcome_type'] != 'continuous' and config['distribution'] is not None:
        raise ValueError(
            f"❌ Invalid combo: outcome_type = '{config['outcome_type']}' requires distribution = None "
            "(normality applies only to continuous outcomes)."
        )

    # variance_equal only applicable for continuous + (two-sample or multi-sample) + independent
    if config['variance_equal'] is not None:
        if config['outcome_type'] != 'continuous':
            raise ValueError(
                f"❌ Invalid combo: outcome_type = '{config['outcome_type']}' requires variance_equal = None."
            )
        if config['group_count'] == 'one-sample':
            raise ValueError("❌ Invalid combo: one-sample requires variance_equal = None.")
        if config['group_relationship'] == 'paired':
            raise ValueError("❌ Invalid combo: paired design requires variance_equal = None.")

    # Effect size unusually large or small (soft validation)
    if config['effect_size'] < 0 or config['effect_size'] > 2:
        print("⚠️ Effect size is unusually extreme. Are you simulating a realistic scenario?")

    # Optional: variance check mismatch
    if config['variance_equal'] not in valid_variance_flags:
        raise ValueError(f"❌ Invalid variance_equal flag: {config['variance_equal']}")

    # Optional: group relationship irrelevant in one-sample, but present
    if config['group_count'] == 'one-sample' and config.get('group_relationship') != 'independent':
        print("⚠️ One-sample tests don’t require `group_relationship`. Defaulting to 'independent'.")
        config['group_relationship'] = 'independent'

    print("✅ Config validated successfully.")

# ==========================================================
# region Synthetic Data Generation
# ==========================================================
def generate_data_from_config(config, seed=1995):
    """
    Generates synthetic data based on full config surface.

    Supports:
    - outcome_type: continuous, binary, categorical, count
    - group_count: one-sample, two-sample, multi-sample
    - group_relationship: independent, paired (paired only valid for two-sample)
    - distribution: normal, non-normal (continuous only)
    - variance_equal: equal, unequal (continuous independent only)

    Returns
    -------
    pandas.DataFrame
        Shape and columns depend on group_count and group_relationship.
        Example outputs (5 rows, dummy data):

        One-sample (columns: value):
             value
        0   1.23
        1  -0.45
        2   2.10
        3   0.67
        4   1.88

        Two-sample independent (columns: group, value):
          group  value
        0     A   0.52
        1     A   1.11
        2     B   1.03
        3     B   0.89
        4     A   0.41

        Two-sample paired (columns: id, group_A, group_B):
           id  group_A  group_B
        0   0     0.12     0.58
        1   1     1.02     1.49
        2   2    -0.33     0.21
        3   3     0.77     1.22
        4   4     0.45     0.91

        Multi-sample independent (columns: group, value):
          group  value
        0     A   0.31
        1     B   0.95
        2     C   1.12
        3     A   0.44
        4     B   0.78
    """

    import numpy as np
    import pandas as pd

    np.random.seed(seed)

    outcome = config['outcome_type']
    group_count = config['group_count']
    relationship = config['group_relationship']
    size = config['sample_size']
    effect = config['effect_size']
    distribution = config['distribution']
    variance_equal = config['variance_equal']

    # ----------------------------
    # ONE-SAMPLE
    # ----------------------------
    if group_count == 'one-sample':
        if outcome == 'count':
            raise ValueError(
                "One-sample tests for count data are not supported by this module. "
                "Use two-sample or multi-sample for count outcomes."
            )

        if outcome == 'continuous':
            if distribution == 'non-normal':
                values = np.random.lognormal(mean=effect, sigma=0.5, size=size)
            else:
                values = np.random.normal(loc=effect, scale=1.0, size=size)

        elif outcome == 'binary':
            p = min(max(0.5 + effect, 0), 1)
            values = np.random.binomial(1, p, size=size)

        elif outcome == 'categorical':
            categories = ['A', 'B', 'C']
            probs = np.array([0.4, 0.4, 0.2])
            values = np.random.choice(categories, size=size, p=probs)

        elif outcome == 'count':
            lam = max(1 + effect, 0.1)
            values = np.random.poisson(lam=lam, size=size)

        else:
            raise ValueError("Invalid outcome_type.")

        return pd.DataFrame({'value': values})

    # ----------------------------
    # TWO-SAMPLE
    # ----------------------------
    if group_count == 'two-sample':

        if relationship == 'paired':

            if outcome == 'continuous':
                if distribution == 'non-normal':
                    before = np.random.lognormal(mean=0, sigma=0.5, size=size)
                else:
                    before = np.random.normal(loc=0, scale=1.0, size=size)

                after = before + effect

            elif outcome == 'binary':
                before = np.random.binomial(1, 0.4, size=size)
                after = np.clip(before + np.random.binomial(1, effect, size=size), 0, 1)

            else:
                raise ValueError("Paired supported only for continuous and binary.")

            return pd.DataFrame({
                'id': np.arange(size),
                'group_A': before,
                'group_B': after
            })

        # ------------------------
        # Independent
        # ------------------------

        groups = ['A', 'B']
        data = []

        for i, g in enumerate(groups):

            shift = effect if g == 'B' else 0

            if outcome == 'continuous':
                std = 1.0
                if variance_equal == 'unequal' and g == 'B':
                    std = 1.5

                if distribution == 'non-normal':
                    values = np.random.lognormal(mean=shift, sigma=0.5, size=size)
                else:
                    values = np.random.normal(loc=shift, scale=std, size=size)

            elif outcome == 'binary':
                p = min(max(0.4 + shift, 0), 1)
                values = np.random.binomial(1, p, size=size)

            elif outcome == 'categorical':
                categories = ['A', 'B', 'C']
                probs = np.array([0.4, 0.4, 0.2])
                probs = probs + shift * np.array([0, 0.1, -0.1])
                probs = probs / probs.sum()
                values = np.random.choice(categories, size=size, p=probs)

            elif outcome == 'count':
                lam = max(3 + shift, 0.1)
                values = np.random.poisson(lam=lam, size=size)

            else:
                raise ValueError("Invalid outcome_type.")

            for v in values:
                data.append((g, v))

        return pd.DataFrame(data, columns=['group', 'value'])

    # ----------------------------
    # MULTI-SAMPLE (Independent Only)
    # ----------------------------
    if group_count == 'multi-sample':

        if relationship == 'paired':
            raise ValueError("Paired multi-sample not supported.")

        groups = ['A', 'B', 'C']
        data = []

        for i, g in enumerate(groups):

            shift = effect * i

            if outcome == 'continuous':
                std = 1.0
                if variance_equal == 'unequal':
                    std = 1.0 + 0.5 * i

                if distribution == 'non-normal':
                    values = np.random.lognormal(mean=shift, sigma=0.5, size=size)
                else:
                    values = np.random.normal(loc=shift, scale=std, size=size)

            elif outcome == 'binary':
                p = min(max(0.4 + shift, 0), 1)
                values = np.random.binomial(1, p, size=size)

            elif outcome == 'categorical':
                categories = ['A', 'B', 'C']
                probs = np.array([0.4, 0.4, 0.2])
                probs = probs + shift * np.array([0, 0.1, -0.1])
                probs = probs / probs.sum()
                values = np.random.choice(categories, size=size, p=probs)

            elif outcome == 'count':
                lam = max(3 + shift, 0.1)
                values = np.random.poisson(lam=lam, size=size)

            else:
                raise ValueError("Invalid outcome_type.")

            for v in values:
                data.append((g, v))

        return pd.DataFrame(data, columns=['group', 'value'])

    raise ValueError("Invalid group_count.")

# ==========================================================
# region EDA functions - Sample Size
# ==========================================================
def infer_sample_size_from_data(config, df):
    """
    Infers and updates sample_size in config based on the dataset structure.

    Rules:
    - one-sample → total rows
    - two-sample independent → minimum group size (per-group n)
    - two-sample paired → number of paired rows
    - multi-sample → total rows (can refine later)
    """

    group_count = config['group_count']
    relationship = config.get('group_relationship')

    if group_count == 'one-sample':
        n = len(df)

    elif group_count == 'two-sample' and relationship == 'independent':
        if 'group' not in df.columns:
            raise ValueError("Expected 'group' column for two-sample independent test.")
        group_sizes = df['group'].value_counts()
        n = group_sizes.min()  # conservative per-group size

    elif group_count == 'two-sample' and relationship == 'paired':
        n = len(df)

    elif group_count == 'multi-sample':
        if 'group' not in df.columns:
            raise ValueError("Expected 'group' column for multi-sample test.")
        n = len(df)

    else:
        raise ValueError("Invalid group configuration.")

    config['sample_size'] = int(n)

    print(f"📊 Synced sample_size → {config['sample_size']}")

    return config['sample_size']

# ==========================================================
# region EDA functions - Group Count
# ==========================================================
def infer_group_count_from_data(config, df):
    """
    Infers group_count ('one-sample', 'two-sample', 'multi-sample')
    based on dataframe structure and group_relationship.

    Rules:
    - If paired → must be two-sample
    - If independent:
        • No 'group' column → one-sample
        • 1 unique group → one-sample
        • 2 unique groups → two-sample
        • >2 unique groups → multi-sample
    """

    relationship = config.get('group_relationship')

    print("\n🔄 Step: Infer Group Count from Dataset")

    # --------------------------
    # 1️⃣ Paired Structure
    # --------------------------
    if relationship == 'paired':
        required_cols = {'group_A', 'group_B'}
        if required_cols.issubset(df.columns):
            config['group_count'] = 'two-sample'
            print("📊 Detected paired structure → group_count = 'two-sample'")
        else:
            raise ValueError("Paired design requires 'group_A' and 'group_B' columns.")

        return config['group_count']

    # --------------------------
    # 2️⃣ Independent Structure
    # --------------------------
    if 'group' not in df.columns:
        config['group_count'] = 'one-sample'
        print("📊 No 'group' column found → group_count = 'one-sample'")
        return config['group_count']

    n_groups = df['group'].nunique()

    if n_groups == 1:
        config['group_count'] = 'one-sample'
    elif n_groups == 2:
        config['group_count'] = 'two-sample'
    elif n_groups > 2:
        config['group_count'] = 'multi-sample'
    else:
        raise ValueError("Unable to infer group_count.")

    print(f"📊 Detected {n_groups} group(s) → group_count = '{config['group_count']}'")

    return config['group_count']


# ==========================================================
# region EDA functions - Outcome Type
# ==========================================================
def infer_outcome_type_from_data(config, df):
    """
    Infers outcome_type based on dataframe structure and values.
    """

    print("\n🔄 Step: Infer Outcome Type from Dataset")

    group_count = config['group_count']
    relationship = config.get('group_relationship')

    # --------------------------
    # Identify outcome column
    # --------------------------
    if group_count == 'one-sample':
        outcome_series = df['value']

    elif group_count == 'two-sample' and relationship == 'independent':
        outcome_series = df['value']

    elif group_count == 'two-sample' and relationship == 'paired':
        outcome_series = df['group_A']  # use one column to infer type

    elif group_count == 'multi-sample':
        outcome_series = df['value']

    else:
        raise ValueError("Unable to determine outcome column.")

    unique_vals = outcome_series.dropna().unique()
    n_unique = len(unique_vals)

    # --------------------------
    # Binary Check
    # --------------------------
    if set(unique_vals).issubset({0, 1}):
        inferred = 'binary'

    # --------------------------
    # Categorical Check
    # --------------------------
    elif outcome_series.dtype == 'object':
        inferred = 'categorical'

    # --------------------------
    # Count Check
    # --------------------------
    elif pd.api.types.is_integer_dtype(outcome_series):
        if outcome_series.min() >= 0 and n_unique > 2:
            inferred = 'count'
        else:
            inferred = 'continuous'

    # --------------------------
    # Continuous Default
    # --------------------------
    elif pd.api.types.is_numeric_dtype(outcome_series):
        inferred = 'continuous'

    else:
        inferred = 'continuous'

    config['outcome_type'] = inferred

    print(f"📊 Inferred outcome_type → '{inferred}'")

    return config['outcome_type']


# ==========================================================
# region EDA functions - Normality Infer
# ==========================================================
def infer_distribution_from_data(config, df):
    """
    Infers whether the outcome variable follows a normal distribution using the Shapiro-Wilk test.

    This function:
    - Checks if the outcome type is continuous (required for normality testing)
    - Applies Shapiro-Wilk test to the outcome (one group, both groups, or each group for multi-sample)
    - Updates the 'distribution' key in the config as 'normal', 'non-normal', or None (when not applicable)
    - Logs interpretation and decision in a reader-friendly format

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing 'outcome_type', 'group_count', and 'group_relationship'
    df : pandas.DataFrame
        Input dataframe containing outcome values and group assignments

    Returns:
    --------
    str
        The inferred distribution: 'normal', 'non-normal', or None (when not applicable).
        Callers typically assign to config, e.g. config['distribution'] = infer_distribution_from_data(config, df).
    """

    print("\n🔍 Step: Infer Distribution of Outcome Variable")

    group_count = config['group_count']
    relationship = config['group_relationship']
    outcome = config['outcome_type']

    if outcome != 'continuous':
        print(f"⚠️ Skipping: Outcome type = `{outcome}` → normality check not applicable.")
        config['distribution'] = None
        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return config['distribution']

    print("📘 Checking if the outcome variable follows a normal distribution")
    print("   Using Shapiro-Wilk Test")
    print("   H₀: Data comes from a normal distribution")
    print("   H₁: Data does NOT come from a normal distribution\n")

    if group_count == 'one-sample':
        print("• One-sample case → testing entire column")
        stat, p = shapiro(df['value'])
        print(f"• Shapiro-Wilk p-value = {p:.4f}")

        if p > 0.05:
            print("✅ Fail to reject H₀ → Data is likely a normal distribution")
            config['distribution'] = 'normal'
        else:
            print("⚠️ Reject H₀ → Data is likely a non-normal distribution")
            config['distribution'] = 'non-normal'

        print(f"📦 Final Decision → config['distribution'] = `{config['distribution']}`")
        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return config['distribution']

    elif group_count == 'two-sample':
        print(f"• Two-sample ({relationship}) case → testing both groups")

        if relationship == 'independent':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
        elif relationship == 'paired':
            a = df['group_A']
            b = df['group_B']
        else:
            print("❌ Invalid group relationship")
            config['distribution'] = None
            # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
            return config['distribution']

        p1 = shapiro(a).pvalue
        p2 = shapiro(b).pvalue

        print(f"• Group A → Shapiro-Wilk p = {p1:.4f} →", 
              "Fail to reject H₀ ✅ (likely a normal distribution)" if p1 > 0.05 
              else "Reject H₀ ⚠️ (likely a non-normal distribution)")

        print(f"• Group B → Shapiro-Wilk p = {p2:.4f} →", 
              "Fail to reject H₀ ✅ (likely a normal distribution)" if p2 > 0.05 
              else "Reject H₀ ⚠️ (likely a non-normal distribution)")

        if p1 > 0.05 and p2 > 0.05:
            print("✅ Both groups are likely drawn from normal distributions")
            config['distribution'] = 'normal'
        else:
            print("⚠️ At least one group does not appear normally distributed")
            config['distribution'] = 'non-normal'

        print(f"📦 Final Decision → config['distribution'] = `{config['distribution']}`")
        return config['distribution']

    elif group_count == 'multi-sample':
        if 'group' not in df.columns:
            print("❌ Multi-sample distribution check requires 'group' column.")
            config['distribution'] = None
            return config['distribution']
        print("• Multi-sample case → testing each group")
        groups = df['group'].unique()
        all_normal = True
        for grp in groups:
            series = df[df['group'] == grp]['value']
            p = shapiro(series).pvalue
            status = "Fail to reject H₀ ✅ (likely normal)" if p > 0.05 else "Reject H₀ ⚠️ (likely non-normal)"
            print(f"• Group {grp} → Shapiro-Wilk p = {p:.4f} → {status}")
            if p <= 0.05:
                all_normal = False
        if all_normal:
            print("✅ All groups are likely drawn from normal distributions")
            config['distribution'] = 'normal'
        else:
            print("⚠️ At least one group does not appear normally distributed")
            config['distribution'] = 'non-normal'
        print(f"📦 Final Decision → config['distribution'] = `{config['distribution']}`")
        return config['distribution']

    else:
        print("❌ Unsupported group count for distribution check.")
        config['distribution'] = None
        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return config['distribution']

# ==========================================================
# region EDA functions - Normality Infer KS Test
# ==========================================================
def infer_distribution_from_data_ks(config, df):
    """
    Infers whether the outcome variable follows a normal distribution using the
    Kolmogorov-Smirnov (KS) test against a fitted normal distribution.

    This function:
    - Checks if the outcome type is continuous (required for normality testing)
    - Fits a normal distribution using sample mean and std
    - Applies KS test to the outcome (one group, both groups, or each group for multi-sample)
    - Updates config['distribution'] as 'normal', 'non-normal', or None (when not applicable)
    - Logs interpretation clearly for readability

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing 'outcome_type', 'group_count', and 'group_relationship'
    df : pandas.DataFrame
        Input dataframe containing outcome values and group assignments

    Returns:
    --------
    str
        The inferred distribution: 'normal', 'non-normal', or None (when not applicable).
        Callers typically assign to config, e.g. config['distribution'] = infer_distribution_from_data_ks(config, df).
    """

    print("\n🔍 Step: Infer Distribution of Outcome Variable (KS Test)")

    group_count = config['group_count']
    relationship = config['group_relationship']
    outcome = config['outcome_type']

    if outcome != 'continuous':
        print(f"⚠️ Skipping: Outcome type = `{outcome}` → normality check not applicable.")
        config['distribution'] = None
        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return config['distribution']

    print("📘 Checking if the outcome variable follows a normal distribution")
    print("   Using Kolmogorov-Smirnov Test (against fitted normal)")
    print("   H₀: Data comes from a normal distribution")
    print("   H₁: Data does NOT come from a normal distribution\n")

    def ks_normal_test(sample):
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        stat, p = kstest(sample, 'norm', args=(mean, std))
        return p

    if group_count == 'one-sample':
        print("• One-sample case → testing entire column")

        p = ks_normal_test(df['value'])
        print(f"• KS p-value = {p:.4f}")

        if p > 0.05:
            print("✅ Fail to reject H₀ → Data is likely a normal distribution")
            config['distribution'] = 'normal'
        else:
            print("⚠️ Reject H₀ → Data is likely a non-normal distribution")
            config['distribution'] = 'non-normal'

        print(f"📦 Final Decision → config['distribution'] = `{config['distribution']}`")
        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return config['distribution']

    elif group_count == 'two-sample':
        print(f"• Two-sample ({relationship}) case → testing both groups")

        if relationship == 'independent':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
        elif relationship == 'paired':
            a = df['group_A']
            b = df['group_B']
        else:
            print("❌ Invalid group relationship")
            config['distribution'] = None
            # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
            return config['distribution']

        p1 = ks_normal_test(a)
        p2 = ks_normal_test(b)

        print(f"• Group A → KS p = {p1:.4f} →",
              "Fail to reject H₀ ✅ (likely normal)" if p1 > 0.05
              else "Reject H₀ ⚠️ (likely non-normal)")

        print(f"• Group B → KS p = {p2:.4f} →",
              "Fail to reject H₀ ✅ (likely normal)" if p2 > 0.05
              else "Reject H₀ ⚠️ (likely non-normal)")

        if p1 > 0.05 and p2 > 0.05:
            print("✅ Both groups are likely drawn from normal distributions")
            config['distribution'] = 'normal'
        else:
            print("⚠️ At least one group does not appear normally distributed")
            config['distribution'] = 'non-normal'

        print(f"📦 Final Decision → config['distribution'] = `{config['distribution']}`")
        return config['distribution']

    elif group_count == 'multi-sample':
        if 'group' not in df.columns:
            print("❌ Multi-sample distribution check requires 'group' column.")
            config['distribution'] = None
            return config['distribution']
        print("• Multi-sample case → testing each group")
        groups = df['group'].unique()
        all_normal = True
        for grp in groups:
            series = df[df['group'] == grp]['value']
            p = ks_normal_test(series)
            status = "Fail to reject H₀ ✅ (likely normal)" if p > 0.05 else "Reject H₀ ⚠️ (likely non-normal)"
            print(f"• Group {grp} → KS p = {p:.4f} → {status}")
            if p <= 0.05:
                all_normal = False
        if all_normal:
            print("✅ All groups are likely drawn from normal distributions")
            config['distribution'] = 'normal'
        else:
            print("⚠️ At least one group does not appear normally distributed")
            config['distribution'] = 'non-normal'
        print(f"📦 Final Decision → config['distribution'] = `{config['distribution']}`")
        return config['distribution']

    else:
        print("❌ Unsupported group count for distribution check.")
        config['distribution'] = None
        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return config['distribution']


# ==========================================================
# region EDA functions - Normality Visualization
# ==========================================================
def qq_plot_normality(config, df):
    """
    Generates Q-Q plots to visually assess normality.

    - For one-sample: plots entire dataset
    - For two-sample: plots each group separately
    - For multi-sample: plots one Q-Q per group in a row
    """

    print("\n📊 Step: Visual Normality Check using Q-Q Plot")
    print("If points fall approximately along the straight line → data is likely normal.\n")

    group_count = config['group_count']
    relationship = config['group_relationship']
    outcome = config['outcome_type']

    if outcome != 'continuous':
        print(f"⚠️ Skipping: Outcome type = `{outcome}` → Q-Q plot not applicable.")
        return

    if group_count == 'one-sample':
        plt.figure(figsize=(6, 6))
        stats.probplot(df['value'], dist="norm", plot=plt)
        plt.title("Q-Q Plot (One-Sample)")
        plt.show()

    elif group_count == 'two-sample':
        if relationship == 'independent':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
        elif relationship == 'paired':
            a = df['group_A']
            b = df['group_B']
        else:
            print("❌ Invalid group relationship.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        stats.probplot(a, dist="norm", plot=axes[0])
        axes[0].set_title("Q-Q Plot: Group A")

        stats.probplot(b, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot: Group B")

        plt.tight_layout()
        plt.show()

    elif group_count == 'multi-sample':
        if 'group' not in df.columns:
            print("❌ Multi-sample Q-Q plot requires 'group' column.")
            return
        groups = df['group'].unique()
        n_groups = len(groups)
        fig_width = min(4 * n_groups, 16)
        fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 5))
        if n_groups == 1:
            axes = [axes]
        for i, grp in enumerate(groups):
            series = df[df['group'] == grp]['value']
            stats.probplot(series, dist="norm", plot=axes[i])
            axes[i].set_title(f"Q-Q Plot: Group {grp}")
        plt.tight_layout()
        plt.show()

    else:
        print("❌ Unsupported group count.")

# ==========================================================
# region EDA functions - Variance
# ==========================================================
def infer_variance_equality(config, df):
    """
    Infers whether the variances across independent groups are equal using Levene's test.

    This function:
    - Checks if the variance assumption is relevant (two-sample or multi-sample independent)
    - Runs Levene's test to compare variances across groups
    - Updates the 'variance_equal' key in the config as 'equal', 'unequal', or None (when not applicable)
    - Logs interpretation of the test result

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing 'group_count' and 'group_relationship'
    df : pandas.DataFrame
        Input dataframe containing 'group' and 'value' columns

    Returns:
    --------
    str
        The inferred variance assumption: 'equal', 'unequal', or None (when not applicable).
        Callers typically assign to config, e.g. config['variance_equal'] = infer_variance_equality(config, df).
    """
    print("\n📏 **Step: Infer Equality of Variance Across Groups**")

    group_count = config['group_count']
    relationship = config['group_relationship']

    # Skip if not applicable (only for independent two-sample or multi-sample)
    if relationship != 'independent':
        print("⚠️ Skipping variance check: Only applicable for independent groups (two-sample or multi-sample).")
        config['variance_equal'] = None
        return config['variance_equal']
    if group_count not in ('two-sample', 'multi-sample'):
        print("⚠️ Skipping variance check: Only applicable for two-sample or multi-sample independent tests.")
        config['variance_equal'] = None
        return config['variance_equal']
    if 'group' not in df.columns:
        print("⚠️ Skipping variance check: 'group' column required.")
        config['variance_equal'] = None
        return config['variance_equal']

    groups = df['group'].unique()
    n_groups = len(groups)

    print("📘 We're checking if the spread (variance) of the outcome variable is similar across all groups.")
    if group_count == 'two-sample':
        print("   This is important for choosing between a **pooled t-test** vs **Welch's t-test**.")
    else:
        print("   This is important for choosing between **ANOVA** vs **Welch ANOVA**.")
    print("🔬 Test Used: Levene's Test for Equal Variance")
    print("   H₀: Variances are equal across groups")
    print("   H₁: Variances are not equal across groups")

    # Extract data: one array per group
    group_arrays = [df[df['group'] == g]['value'].values for g in groups]

    # Run Levene's test (supports 2 or more groups)
    stat, p = levene(*group_arrays)
    print(f"\n📊 Levene's Test Result:")
    print(f"• Test Statistic = {stat:.4f}")
    print(f"• p-value        = {p:.4f}")

    if p > 0.05:
        print("✅ Fail to reject H₀ → Variances appear equal across groups")
        config['variance_equal'] = 'equal'
    else:
        print("⚠️ Reject H₀ → Variances appear unequal across groups")
        config['variance_equal'] = 'unequal'

    print(f"\n📦 Final Decision → config['variance_equal'] = `{config['variance_equal']}`")
    return config['variance_equal']

# ==========================================================
# region EDA functions - Visualization
# ==========================================================
def visualize_distribution(config, df):
    """
    Shows distributions side-by-side for comparison.

    - For one-sample: single panel
    - For two-sample: two panels (Group A, Group B)
    - For multi-sample: one panel per group in a row
    """

    print("\n📊 Step: Visual Distribution Overview (Side-by-Side)\n")

    group_count = config['group_count']
    relationship = config['group_relationship']
    outcome = config['outcome_type']

    if outcome != 'continuous':
        print(f"⚠️ Skipping: Outcome type = `{outcome}` → Not applicable.")
        return

    def plot_on_axis(sample, ax, title):
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        median = np.median(sample)

        sns.histplot(sample, kde=True, stat='density', bins=20, ax=ax)

        x = np.linspace(min(sample), max(sample), 200)
        ax.plot(x, norm.pdf(x, mean, std), linestyle='--')

        ax.axvline(mean, linestyle='-', label='Mean')
        ax.axvline(median, linestyle=':', label='Median')

        ax.set_title(title)
        ax.legend()

    if group_count == 'one-sample':
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plot_on_axis(df['value'], ax, "Distribution")
        plt.tight_layout()
        plt.show()

    elif group_count == 'two-sample':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if relationship == 'independent':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']

        elif relationship == 'paired':
            a = df['group_A']
            b = df['group_B']

        else:
            print("❌ Invalid group relationship.")
            return

        plot_on_axis(a, axes[0], "Group A")
        plot_on_axis(b, axes[1], "Group B")

        # Standardize y-axis (density) so both panels are comparable
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(0, y_max)
        axes[1].set_ylim(0, y_max)

        plt.tight_layout()
        plt.show()

    elif group_count == 'multi-sample':
        if 'group' not in df.columns:
            print("❌ Multi-sample distribution plot requires 'group' column.")
            return
        groups = df['group'].unique()
        n_groups = len(groups)
        fig_width = min(4 * n_groups, 16)
        fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 5))
        if n_groups == 1:
            axes = [axes]
        for i, grp in enumerate(groups):
            series = df[df['group'] == grp]['value']
            plot_on_axis(series, axes[i], f"Group {grp}")
        # Standardize y-axis across panels
        y_max = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(0, y_max)
        plt.tight_layout()
        plt.show()

    else:
        print("❌ Unsupported group count.")

def visualize_variance_boxplot_annotated(config, df):
    """
    Boxplot of outcome by group with std/IQR annotations.
    Applicable for two-sample independent and multi-sample independent (spread comparison across groups).
    """
    print("\n📊 Visual Check: Spread Comparison Between Groups")
    print("   (Spread = how much values vary within each group)\n")

    group_count = config['group_count']
    relationship = config['group_relationship']

    if (group_count == 'two-sample' and relationship == 'independent') or (
            group_count == 'multi-sample' and relationship == 'independent'):

        if 'group' not in df.columns:
            print("⚠️ Spread comparison requires 'group' column.")
            return

        summary = df.groupby('group')['value'].agg(['std', 'var', 'median',
                                                    lambda x: np.percentile(x, 75) - np.percentile(x, 25)])
        summary.columns = ['std_dev', 'variance', 'median', 'IQR']

        print("📋 Spread Summary:")
        display(summary)

        n_groups = len(summary.index)
        fig_width = min(4 * n_groups, 16)
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        sns.boxplot(x='group', y='value', data=df, ax=ax)

        # Annotate each group
        for i, group in enumerate(summary.index):
            std = summary.loc[group, 'std_dev']
            iqr = summary.loc[group, 'IQR']
            ax.text(i,
                    df['value'].max() * 0.95,
                    f"Std Dev: {std:.2f}\nIQR: {iqr:.2f}",
                    horizontalalignment='center',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6))

        ax.set_title("Comparison of Value Spread by Group")
        ax.set_ylabel("Outcome Value")
        ax.set_xlabel("Group")
        fig.set_size_inches(fig_width, 5)
        plt.tight_layout()
        plt.show()

        # Business interpretation
        std_values = summary['std_dev']
        ratio = std_values.max() / std_values.min()

        print("\n🧠 Business Interpretation:")
        if ratio < 1.2:
            print("✅ The spread of values across groups is very similar.")
            print("   Variability does not appear meaningfully different.")
        elif ratio < 1.5:
            print("⚠️ One group shows moderately higher variability.")
            print("   Worth confirming with a statistical variance test.")
        else:
            print("🚨 One group shows substantially higher variability.")
            print("   Statistical tests assuming equal variance may not be appropriate.")

    else:
        print("⚠️ Spread comparison is only applicable for independent groups (two-sample or multi-sample).")

# ==========================================================
# region Test Determination Functions
# ==========================================================
def infer_parametric_flag(config):

    print("\n📏 Step: Decide Between Parametric vs Non-Parametric Approach")

    if config['outcome_type'] != 'continuous':
        print(f"⚠️ Skipping: Outcome type = `{config['outcome_type']}` → Parametric logic not applicable.")
        config['parametric'] = None
        return config['parametric']

    print(f"🔍 Distribution of outcome = `{config['distribution']}`")

    if config['distribution'] == 'normal':
        print("✅ Normal distribution → Proceeding with a parametric test")
        config['parametric'] = True
    else:
        print("⚠️ Non-normal distribution → Using non-parametric alternative")
        config['parametric'] = False

    print(f"\n📦 Final Decision → config['parametric'] = `{config['parametric']}`")
    # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))

    return config['parametric']

def determine_test_to_run(config):
    """
    Determines the appropriate statistical test based on the provided configuration.

    This function:
    - Maps outcome type, group count, group relationship, distribution, and parametric flags
    to the correct hypothesis test
    - Prints the reasoning and selected test
    - Returns a string identifier for the test to be used

    Parameters:
    -----------
    config : dict
        A dictionary containing keys like 'outcome_type', 'group_count', 'group_relationship',
        'distribution', 'variance_equal', and 'parametric'.

    Returns:
    --------
    str
        A string representing the selected test name (e.g., 'two_sample_ttest_welch', 'mcnemar', etc.)
    """

    print("\n🧭 Step: Determine Which Statistical Test to Use")
    
    outcome = config['outcome_type']
    group_rel = config['group_relationship']
    group_count = config['group_count']
    dist = config['distribution']
    equal_var = config['variance_equal']
    parametric = config['parametric']

    print("📦 Inputs:")
    print(f"• Outcome Type         = `{outcome}`")
    print(f"• Group Count          = `{group_count}`")
    print(f"• Group Relationship   = `{group_rel}`")
    print(f"• Distribution         = `{dist}`")
    print(f"• Equal Variance       = `{equal_var}`")
    print(f"• Parametric Flag      = `{parametric}`")

    print("\n🔍 Matching against known test cases...")

    # One-sample
    if group_count == 'one-sample':
        if outcome == 'continuous':
            test_name = 'one_sample_ttest' if dist == 'normal' else 'one_sample_wilcoxon'
        elif outcome == 'binary':
            test_name = 'one_proportion_ztest'
        else:
            test_name = 'test_not_found'

    # Two-sample independent
    elif group_count == 'two-sample' and group_rel == 'independent':
        if outcome == 'continuous':
            if parametric:
                test_name = 'two_sample_ttest_pooled' if equal_var == 'equal' else 'two_sample_ttest_welch'
            else:
                test_name = 'mann_whitney_u'
        elif outcome == 'binary':
            test_name = 'two_proportion_ztest'
        elif outcome == 'categorical':
            test_name = 'chi_square'
        elif outcome == 'count':
            test_name = 'poisson_test'
        else:
            test_name = 'test_not_found'

    # Two-sample paired
    elif group_count == 'two-sample' and group_rel == 'paired':
        if outcome == 'continuous':
            test_name = 'paired_ttest' if parametric else 'wilcoxon_signed_rank'
        elif outcome == 'binary':
            test_name = 'mcnemar'
        else:
            test_name = 'test_not_found'

    # Multi-group
    elif group_count == 'multi-sample':
        if outcome == 'continuous':
            if parametric:
                test_name = 'anova' if equal_var == 'equal' else 'welch_anova'
            else:
                test_name = 'kruskal_wallis'
        elif outcome == 'binary':
            test_name = 'chi_square'
        elif outcome == 'categorical':
            test_name = 'chi_square'
        elif outcome == 'count':
            test_name = 'poisson_test'
        else:
            test_name = 'test_not_found'

    else:
        test_name = 'test_not_found'

    print(f"\n✅ Selected Test: `{test_name}`")
    # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
    return test_name

def print_hypothesis_statement(config):
    """
    Prints the null and alternative hypothesis statements based on the selected test and tail type.

    This function:
    - Uses the provided config to determine which statistical test applies
    - Displays clear H₀ and H₁ statements tailored to the test type and direction (one-tailed vs two-tailed)
    - Aims to bridge technical and business understanding of what is being tested

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing at least 'tail_type' and the required fields to determine the test

    Returns:
    --------
    tuple of (str, str)
        (H_0, H_a) — the null and alternative hypothesis strings.
        Callers may assign to config, e.g. config['H_0'], config['H_a'] = print_hypothesis_statement(config).
        Also prints the statements to output.
    """

    print("\n🧠 Step: Generate Hypothesis Statement")

    tail = config['tail_type']
    test_name = config['test_name']
    print(f"🔍 Selected Test        : `{test_name}`")
    print(f"🔍 Tail Type            : `{tail}`\n")

    H_0 = None
    H_a = None

    if test_name == 'one_sample_ttest':
        H_0 = "The sample mean equals the reference value."
        H_a = "The sample mean is different from the reference." if tail == 'two-tailed' else "The sample mean is greater/less than the reference."

    elif test_name == 'one_sample_wilcoxon':
        H_0 = "The population median (or symmetric center) equals the reference value."
        H_a = "The median is different from the reference." if tail == 'two-tailed' else "The median is greater/less than the reference."

    elif test_name == 'one_proportion_ztest':
        H_0 = "The sample proportion equals the baseline rate."
        H_a = "The sample proportion is different from the baseline." if tail == 'two-tailed' else "The sample proportion is greater/less than the baseline."

    elif test_name in ['two_sample_ttest_pooled', 'two_sample_ttest_welch', 'mann_whitney_u', 'two_proportion_ztest']:
        H_0 = "The outcome (mean/proportion) is the same across groups A and B."
        H_a = "The outcome differs between groups." if tail == 'two-tailed' else "Group B is greater/less than Group A."

    elif test_name in ['paired_ttest', 'wilcoxon_signed_rank']:
        H_0 = "The average difference between paired values (before vs after) is zero."
        H_a = "There is a difference in paired values." if tail == 'two-tailed' else "After is greater/less than before."

    elif test_name == 'mcnemar':
        H_0 = "Proportion of success is the same before and after."
        H_a = "Proportion of success changed after treatment."

    elif test_name in ['anova', 'welch_anova', 'kruskal_wallis']:
        H_0 = "All group means (or distributions) are equal."
        H_a = "At least one group differs."

    elif test_name == 'chi_square':
        H_0 = "The distribution of the outcome (across categories) is the same in all groups."
        H_a = "The distribution differs across groups."

    elif test_name == 'poisson_test':
        H_0 = "The count rate (λ) is the same across groups."
        H_a = "Count rate differs between groups."

    elif test_name == 'bayesian_ab':
        H_0 = "Posterior probability that Group B is not better than Group A."
        H_a = "Posterior probability that Group B is better than Group A."

    elif test_name == 'permutation_test':
        H_0 = "Observed difference is due to chance."
        H_a = "Observed difference is unlikely under random shuffling."

    else:
        H_0 = f"Unable to generate hypothesis statement for test: `{test_name}`"
        H_a = ""

    print("📜 Hypothesis Statement:")
    print(f"• H₀: {H_0}")
    print(f"• H₁: {H_a}")

    return H_0, H_a

# ==========================================================
# region Conduct Hypothesis Test - Visualization
# ==========================================================
def visualize_test_result(stat, alpha, test_label, tail='two-tailed', df1=None, df2=None):
    """
    Plot the null distribution and show where the observed statistic falls.

    Supported test_label values (valid 5-key tests):
    - t-statistic, z-statistic, F-statistic → full plot
    - Chi-square statistic (contingency, mcnemar, LRT) → chi2 plot (pass df1 for df)
    - U-, W-, H-statistic (Mann-Whitney, Wilcoxon signed-rank, Kruskal-Wallis) → no plot (message only)
    """
    plt.figure(figsize=(10, 5))

    # ----- Select Distribution -----
    if test_label == 't-statistic':
        df_used = df1 if df1 else 30
        dist = t(df=df_used)
        x = np.linspace(-4, 4, 400)
        y = dist.pdf(x)
        critical = dist.ppf(1 - alpha/2)

    elif test_label == 'z-statistic':
        dist = norm()
        x = np.linspace(-4, 4, 400)
        y = dist.pdf(x)
        critical = dist.ppf(1 - alpha/2)

    elif test_label == 'F-statistic':
        df1_used = df1 if df1 else 2
        df2_used = df2 if df2 else 30
        dist = f(df1_used, df2_used)
        x = np.linspace(0, 5, 400)
        y = dist.pdf(x)
        critical = dist.ppf(1 - alpha)

    elif test_label is not None and 'Chi-square' in str(test_label):
        # Chi-square statistic (contingency, mcnemar, LRT): one-tailed upper
        df_used = int(df1) if df1 is not None else 1
        dist = chi2(df=df_used)
        x = np.linspace(0, max(15, df_used * 2 + 5), 400)
        y = dist.pdf(x)
        critical = dist.ppf(1 - alpha)

    else:
        # Non-parametric rank tests (U, W, H) have no simple null-distribution plot here
        print("Visualization not supported for this test statistic (no null-distribution plot).")
        return

    # ----- Plot Curve -----
    plt.plot(x, y)

    # One-tailed upper (F, Chi-square); two-tailed (t, z)
    one_tailed_upper = test_label == 'F-statistic' or (test_label is not None and 'Chi-square' in str(test_label))

    # ----- Shade Rejection Region -----
    if not one_tailed_upper:
        plt.fill_between(x, y, where=(x <= -critical), alpha=0.3)
        plt.fill_between(x, y, where=(x >= critical), alpha=0.3)
    else:
        plt.fill_between(x, y, where=(x >= critical), alpha=0.3)

    # ----- Observed Statistic -----
    plt.axvline(stat, linewidth=3)
    plt.axvline(critical, linestyle='--')
    if not one_tailed_upper:
        plt.axvline(-critical, linestyle='--')

    # ----- Business Annotations -----
    plt.text(stat, max(y)*0.6, f"Observed {test_label}\n({stat:.2f})",
             ha='right' if stat < 0 else 'left')

    plt.text(critical, max(y)*0.1,
             f"Critical cutoff\n(α = {alpha})",
             ha='left')

    if not one_tailed_upper:
        plt.text(-critical, max(y)*0.1,
                 f"Critical cutoff\n(α = {alpha})",
                 ha='right')

    if one_tailed_upper:
        in_rejection = stat >= critical
    else:
        in_rejection = abs(stat) > critical
    if in_rejection:
        decision_text = "Result falls inside rejection region → Reject H₀"
    else:
        decision_text = "Result falls inside safe region → Fail to Reject H₀"

    plt.text(0, max(y)*0.85, decision_text, ha='center')

    plt.title(f"{test_label} — What This Decision Means")
    plt.xlabel("Test Statistic Value")
    plt.ylabel("Probability Density")

    plt.show()


# ==========================================================
# region Conduct Hypothesis Test
# ==========================================================
def run_hypothesis_test(config, df):
    """
    Runs the appropriate hypothesis test based on the provided configuration and dataset.

    This function:
    - Identifies the correct test using `determine_test_to_run(config)`
    - Executes the corresponding statistical test (e.g., t-test, z-test, Mann-Whitney, ANOVA, etc.)
    - Prints a guided explanation of inputs, selected test, test statistic, p-value, and business interpretation
    - Returns a result dictionary with test details and significance flag

    Parameters:
    -----------
    config : dict
        Dictionary specifying test configuration (e.g., outcome_type, group_relationship, parametric, etc.)
    df : pandas.DataFrame
        Input dataset containing outcome values (and group labels if applicable)

    Returns:
    --------
    dict
        {
            'test': str,              # Name of the statistical test run
            'statistic': float,       # Test statistic
            'p_value': float,         # P-value of the test
            'significant': bool,      # True if p < alpha
            'alpha': float            # Significance threshold used
        }
    """
    # Re-print hypothesis for clarity before execution
    print("📜 Hypothesis Being Tested:")
    print(f"• H₀: {config['H_0']}")
    print(f"• H₁: {config['H_a']}\n")
    
    print("\n🧪 Step: Run Hypothesis Test")

    alpha = config.get('alpha', 0.05)
    test_name = config['test_name']

    test_statistic_map = {
        'one_sample_ttest': 't-statistic',
        'two_sample_ttest_pooled': 't-statistic',
        'two_sample_ttest_welch': 't-statistic',
        'paired_ttest': 't-statistic',
        'anova': 'F-statistic',
        'welch_anova': 'F-statistic',
        'mann_whitney_u': 'U-statistic',
        'wilcoxon_signed_rank': 'W-statistic',
        'kruskal_wallis': 'H-statistic',
        'chi_square': 'Chi-square statistic',
        'mcnemar': 'Chi-square statistic',
        'one_proportion_ztest': 'z-statistic',
        'two_proportion_ztest': 'z-statistic',
        'poisson_test': 'Chi-square statistic (LRT)'
    }
    
    test_statistic_label = test_statistic_map.get(test_name, 'Test statistic')

    print(f"✅ Selected Test        : `{test_name}`")
    print(f"🔍 Significance Threshold (α) : {alpha:.2f}\n")

    result = {
        'test': test_name,
        'statistic': None,
        'p_value': None,
        'significant': None,
        'alpha': alpha
    }

    try:
        print("🚀 Executing statistical test...")

        # --- Run appropriate test ---
        if test_name == 'one_sample_ttest':
            stat, p = ttest_1samp(df['value'], config['population_mean'])

        elif test_name == 'one_sample_wilcoxon':
            stat, p = wilcoxon(df['value'] - config['population_mean'])

        elif test_name == 'one_proportion_ztest':
            x = np.sum(df['value'])
            n = len(df)
            stat, p = proportions_ztest(x, n, value=config['population_mean'])

        elif test_name == 'two_sample_ttest_pooled':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
            stat, p = ttest_ind(a, b, equal_var=True)

        elif test_name == 'two_sample_ttest_welch':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
            stat, p = ttest_ind(a, b, equal_var=False)

        elif test_name == 'mann_whitney_u':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
            stat, p = mannwhitneyu(a, b, alternative='two-sided')

        elif test_name == 'paired_ttest':
            stat, p = ttest_rel(df['group_A'], df['group_B'])

        elif test_name == 'wilcoxon_signed_rank':
            stat, p = wilcoxon(df['group_A'], df['group_B'])

        elif test_name == 'two_proportion_ztest':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
            counts = [np.sum(a), np.sum(b)]
            nobs = [len(a), len(b)]
            stat, p = proportions_ztest(count=counts, nobs=nobs)

        elif test_name == 'mcnemar':
            before = df['group_A']
            after = df['group_B']
            both = np.sum((before == 1) & (after == 1))
            before_only = np.sum((before == 1) & (after == 0))
            after_only = np.sum((before == 0) & (after == 1))
            neither = np.sum((before == 0) & (after == 0))
            table = np.array([[both, before_only], [after_only, neither]])
            stat, p = chi2_contingency(table, correction=True)[:2]

        elif test_name == 'anova':
            groups = [g['value'].values for _, g in df.groupby('group')]
            stat, p = f_oneway(*groups)

        elif test_name == 'welch_anova':
            groups = [g['value'].values for _, g in df.groupby('group')]
            res = anova_oneway(groups, use_var='unequal')
            stat = res.statistic
            p = res.pvalue
        
        elif test_name == 'kruskal_wallis':
            groups = [g['value'].values for _, g in df.groupby('group')]
            stat, p = kruskal(*groups)

        elif test_name == 'chi_square':
            contingency = pd.crosstab(df['group'], df['value'])
            stat, p, _, _ = chi2_contingency(contingency)

        elif test_name == 'poisson_test':
            # Compare Poisson rates across groups via GLM likelihood-ratio test
            y = df['value'].values
            X_dummies = pd.get_dummies(df['group'], drop_first=True)
            X = sm.add_constant(X_dummies)
            full = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            null_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson()).fit()
            lr_stat = 2 * (full.llf - null_model.llf)
            df_diff = full.df_model - null_model.df_model
            p = float(stats.chi2.sf(lr_stat, df_diff))
            stat = lr_stat

        else:
            warnings.warn(f"❌ Test not implemented: `{test_name}`")
            return result

        result['statistic'] = stat
        result['p_value'] = p
        result['significant'] = p < alpha

        # --- Final Output Block ---
        print(f"\n📊 Test Summary: {test_name.replace('_', ' ').title()}")

        print("\n🧪 Technical Result")
        print(f"• Test Statistic ({test_statistic_label}) = {stat:.4f}")
        print(f"• P-value            = {p:.4f}")
        print(f"• Alpha (α)          = {alpha:.2f}")

        # Degrees of freedom for visualization (t and F distributions)
        vis_df1, vis_df2 = None, None
        if test_name == 'one_sample_ttest':
            vis_df1 = len(df) - 1
        elif test_name == 'two_sample_ttest_pooled':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
            vis_df1 = len(a) + len(b) - 2
        elif test_name == 'two_sample_ttest_welch':
            a = df[df['group'] == 'A']['value']
            b = df[df['group'] == 'B']['value']
            var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
            n_a, n_b = len(a), len(b)
            num = (var_a / n_a + var_b / n_b) ** 2
            den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
            vis_df1 = num / den if den > 0 else (n_a + n_b - 2)
        elif test_name == 'paired_ttest':
            vis_df1 = len(df) - 1
        elif test_name in ('anova', 'welch_anova'):
            k = df['group'].nunique()
            n = len(df)
            vis_df1 = k - 1
            vis_df2 = n - k
        elif test_name == 'chi_square':
            contingency = pd.crosstab(df['group'], df['value'])
            vis_df1 = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
        elif test_name == 'mcnemar':
            vis_df1 = 1
        elif test_name == 'poisson_test':
            vis_df1 = df['group'].nunique() - 1

        visualize_test_result(
            stat,
            alpha,
            test_statistic_label,
            tail=config.get('tail_type', 'two-tailed'),
            df1=vis_df1,
            df2=vis_df2)

        if p < alpha:
            print(f"• Conclusion         = ✅ Statistically significant → Reject H₀")
            print("\n📈 Interpretation")
            print("• The observed difference is unlikely due to random variation.")

            # Business Interpretation
            if config['group_count'] == 'two-sample' and config['group_relationship'] == 'independent':
                a = df[df['group'] == 'A']['value']
                b = df[df['group'] == 'B']['value']
                mean_a = np.mean(a)
                mean_b = np.mean(b)
                lift = mean_b - mean_a
                pct_lift = (lift / mean_a) * 100

                label = "mean" if config['outcome_type'] == 'continuous' else 'conversion rate'
                print("\n💼 Business Insight")
                print(f"• Group A {label} = {mean_a:.2f}")
                print(f"• Group B {label} = {mean_b:.2f}")
                print(f"• Lift = {lift:.2f} ({pct_lift:+.2f}%)")

                if lift > 0:
                    print("🏆 Group B outperforms Group A — and the effect is statistically significant.")
                else:
                    print("📉 Group B underperforms Group A — and the drop is statistically significant.")
        else:
            print(f"• Conclusion         = ❌ Not statistically significant → Fail to reject H₀")
            print("\n📈 Interpretation")
            print("• The observed difference could be explained by randomness.")
            print("\n💼 Business Insight")
            print("• No strong evidence of difference between the groups.")

        # display(HTML("<hr style='border: none; height: 1px; background-color: #ddd;' />"))
        return result

    except Exception as e:
        warnings.warn(f"🚨 Error during test execution: {e}")
        return result
