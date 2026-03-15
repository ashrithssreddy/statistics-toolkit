# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python [conda env:base]
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# <a id="table-of-contents"></a>
#
# ![Status: Complete](https://img.shields.io/badge/status-complete-brightgreen)
# ![Python](https://img.shields.io/badge/python-3.10-blue)
# ![Coverage](https://img.shields.io/badge/coverage-95%25-yellowgreen)
# ![License](https://img.shields.io/badge/license-MIT-green)
#
# <h1>📖 AB Testing</h1>
#
# [🗂️ Data Setup](#data-setup)  
# - [⚙️ Environment Setup](#environment-setup)  
# - [🛠️ Experiment Setup](#experiment-setup)  
# - [🔧 Central Control Panel](#central-control-panel)
# - [📥 Read/Generate Data](#read-data)
#
# [⚡ Power Analysis](#power-analysis)  
# - [⚙️ Setup Inputs + Config Values](#setup-inputs--config-values)  
# - [📈 Baseline Estimation from Data](#baseline-from-data)  
# - [📈 Minimum Detectable Effect](#minimum-detectable-effect)  
# - [🔍 Test Family](#test-family)
# - [📐 Required Sample Size](#required-sample-size)  
#
# [🔀 Randomization](#randomization)  
# - [🎲 Apply Randomization](#apply-randomization)  
# - [🕸️ Network Effects & SUTVA Violations](#network-effects)
# - [⚖️ Sample Ratio Mismatch](#sample-ratio-mismatch)
#
# [📈 EDA](#eda)  
# - [🔍 Normality](#normality)  
# - [🔍 Variance Homogeneity Check](#variance-homogeneity-check)  
#
# [🧪 AA Testing](#aa-testing)
# - [🧬 Outcome Similarity Test](#outcome-similarity-test)
# - [📊 AA Test Visualization](#aa-test-visualization)
# - [🎲 Type I Error Simulation](#type-i-error-simulation)
#
# [🧪 A/B Testing](#ab-testing)
# - [🧾 Summaries](#summaries)  
# - [📊 Visualization](#results-visualization)  
# - [🎯 95% Confidence Intervals](#confidence-intervals)  
# - [📈 Lift Analysis](#lift-analysis)  
# - [✅ Final Conclusion](#final-conclusion)
# - [⏱️ How Long](#how-long)
#
# [🔍 Post Hoc Analysis](#post-hoc-analysis)  
# - [🧩 Segmented Lift](#segmented-lift)  
# - [🚦 Guardrail Metrics](#guardrail-metrics)  
# - [🔄 CUPED](#cuped)
# - [🧠 Correcting for Multiple Comparisons](#multiple-comparisons)
# - [🪄 Novelty Effects & Behavioral Decay](#novelty-effects)
# - [🎯 Primacy Effect & Order Bias](#primacy-effect)
# - [🎲 Rollout Simulation](#rollout-simulation)
# - [🧪 A/B Test Holdouts](#ab-test-holdouts)
# - [🚫 Limits & Alteratives](#ab-test-limits)
#
# <hr style="border: none; height: 1px; background-color: #ddd;">
#

# %% [markdown]
# <a id="data-setup"></a>
# <h1>🗂️ Data Setup</h1>

# %% [markdown]
# <a id="environment-setup"></a>
# <h4>⚙️ Environment Setup</h4>

# %%
# Display Settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display, HTML

# Set Seed 
my_seed=1995

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
from scipy.stats import (
    ttest_ind,
    ttest_rel,
    chi2_contingency,
    mannwhitneyu,
    levene,
    shapiro
)
import statsmodels.api as sm
from statsmodels.stats.power import (
    TTestIndPower,
    TTestPower,
    FTestAnovaPower,
    NormalIndPower
)
from sklearn.model_selection import train_test_split

from ab_utils import *
import sys
sys.path.insert(0, '..')
from Hypothesis_Testing.ht_utils import print_config_summary

# %% [markdown]
# <a id="experiment-setup"></a>
# <h4>🛠️ Experiment Setup</h4>

# %%
# 1. Main outcome variable you're testing
outcome_metric_col = 'engagement_score'

# 2. Metric type: 'binary', 'continuous', or 'categorical'
outcome_metric_datatype = 'continuous'

# 3. Group assignment (to be generated)
group_labels = ('control', 'treatment')

# 3b. Number of groups in the experiment (e.g., 2 for A/B test, 3 for A/B/C test)
group_count = len(group_labels)

# 4. Experimental design variant: independent or paired
variant = 'independent'  # Options: 'independent' (supported), 'paired' (not supported yet, todo)

# 5. Optional: Unique identifier for each observation (can be user_id, session_id, etc.)
observation_id_col = 'user_id'

# 6. Optional: Pre-experiment metric for CUPED, if used
pre_experiment_metric = 'past_purchase_count'  # Can be None

# Column name used to store assigned group after randomization
group_col = 'group'

# Randomization method to assign users to groups
# Options: 'simple', 'stratified', 'block', 'matched_pair', 'cluster', 'cuped'
randomization_method = "simple"

# Optional: guardrail metric column for simulated outcome data. Set to None to omit.
guardrail_metric_col = 'bounce_rate'


# %% [markdown]
# <a id="central-control-panel"></a>
# <h4>🔧 Central Control Panel</h4>

# %%
test_config = {
    # Core experiment setup
    'outcome_metric_col'     : outcome_metric_col,         # Main metric to analyze (e.g., 'engagement_score')
    'observation_id_col'     : observation_id_col,         # Unique identifier for each observation
    'pre_experiment_metric'  : pre_experiment_metric,      # Used for CUPED adjustment (if any)
    'outcome_metric_datatype': outcome_metric_datatype,    # One of: 'binary', 'continuous', 'categorical'
    'group_labels'           : group_labels,               # Tuple of (control, treatment) group names
    'group_count'            : group_count,                # Number of groups (usually 2 for A/B tests)
    'variant'                : variant,                    # 'independent' or 'paired'
    'guardrail_metric_col'   : guardrail_metric_col,       # Optional: e.g. 'bounce_rate'; None to omit

    # Diagnostic results — filled after EDA/assumptions check
    'normality'              : None,  # Will be set based on Shapiro-Wilk or visual tests
    'equal_variance'         : None,  # Will be set using Levene’s/Bartlett’s test
    'family'                 : None   # Test family → 'z_test', 't_test', 'anova', 'chi_square', etc.
}

print_config_summary(test_config)

# %% [markdown]
# <a id="read-data"></a>
# <h4>📥 Read/Generate Data</h4>

# %%
observations_count = 1000
df = create_dummy_ab_data(observations_count, seed=my_seed, outcome_metric_col=outcome_metric_col, guardrail_metric_col=guardrail_metric_col)
historical_df = create_historical_df(df, outcome_metric_col, guardrail_metric_col, seed=my_seed)
df
historical_df

# %% [markdown]
# [Back to the top](#table-of-contents)
# ___
#

# %% [markdown]
# <a id="power-analysis"></a>
#
# <h1>⚡ Power Analysis</h1>
#
# <details><summary><strong>📖 Click to Expand </strong></summary>
#
# <p>Power analysis helps determine the <strong>minimum sample size</strong> required to detect a true effect with statistical confidence.</p>
#
# <h5>Why It Matters:</h5>
# <ul>
#   <li>Avoids <strong>underpowered tests</strong> (risk of missing real effects)</li>
#   <li>Balances tradeoffs between Sample size, Minimum Detectable Effect (MDE), Significance level (α), Statistical power (1 - β)</li>
# </ul>
#
# <h5>Key Inputs:</h5>
# <table>
# <thead>
# <tr>
#   <th>Parameter</th>
#   <th>Meaning</th>
# </tr>
# </thead>
# <tbody>
# <tr>
#   <td><strong>alpha (α)</strong></td>
#   <td>Significance level (probability of false positive), e.g. 0.05</td>
# </tr>
# <tr>
#   <td><strong>Power (1 - β)</strong></td>
#   <td>Probability of detecting a true effect, e.g. 0.80 or 0.90</td>
# </tr>
# <tr>
#   <td><strong>Baseline</strong></td>
#   <td>Current outcome (e.g., 10% conversion, $50 revenue)</td>
# </tr>
# <tr>
#   <td><strong>MDE</strong></td>
#   <td>Minimum detectable effect — the smallest meaningful lift (e.g., +2% or +$5)</td>
# </tr>
# <tr>
#   <td><strong>Std Dev</strong></td>
#   <td>Standard deviation of the metric (for continuous outcomes)</td>
# </tr>
# <tr>
#   <td><strong>Effect Size</strong></td>
#   <td>Optional: Cohen's d (for t-tests) or f (for ANOVA)</td>
# </tr>
# <tr>
#   <td><strong>Groups</strong></td>
#   <td>Number of groups (relevant for ANOVA)</td>
# </tr>
# </tbody>
# </table>
#
# <p>This notebook automatically selects the correct formula based on <code>experiment_type</code> variable.</p>
#
# </details>
#

# %% [markdown]
# <a id="setup-inputs--config-values"></a>
#
# <h4>⚙️ Setup Inputs + Config Values</h4>
#
# <details><summary><strong>📖 Click to Expand </strong></summary>
#
# <p>These are the <strong>core experiment design parameters</strong> required for power analysis and statistical testing.</p>
#
# <ul>
#   <li><code>alpha</code>: Significance level — the <strong>tolerance for false positives</strong> (commonly set at 0.05).</li>
#   <li><code>power</code>: Probability of detecting a true effect — typically <strong>0.80 or 0.90</strong>.</li>
#   <li><code>group_labels</code>: The names of the experimental groups (e.g., <code>'control'</code>, <code>'treatment'</code>).</li>
#   <li><code>metric_col</code>: Outcome metric column you're analyzing.</li>
#   <li><code>test_family</code>: Chosen statistical test (e.g., <code>'t_test'</code>, <code>'z_test'</code>, <code>'chi_square'</code>) based on assumptions.</li>
#   <li><code>variant</code>: Experimental design structure — <code>'independent'</code> or <code>'paired'</code>.</li>
# </ul>
#
# <p>These inputs drive sample size estimation, test choice, and downstream analysis logic.</p>
#
# </details>
#

# %%
# Define Core Inputs

# Use values from your config or plug in manually
alpha = 0.05  # False positive tolerance (Type I error)
power = 0.80  # Statistical power (1 - Type II error)


# %% [markdown]
# <a id="baseline-from-data"></a>
#
# <h4>📈 Baseline Estimation (Pre-Experiment)</h4>
#
# <details>
# <summary><strong>📖 Click to Expand </strong></summary>
#
# <p>Before running power analysis, we need a <strong>baseline estimate</strong> of the outcome metric.</p>
#
# <ul>
#   <li>These values must come from <strong>historical data collected before the experiment</strong>.</li>
#   <li>They represent the expected behavior of users under the <strong>current system (control condition)</strong>.</li>
# </ul>
#
# <ul>
#   <li>For <strong>binary metrics</strong> (e.g., conversion), the baseline is the historical <strong>conversion rate</strong>.</li>
#   <li>For <strong>continuous metrics</strong> (e.g., revenue, engagement), we estimate the historical <strong>mean and standard deviation</strong>.</li>
# </ul>
#
# <p>
# These estimates allow us to translate the <strong>Minimum Detectable Effect (MDE)</strong> into a statistical
# <strong>effect size</strong> and compute the required sample size.
# </p>
#
# <blockquote>
# ⚠️ Baselines must be computed from <strong>pre-experiment data</strong>.  
# Using outcome data from the experiment itself would introduce <strong>data leakage</strong>.
# </blockquote>
#
# </details>

# %%
# 🧮 Data-Driven Baseline Metric from historical data (stored in test_config; only relevant keys are set per test family)
_b = compute_baseline_from_data(historical_df, test_config)
test_config['baseline_rate'] = _b['baseline_rate']
test_config['baseline_mean'] = _b['baseline_mean']
test_config['std_dev'] = _b['std_dev']


# %% [markdown]
# <a id="minimum-detectable-effect"></a>
#
# <h4>📈 Minimum Detectable Effect</h4>
#
# <details>
#   <summary><strong>📖 Click to Expand</strong></summary>
#
#   🎯 <strong>Minimum Detectable Effect (MDE)</strong> is the smallest <strong>business-relevant difference</strong> you want your test to catch.
#
#   <ul>
#     <li>It reflects <strong>what matters</strong> — not what the data happens to show</li>
#     <li>Drives required sample size:
#       <ul>
#         <li>Smaller MDE → larger sample</li>
#         <li>Larger MDE → smaller sample</li>
#       </ul>
#     </li>
#   </ul>
#
#   🧠 <strong>Choose an MDE based on:</strong>
#   <ul>
#     <li>What level of uplift would justify launching the feature?</li>
#     <li>What's a meaningful change in your metric — not just statistical noise?</li>
#   </ul>
#
# </details>
#

# %%
# Minimum Detectable Effect (MDE)
# This is NOT data-driven — it reflects the minimum improvement you care about detecting.
# It should be small enough to catch valuable changes, but large enough to avoid inflating sample size.

# Examples by Metric Type:
# - Binary       : 0.02 → detect a 2% lift in conversion rate (e.g., from 10% to 12%)
# - Categorical  : 0.05 → detect a 5% shift in plan preference (e.g., more users choosing 'premium' over 'basic')
# - Continuous   : 3.0  → detect a 3-point gain in engagement score (e.g., from 50 to 53 avg. score)

mde = 5  # TODO: Change this based on business relevance


# %% [markdown]
# <a id="test-family"></a>
#
# <h4>🔍 Test Family</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Selects the appropriate statistical test based on:</p>
# <ul>
#   <li>Outcome data type (binary, continuous, categorical)</li>
#   <li>Distributional assumptions (normality, variance)</li>
#   <li>Number of groups and experiment structure (independent vs paired)</li>
# </ul>
#
# <p>This step <strong>automatically maps to the correct test</strong> (e.g., t-test, z-test, chi-square, ANOVA).</p>
#
# <h6>🧪 Experiment Type → Test Family Mapping</h6>
#
# <table>
#   <tr><th>Outcome Metric</th><th>Normality</th><th>Group Count</th><th>Selected Test Family</th></tr>
#   <tr><td><strong>binary</strong></td><td>—</td><td>2</td><td><code>z_test</code></td></tr>
#   <tr><td><strong>binary</strong></td><td>—</td><td>3+</td><td><code>chi_square</code></td></tr>
#   <tr><td><strong>continuous</strong></td><td>✅</td><td>2</td><td><code>t_test</code></td></tr>
#   <tr><td><strong>continuous</strong></td><td>✅</td><td>3+</td><td><code>anova</code></td></tr>
#   <tr><td><strong>continuous</strong></td><td>❌</td><td>2</td><td><code>non_parametric</code> (Mann-Whitney U)</td></tr>
#   <tr><td><strong>continuous</strong></td><td>❌</td><td>3+</td><td><code>non_parametric</code> (Kruskal-Wallis)</td></tr>
#   <tr><td><strong>categorical</strong></td><td>—</td><td>2</td><td><code>chi_square</code></td></tr>
#   <tr><td><strong>categorical</strong></td><td>—</td><td>3+</td><td><code>chi_square</code></td></tr>
# </table>
#
# </details>
#
#
#

# %%
test_config['family']

# %%
test_config['family'] = determine_test_family(test_config)
# test_config
print_config_summary(test_config)

print(f"✅ Selected test family: {test_config['family']}")

# %% [markdown]
# <a id="required-sample-size"></a>
#
# <h4>📐 Required Sample Size</h4>

# %%
required_sample_size = calculate_power_sample_size(
    test_family=test_config['family'],
    variant=test_config.get('variant'),
    alpha=alpha,
    power=power,
    baseline_rate=test_config.get('baseline_rate'),
    mde=mde,
    std_dev=test_config.get('std_dev'),
    effect_size=None,  # Let it compute internally via mde/std
    num_groups=2
)

test_config['required_sample_size'] = required_sample_size
print(f"✅ Required sample size per group: {required_sample_size}")
print(f"👥 Total sample size: {required_sample_size * 2}")

# %%
print_power_summary(
    test_family=test_config['family'],
    variant=test_config.get('variant'),
    alpha=alpha,
    power=power,
    baseline_rate=test_config.get('baseline_rate'),
    mde=mde,
    std_dev=test_config.get('std_dev'),
    required_sample_size=required_sample_size
)


# %% [markdown]
# [Back to the top](#table-of-contents)
# ___

# %% [markdown]
# <a id="randomization"></a>
# <h1>🔀 Randomization</h1>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Randomization is used to ensure that observed differences in outcome metrics are due to the experiment, not pre-existing differences.</p>
#
# <ul>
#   <li>Prevents <strong>selection bias</strong> (e.g., users self-selecting into groups)</li>
#   <li>Balances <strong>confounding factors</strong> like platform, region, or past behavior</li>
#   <li>Enables <strong>valid inference</strong> through statistical testing</li>
# </ul>
#
# </details>
#

# %% [markdown]
# <a id="simple-randomization"></a>
#
# <details>
#   <summary><strong>📖 Simple Randomization (Click to Expand)</strong></summary>
#
#   <p><strong>Each user is assigned to control or treatment with equal probability, independent of any characteristics.</strong></p>
#
#   <strong>✅ When to Use:</strong>
#   <ul>
#     <li>Sample size is <em>large enough</em> to ensure natural balance</li>
#     <li>No strong concern about <em>confounding variables</em></li>
#     <li>Need a <em>quick, default assignment</em> strategy</li>
#   </ul>
#
#   <strong>🛠️ How It Works:</strong>
#   <ul>
#     <li>Assign each user randomly (e.g., 50/50 split)</li>
#     <li>No grouping, segmentation, or blocking involved</li>
#     <li>Groups are expected to balance out on average</li>
#   </ul>
# </details>
#

# %% [markdown]
# <a id="stratified-sampling"></a>
#
# <details><summary><strong>📖 Stratified Sampling (Click to Expand)</strong></summary>
#
# <p>Ensures that key segments (e.g., platform, region) are evenly represented across control and treatment.</p>
#
# <h5>When to Use</h5>
# <ul>
#   <li>User base is <strong>naturally skewed</strong> (e.g., 70% mobile, 30% desktop)</li>
#   <li>Important to control for <strong>known confounders</strong> like geography or device</li>
#   <li>You want balance <strong>within subgroups</strong>, not just overall</li>
# </ul>
#
# <h5>How It Works</h5>
# <ul>
#   <li>Pick a stratification variable (e.g., platform)</li>
#   <li>Split population into strata (groups)</li>
#   <li>Randomly assign users <strong>within each stratum</strong></li>
# </ul>
#
# </details>
#

# %% [markdown]
# <a id="block-randomization"></a>
#
# <details><summary><strong>📖 Block Randomization (Click to Expand)</strong></summary>
#
# <p>Groups users into fixed-size blocks and randomly assigns groups within each block.</p>
#
# <h5>When to Use</h5>
# <ul>
#   <li>Users arrive in <strong>time-based batches</strong> (e.g., daily cohorts)</li>
#   <li>Sample size is <strong>small</strong> and needs enforced balance</li>
#   <li>You want to minimize <strong>temporal or ordering effects</strong></li>
# </ul>
#
# <h5>How It Works</h5>
# <ul>
#   <li>Create blocks based on order or ID (e.g., every 10 users)</li>
#   <li>Randomize assignments <strong>within each block</strong></li>
#   <li>Ensures near-equal split in every batch</li>
# </ul>
#
# </details>
#

# %% [markdown]
# <a id="match-pair-randomization"></a>
#
# <details><summary><strong>📖 Match Pair Randomization (Click to Expand)</strong></summary>
#
# <p>Participants are <strong>paired based on similar characteristics</strong> before random group assignment.  
# This reduces variance and improves <strong>statistical power</strong> by ensuring balance on key covariates.</p>
#
# <h5>When to Use</h5>
# <ul>
#   <li>Small sample size with high risk of <strong>confounding</strong></li>
#   <li>Outcomes influenced by user traits (e.g., <strong>age, income, tenure</strong>)</li>
#   <li>Need to <strong>minimize variance</strong> across groups</li>
# </ul>
#
# <h5>How It Works</h5>
# <ol>
#   <li>Identify important covariates (e.g., age, purchase history)</li>
#   <li>Sort users by those variables</li>
#   <li>Create matched pairs (or small groups)</li>
#   <li>Randomly assign one to <strong>control</strong>, the other to <strong>treatment</strong></li>
# </ol>
#
# </details>
#

# %% [markdown]
# <a id="cluster-randomization"></a>
#
# <details><summary><strong>📖 Cluster Randomization (Click to Expand)</strong></summary>
#
# <p>Entire <strong>groups or clusters</strong> (e.g., cities, stores, schools) are assigned to control or treatment.  
# Used when it's impractical or risky to randomize individuals within a cluster.</p>
#
# <h5>When to Use</h5>
# <ul>
#   <li>Users naturally exist in <strong>groups</strong> (e.g., teams, locations, devices)</li>
#   <li>There's a risk of <strong>interference</strong> between users (e.g., word-of-mouth)</li>
#   <li>Operational or tech constraints prevent individual-level randomization</li>
# </ul>
#
# <h5>How It Works</h5>
# <ol>
#   <li>Define the cluster unit (e.g., store, city)</li>
#   <li>Randomly assign each cluster to control or treatment</li>
#   <li>All users within the cluster inherit the group assignment</li>
# </ol>
#
# </details>
#

# %% [markdown]
# <a id="apply-randomization"></a>
# <h4>🎲 Apply Randomization</h4>
#

# %% [markdown]
#
# <details>
# <summary><strong>📖 Click to Expand </strong></summary>
#
# <p>
# In this notebook we randomize the <strong>entire dataset</strong> (e.g., all 1000 simulated users). 
# This is done purely for demonstration so that downstream analysis has enough data to run.
# </p>
#
# <p>
# In real A/B experiments the workflow is different:
# </p>
#
# <ul>
# <li><strong>Power analysis</strong> determines the required sample size (e.g., 272 users per group).</li>
# <li>Users are then <strong>randomized as they arrive in the product</strong>.</li>
# <li>The experiment continues until the required sample size is reached.</li>
# <li>Analysis is performed once the target sample size is collected.</li>
# </ul>
#
# <p>
# In other words, real experiments do <strong>not randomize a fixed dataset upfront</strong>. 
# Instead, randomization happens dynamically during the experiment as new users enter the system.
# </p>
#
# <blockquote>
# ⚠️ In this notebook the dataset is simulated beforehand, so randomization is applied to all users at once. 
# In production experimentation platforms (e.g., Optimizely, Statsig, internal experimentation systems), 
# users are assigned to variants <strong>at runtime</strong>.
# </blockquote>
#
# </details>

# %%
n_required = test_config['required_sample_size'] * test_config['group_count']
df = df.sample(n=n_required, random_state=42)

df

# %%
# Apply randomization method
if randomization_method == "simple":
    df = apply_simple_randomization(df, group_col=group_col, seed=my_seed)

elif randomization_method == "stratified":
    df = apply_stratified_randomization(df, stratify_col='platform', group_col=group_col, seed=my_seed)

elif randomization_method == "block":
    df = apply_block_randomization(df, observation_id_col='user_id', group_col=group_col, block_size=10, seed=my_seed)

elif randomization_method == "matched_pair":
    df = apply_matched_pair_randomization(df, sort_col=pre_experiment_metric, group_col=group_col, group_labels=test_config['group_labels'])

elif randomization_method == "cluster":
    df = apply_cluster_randomization(df, cluster_col='city', group_col=group_col, seed=my_seed)

# TODO: remove this
# elif randomization_method == "cuped":
#     df = add_outcome_metrics(df, group_col=group_col, group_labels=test_config['group_labels'], outcome_metric_col=test_config['outcome_metric_col'], guardrail_metric_col=test_config.get('guardrail_metric_col') or guardrail_metric_col, seed=my_seed)
#     df = apply_cuped(df, pre_metric='past_purchase_count', outcome_metric_col=test_config['outcome_metric_col'], group_col=group_col, group_labels=test_config['group_labels'], seed=my_seed)
#     test_config['outcome_metric_col'] = f"{test_config['outcome_metric_col']}_cuped_adjusted"
else:
    raise ValueError(f"❌ Unsupported randomization method: {randomization_method}")

# Randomization only assigns group. Outcome data is collected in the next section (Outcome data).
df

# %% [markdown]
# <a id="network-effects"></a>
# <h4>🕸️ Network Effects & SUTVA Violations</h4>

# %% [markdown]
# <details><summary><strong>📖 When Randomization Assumptions Break (Click to Expand)</strong></summary>
#
# Most A/B tests assume the **Stable Unit Treatment Value Assumption (SUTVA)** — meaning:
# - A user's outcome depends *only* on their own treatment assignment.
# - One unit's treatment **does not influence** another unit’s outcome.
#
# ##### 🧪 Why It Matters
# If users in different groups interact:
# - Control group behavior may be **influenced by treatment group exposure**.
# - This **biases your estimates** and **dilutes treatment effect**.
# - Standard tests may incorrectly **accept the null hypothesis** due to **spillover**.
#
# This assumption **breaks down** in experiments involving **social behavior**, **multi-user platforms**, or **ecosystem effects**.
# ##### ⚠️ Common Violation Scenarios
# - 🛍️ **Marketplace platforms** (e.g., sellers and buyers interact)
# - 🧑‍🤝‍🧑 **Social features** (e.g., follows, likes, comments, feeds)
# - 📲 **Referrals / network effects** (e.g., invites, rewards)
# - 💬 **Chat and collaboration tools** (e.g., Slack, Teams)
#
# ##### 🧩 Solutions (If You Suspect Interference)
# | Strategy                  | Description                                                                 |
# |---------------------------|-----------------------------------------------------------------------------|
# | Cluster Randomization     | Randomize **at group level** (e.g., friend group, region, org ID)          |
# | Isolation Experiments     | Only roll out to **fully disconnected segments** (e.g., one region only)   |
# | Network-Based Metrics     | Include **network centrality / exposure** as covariates                    |
# | Post-Experiment Checks    | Monitor if control group was exposed indirectly (e.g., referrals, shared UIs) |
# | Simulation-Based Designs  | Use agent-based or graph simulations to estimate contamination risk        |
# </details>

# %% [markdown]
# <a id="sample-ratio-mismatch"></a>
#
# <h4>⚖️ Sample Ratio Mismatch</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Is group assignment balanced?</p>
# <ul>
#   <li>SRM (Sample Ratio Mismatch) checks whether the observed group sizes match the expected ratio.</li>
#   <li>In a perfect world, random assignment to 'A1' and 'A2' should give ~50/50 split.</li>
#   <li>SRM helps catch bugs in randomization, data logging, or user eligibility filtering.</li>
# </ul>
#
# <p><strong>Real-World Experiment Split Ratios</strong></p>
# <table>
#   <thead>
#     <tr>
#       <th><strong>Scenario</strong></th>
#       <th><strong>Split</strong></th>
#       <th><strong>Why</strong></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>Default A/B</td>
#       <td>50 / 50</td>
#       <td>Maximizes power and ensures fairness</td>
#     </tr>
#     <tr>
#       <td>Risky feature</td>
#       <td>10 / 90 or 20 / 80</td>
#       <td>Limits user exposure to minimize risk</td>
#     </tr>
#     <tr>
#       <td>Ramp-up</td>
#       <td>Step-wise (1-5-25-50…)</td>
#       <td>Gradual rollout to catch issues early</td>
#     </tr>
#     <tr>
#       <td>A/B/C Test</td>
#       <td>33 / 33 / 33 or weighted</td>
#       <td>Compare multiple variants fairly or with bias</td>
#     </tr>
#     <tr>
#       <td>High control confidence needed</td>
#       <td>70 / 30 or 60 / 40</td>
#       <td>More stability in baseline comparisons</td>
#     </tr>
#   </tbody>
# </table>
#
# </details>

# %%
check_sample_ratio_mismatch(df, group_col=group_col, group_labels=test_config['group_labels'], expected_ratios=[0.5, 0.5], alpha=0.05)

# %%
df


# %% [markdown]
# [Back to the top](#table-of-contents)
# ___
#

# %% [markdown]
# <a id="eda"></a>
# <h1>📈 EDA</h1>
#
# Exploratory Data Analysis validates core statistical assumptions before testing begins.

# %%
df

# %%
# TODO: In a real experiment this data comes from production logs after the
# experiment has run. Replace add_outcome_metrics() with real outcome data.
df = add_outcome_metrics(df, group_col=group_col, group_labels=test_config['group_labels'], outcome_metric_col=test_config['outcome_metric_col'], guardrail_metric_col=test_config.get('guardrail_metric_col') or guardrail_metric_col, seed=my_seed)

# Outcome data (post-assignment): simulate collection so we have primary outcome, converted, bounce_rate for analysis.
if randomization_method == "cuped":
    df = apply_cuped(df, pre_metric='past_purchase_count', outcome_metric_col=test_config['outcome_metric_col'], group_col=group_col, group_labels=test_config['group_labels'], seed=my_seed)
    test_config['outcome_metric_col'] = f"{test_config['outcome_metric_col']}_cuped_adjusted"

# %%
df

# %% [markdown]
# <a id="normality"></a>
#
# <h4>🔍 Normality</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Checks whether your outcome metric follows a <strong>normal distribution</strong>, which is a key assumption for <strong>parametric tests</strong> like t-test or ANOVA.</p>
#
# <ul>
#   <li>Use <strong>Shapiro-Wilk test</strong> or visual tools (histograms, Q-Q plots).</li>
#   <li>Helps determine whether to use parametric or non-parametric tests.</li>
#   <li>If data is non-normal, switch to <strong>Mann-Whitney U</strong> or <strong>Wilcoxon</strong>.</li>
# </ul>
#
# </details>
#

# %%
normality_results = test_normality(df, outcome_metric_col=test_config['outcome_metric_col'], group_col='group', group_labels=test_config['group_labels'])

print("Normality test (Shapiro-Wilk) results:")
for group, result in normality_results.items():
    print(f"{group}: p = {result['p_value']:.4f} → {'Normal' if result['normal'] else 'Non-normal'}")

# %%
# Assume both groups must be normal to proceed with parametric tests
test_config['normality'] = all(result['normal'] for result in normality_results.values())
test_config


# %% [markdown]
# <a id="variance-homogeneity-check"></a>
#
# <h4>🔍 Variance Homogeneity Check</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Tests whether the <strong>variances between groups are equal</strong>, which affects the validity of t-tests and ANOVA.</p>
#
# <ul>
#   <li>Performed using <strong>Levene’s test</strong> or <strong>Bartlett’s test</strong>.</li>
#   <li>If variances are unequal, use <strong>Welch's t-test</strong> instead.</li>
#   <li>Unequal variances do not invalidate analysis but change the test used.</li>
# </ul>
#
# </details>
#

# %%
variance_result = test_equal_variance(df, outcome_metric_col=test_config['outcome_metric_col'], group_col='group', group_labels=test_config['group_labels'])
variance_result

# %%
print(f"Levene’s test: p = {variance_result['p_value']:.4f} → {'Equal variances' if variance_result['equal_variance'] else 'Unequal variances'}")
test_config['equal_variance'] = variance_result['equal_variance']
test_config

# %% [markdown]
# [Back to the top](#table-of-contents)
# ___
#

# %% [markdown]
# <a id="aa-testing"></a>
#
# <h1>🧪 AA Testing</h1>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>A/A testing is a <strong>preliminary experiment</strong> where both groups (e.g., “control” and “treatment”) receive the exact same experience. It's used to validate the experimental setup before running an actual A/B test.</p>
#
# <p><strong>What Are We Checking?</strong></p>
# <ul>
#   <li>Are users being assigned fairly and randomly?</li>
#   <li>Are key outcome metrics statistically similar across groups?</li>
#   <li>Can we trust the experimental framework?</li>
# </ul>
#
# <p><strong>Why A/A Testing Matters</strong></p>
# <ul>
#   <li><strong>Validates Randomization</strong> — Confirms the groups are balanced at baseline (no bias or leakage)</li>
#   <li><strong>Detects SRM (Sample Ratio Mismatch)</strong> — Ensures the actual split (e.g., 50/50) matches what was intended</li>
#   <li><strong>Estimates Variability</strong> — Helps calibrate variance for accurate power calculations later</li>
#   <li><strong>Trust Check</strong> — Catches bugs in assignment logic, event tracking, or instrumentation</li>
# </ul>
#
# <p><strong>A/A Test Process</strong></p>
# <ol>
#   <li><strong>Randomly assign users</strong> into two equal groups — Just like you would for an A/B test (e.g., control vs treatment)</li>
#   <li><strong>Measure key outcome</strong> — This depends on your experiment type:
#     <ul>
#       <li><code>binary</code> → conversion rate</li>
#       <li><code>continuous</code> → avg. revenue, time spent</li>
#       <li><code>categorical</code> → feature adoption, plan selected</li>
#     </ul>
#   </li>
#   <li><strong>Run statistical test</strong>:
#     <ul>
#       <li><code>binary</code> → Z-test or Chi-square</li>
#       <li><code>continuous</code> → t-test</li>
#       <li><code>categorical</code> → Chi-square test</li>
#     </ul>
#   </li>
#   <li><strong>Check SRM</strong> — Use a chi-square goodness-of-fit test to detect assignment imbalances</li>
# </ol>
#
# <p><strong>Possible Outcomes</strong></p>
# <table>
#   <thead>
#     <tr>
#       <th><strong>Result</strong></th>
#       <th><strong>Interpretation</strong></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>No significant difference</td>
#       <td>✅ Randomization looks good. Test setup is sound.</td>
#     </tr>
#     <tr>
#       <td>Statistically significant difference</td>
#       <td>⚠️ Something’s off — check assignment logic, instrumentation, or sample leakage</td>
#     </tr>
#   </tbody>
# </table>
#
# <p><em>Run A/A tests whenever you launch a new experiment framework, roll out a new randomizer, or need to build stakeholder trust.</em></p>
#
# </details>
#

# %% [markdown]
# <a id="outcome-similarity-test"></a>
#
# <h4>🧬 Outcome Similarity Test</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Compares the <strong>outcome metric across groups</strong> to ensure no significant differences exist when there shouldn't be any — usually used during <strong>A/A testing</strong> or pre-experiment validation.</p>
#
# <ul>
#   <li>Helps detect setup issues like <strong>biased group assignment</strong> or <strong>data leakage</strong>.</li>
#   <li>Null Hypothesis: <strong>No difference</strong> in outcomes between control and treatment.</li>
#   <li>Uses the same statistical test as the main A/B test (e.g., t-test, z-test, chi-square).</li>
# </ul>
#
# </details>
#

# %% [markdown]
# <a id="aa-test-visualization"></a>
#
# <h4>📊 AA Test Visualization</h4>

# %%
# TODO: Replace simulated outcomes with real experiment logs when running a real AA test.
run_aa_testing_generalized(
    df=df,
    group_col='group',
    metric_col=test_config['outcome_metric_col'],
    group_labels=test_config['group_labels'],
    test_family=test_config['family'],
    variant=test_config.get('variant'),
    alpha=0.05
)


# %% [markdown]
# todo: pseudo AA test

# %% [markdown]
# <a id="type-i-error-simulation"></a>
#
# <h4>🎲 Type I Error Simulation</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <h5>🔁 Repeated A/A Tests</h5>
#
# <p>
# While a single A/A test helps detect obvious flaws in group assignment (like SRM or data leakage), it’s still a one-off check.  
# To gain confidence in your randomization method, we simulate <strong>multiple A/A tests</strong> using the same logic:
# </p>
#
# <ul>
#   <li>Each run reassigns users randomly into <code>control</code> and <code>treatment</code> (with no actual change)</li>
#   <li>We then run the statistical test between groups for each simulation</li>
#   <li>We track how often the test reports a <strong>false positive</strong> (p &lt; α), which estimates the <strong>Type I error rate</strong></li>
# </ul>
#
# <blockquote>
#   In theory, if your setup is unbiased and α = 0.05, you'd expect about 5% of simulations to return a significant result — this validates your A/B framework isn’t "trigger-happy."
# </blockquote>
#
# <h5>📊 What this tells you:</h5>
#
# <ul>
#   <li>Too many significant p-values → your framework is too noisy (bad randomization, poor test choice)</li>
#   <li>Near 5% = healthy noise level, expected by design</li>
# </ul>
#
# <p>This step is optional but highly recommended when you're:</p>
# <ul>
#   <li>Trying out a new randomization strategy</li>
#   <li>Validating an internal experimentation framework</li>
#   <li>Stress-testing your end-to-end pipeline</li>
# </ul>
#
# </details>
#

# %%
_ = simulate_aa_type1_error_rate(
    df=df,
    metric_col=test_config['outcome_metric_col'],
    group_labels=test_config['group_labels'],
    test_family=test_config['family'],
    variant=test_config.get('variant'),
    runs=100,
    alpha=0.05
)


# %% [markdown]
# [Back to the top](#table-of-contents)
# ___
#

# %% [markdown]
# <a id="ab-testing"></a>
#
# <h1>🧪 A/B Testing</h1>

# %% [markdown]
# 🔗 For test selection (e.g., Z-test, t-test), refer to [📖 Hypothesis Testing Notebook](https://ashrithssreddy.github.io/statistics-toolkit/Hypothesis_Testing/Hypothesis_Testing.html)
#
# <details><summary><strong>📖 Click to Expand </strong></summary>
#
# <h5>🧪 A/B Testing - Outcome Comparison</h5>
#
# <p>This section compares the outcome metric between control and treatment groups using the appropriate statistical test based on the experiment type.</p>
#
# <h5>📌 Metric Tracked:</h5>
# <ul>
#   <li><strong>Primary metric:</strong> Depends on use case:
#     <ul>
#       <li><strong>Binary:</strong> Conversion rate (clicked or not)</li>
#       <li><strong>Continuous:</strong> Average engagement, revenue, time spent</li>
#       <li><strong>Categorical:</strong> Plan type, user tier, etc.</li>
#     </ul>
#   </li>
#   <li><strong>Unit of analysis:</strong> Unique user or unique observation</li>
# </ul>
#
# <h5>🔬 Outcome Analysis Steps:</h5>
# <ul>
#   <li>Choose the <strong>right statistical test</strong> based on <code>experiment_type</code>:
#     <ul>
#       <li><code>'binary'</code> → <strong>Z-test for proportions</strong></li>
#       <li><code>'continuous_independent'</code> → <strong>Two-sample t-test</strong></li>
#       <li><code>'continuous_paired'</code> → <strong>Paired t-test</strong></li>
#       <li><code>'categorical'</code> → <strong>Chi-square test of independence</strong></li>
#     </ul>
#   </li>
#   <li>Calculate test statistics, p-values, and confidence intervals</li>
#   <li>Visualize the comparison to aid interpretation</li>
# </ul>
#
# </details>
#

# %%
result = run_ab_test(
    df=df,
    group_col='group',
    metric_col=test_config['outcome_metric_col'],
    group_labels=test_config['group_labels'],
    test_family=test_config['family'],
    variant=test_config.get('variant'),
    alpha=0.05
)
result

# %% [markdown]
# <a id="summaries"></a>
# <h4>🧾 Summaries</h4>

# %%
summarize_ab_test_result(result)

# %% [markdown]
# <a id="results-visualization"></a>
#
# <h4>📊 Visualization</h4>

# %%
plot_ab_test_results(result)

# %% [markdown]
# <a id="confidence-intervals"></a>
#
# <h4>🎯 95% Confidence Intervals<br><small>for <code>outcome in groups</code></small></h4>
#
# <details><summary><strong>📖 Click to Expand </strong></summary>
#
# <ul>
#   <li>The 95% confidence interval gives a range in which we expect the <strong>true conversion rate</strong> to fall for each group.</li>
#   <li>If the confidence intervals <strong>do not overlap</strong>, it's strong evidence that the difference is statistically significant.</li>
#   <li>If they <strong>do overlap</strong>, it doesn't guarantee insignificance — you still need the p-value to decide — but it suggests caution when interpreting lift.</li>
# </ul>
#
# </details>
#

# %%
plot_confidence_intervals(result)


# %% [markdown]
# <a id="lift-analysis"></a>
#
# <h4>📈 Lift Analysis<br><small>AKA 95% Confidence Intervals for (difference in outcomes)</small></h4>
#
# <details><summary><strong>📖 Click to Expand </strong></summary>
#
# <p>This confidence interval helps quantify uncertainty around the observed <strong>lift</strong> between treatment and control groups. It answers:</p>
#
# <ul>
#   <li><em>How large is the difference between groups?</em></li>
#   <li><em>How confident are we in this lift estimate?</em></li>
# </ul>
#
# <p>
# We compute a 95% CI for the difference in means (or proportions), not just for each group. If this interval <strong>does not include 0</strong>, we can reasonably trust there's a true difference.  
# If it <strong>does include 0</strong>, the observed difference might be due to random chance.
# </p>
#
# <p>
# This complements the p-value — while p-values tell us <em>if</em> the difference is significant, <strong>CIs tell us how big the effect is, and how uncertain we are.</strong>
# </p>
#
# </details>
#

# %%
compute_lift_confidence_interval(result)

# %% [markdown]
# <a id="final-conclusion"></a>
#
# <h4>✅ Final Conclusion</h4>
#

# %%
print_final_ab_test_summary(result)

# %% [markdown]
# <a id="how-long"></a>
# <h4>⏱️ How Long</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>The duration of an A/B test depends on how quickly you reach the required sample size per group, as estimated during your power analysis.</p>
#
# <h5>✅ Key Inputs</h5>
# <ul>
#   <li>Daily volume of eligible observations (users, sessions, or orders — depends on your unit of analysis)</li>
#   <li>Required sample size per group (from power analysis)</li>
#   <li>Traffic split ratio (e.g., 50/50, 10/90, 33/33/33)</li>
# </ul>
#
# <h5>🧮 Formula</h5>
# <blockquote>
#   Test Duration (in days) =<br>
#   Required Sample Size per Group ÷ (Daily Eligible Observations × Group Split Proportion)
# </blockquote>
#
# <p>This ensures the experiment runs long enough to detect the expected effect with the desired confidence and power.</p>
#
# <h5>💡 Planning Tips</h5>
# <ol>
#   <li>Estimate required sample size using power analysis (based on effect size, baseline, alpha, and power)</li>
#   <li>Understand your traffic: 
#     <ul>
#       <li>What’s your average daily eligible traffic?</li>
#       <li>What unit of analysis is used (user, session, impression)?</li>
#     </ul>
#   </li>
#   <li>Apply group split:
#     <ul>
#       <li>e.g., for a 50/50 A/B test, each group gets 50% of traffic</li>
#     </ul>
#   </li>
#   <li>Estimate days using the formula above.</li>
# </ol>
#
# <h5>🧠 Real-World Considerations</h5>
# <ul>
#   <li><strong>✅ Ramp-Up Period</strong><br>
#     Gradually increase traffic exposure: 5% → 25% → 50% → full traffic.<br>
#     Helps catch bugs, stability issues, and confounding edge cases early.
#   </li>
#   <li><strong>✅ Cool-Down Buffer</strong><br>
#     Avoid ending tests on weekends, holidays, or during unusual traffic spikes.<br>
#     Add buffer days so your conclusions aren’t skewed by anomalies.
#   </li>
#   <li><strong>✅ Trust Checks Before Analysis</strong>
#     <ul>
#       <li>A/A testing to verify setup</li>
#       <li>SRM checks to confirm user distribution</li>
#       <li>Monitor guardrail metrics (e.g., bounce rate, latency, load time)</li>
#     </ul>
#   </li>
# </ul>
#
# <h5>🗣️ Common Practitioner Advice</h5>
# <blockquote>
#   “We calculate sample size using power analysis, then divide by daily traffic per group. But we always factor in buffer days — for ramp-up, trust checks, and stability. Better safe than sorry.”
#   <br><br>
#   “Power analysis is the starting point. But we don’t blindly stop when we hit N. We monitor confidence intervals, metric stability, and coverage to make sure we’re making decisions the business can trust.”
# </blockquote>
#
# </details>
#

# %% [markdown]
# <details><summary><strong>📖 Monitoring Dashboard (Click to Expand)</strong></summary>
#
#
# <ul>
#   <li><strong>Overall Test Health</strong>
#     <ul>
#       <li>Start/end date, traffic ramp-up %, time remaining</li>
#       <li>SRM (Sample Ratio Mismatch) indicator</li>
#       <li>P-value and effect size summary (updated daily)</li>
#     </ul>
#   </li>
#   <li><strong>Primary Metric Tracking</strong>
#     <ul>
#       <li>Daily trends for primary outcome (conversion, revenue, etc.)</li>
#       <li>Cumulative lift + confidence intervals</li>
#       <li>Statistical significance tracker (p-value, test stat)</li>
#     </ul>
#   </li>
#   <li><strong>Guardrail Metrics</strong>
#     <ul>
#       <li>Bounce rate, load time, checkout errors, etc.</li>
#       <li>Alert thresholds (e.g., +10% increase in latency)</li>
#       <li>Trend vs baseline and prior experiments</li>
#     </ul>
#   </li>
#   <li><strong>Segment Drilldowns</strong>
#     <ul>
#       <li>Platform (iOS vs Android), geography, user tier</li>
#       <li>Detect heterogeneous treatment effects</li>
#       <li>Option to toggle test results per segment</li>
#     </ul>
#   </li>
#   <li><strong>Cohort Coverage</strong>
#     <ul>
#       <li>Total users assigned vs eligible</li>
#       <li>Daily inclusion and exclusion trends</li>
#       <li>Debugging filters (e.g., why user X didn’t get assigned)</li>
#     </ul>
#   </li>
#   <li><strong>Variance & Stability Checks</strong>
#     <ul>
#       <li>Volatility of key metrics</li>
#       <li>Pre vs post baseline comparisons</li>
#       <li>Funnel conversion variance analysis</li>
#     </ul>
#   </li>
#   <li><strong>Notes & Annotations</strong>
#     <ul>
#       <li>Manual tagging of major incidents (e.g., bug fix deployed, pricing change)</li>
#       <li>Timeline of changes affecting experiment interpretation</li>
#     </ul>
#   </li>
# </ul>
#
# </details>
#

# %%
daily_eligible_users = 1000
allocation_ratios = (0.5, 0.5)
buffer_days = 2

test_duration_result = estimate_test_duration(
    required_sample_size_per_group=test_config['required_sample_size'],
    daily_eligible_users=daily_eligible_users,
    allocation_ratios=allocation_ratios,
    buffer_days=buffer_days,
    test_family=test_config['family']
)
test_duration_result

# %% [markdown]
# [Back to the top](#table-of-contents)
# ___
#

# %% [markdown]
# <a id="post-hoc-analysis"></a>
#
# <h1>🔍 Post Hoc Analysis</h1>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <blockquote>
#   After statistical significance, post-hoc analysis helps <strong>connect results to business confidence</strong>.<br>
#   It's not just <em>did it work</em> — but <em>how, for whom, and at what cost or benefit?</em>
# </blockquote>
#
# <p><strong>🧠 Why Post Hoc Analysis Matters</strong></p>
# <ul>
#   <li>Segments may <strong>respond differently</strong> — average lift may hide underperformance in subgroups</li>
#   <li>Guardrails may show <strong>collateral damage</strong> (e.g., slower load time, higher churn)</li>
#   <li>Stakeholders need <strong>impact translation</strong> — what does this mean in revenue, retention, or strategy?</li>
# </ul>
#
# <p><strong>🔎 Typical Post Hoc Questions</strong></p>
# <ul>
#   <li><strong>Segment Lift</strong>
#     <ul>
#       <li>Did certain platforms, geos, cohorts, or user types benefit more?</li>
#       <li>Any negative lift in high-value user segments?</li>
#     </ul>
#   </li>
#   <li><strong>Guardrail Checks</strong>
#     <ul>
#       <li>Did the treatment impact non-primary metrics (e.g., latency, engagement, bounce rate)?</li>
#       <li>Were alert thresholds breached?</li>
#     </ul>
#   </li>
#   <li><strong>Business Impact Simulation</strong>
#     <ul>
#       <li>How does the observed lift scale to 100% of eligible users?</li>
#       <li>What’s the projected change in conversions, revenue, or user satisfaction?</li>
#     </ul>
#   </li>
#   <li><strong>Edge Case Discovery</strong>
#     <ul>
#       <li>Any bugs, instrumentation gaps, or unexpected usage patterns?</li>
#       <li>Did any user types get excluded disproportionately?</li>
#     </ul>
#   </li>
# </ul>
#
# <p><strong>📊 What to Report</strong></p>
#
# <table>
#   <thead>
#     <tr>
#       <th>Area</th>
#       <th>What to Show</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>Segment Analysis</td>
#       <td>Table or chart showing lift per segment, sorted by effect size or risk</td>
#     </tr>
#     <tr>
#       <td>Guardrail Metrics</td>
#       <td>Summary table of guardrails vs baseline, with thresholds or annotations</td>
#     </tr>
#     <tr>
#       <td>Revenue Simulation</td>
#       <td>Projected uplift × traffic volume × conversion = business impact</td>
#     </tr>
#     <tr>
#       <td>Confidence Range</td>
#       <td>95% CI for key metrics per segment (wherever possible)</td>
#     </tr>
#     <tr>
#       <td>Rollout Readiness</td>
#       <td>Any blockers, mitigations, or next steps if full rollout is considered</td>
#     </tr>
#   </tbody>
# </table>
#
# <p><strong>💡 Pro Tip</strong><br>
# Even if your p-value says “yes,” <strong>business rollout is a risk-based decision</strong>.<br>
# Post-hoc analysis is where <strong>statistical rigor meets product judgment</strong>.
# </p>
#
# </details>
#

# %% [markdown]
# <a id="segmented-lift"></a>
#
# <h4>🧩 Segmented Lift</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Segmented lift tells us <strong>how different user segments responded</strong> to the treatment.</p>
#
# <p><strong>Why It Matters:</strong></p>
# <ul>
#   <li><strong>Uncovers hidden heterogeneity</strong> — The overall average might mask variation across platforms, geographies, or user tiers.</li>
#   <li><strong>Identifies high-risk or high-reward cohorts</strong> — Some segments might benefit more, while others could be negatively impacted.</li>
#   <li><strong>Guides rollout and targeting decisions</strong> — Helps decide where to prioritize feature exposure, or where to mitigate risk.</li>
# </ul>
#
# <p><strong>Typical Segments:</strong></p>
# <ul>
#   <li>Device type (e.g., mobile vs desktop)</li>
#   <li>Region (e.g., North vs South)</li>
#   <li>User lifecycle (e.g., new vs returning)</li>
#   <li>Platform (e.g., iOS vs Android)</li>
# </ul>
#
# <blockquote>
#   <em>"Segmentation answers <strong>who is benefiting (or suffering)</strong> — not just <strong>whether it worked on average.</strong>"</em>
# </blockquote>
#
# </details>
#

# %%
analyze_segment_lift(
    df=df,
    test_config=test_config,
    segment_cols=['platform', 'device_type'], # , 'user_tier', 'region'
    min_count_per_group=30,
    visualize=True
)


# %% [markdown]
# <a id="guardrail-metrics"></a>
#
# <h4>🚦 Guardrail Metrics</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Guardrail metrics are <strong>non-primary metrics</strong> tracked during an experiment to ensure the feature doesn't create <strong>unintended negative consequences</strong>.</p>
#
# <p>We monitor them alongside the main success metric to:</p>
# <ul>
#   <li>📉 Catch regressions in user behavior or system performance</li>
#   <li>🔍 Detect trade-offs (e.g., conversion ↑ but bounce rate ↑ too)</li>
#   <li>🛑 Block rollouts if a feature does more harm than good</li>
# </ul>
#
# <h6>🧪 How We Check</h6>
# <ul>
#   <li>Run <strong>statistical tests</strong> on each guardrail metric just like we do for the primary metric</li>
#   <li>Use the <strong>same experiment type</strong> (binary, continuous, etc.) for evaluation</li>
#   <li>Report <strong>p-values and lift</strong> to assess significance and direction</li>
#   <li>Focus more on <strong>risk detection</strong> than optimization</li>
# </ul>
#
# <h6>📊 Common Guardrail Metrics</h6>
# <table>
#   <tr><th>Type</th><th>Examples</th></tr>
#   <tr><td><strong>UX Health</strong></td><td>Bounce Rate, Session Length, Engagement</td></tr>
#   <tr><td><strong>Performance</strong></td><td>Page Load Time, API Latency, CPU Usage</td></tr>
#   <tr><td><strong>Reliability</strong></td><td>Error Rate, Crash Rate, Timeout Errors</td></tr>
#   <tr><td><strong>Behavioral</strong></td><td>Scroll Depth, Page Views per Session</td></tr>
# </table>
#
# <h6>✅ When to Act</h6>
# <ul>
#   <li>If the <strong>treatment significantly worsens</strong> a guardrail metric → investigate</li>
#   <li>If the <strong>primary metric improves</strong> but <strong>guardrails suffer</strong>, assess trade-offs</li>
#   <li>Use <strong>p-values</strong>, <strong>lift</strong>, and <strong>domain context</strong> to guide decision-making</li>
# </ul>
#
# <h6>🧠 Why Guardrails Matter</h6>
# <blockquote>
#   “We don’t just care <em>if</em> a metric moves — we care <em>what else</em> it moved. Guardrails give us confidence that improvements aren’t hiding regressions elsewhere.”
# </blockquote>
#
# </details>
#

# %%
df.head(20)

# %%
# Quick average check by group (if guardrail metric is configured)
guardrail_col = test_config.get('guardrail_metric_col')
if guardrail_col and guardrail_col in df.columns:
    guardrail_avg = df.groupby('group')[guardrail_col].mean()
    print(f"🚦 Average {guardrail_col} by Group:")
    for grp, val in guardrail_avg.items():
        print(f"- {grp}: {val:.4f}")

# %%
if guardrail_col and guardrail_col in df.columns:
    evaluate_guardrail_metric(
        df=df,
        test_config=test_config,
        guardrail_metric_col=guardrail_col,
        alpha=0.05
    )
else:
    print("🚦 No guardrail metric configured (set guardrail_metric_col in Experiment Setup to evaluate one).")


# %% [markdown]
# <a id="cuped"></a>
# <h4>🔄 CUPED</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Controlled Pre-Experiment Data: A statistical adjustment that uses <strong>pre-experiment behavior</strong> to reduce variance and improve power.  
# It helps detect smaller effects without increasing sample size.</p>
#
# <h5>When to Use</h5>
# <ul>
#   <li>You have reliable <strong>pre-experiment metrics</strong> (e.g., past spend, engagement)</li>
#   <li>You want to <strong>reduce variance</strong> and improve test sensitivity</li>
#   <li>You’re dealing with <strong>small lifts</strong> or <strong>costly sample sizes</strong></li>
# </ul>
#
# <h5>How It Works</h5>
# <ol>
#   <li>Identify a pre-period metric <strong>correlated with your outcome</strong></li>
#   <li>Use regression to compute an adjustment (theta)</li>
#   <li>Subtract the correlated component from your outcome metric</li>
#   <li>Analyze the adjusted metric instead of the raw one</li>
# </ol>
#
# </details>
#

# %%

# %% [markdown]
# <a id="multiple-comparisons"></a>
# <h4>🧠 Correcting for Multiple Comparisons</h4>

# %% [markdown]
# <details>
# <summary><strong>📖 Why p-values can't always be trusted</strong></summary>
#
# When we test multiple segments, multiple metrics or multiple variants, we increase the risk of **false positives** (Type I errors).
# This is known as the **Multiple Comparisons Problem** — and it’s dangerous in data-driven decision-making.
#
# ##### 📉 Example Scenario:
# We run A/B tests on:
# - Overall population ✅
# - By platform ✅
# - By user tier ✅
# - By region ✅
#
# If we test 10 hypotheses at 0.05 significance level, the chance of *at least one false positive* ≈ 40%.
#
# ##### ✅ Correction Methods
#
# | Method                 | Use Case                                        | Risk |
# |------------------------|------------------------------------------------|------|
# | **Bonferroni**         | Very strict, controls **Family-Wise Error Rate (FWER)** | ❄️ Conservative |
# | **Benjamini-Hochberg** | Controls **False Discovery Rate (FDR)**        | 🔥 Balanced |
#
# ##### 🧠 In Practice:
# We calculate raw p-values for each segment, and then apply corrections to get adjusted p-values.
# > If even the adjusted p-values are significant → result is robust.
#
# </details>

# %% [markdown]
# ##### ❄️ Bonferroni Correction
#
# <details>
# <summary><strong>📖 FWER Control (Click to Expand)</strong></summary>
# Bonferroni is the most **conservative** correction method.  
# It adjusts the p-value threshold by dividing it by the number of comparisons.
#
# - Formula: `adjusted_alpha = alpha / num_tests`
# - Or: `adjusted_p = p * num_tests`
# - If even one adjusted p-value < 0.05, it’s **very likely real**
#
# 📌 **Best for:** High-risk decisions (e.g., medical trials, irreversible launches)    
# ⚠️ **Drawback:** May **miss true positives** (higher Type II error)
#
# </details>

# %% [markdown]
# ##### 🔬 Benjamini-Hochberg (BH) Procedure
#
# <details>
# <summary><strong>📖 FDR Control (Click to Expand)</strong></summary>
#
# BH controls the **expected proportion of false discoveries** (i.e., false positives among all positives). It:
# - Ranks p-values from smallest to largest
# - Compares each to `(i/m) * alpha`, where:
#    - `i` = rank
#    - `m` = total number of tests
#
# **🧠 Important:** After adjustment, BH enforces **monotonicity** by **capping earlier (smaller) ranks to not exceed later ones**.  
# > In simple terms: **adjusted p-values can only decrease as rank increases.**
#
# The **largest p-value** that satisfies this inequality becomes the threshold — all smaller p-values are considered significant.
#
# 📌 **Best for:** Exploratory research, product experiments with many segments  
# 💡 **Advantage:** More power than Bonferroni, still controls errors
#
# </details>

# %%
import pandas as pd
from statsmodels.stats.multitest import multipletests

# Original inputs
segment_names = ['North', 'South', 'East', 'West']
p_vals = [0.03, 0.06, 0.02, 0.10]

# Create DataFrame and sort by raw p-values BEFORE correction
df_pvalues = pd.DataFrame({
    'Segment': segment_names,
    'Raw_pValue': p_vals
}).sort_values('Raw_pValue').reset_index(drop=True)

# Apply corrections to the sorted p-values
_, bonf, _, _ = multipletests(df_pvalues['Raw_pValue'], alpha=0.05, method='bonferroni')
_, bh, _, _ = multipletests(df_pvalues['Raw_pValue'], alpha=0.05, method='fdr_bh')

# Add to DataFrame
df_pvalues['Bonferroni_Adj_pValue'] = bonf
df_pvalues['BH_Adj_pValue'] = bh
df_pvalues

# %%
# Plot p values - raw and adjusted

plt.figure(figsize=(8, 5))

# Plot lines
plt.plot(df_pvalues.index + 1, df_pvalues['Raw_pValue'], marker='o', label='Raw p-value')
plt.plot(df_pvalues.index + 1, df_pvalues['Bonferroni_Adj_pValue'], marker='^', label='Bonferroni Adj p-value')
plt.plot(df_pvalues.index + 1, df_pvalues['BH_Adj_pValue'], marker='s', label='BH Adj p-value')

# Add value labels next to each point
for i in range(len(df_pvalues)):
    x = i + 1
    plt.text(x + 0.05, df_pvalues['Raw_pValue'][i], f"{df_pvalues['Raw_pValue'][i]:.2f}", va='center')
    plt.text(x + 0.05, df_pvalues['Bonferroni_Adj_pValue'][i], f"{df_pvalues['Bonferroni_Adj_pValue'][i]:.2f}", va='center')
    plt.text(x + 0.05, df_pvalues['BH_Adj_pValue'][i], f"{df_pvalues['BH_Adj_pValue'][i]:.2f}", va='center')

# Axis & labels
plt.xticks(df_pvalues.index + 1, df_pvalues['Segment']);
plt.axhline(0.05, color='gray', linestyle='--', label='α = 0.05');
plt.xlabel("Segment (Ranked by Significance)");
plt.ylabel("p-value");
plt.title("p-value Correction: Bonferroni vs Benjamini Hochberg (FDR)");
plt.legend();
plt.tight_layout();
plt.show();

# %% [markdown]
# <a id="novelty-effects"></a>
# <h4>🪄 Novelty Effects & Behavioral Decay</h4>
#
# <details><summary><strong>📖 Why First Impressions Might Lie (Click to Expand) </strong></summary>
#
# ##### 🪄 Novelty Effects & Behavioral Decay
# Even if an A/B test shows a statistically significant lift, that improvement may **not last**.
#
# This often happens due to **novelty effects** — short-term spikes in engagement driven by:
# - Curiosity (“What’s this new feature?”)
# - Surprise (“This looks different!”)
# - Visual attention (e.g., placement or color changes)
#
# ##### 📉 Common Signs of Novelty Effects
# - Strong lift in week 1 → drops by week 3.
# - High initial usage → no long-term retention.
# - Positive metrics in one segment only (e.g., “new users”).
#
# ##### 🧭 What We Do About It
#
# To address this risk during rollouts:
# - ✅ Monitor **metrics over time** post-launch (e.g., 7, 14, 28-day retention)
# - ✅ Compare results across **early adopters vs late adopters**
# - ✅ Run **holdout experiments** during phased rollout to detect fading impact
#
# </details>

# %% [markdown]
# <a id="primacy-effect"></a>
# <h4>🎯 Primacy Effect & Order Bias</h4>
#
# <details><summary><strong>📖 When First = Best (Click to Expand)</strong></summary>
#
# Sometimes, the **position** of a variant or option can distort results — especially if it's shown **first**. This is called the **primacy effect**, a type of cognitive bias.
#
# It often shows up in:
# - Feed ranking or content ordering experiments
# - Option selection (e.g., first dropdown item)
# - Surveys or in-app prompts
#
# ##### 🚩 Common Indicators
# - Variant A always performs better **regardless of content**
# - Metrics drop when position is swapped
# - Discrepancy between test and real-world usage
#
# ##### 🧭 What We Do About It
# To minimize primacy bias:
# - ✅ Randomize order of options or content
# - ✅ Use **position-aware metrics** (e.g., click-through by slot)
# - ✅ Validate with **follow-up tests** using rotated or reversed orders
#
# </details>
#

# %% [markdown]
# <a id="rollout-simulation"></a>
#
# <h4>🎲 Rollout Simulation</h4>
#
# <details><summary><strong>📖 Click to Expand</strong></summary>
#
# <p>Once statistical significance is established, it's useful to simulate <strong>business impact</strong> from full rollout.</p>
#
# <p>Assume full exposure to <strong>eligible daily traffic</strong>, and estimate <strong>incremental impact</strong> from the observed lift.</p>
#
# <p>This helps stakeholders understand the real-world benefit of implementing the change.</p>
#
# <p>We typically estimate:</p>
#
# <ul>
#   <li>📈 Daily lift (e.g., additional conversions, dollars, sessions)</li>
#   <li>📈 Monthly extrapolation (daily lift × 30)</li>
# </ul>
#
# </details>
#

# %%
# Derive daily volume from actual data
daily_traffic_estimate = df.shape[0]  # Assuming full traffic per day

simulate_rollout_impact(
    experiment_result=result,                         # Output from run_ab_test()
    daily_eligible_observations=daily_traffic_estimate,
    metric_unit=test_config['outcome_metric_col']     # Dynamic label like 'engagement_score' or 'revenue'
)


# %% [markdown]
# <a id="ab-test-holdouts"></a>
# <h4>🧪 A/B Test Holdouts</h4>
#
# <details><summary><strong>📖 Why We Sometimes Don't Ship to 100% (Click to Expand)</strong></summary>
#
# ##### 🧪 A/B Test Holdouts  
# Even after a successful A/B test, we often maintain a **small holdout group** during rollout.
#
# This helps us:
# - Track **long-term impact** beyond the experiment window.
# - Detect **novelty fade** or **unexpected side effects**.
# - Maintain a clean “control” for system-wide benchmarking.
#
# ##### 🏢 Industry Practice  
# - Common at large orgs like **Facebook**, where teams share a holdout pool for all feature launches.
# - Holdouts help leadership evaluate **true impact** during performance reviews and roadmap planning.
#
# ##### ⚠️ When We Skip Holdouts  
# - **Bug fixes** or critical updates (e.g., spam, abuse, policy violations).
# - **Sensitive changes** like content filtering (e.g., child safety flags).
#
# </details>
#

# %% [markdown]
# <a id="ab-test-limits"></a>
# <h4>🚫 Limits & Alteratives</h4>
#
# <details><summary><strong>📉 When Not to A/B Test & What to Do Instead (Click to Expand)</strong></summary>
#
# ##### 🙅‍♀️ When Not to A/B Test
# - **Lack of infrastructure** → No tracking, engineering, or experiment setup.
# - **Lack of impact** → Not worth the effort if the feature has minimal upside, shipping features has downstream implications (support, bugs, operations)..
# - **Lack of traffic** → Can’t reach stat sig in a reasonable time.
# - **Lack of conviction** → No strong hypothesis; testing dozens of variants blindly.
# - **Lack of isolation** → Hard to contain exposure (e.g., testing a new logo everyone sees).
#
# ##### 🧪 Alternatives & Edge Cases
# - Use **user interviews or logs** to gather directional signals.
# - Leverage **retrospective data** for pre/post comparisons.
# - Consider **sequential testing** or **soft rollouts** for low-risk changes.
# - Use **design experiments** (e.g., multivariate, observational) when randomization isn't feasible.
#
# </details>

# %% [markdown]
# [Back to the top](#table-of-contents)
# ___
#
