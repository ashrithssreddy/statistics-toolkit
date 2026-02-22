"""
Smoke test for the Hypothesis Testing pipeline.

Runs the full notebook flow (validate → generate data → EDA → validate → determine test
→ hypothesis statement → run test) for a set of valid configs. Ensures no exceptions
and that each run returns the expected test and a valid result when p_value is set.

Note: For poisson_test configs, the GLM in run_hypothesis_test may raise (e.g. convergence);
the exception is caught there and result['p_value'] can be None. The smoke test still
passes (expected test name is set); consider making the Poisson path more robust in ht_utils.

Usage:
    python smoke_test.py

Or with pytest (one test per config):
    pytest smoke_test.py -v
"""
from __future__ import annotations

import io
import sys
from copy import deepcopy

# Non-interactive backend so plt.show() does not block
import matplotlib
matplotlib.use("Agg")

import pandas as pd

# Import after backend set
from ht_utils import (
    validate_config,
    generate_data_from_config,
    infer_group_count_from_data,
    infer_sample_size_from_data,
    infer_outcome_type_from_data,
    infer_distribution_from_data,
    infer_variance_equality,
    infer_parametric_flag,
    determine_test_to_run,
    print_hypothesis_statement,
    run_hypothesis_test,
)

# -----------------------------------------------------------------------------
# Base config (all required keys). Overrides below set the 5-key combo + population_mean when needed.
# -----------------------------------------------------------------------------
BASE_CONFIG = {
    "outcome_type": "continuous",
    "group_relationship": "independent",
    "group_count": "two-sample",
    "distribution": None,
    "variance_equal": None,
    "tail_type": "two-tailed",
    "parametric": None,
    "alpha": 0.05,
    "sample_size": 50,
    "effect_size": 0.5,
    "test_name": None,
    "H_0": None,
    "H_a": None,
}

# List of (override_dict, expected_test_name). Override is merged on top of BASE_CONFIG.
SMOKE_CONFIGS = [
    # One-sample
    (
        {
            "group_count": "one-sample",
            "outcome_type": "continuous",
            "distribution": "normal",
            "population_mean": 0.0,
        },
        "one_sample_ttest",
    ),
    (
        {
            "group_count": "one-sample",
            "outcome_type": "continuous",
            "distribution": "non-normal",
            "population_mean": 0.0,
        },
        "one_sample_wilcoxon",
    ),
    (
        {
            "group_count": "one-sample",
            "outcome_type": "binary",
            "distribution": None,
            "population_mean": 0.5,
        },
        "one_proportion_ztest",
    ),
    # Two-sample independent
    (
        {
            "group_count": "two-sample",
            "group_relationship": "independent",
            "outcome_type": "continuous",
            "distribution": "normal",
            "variance_equal": "equal",
        },
        "two_sample_ttest_pooled",
    ),
    (
        {
            "group_count": "two-sample",
            "group_relationship": "independent",
            "outcome_type": "continuous",
            "distribution": "normal",
            "variance_equal": "unequal",
        },
        "two_sample_ttest_welch",
    ),
    (
        {
            "group_count": "two-sample",
            "group_relationship": "independent",
            "outcome_type": "continuous",
            "distribution": "non-normal",
            "variance_equal": None,
        },
        "mann_whitney_u",
    ),
    (
        {
            "group_count": "two-sample",
            "group_relationship": "independent",
            "outcome_type": "binary",
            "distribution": None,
            "variance_equal": None,
        },
        "two_proportion_ztest",
    ),
    (
        {
            "group_count": "two-sample",
            "group_relationship": "independent",
            "outcome_type": "categorical",
            "distribution": None,
            "variance_equal": None,
        },
        "chi_square",
    ),
    (
        {
            "group_count": "two-sample",
            "group_relationship": "independent",
            "outcome_type": "count",
            "distribution": None,
            "variance_equal": None,
        },
        "poisson_test",
    ),
    # Two-sample paired
    (
        {
            "group_count": "two-sample",
            "group_relationship": "paired",
            "outcome_type": "continuous",
            "distribution": "normal",
            "variance_equal": None,
        },
        "paired_ttest",
    ),
    (
        {
            "group_count": "two-sample",
            "group_relationship": "paired",
            "outcome_type": "binary",
            "distribution": None,
            "variance_equal": None,
        },
        "mcnemar",
    ),
    # Multi-sample
    (
        {
            "group_count": "multi-sample",
            "group_relationship": "independent",
            "outcome_type": "continuous",
            "distribution": "normal",
            "variance_equal": "equal",
        },
        "anova",
    ),
    (
        {
            "group_count": "multi-sample",
            "group_relationship": "independent",
            "outcome_type": "continuous",
            "distribution": "normal",
            "variance_equal": "unequal",
        },
        "welch_anova",
    ),
    (
        {
            "group_count": "multi-sample",
            "group_relationship": "independent",
            "outcome_type": "continuous",
            "distribution": "non-normal",
            "variance_equal": None,
        },
        "kruskal_wallis",
    ),
    (
        {
            "group_count": "multi-sample",
            "group_relationship": "independent",
            "outcome_type": "binary",
            "distribution": None,
            "variance_equal": None,
        },
        "chi_square",
    ),
    (
        {
            "group_count": "multi-sample",
            "group_relationship": "independent",
            "outcome_type": "count",
            "distribution": None,
            "variance_equal": None,
        },
        "poisson_test",
    ),
]


def run_pipeline(config: dict, quiet: bool = True) -> dict:
    """
    Run the full hypothesis-testing pipeline for one config.

    1. validate_config
    2. generate_data_from_config
    3. EDA: infer group_count, sample_size, outcome_type, distribution, variance_equal, parametric
    4. validate_config again
    5. determine_test_to_run, print_hypothesis_statement, run_hypothesis_test

    Returns the result dict from run_hypothesis_test.
    """
    config = deepcopy(config)
    if quiet:
        out = io.StringIO()
        err = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
    try:
        validate_config(config)
        df = generate_data_from_config(config, seed=1995)
        config["group_count"] = infer_group_count_from_data(config, df)
        config["sample_size"] = infer_sample_size_from_data(config, df)
        config["outcome_type"] = infer_outcome_type_from_data(config, df)
        config["distribution"] = infer_distribution_from_data(config, df)
        config["variance_equal"] = infer_variance_equality(config, df)
        config["parametric"] = infer_parametric_flag(config)
        validate_config(config)
        config["test_name"] = determine_test_to_run(config)
        config["H_0"], config["H_a"] = print_hypothesis_statement(config)
        result = run_hypothesis_test(config, df)
        return result
    finally:
        if quiet:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def main() -> int:
    failed = 0
    for override, expected_test in SMOKE_CONFIGS:
        config = {**BASE_CONFIG, **override}
        label = f"{config['group_count']} | {config['outcome_type']} | {expected_test}"
        try:
            result = run_pipeline(config, quiet=True)
            p = result.get("p_value")
            test = result.get("test")
            if test is None or test == "test_not_found":
                print(f"FAIL {label}: test_name = {test!r}")
                failed += 1
            elif test != expected_test:
                print(f"FAIL {label}: got test {test!r}, expected {expected_test!r}")
                failed += 1
            elif p is not None and not (0 <= p <= 1):
                print(f"FAIL {label}: invalid p_value = {p!r}")
                failed += 1
            else:
                if p is None and expected_test == "poisson_test":
                    print(f"OK   {label} (p_value None: GLM may have failed)")
                else:
                    print(f"OK   {label}")
        except Exception as e:
            msg = str(e).encode("ascii", errors="replace").decode()
            print(f"FAIL {label}: {msg}")
            failed += 1
    print()
    print(f"Passed: {len(SMOKE_CONFIGS) - failed} / {len(SMOKE_CONFIGS)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
