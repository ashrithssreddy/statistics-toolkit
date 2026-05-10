from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


# Same import setup needed when running this from AB_Testing/tests.
AB_TESTING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AB_TESTING_DIR.parent
sys.path.insert(0, str(AB_TESTING_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ab_utils_01_data_setup import *  # noqa: F401,F403,E402
from utils.ab_utils_02_power_analysis import *  # noqa: F401,F403,E402
from utils.ab_utils_03_randomization import *  # noqa: F401,F403,E402
from utils.ab_utils_04_aa_testing import *  # noqa: F401,F403,E402
from utils.ab_utils_05_ab_testing import *  # noqa: F401,F403,E402
from utils.ab_utils_06_post_hoc import *  # noqa: F401,F403,E402


my_seed = 1995

# Notebook variables.
outcome_metric_col = "engagement_score"
observation_id_col = "user_id"
group_col = "group"
group_labels = ("control", "treatment")
group_count = len(group_labels)

# Change these lists to test fewer/more combinations.
outcome_metric_datatype_values = ["continuous", "binary", "categorical"]
group_relationship_values = ["independent", "paired"]
hypothesis_type_values = ["two_sided", "greater", "less"]
randomization_method_values = ["simple", "stratified", "block", "matched_pair", "cluster"]
historical_normality_values = ["normal", "non_normal"]
pre_experiment_metric_col_values = ["past_purchase_revenue", None]
guardrail_metric_col_values = ["bounce_rate", None]

rows = []

for outcome_metric_datatype in outcome_metric_datatype_values:
    for group_relationship in group_relationship_values:
        for hypothesis_type in hypothesis_type_values:
            for randomization_method in randomization_method_values:
                for historical_normality in historical_normality_values:
                    for pre_experiment_metric_col in pre_experiment_metric_col_values:
                        for guardrail_metric_col in guardrail_metric_col_values:

                            print(
                                outcome_metric_datatype,
                                group_relationship,
                                hypothesis_type,
                                randomization_method,
                                historical_normality,
                                pre_experiment_metric_col,
                                guardrail_metric_col,
                            )

                            try:
                                # ------------------------------------------------------------------
                                # Central Control Panel
                                # ------------------------------------------------------------------
                                test_config = {
                                    "outcome_metric_col": outcome_metric_col,
                                    "observation_id_col": observation_id_col,
                                    "pre_experiment_metric_col": pre_experiment_metric_col,
                                    "guardrail_metric_col": guardrail_metric_col,
                                    "outcome_metric_datatype": outcome_metric_datatype,
                                    "group_labels": group_labels,
                                    "group_count": group_count,
                                    "group_relationship": group_relationship,
                                    "hypothesis_type": hypothesis_type,
                                    "normality": None,
                                    "equal_variance": None,
                                    "family": None,
                                }

                                # ------------------------------------------------------------------
                                # Read/Generate Data
                                # ------------------------------------------------------------------
                                observations_count = 1000
                                df = create_dummy_ab_data(
                                    observations_count,
                                    seed=my_seed,
                                    outcome_metric_col=outcome_metric_col,
                                    guardrail_metric_col=guardrail_metric_col,
                                )

                                historical_df = create_historical_df(
                                    df,
                                    outcome_metric_col,
                                    guardrail_metric_col,
                                    seed=my_seed,
                                    historical_normality=historical_normality,
                                )

                                # The notebook is continuous by default. This inline block lets the
                                # same notebook calls run against binary/categorical combos.
                                if outcome_metric_datatype == "binary":
                                    historical_df[outcome_metric_col] = (
                                        historical_df[outcome_metric_col]
                                        >= historical_df[outcome_metric_col].median()
                                    ).astype(int)

                                if outcome_metric_datatype == "categorical":
                                    q1 = historical_df[outcome_metric_col].quantile(0.33)
                                    q2 = historical_df[outcome_metric_col].quantile(0.66)
                                    historical_df[outcome_metric_col] = np.where(
                                        historical_df[outcome_metric_col] < q1,
                                        "low",
                                        np.where(
                                            historical_df[outcome_metric_col] < q2,
                                            "mid",
                                            "high",
                                        ),
                                    )

                                # ------------------------------------------------------------------
                                # Power Analysis
                                # ------------------------------------------------------------------
                                historical_df_for_tests = apply_simple_randomization(
                                    historical_df.copy(),
                                    group_col=group_col,
                                    group_labels=group_labels,
                                    seed=my_seed,
                                )

                                if outcome_metric_datatype == "continuous":
                                    test_config = test_normality(
                                        df=historical_df_for_tests,
                                        group_col=group_col,
                                        test_config=test_config,
                                        update_config=True,
                                    )

                                    test_config = test_equal_variance(
                                        df=historical_df_for_tests,
                                        group_col=group_col,
                                        test_config=test_config,
                                        update_config=True,
                                    )

                                test_config["family"] = determine_test_family(test_config)

                                baseline = compute_baseline_from_data(
                                    historical_df,
                                    test_config,
                                    verbose=False,
                                )

                                mde = 5
                                required_sample_size = None

                                if test_config["family"] != "mcnemar_test":
                                    if outcome_metric_datatype == "continuous":
                                        required_sample_size = calculate_power_sample_size(
                                            test_family=test_config["family"],
                                            group_relationship=group_relationship,
                                            hypothesis_type=hypothesis_type,
                                            alpha=0.05,
                                            power=0.80,
                                            std_dev=baseline["baseline_std_dev"],
                                            mde=mde,
                                            num_groups=group_count,
                                            verbose=False,
                                        )

                                    if outcome_metric_datatype == "binary":
                                        required_sample_size = calculate_power_sample_size(
                                            test_family=test_config["family"],
                                            group_relationship=group_relationship,
                                            hypothesis_type=hypothesis_type,
                                            alpha=0.05,
                                            power=0.80,
                                            baseline_rate=baseline["baseline_rate"],
                                            mde=0.05,
                                            num_groups=group_count,
                                            verbose=False,
                                        )

                                    if outcome_metric_datatype == "categorical":
                                        required_sample_size = calculate_power_sample_size(
                                            test_family=test_config["family"],
                                            group_relationship=group_relationship,
                                            hypothesis_type=hypothesis_type,
                                            alpha=0.05,
                                            power=0.80,
                                            baseline_rate=0.33,
                                            mde=0.05,
                                            num_groups=group_count,
                                            verbose=False,
                                        )

                                # Ensure the dataset matches the minimum required experiment size.
                                if required_sample_size is None:
                                    n_required = min(len(df), 800)
                                else:
                                    n_required = min(
                                        len(df),
                                        required_sample_size * test_config["group_count"],
                                        800,
                                    )

                                df = df.sample(n=n_required, random_state=42)

                                # ------------------------------------------------------------------
                                # Apply Randomization
                                # ------------------------------------------------------------------
                                if randomization_method == "simple":
                                    df = apply_simple_randomization(
                                        df,
                                        group_col=group_col,
                                        group_labels=group_labels,
                                        seed=my_seed,
                                    )

                                elif randomization_method == "stratified":
                                    df = apply_stratified_randomization(
                                        df,
                                        stratify_col="platform",
                                        group_col=group_col,
                                        group_labels=group_labels,
                                        seed=my_seed,
                                    )

                                elif randomization_method == "block":
                                    df = apply_block_randomization(
                                        df,
                                        observation_id_col=observation_id_col,
                                        group_col=group_col,
                                        block_size=10,
                                        group_labels=group_labels,
                                        seed=my_seed,
                                    )

                                elif randomization_method == "matched_pair":
                                    df = apply_matched_pair_randomization(
                                        df,
                                        sort_col=pre_experiment_metric_col
                                        or "past_purchase_revenue",
                                        group_col=group_col,
                                        group_labels=test_config["group_labels"],
                                    )

                                elif randomization_method == "cluster":
                                    if "city" not in df.columns:
                                        rng = np.random.default_rng(my_seed)
                                        df["city"] = rng.choice(
                                            ["ny", "sf", "chicago", "austin"],
                                            size=len(df),
                                        )

                                    df = apply_cluster_randomization(
                                        df,
                                        cluster_col="city",
                                        group_col=group_col,
                                        group_labels=group_labels,
                                        seed=my_seed,
                                    )

                                else:
                                    raise ValueError(
                                        f"Unsupported randomization method: {randomization_method}"
                                    )

                                check_sample_ratio_mismatch(
                                    df,
                                    group_col=group_col,
                                    group_labels=test_config["group_labels"],
                                    expected_ratios=[0.5, 0.5],
                                )

                                # ------------------------------------------------------------------
                                # AA Testing
                                # ------------------------------------------------------------------
                                df = add_outcome_metrics(
                                    df,
                                    group_col=group_col,
                                    group_labels=test_config["group_labels"],
                                    outcome_metric_col=test_config["outcome_metric_col"],
                                    guardrail_metric_col=test_config.get(
                                        "guardrail_metric_col"
                                    )
                                    or guardrail_metric_col,
                                    treatment_effect=False,
                                    seed=my_seed,
                                )

                                if outcome_metric_datatype == "binary":
                                    df[outcome_metric_col] = (
                                        df[outcome_metric_col]
                                        >= df[outcome_metric_col].median()
                                    ).astype(int)

                                if outcome_metric_datatype == "categorical":
                                    q1 = df[outcome_metric_col].quantile(0.33)
                                    q2 = df[outcome_metric_col].quantile(0.66)
                                    df[outcome_metric_col] = np.where(
                                        df[outcome_metric_col] < q1,
                                        "low",
                                        np.where(
                                            df[outcome_metric_col] < q2,
                                            "mid",
                                            "high",
                                        ),
                                    )

                                aa_result = run_outcome_similarity_test(
                                    df=df,
                                    group_col=group_col,
                                    metric_col=test_config["outcome_metric_col"],
                                    test_family=test_config["family"],
                                    group_relationship=test_config.get(
                                        "group_relationship"
                                    ),
                                    hypothesis_type=test_config.get(
                                        "hypothesis_type",
                                        "two_sided",
                                    ),
                                    group_labels=test_config["group_labels"],
                                    alpha=0.05,
                                    verbose=False,
                                )

                                # ------------------------------------------------------------------
                                # AB Testing
                                # ------------------------------------------------------------------
                                df = add_outcome_metrics(
                                    df,
                                    group_col=group_col,
                                    group_labels=test_config["group_labels"],
                                    outcome_metric_col=test_config["outcome_metric_col"],
                                    guardrail_metric_col=test_config.get(
                                        "guardrail_metric_col"
                                    )
                                    or guardrail_metric_col,
                                    treatment_effect=True,
                                    seed=my_seed,
                                )

                                if outcome_metric_datatype == "binary":
                                    df[outcome_metric_col] = (
                                        df[outcome_metric_col]
                                        >= df[outcome_metric_col].median()
                                    ).astype(int)

                                if outcome_metric_datatype == "categorical":
                                    q1 = df[outcome_metric_col].quantile(0.33)
                                    q2 = df[outcome_metric_col].quantile(0.66)
                                    df[outcome_metric_col] = np.where(
                                        df[outcome_metric_col] < q1,
                                        "low",
                                        np.where(
                                            df[outcome_metric_col] < q2,
                                            "mid",
                                            "high",
                                        ),
                                    )

                                if outcome_metric_datatype == "continuous":
                                    test_config = test_normality(
                                        df=df,
                                        group_col=group_col,
                                        test_config=test_config,
                                        update_config=True,
                                    )

                                    test_config = test_equal_variance(
                                        df=df,
                                        group_col=group_col,
                                        test_config=test_config,
                                        update_config=True,
                                    )

                                    test_config["family"] = determine_test_family(
                                        test_config
                                    )

                                ab_result = run_ab_test(
                                    df=df,
                                    group_col=group_col,
                                    metric_col=test_config["outcome_metric_col"],
                                    group_labels=test_config["group_labels"],
                                    test_family=test_config["family"],
                                    group_relationship=test_config.get(
                                        "group_relationship"
                                    ),
                                    hypothesis_type=test_config.get(
                                        "hypothesis_type",
                                        "two_sided",
                                    ),
                                    alpha=0.05,
                                )

                                # ------------------------------------------------------------------
                                # Post Hoc Analysis
                                # ------------------------------------------------------------------
                                if guardrail_metric_col is not None:
                                    run_guardrail_analysis(
                                        df,
                                        test_config,
                                        group_col=group_col,
                                        alpha=0.05,
                                    )

                                cuped_result = None
                                if (
                                    pre_experiment_metric_col is not None
                                    and outcome_metric_datatype == "continuous"
                                ):
                                    df = apply_cuped(
                                        df,
                                        pre_metric=pre_experiment_metric_col,
                                        outcome_metric=outcome_metric_col,
                                    )

                                    cuped_result = run_ab_test(
                                        df=df,
                                        group_col=group_col,
                                        metric_col=f"{outcome_metric_col}_cuped",
                                        group_labels=test_config["group_labels"],
                                        test_family=test_config["family"],
                                        group_relationship=test_config.get(
                                            "group_relationship"
                                        ),
                                        hypothesis_type=test_config.get(
                                            "hypothesis_type",
                                            "two_sided",
                                        ),
                                        alpha=0.05,
                                    )

                                rows.append(
                                    {
                                        "outcome_metric_datatype": outcome_metric_datatype,
                                        "group_relationship": group_relationship,
                                        "hypothesis_type": hypothesis_type,
                                        "randomization_method": randomization_method,
                                        "historical_normality": historical_normality,
                                        "pre_experiment_metric_col": pre_experiment_metric_col,
                                        "guardrail_metric_col": guardrail_metric_col,
                                        "family": test_config["family"],
                                        "n_required": n_required,
                                        "status": "pass",
                                        "aa_p_value": aa_result.get("p_value")
                                        if isinstance(aa_result, dict)
                                        else None,
                                        "ab_p_value": ab_result.get("p_value"),
                                        "cuped_p_value": cuped_result.get("p_value")
                                        if cuped_result is not None
                                        else None,
                                        "error": None,
                                    }
                                )

                            except Exception as exc:
                                rows.append(
                                    {
                                        "outcome_metric_datatype": outcome_metric_datatype,
                                        "group_relationship": group_relationship,
                                        "hypothesis_type": hypothesis_type,
                                        "randomization_method": randomization_method,
                                        "historical_normality": historical_normality,
                                        "pre_experiment_metric_col": pre_experiment_metric_col,
                                        "guardrail_metric_col": guardrail_metric_col,
                                        "family": test_config.get("family")
                                        if "test_config" in locals()
                                        else None,
                                        "n_required": None,
                                        "status": "fail",
                                        "aa_p_value": None,
                                        "ab_p_value": None,
                                        "cuped_p_value": None,
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc(),
                                    }
                                )


report = pd.DataFrame(rows)
out_path = Path(__file__).with_name("notebook_nested_combo_report.csv")
report.to_csv(out_path, index=False)

print("\nStatus counts:")
print(report["status"].value_counts(dropna=False))
print(f"\nSaved report to: {out_path}")

