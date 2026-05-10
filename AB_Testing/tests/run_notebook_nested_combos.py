from itertools import product
import os
from pathlib import Path
import sys

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

# Set this to the row number printed before a failure to resume from there.
START_ROW = 0

combo_space = pd.DataFrame(
    list(
        product(
            outcome_metric_datatype_values,
            group_relationship_values,
            hypothesis_type_values,
            randomization_method_values,
            historical_normality_values,
            pre_experiment_metric_col_values,
            guardrail_metric_col_values,
        )
    ),
    columns=[
        "outcome_metric_datatype",
        "group_relationship",
        "hypothesis_type",
        "randomization_method",
        "historical_normality",
        "pre_experiment_metric_col",
        "guardrail_metric_col",
    ],
)
combo_space["run_combo"] = True
combo_space.loc[
    (combo_space["randomization_method"] == "matched_pair")
    & (combo_space["pre_experiment_metric_col"].isna()),
    "run_combo",
] = False

rows = []

os.system("cls" if os.name == "nt" else "clear")

for combo_row, combo in combo_space[
    (combo_space.index >= START_ROW) & (combo_space["run_combo"])
].iterrows():
    for _var_name in [
        "test_config",
        "observations_count",
        "df",
        "historical_df",
        "alpha",
        "power",
        "_b",
        "mde",
        "n_required",
        "aa_result",
        "result",
        "result_cuped",
        "out_path",
        "report",
    ]:
        globals().pop(_var_name, None)

    outcome_metric_datatype = combo["outcome_metric_datatype"]
    group_relationship = combo["group_relationship"]
    hypothesis_type = combo["hypothesis_type"]
    randomization_method = combo["randomization_method"]
    historical_normality = combo["historical_normality"]
    pre_experiment_metric_col = combo["pre_experiment_metric_col"]
    guardrail_metric_col = combo["guardrail_metric_col"]

    print(
        "combo_row=",
        combo_row,
        "of",
        len(combo_space) - 1,
        "|",
        outcome_metric_datatype,
        group_relationship,
        hypothesis_type,
        randomization_method,
        historical_normality,
        pre_experiment_metric_col,
        guardrail_metric_col,
    )

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
    observations_count = 5000
    df = create_dummy_ab_data(
        observations_count,
        seed=my_seed,
        outcome_metric_col=outcome_metric_col,
        guardrail_metric_col=guardrail_metric_col,
        randomization_method=randomization_method,
    )

    historical_df = create_historical_df(
        df,
        outcome_metric_col,
        guardrail_metric_col,
        seed=my_seed,
        historical_normality=historical_normality,
        outcome_metric_datatype=outcome_metric_datatype,
    )

    # ------------------------------------------------------------------
    # Power Analysis
    # ------------------------------------------------------------------
    alpha = 0.05
    power = 0.80

    _b = compute_baseline_from_data(historical_df, test_config)
    test_config["baseline_rate"] = _b["baseline_rate"]
    test_config["baseline_mean"] = _b["baseline_mean"]
    test_config["baseline_std_dev"] = _b["baseline_std_dev"]

    if outcome_metric_datatype in ["binary", "categorical"]:
        mde = 0.05
    else:
        mde = 5

    test_config = test_normality(
        df=historical_df,
        group_col=group_col,
        test_config=test_config,
        update_config=True,
    )

    test_config["family"] = determine_test_family(test_config)

    test_config["required_sample_size"] = calculate_power_sample_size(
        test_family=test_config["family"],
        group_relationship=test_config.get("group_relationship"),
        hypothesis_type=test_config.get("hypothesis_type", "two_sided"),
        alpha=alpha,
        power=power,
        baseline_rate=test_config.get("baseline_rate"),
        mde=mde,
        std_dev=test_config.get("baseline_std_dev"),
        effect_size=None,
        num_groups=test_config["group_count"],
    )

    # Ensure the dataset matches the minimum required experiment size.
    n_required = test_config["required_sample_size"] * test_config["group_count"]

    df = df.sample(n=n_required, random_state=42)

    # ------------------------------------------------------------------
    # Apply Randomization
    # ------------------------------------------------------------------
    if randomization_method == "simple":
        df = apply_simple_randomization(df, group_col=group_col, seed=my_seed)

    elif randomization_method == "stratified":
        df = apply_stratified_randomization(
            df,
            stratify_col="platform",
            group_col=group_col,
            seed=my_seed,
        )

    elif randomization_method == "block":
        df = apply_block_randomization(
            df,
            observation_id_col=observation_id_col,
            group_col=group_col,
            block_size=10,
            seed=my_seed,
        )

    elif randomization_method == "matched_pair":
        df = apply_matched_pair_randomization(
            df,
            sort_col=pre_experiment_metric_col,
            group_col=group_col,
            group_labels=test_config["group_labels"],
        )

    elif randomization_method == "cluster":
        df = apply_cluster_randomization(
            df,
            cluster_col="city",
            group_col=group_col,
            seed=my_seed,
        )

    else:
        raise ValueError(f"Unsupported randomization method: {randomization_method}")

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
        guardrail_metric_col=test_config.get("guardrail_metric_col") or guardrail_metric_col,
        treatment_effect=False,
        seed=my_seed,
        outcome_metric_datatype=outcome_metric_datatype,
        historical_normality=historical_normality,
    )

    aa_result = run_outcome_similarity_test(
        df=df,
        group_col="group",
        metric_col=test_config["outcome_metric_col"],
        test_family=test_config["family"],
        group_relationship=test_config.get("group_relationship"),
        hypothesis_type=test_config.get("hypothesis_type", "two_sided"),
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
        guardrail_metric_col=test_config.get("guardrail_metric_col") or guardrail_metric_col,
        treatment_effect=True,
        seed=my_seed,
        outcome_metric_datatype=outcome_metric_datatype,
        historical_normality=historical_normality,
    )

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

    test_config["family"] = determine_test_family(test_config)

    result = run_ab_test(
        df=df,
        group_col="group",
        metric_col=test_config["outcome_metric_col"],
        group_labels=test_config["group_labels"],
        test_family=test_config["family"],
        group_relationship=test_config.get("group_relationship"),
        hypothesis_type=test_config.get("hypothesis_type", "two_sided"),
        alpha=0.05,
    )

    # ------------------------------------------------------------------
    # Post Hoc Analysis
    # ------------------------------------------------------------------
    run_guardrail_analysis(df, test_config, group_col="group", alpha=0.05)

    # CUPED is a linear adjustment on numeric outcomes; categorical strings break OLS.
    if outcome_metric_datatype == "continuous":
        df = apply_cuped(
            df=df,
            pre_metric="past_purchase_revenue",
            outcome_metric_col=test_config["outcome_metric_col"],
            group_col="group",
            group_labels=test_config["group_labels"],
        )

        result_cuped = run_ab_test(
            df=df,
            group_col="group",
            metric_col=f"{test_config['outcome_metric_col']}_cuped_adjusted",
            group_labels=test_config["group_labels"],
            test_family=test_config["family"],
            group_relationship=test_config.get("group_relationship"),
            hypothesis_type=test_config.get("hypothesis_type", "two_sided"),
        )
    else:
        result_cuped = {}

    rows.append(
        {
            "combo_row": combo_row,
            "outcome_metric_datatype": outcome_metric_datatype,
            "group_relationship": group_relationship,
            "hypothesis_type": hypothesis_type,
            "randomization_method": randomization_method,
            "historical_normality": historical_normality,
            "pre_experiment_metric_col": pre_experiment_metric_col,
            "guardrail_metric_col": guardrail_metric_col,
            "family": test_config["family"],
            "n_required": n_required,
            # run_outcome_similarity_test returns a float p_value (or None), not a dict.
            "aa_p_value": aa_result.get("p_value") if isinstance(aa_result, dict) else aa_result,
            "ab_p_value": result.get("p_value"),
            "cuped_p_value": (result_cuped or {}).get("p_value"),
        }
    )


report = pd.DataFrame(rows)
out_path = Path(__file__).with_name("notebook_nested_combo_report.csv")
report.to_csv(out_path, index=False)

print("\nFinished all combinations.")
print(f"Saved report to: {out_path}")
