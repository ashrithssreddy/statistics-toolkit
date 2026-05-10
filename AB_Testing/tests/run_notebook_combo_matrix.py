from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
import subprocess
from typing import Any

import numpy as np
import pandas as pd

# Allow direct imports used by AB utils.
AB_TESTING_DIR = Path(__file__).resolve().parents[1]
if str(AB_TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(AB_TESTING_DIR))

from utils.ab_utils_01_data_setup import add_outcome_metrics, create_dummy_ab_data
from utils.ab_utils_02_power_analysis import determine_test_family
from utils.ab_utils_03_randomization import (
    apply_block_randomization,
    apply_cluster_randomization,
    apply_matched_pair_randomization,
    apply_simple_randomization,
    apply_stratified_randomization,
)
from utils.ab_utils_05_ab_testing import run_ab_test


SUPPORTED_RUN_AB_FAMILIES = {
    "z_test",
    "t_test",
    "paired_t_test",
    "mcnemar_test",
    "non_parametric",
    "mann_whitney_u_test",
    "chi_square",
}


def normalize_family_for_runner(family: str) -> str:
    mapping = {
        "two_proportion_z_test": "z_test",
        "chi_square_test": "chi_square",
        "two_sample_t_test": "t_test",
        "welch_t_test": "t_test",
        "paired_t_test": "paired_t_test",
        "mann_whitney_u_test": "mann_whitney_u_test",
    }
    return mapping.get(family, family)


@dataclass(frozen=True)
class Scenario:
    datatype: str
    group_relationship: str
    randomization_method: str
    group_labels: tuple[str, str]
    pre_experiment_metric: str | None
    guardrail_metric_col: str | None


def _apply_randomization(df: pd.DataFrame, s: Scenario, group_col: str) -> pd.DataFrame:
    if s.randomization_method == "simple":
        return apply_simple_randomization(df, group_labels=s.group_labels, group_col=group_col, seed=1995)
    if s.randomization_method == "stratified":
        return apply_stratified_randomization(
            df, stratify_col="platform", group_labels=s.group_labels, group_col=group_col, seed=1995
        )
    if s.randomization_method == "block":
        return apply_block_randomization(
            df,
            observation_id_col="user_id",
            group_labels=s.group_labels,
            group_col=group_col,
            block_size=10,
            seed=1995,
        )
    if s.randomization_method == "matched_pair":
        sort_col = s.pre_experiment_metric or "past_purchase_count"
        return apply_matched_pair_randomization(
            df, sort_col=sort_col, group_col=group_col, group_labels=s.group_labels
        )
    if s.randomization_method == "cluster":
        if "city" not in df.columns:
            df = df.copy()
            rng = np.random.default_rng(1995)
            df["city"] = rng.choice(["ny", "sf", "chicago", "austin"], size=len(df))
        return apply_cluster_randomization(
            df, cluster_col="city", group_labels=s.group_labels, group_col=group_col, seed=1995
        )
    raise ValueError(f"Unsupported randomization_method: {s.randomization_method}")


def _coerce_metric_for_datatype(df: pd.DataFrame, metric_col: str, datatype: str) -> pd.DataFrame:
    df = df.copy()
    if datatype == "binary":
        threshold = float(df[metric_col].median())
        df[metric_col] = (df[metric_col] >= threshold).astype(int)
    elif datatype == "categorical":
        q1 = float(df[metric_col].quantile(0.33))
        q2 = float(df[metric_col].quantile(0.66))
        df[metric_col] = np.where(
            df[metric_col] < q1,
            "low",
            np.where(df[metric_col] < q2, "mid", "high"),
        )
    return df


def run_one_scenario(s: Scenario) -> dict[str, Any]:
    outcome_metric_col = "engagement_score"
    group_col = "group"

    record: dict[str, Any] = {
        "datatype": s.datatype,
        "group_relationship": s.group_relationship,
        "randomization_method": s.randomization_method,
        "group_labels": s.group_labels,
        "status": "pass",
        "reason": "",
        "selected_family": None,
        "runner_family": None,
        "p_value": None,
        "n_total": None,
    }

    try:
        df = create_dummy_ab_data(
            observations_count=600,
            seed=1995,
            outcome_metric_col=outcome_metric_col,
            guardrail_metric_col=s.guardrail_metric_col,
        )

        df = _apply_randomization(df=df, s=s, group_col=group_col)
        df = add_outcome_metrics(
            df,
            group_col=group_col,
            group_labels=s.group_labels,
            outcome_metric_col=outcome_metric_col,
            guardrail_metric_col=s.guardrail_metric_col,
            treatment_effect=True,
            seed=1995,
        )
        df = _coerce_metric_for_datatype(df, outcome_metric_col, s.datatype)

        cfg = {
            "outcome_metric_datatype": s.datatype,
            "group_count": len(s.group_labels),
            "group_relationship": s.group_relationship,
            "normality": True if s.datatype == "continuous" else None,
            "variance_equal": True if s.datatype == "continuous" else None,
        }

        selected_family = determine_test_family(cfg)
        runner_family = normalize_family_for_runner(selected_family)
        record["selected_family"] = selected_family
        record["runner_family"] = runner_family
        record["n_total"] = len(df)

        if runner_family not in SUPPORTED_RUN_AB_FAMILIES:
            record["status"] = "unsupported"
            record["reason"] = f"Family '{selected_family}' is not implemented in run_ab_test."
            return record

        result = run_ab_test(
            df=df,
            group_col=group_col,
            metric_col=outcome_metric_col,
            group_labels=s.group_labels,
            test_family=runner_family,
            group_relationship=s.group_relationship,
            alpha=0.05,
        )

        p = result.get("p_value")
        if p is None or not (0 <= p <= 1):
            record["status"] = "fail"
            record["reason"] = "Invalid p-value produced."
            record["p_value"] = p
            return record

        record["p_value"] = float(p)
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = "fail"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record


def build_scenarios() -> list[Scenario]:
    datatypes = ["continuous", "binary", "categorical"]
    variants = ["independent", "paired"]
    randomization_methods = ["simple", "stratified", "block", "matched_pair", "cluster"]
    scenarios = []
    for datatype, group_relationship, randomization_method in product(datatypes, variants, randomization_methods):
        scenarios.append(
            Scenario(
                datatype=datatype,
                group_relationship=group_relationship,
                randomization_method=randomization_method,
                group_labels=("control", "treatment"),
                pre_experiment_metric="past_purchase_count",
                guardrail_metric_col="bounce_rate",
            )
        )
    return scenarios


def run_combo_matrix() -> pd.DataFrame:
    rows = [run_one_scenario(s) for s in build_scenarios()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    report = run_combo_matrix()
    cols = [
        "datatype",
        "group_relationship",
        "randomization_method",
        "selected_family",
        "runner_family",
        "status",
        "p_value",
        "reason",
    ]
    print("\n=== Notebook Combo Matrix Report ===")
    print(report[cols].to_string(index=False))

    print("\n=== Status Counts ===")
    print(report["status"].value_counts().to_string())

    out_path = "AB_Testing/tests/combo_matrix_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nSaved full report to: {out_path}")

    # Auto-open report in default app (Excel/Sheets) when run locally.
    out_abs = Path(out_path).resolve()
    try:
        if sys.platform.startswith("win"):
            import os

            os.startfile(str(out_abs))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(out_abs)], check=False)
        else:
            subprocess.run(["xdg-open", str(out_abs)], check=False)
        print(f"Opened report: {out_abs}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not auto-open report: {exc}")

