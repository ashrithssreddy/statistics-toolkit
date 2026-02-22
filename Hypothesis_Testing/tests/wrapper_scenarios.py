"""
Run the hypothesis-testing pipeline for every 5-key combination (216 rows).

Builds the full table: outcome_type × group_relationship × group_count × distribution × variance_equal,
then for each row runs the full pipeline. Prints all rows to the terminal (CSV-style), then lists
every failure with its 5-key combo and full error message so you can see exactly what is rejected
or implemented poorly.

Usage:
    python run_all_combos.py
"""
from __future__ import annotations

import sys
from itertools import product

# Non-interactive backend before any ht_utils/matplotlib
import matplotlib
matplotlib.use("Agg")

from smoke_test import BASE_CONFIG, run_pipeline


# Full grid of 5 keys (216 = 4×2×3×3×3)
OUTCOME_TYPES = ["continuous", "binary", "categorical", "count"]
GROUP_RELATIONSHIPS = ["independent", "paired"]
GROUP_COUNTS = ["one-sample", "two-sample", "multi-sample"]
DISTRIBUTIONS = ["normal", "non-normal", None]
VARIANCE_EQUAL_OPTS = ["equal", "unequal", None]


def build_config(
    outcome_type: str,
    group_relationship: str,
    group_count: str,
    distribution: str | None,
    variance_equal: str | None,
) -> dict:
    """Build a full config from the 5 keys. Adds population_mean for one-sample."""
    config = {
        **BASE_CONFIG,
        "outcome_type": outcome_type,
        "group_relationship": group_relationship,
        "group_count": group_count,
        "distribution": distribution,
        "variance_equal": variance_equal,
    }
    if group_count == "one-sample":
        config["population_mean"] = 0.5 if outcome_type == "binary" else 0.0
    return config


def run_one_row(
    outcome_type: str,
    group_relationship: str,
    group_count: str,
    distribution: str | None,
    variance_equal: str | None,
) -> tuple[str, str, str]:
    """
    Run the pipeline for one 5-key combo. Returns (status, error_message, test_name).
    status is "OK" or "FAIL". error_message is "" if OK.
    """
    config = build_config(
        outcome_type, group_relationship, group_count, distribution, variance_equal
    )
    try:
        result = run_pipeline(config, quiet=True)
        test = result.get("test") or ""
        return ("OK", "", test)
    except Exception as e:
        msg = str(e).encode("ascii", errors="replace").decode()
        return ("FAIL", msg, "")


def _csv_escape(s: str) -> str:
    """Escape a field for CSV (quote if contains comma or newline)."""
    if s is None:
        return ""
    s = str(s)
    if "," in s or "\n" in s or '"' in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def main() -> int:
    rows = list(
        product(
            OUTCOME_TYPES,
            GROUP_RELATIONSHIPS,
            GROUP_COUNTS,
            DISTRIBUTIONS,
            VARIANCE_EQUAL_OPTS,
        )
    )
    total = len(rows)

    results = []
    for outcome_type, group_relationship, group_count, distribution, variance_equal in rows:
        status, error_message, test_name = run_one_row(
            outcome_type, group_relationship, group_count, distribution, variance_equal
        )
        results.append(
            {
                "outcome_type": outcome_type,
                "group_relationship": group_relationship,
                "group_count": group_count,
                "distribution": distribution if distribution is not None else "None",
                "variance_equal": variance_equal if variance_equal is not None else "None",
                "status": status,
                "error_message": error_message,
                "test_name": test_name,
            }
        )

    # CSV header
    cols = [
        "outcome_type",
        "group_relationship",
        "group_count",
        "distribution",
        "variance_equal",
        "status",
        "error_message",
        "test_name",
    ]
    print(",".join(cols))

    # Print every row to terminal (CSV-style)
    for r in results:
        line = ",".join(
            _csv_escape(r[c]) for c in cols
        )
        print(line)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = total - ok_count

    # Summary
    print()
    print(f"Total: {total}  |  OK: {ok_count}  |  FAIL: {fail_count}")

    # Every failure: full 5-key combo + full error message (so you see what is implemented poorly)
    if fail_count > 0:
        print()
        print("=" * 80)
        print("ALL FAILURES (each combo + full error)")
        print("=" * 80)
        for r in results:
            if r["status"] != "FAIL":
                continue
            combo = (
                f"outcome_type={r['outcome_type']} | group_relationship={r['group_relationship']} | "
                f"group_count={r['group_count']} | distribution={r['distribution']} | "
                f"variance_equal={r['variance_equal']}"
            )
            print(combo)
            print(f"  -> {r['error_message']}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
