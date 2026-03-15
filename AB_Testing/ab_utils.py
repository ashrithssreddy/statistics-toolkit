# AB Testing utils — facade re-exporting section modules for backward compatibility.
# Sections: Data Setup, Power Analysis, Randomization, AA Testing, A/B Testing, Post Hoc Analysis.

from ab_utils_common import my_seed
from ab_utils_data_setup import (
    create_dummy_ab_data,
    create_historical_df,
    add_outcome_metrics,
)
from ab_utils_randomization import (
    apply_simple_randomization,
    apply_stratified_randomization,
    apply_block_randomization,
    apply_matched_pair_randomization,
    apply_cluster_randomization,
    check_sample_ratio_mismatch,
)
from ab_utils_power_analysis import (
    test_normality,
    test_equal_variance,
    determine_test_family,
    compute_baseline_from_data,
    calculate_power_sample_size,
    print_power_summary,
)
from ab_utils_aa_testing import (
    run_outcome_similarity_test,
    run_aa_testing_generalized,
    visualize_aa_distribution,
    simulate_aa_type1_error_rate,
    plot_p_value_distribution,
)
from ab_utils_ab_testing import (
    run_ab_test,
    summarize_ab_test_result,
    plot_ab_test_results,
    plot_confidence_intervals,
    compute_lift_confidence_interval,
    print_final_ab_test_summary,
    estimate_test_duration,
)
from ab_utils_post_hoc import (
    apply_cuped,
    visualize_segment_lift,
    analyze_segment_lift,
    run_guardrail_analysis,
    evaluate_guardrail_metric,
    simulate_rollout_impact,
)

__all__ = [
    'my_seed',
    'create_dummy_ab_data',
    'create_historical_df',
    'add_outcome_metrics',
    'apply_simple_randomization',
    'apply_stratified_randomization',
    'apply_block_randomization',
    'apply_matched_pair_randomization',
    'apply_cluster_randomization',
    'check_sample_ratio_mismatch',
    'test_normality',
    'test_equal_variance',
    'determine_test_family',
    'compute_baseline_from_data',
    'calculate_power_sample_size',
    'print_power_summary',
    'run_outcome_similarity_test',
    'run_aa_testing_generalized',
    'visualize_aa_distribution',
    'simulate_aa_type1_error_rate',
    'plot_p_value_distribution',
    'run_ab_test',
    'summarize_ab_test_result',
    'plot_ab_test_results',
    'plot_confidence_intervals',
    'compute_lift_confidence_interval',
    'print_final_ab_test_summary',
    'estimate_test_duration',
    'apply_cuped',
    'visualize_segment_lift',
    'analyze_segment_lift',
    'run_guardrail_analysis',
    'evaluate_guardrail_metric',
    'simulate_rollout_impact',
]
