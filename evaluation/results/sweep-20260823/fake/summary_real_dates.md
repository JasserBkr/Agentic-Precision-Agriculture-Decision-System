# fake — real_dates

## Layer 2 — Decision quality (real_dates only)

| | |
|---|---|
| n_scored | `3` |
| n_undecided_ref | `0` |
| agreement_3class | `1.0` |
| agreement_2class | `1.0` |
| confusion_matrix | `{"irrigate_now": {"irrigate_now": 1}, "irrigate_soon": {"irrigate_soon": 2}}` |
| boundary_cases | `{"n": 0, "agree": 0, "rate": null}` |
| confidence_mean_agree | `0.8` |
| confidence_mean_disagree | `null` |
| calibration_buckets | `{"(0.75-1]": {"n_agree": 3, "n_disagree": 0}}` |
| per_stratum | `{"seasonal_anchor": {"n": 2, "agreement": 1.0}, "seasonal_anchor+gap": {"n": 1, "agreement": 1.0}}` |
| fertilization_caveat_rate | `1.0` |
| fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY | `null` |

## Layer 3 — Agent quality

| | |
|---|---|
| n_runs | `3` |
| schema_structured_output_failure_rate | `0.0` |
| grounding_first_attempt_proxy_rate | `1.0` |
| retry_rate | `0.0` |
| retry_recovery_rate | `null` |
| unsupported_claims_after_retry_rate | `null` |
| signal_conflict_detected_rate | `0.0` |
| conflict_rule_breakdown_BY_RULE | `{}` |
| contributing_signal_coverage_distinct_names | `{"mean": 4.0}` |
| synthetic_expected_checks | `null` |
| generic_default_scenarios_all_pass | `null` |

## Layer 4 — Operational quality

| | |
|---|---|
| latency_chronos_s | `{"mean": 2.33, "median": 0.985, "p95": 5.06, "max": 5.06}` |
| latency_graph_llm_s | `{"mean": 0.0, "median": 0.002, "p95": 0.006, "max": 0.006}` |
| latency_prep_total_s | `{"mean": 2.35, "median": 1.001, "p95": 5.079, "max": 5.079}` |
| llm_calls_per_run | `{"mean": 0.0, "max": 0, "theoretical_ceiling": 3}` |
| tokens_input_total | `0` |
| tokens_output_total | `0` |
| tokens_per_run_mean_in_out | `[0.0, 0.0]` |
| failure_rate_by_type | `{}` |
| soil_moisture_insufficient_data_rate_REAL_DATES | `0.0` |
