# fake — synthetic_edge_cases

## Layer 2 — Decision quality (real_dates only)

_skipped (not real_dates)_

## Layer 3 — Agent quality

| | |
|---|---|
| n_runs | `10` |
| schema_structured_output_failure_rate | `0.0` |
| grounding_first_attempt_proxy_rate | `0.5` |
| retry_rate | `0.5` |
| retry_recovery_rate | `0.0` |
| unsupported_claims_after_retry_rate | `1.0` |
| signal_conflict_detected_rate | `0.5` |
| conflict_rule_breakdown_BY_RULE | `{"GROUNDING": 2, "R4_CONFIDENCE_CEILING": 2, "R2_STRESS_NOACTION": 1, "R3_FERTILIZE_THRIVING": 1, "R1_RAIN_OFFSET": 1}` |
| contributing_signal_coverage_distinct_names | `{"mean": 4.0}` |
| synthetic_expected_checks | `{"pass": 12, "total": 12}` |
| generic_default_scenarios_all_pass | `true` |

## Layer 4 — Operational quality

| | |
|---|---|
| latency_chronos_s | `null` |
| latency_graph_llm_s | `{"mean": 0.0, "median": 0.002, "p95": 0.006, "max": 0.006}` |
| latency_prep_total_s | `null` |
| llm_calls_per_run | `{"mean": 0.0, "max": 0, "theoretical_ceiling": 3}` |
| tokens_input_total | `0` |
| tokens_output_total | `0` |
| tokens_per_run_mean_in_out | `[0.0, 0.0]` |
| failure_rate_by_type | `{}` |
| soil_moisture_insufficient_data_rate_REAL_DATES | `0.0` |
