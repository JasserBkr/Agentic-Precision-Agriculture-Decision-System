# groq — synthetic_edge_cases

## Layer 2 — Decision quality (real_dates only)

_skipped (not real_dates)_

## Layer 3 — Agent quality

| | |
|---|---|
| n_runs | `10` |
| schema_structured_output_failure_rate | `0.0` |
| grounding_first_attempt_proxy_rate | `0.7` |
| retry_rate | `0.3` |
| retry_recovery_rate | `1.0` |
| unsupported_claims_after_retry_rate | `0.0` |
| signal_conflict_detected_rate | `0.0` |
| conflict_rule_breakdown_BY_RULE | `{}` |
| contributing_signal_coverage_distinct_names | `{"mean": 6.2}` |
| synthetic_expected_checks | `{"pass": 5, "total": 5}` |
| generic_default_scenarios_all_pass | `true` |

## Layer 4 — Operational quality

| | |
|---|---|
| latency_chronos_s | `null` |
| latency_graph_llm_s | `{"mean": 28.07, "median": 27.791, "p95": 58.408, "max": 58.408}` |
| latency_prep_total_s | `null` |
| llm_calls_per_run | `{"mean": 1.3, "max": 2, "theoretical_ceiling": 3}` |
| tokens_input_total | `30607` |
| tokens_output_total | `13932` |
| tokens_per_run_mean_in_out | `[3060.7, 1393.2]` |
| failure_rate_by_type | `{}` |
| soil_moisture_insufficient_data_rate_REAL_DATES | `0.0` |
