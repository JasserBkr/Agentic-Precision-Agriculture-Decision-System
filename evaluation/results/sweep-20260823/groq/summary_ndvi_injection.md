# groq — ndvi_injection

## Layer 2 — Decision quality (real_dates only)

_skipped (not real_dates)_

## Layer 2b — NDVI injection detection

| | |
|---|---|
| precision_anomaly_band | `1.0` |
| recall_anomaly_band | `1.0` |
| false_positive_rate_controls | `0.0` |
| by_sign | `{"stress(-2.5)": {"n": 2, "detected": 2}, "vigor(+2.5)": {"n": 3, "detected": 3}}` |
| recovery_error_mean_abs | `0.257` |
| sparse_origins_excluded | `["groq:ndvi_injection:inj-neg-1", "groq:ndvi_injection:ctrl-1"]` |

## Layer 3 — Agent quality

| | |
|---|---|
| n_runs | `9` |
| schema_structured_output_failure_rate | `0.0` |
| grounding_first_attempt_proxy_rate | `0.444` |
| retry_rate | `0.556` |
| retry_recovery_rate | `1.0` |
| unsupported_claims_after_retry_rate | `0.0` |
| signal_conflict_detected_rate | `0.0` |
| conflict_rule_breakdown_BY_RULE | `{}` |
| contributing_signal_coverage_distinct_names | `{"mean": 6.67}` |
| synthetic_expected_checks | `null` |
| generic_default_scenarios_all_pass | `null` |

## Layer 4 — Operational quality

| | |
|---|---|
| latency_chronos_s | `{"mean": 1.03, "median": 1.193, "p95": 1.776, "max": 1.776}` |
| latency_graph_llm_s | `{"mean": 42.45, "median": 37.717, "p95": 73.098, "max": 73.098}` |
| latency_prep_total_s | `null` |
| llm_calls_per_run | `{"mean": 1.56, "max": 2, "theoretical_ceiling": 3}` |
| tokens_input_total | `45188` |
| tokens_output_total | `16285` |
| tokens_per_run_mean_in_out | `[5020.9, 1809.4]` |
| failure_rate_by_type | `{}` |
| soil_moisture_insufficient_data_rate_REAL_DATES | `0.0` |
