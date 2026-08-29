# gemini — real_dates

## Layer 2 — Decision quality (real_dates only)

| | |
|---|---|
| n_scored | `9` |
| n_undecided_ref | `0` |
| agreement_3class | `0.889` |
| agreement_2class | `0.889` |
| confusion_matrix | `{"irrigate_now": {"irrigate_now": 6}, "irrigate_soon": {"irrigate_soon": 2, "no_action_needed": 1}}` |
| boundary_cases | `{"n": 0, "agree": 0, "rate": null}` |
| confidence_mean_agree | `0.919` |
| confidence_mean_disagree | `0.9` |
| calibration_buckets | `{"(0.75-1]": {"n_agree": 8, "n_disagree": 1}}` |
| per_stratum | `{"clearly_dry": {"n": 2, "agreement": 1.0}, "clearly_dry+heatwave": {"n": 1, "agreement": 1.0}, "seasonal_anchor": {"n": 4, "agreement": 0.75}, "seasonal_anchor+dry": {"n": 1, "agreement": 1.0}, "seasonal_anchor+gap": {"n": 1, "agreement": 1.0}}` |
| fertilization_caveat_rate | `1.0` |
| fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY | `null` |

## Layer 3 — Agent quality

| | |
|---|---|
| n_runs | `10` |
| schema_structured_output_failure_rate | `0.1` |
| grounding_first_attempt_proxy_rate | `0.889` |
| retry_rate | `0.111` |
| retry_recovery_rate | `1.0` |
| unsupported_claims_after_retry_rate | `0.0` |
| signal_conflict_detected_rate | `0.0` |
| conflict_rule_breakdown_BY_RULE | `{}` |
| contributing_signal_coverage_distinct_names | `{"mean": 6.33}` |
| synthetic_expected_checks | `null` |
| generic_default_scenarios_all_pass | `null` |

## Layer 4 — Operational quality

| | |
|---|---|
| latency_chronos_s | `{"mean": 1.56, "median": 1.365, "p95": 3.263, "max": 3.263}` |
| latency_graph_llm_s | `{"mean": 30.17, "median": 27.482, "p95": 59.325, "max": 59.325}` |
| latency_prep_total_s | `{"mean": 1.58, "median": 1.385, "p95": 3.284, "max": 3.284}` |
| llm_calls_per_run | `{"mean": 1.11, "max": 2, "theoretical_ceiling": 3}` |
| tokens_input_total | `0` |
| tokens_output_total | `0` |
| tokens_per_run_mean_in_out | `[0.0, 0.0]` |
| failure_rate_by_type | `{"ChatGoogleGenerativeAIError": 1}` |
| soil_moisture_insufficient_data_rate_REAL_DATES | `0.0` |

## Failures

- `gemini:real_dates:rd-010`: **ChatGoogleGenerativeAIError** — Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. Fo
