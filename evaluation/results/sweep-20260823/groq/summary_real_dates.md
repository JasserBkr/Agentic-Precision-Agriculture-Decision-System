# groq — real_dates

## Layer 2 — Decision quality (real_dates only)

| | |
|---|---|
| n_scored | `22` |
| n_undecided_ref | `0` |
| agreement_3class | `0.818` |
| agreement_2class | `1.0` |
| confusion_matrix | `{"irrigate_now": {"irrigate_now": 9, "irrigate_soon": 4}, "irrigate_soon": {"irrigate_soon": 7}, "no_action_needed": {"no_action_needed": 2}}` |
| boundary_cases | `{"n": 4, "agree": 4, "rate": 1.0}` |
| confidence_mean_agree | `0.863` |
| confidence_mean_disagree | `0.88` |
| calibration_buckets | `{"(0.75-1]": {"n_agree": 17, "n_disagree": 4}, "(0.5-0.75]": {"n_agree": 1, "n_disagree": 0}}` |
| per_stratum | `{"boundary": {"n": 2, "agreement": 1.0}, "boundary+gap": {"n": 1, "agreement": 0.0}, "clearly_dry": {"n": 2, "agreement": 1.0}, "clearly_dry+heatwave": {"n": 2, "agreement": 1.0}, "clearly_wet": {"n": 4, "agreement": 0.5}, "gap_adjacent": {"n": 1, "agreement": 1.0}, "heatwave": {"n": 1, "agreement": 1.0}, "seasonal_anchor": {"n": 4, "agreement": 0.75}, "seasonal_anchor+dry": {"n": 1, "agreement": 1.0}, "seasonal_anchor+gap": {"n": 1, "agreement": 1.0}, "transition": {"n": 2, "agreement": 1.0}, "transition+boundary": {"n": 1, "agreement": 1.0}}` |
| fertilization_caveat_rate | `0.0` |
| fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY | `{"rho": -0.327, "p": 0.1376, "n": 22}` |

## Layer 3 — Agent quality

| | |
|---|---|
| n_runs | `22` |
| schema_structured_output_failure_rate | `0.0` |
| grounding_first_attempt_proxy_rate | `0.773` |
| retry_rate | `0.227` |
| retry_recovery_rate | `0.8` |
| unsupported_claims_after_retry_rate | `0.2` |
| signal_conflict_detected_rate | `0.045` |
| conflict_rule_breakdown_BY_RULE | `{"GROUNDING": 1}` |
| contributing_signal_coverage_distinct_names | `{"mean": 7.14}` |
| synthetic_expected_checks | `null` |
| generic_default_scenarios_all_pass | `null` |

## Layer 4 — Operational quality

| | |
|---|---|
| latency_chronos_s | `{"mean": 1.54, "median": 1.39, "p95": 3.36, "max": 3.439}` |
| latency_graph_llm_s | `{"mean": 32.62, "median": 28.616, "p95": 65.7, "max": 66.869}` |
| latency_prep_total_s | `{"mean": 1.56, "median": 1.408, "p95": 3.377, "max": 3.459}` |
| llm_calls_per_run | `{"mean": 1.23, "max": 2, "theoretical_ceiling": 3}` |
| tokens_input_total | `94443` |
| tokens_output_total | `34936` |
| tokens_per_run_mean_in_out | `[4292.9, 1588.0]` |
| failure_rate_by_type | `{}` |
| soil_moisture_insufficient_data_rate_REAL_DATES | `0.0` |
