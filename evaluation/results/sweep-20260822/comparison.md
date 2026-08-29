# Provider comparison (Layer 3 focus)

| metric | fake | groq |
|---|---|---|
| schema_structured_output_failure_rate | 0.0 | 0.0 |
| grounding_first_attempt_proxy_rate | 0.5 | 0.7 |
| retry_rate | 0.5 | 0.3 |
| retry_recovery_rate | 0.0 | 1.0 |
| unsupported_claims_after_retry_rate | 1.0 | 0.0 |
| signal_conflict_detected_rate | 0.5 | 0.0 |

## Rule breakdown by provider/file

- **fake/real_dates**: `{}`
- **fake/synthetic_edge_cases**: `{'GROUNDING': 2, 'R4_CONFIDENCE_CEILING': 2, 'R2_STRESS_NOACTION': 1, 'R3_FERTILIZE_THRIVING': 1, 'R1_RAIN_OFFSET': 1}`
- **groq/real_dates**: `{'GROUNDING': 1}`
- **groq/synthetic_edge_cases**: `{}`
