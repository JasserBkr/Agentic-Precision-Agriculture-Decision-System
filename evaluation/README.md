# Evaluation Framework (Step 3) — Complete Reference

Evaluates the **agent** (LLM reasoning + validation), NOT the forecaster.
Layer 1 (Chronos-2 quality) is already closed: Chronos-2 beat TFT/N-HiTS on
a 26-fold rolling backtest with paired Wilcoxon significance. This framework
answers three remaining questions:

1. Does the agent make the RIGHT irrigation decision? (Layer 2)
2. Is the agent's BEHAVIOR sound — grounded, honest about uncertainty,
   conflict-aware? (Layer 3)
3. Is the system OPERATIONALLY viable on free-tier quotas? (Layer 4)

## The pipeline of an evaluation campaign

```
 configs/thresholds.yaml ─┐
 data/processed/fused_2years.parquet ─┤
 evaluation/scenarios/*.jsonl ────────┤→ harness.py ──→ results/<stamp>/<provider>/<file>.jsonl
 reference_policy.py (Layer 2 truth) ─┘   (1 JSONL record     │
 ndvi_injection.py (Layer 2b truth) ────┘   per scenario run)  └→ metrics/report.py
                                                                  ├→ <dir>/summary_<file>.md
                                                                  └─ <dir>/comparison.md
```

Raw records are append-only and immutable. Aggregations never mutate them.
Scenario files' results are NEVER merged with each other or across providers;
the only place providers meet is `comparison.md`.

---

## File-by-file reference

### scenarios/real_dates.jsonl (input, hand-curated)
22 historical origin dates spanning 7 condition strata (seasonal anchors,
clearly dry/wet, transition, boundary-to-trigger, gap-adjacent, heatwave).
Each line: `run_id, date, stratum, selection_reason, query`. Selected by
reasoned manual inspection (see `inspect_dataset.py`) — no full-dataset
pre-sweep exists because real LLM runs cost quota; stratification preserves
condition coverage while keeping the sweep at ~22 runs. Selection used only
INPUT covariates (weather/moisture/NDVI), never agent outputs → no leakage
into scores.

### scenarios/synthetic_edge_cases.jsonl (input, generated)
10 hand-built SignalBundles that bypass PREP entirely, stress-testing paths
real dates rarely hit:
| case | tests |
|---|---|
| syn-001 sensor dropout | insufficient_data surfacing + R4 confidence ceiling |
| syn-002 stale satellite | 45-day-old scene surfaced honestly (no validator rule by design) |
| syn-003 heatwave | reasoning under extreme tmax |
| syn-004 stressed-veg trap | z=-2.5 but comfy moisture → R2 fires IF agent says no_action |
| syn-005 thriving-fertilize trap | z=+2.6 tempts fertilization → R3 fires IF applied |
| syn-006 heavy-rain trap | 35mm forecast → R1 fires IF irrigate_now anyway |
| syn-007 unknown crop | generic_default_used substitution flag |
| syn-008 stage misname | 'flower-ing' normalization |
| syn-009 zero forward window | double insufficient + R4 |
| syn-010 clean baseline | sanity; expect no conflicts |

Each carries `expected.deterministic` (must always hold), `expected.conditional`
(IF antecedent THEN rule), `expected.descriptive_tags`. Regenerate:
`uv run python -m evaluation.scenarios.synthetic_cases`

### inspect_dataset.py — selection evidence (read-only)
- `_rolling_stats(df)`: adds 7d precip/ET0 sums, 14d moisture mean+slope,
  monthly tmax percentiles — the columns curation eyes.
- `print_monthly(df)`: seasonality table (Kairouan wet Nov–Mar / dry Jun–Aug).
- `print_candidates(df)`: per-stratum candidate listings incl. boundary days
  near trigger = FC − MAD·(FC−WP) = 0.201 m³/m³.
- `main()`: prints everything. Run: `uv run python -m evaluation.inspect_dataset`

### reference_policy.py — independent Layer-2 scorer
Zero imports from `agri_agent.agent.validator` (no shared code path);
constants coincide BY SPEC only.
- `_sum_forecast(weather_sub, key)`: NaN-safe forecast-row sum (rain total).
- `_p50_min(soil_sub)`: driest median day of the horizon.
- `_horizon_crossing_days(p50_first, p50_last, trig, horizon)`: linear
  extrapolation — does falling moisture reach the trigger within horizon?
- `reference_irrigation_decision(weather, soil, thresholds)`: the policy.
  First match wins: missing input → `UNDECIDED` (excluded from denominators,
  counted separately); below trigger → now/soon by ≥5mm rain offset;
  trending-into-trigger → soon; below target range → soon; else no_action.
  Returns action + all intermediate quantities (`rule_fired`,
  `distance_to_trigger`, …) so aggregation never re-derives anything.
- `collapsed_2class(action)`: {now, soon} → "irrigate" for the secondary metric.

### ndvi_injection.py — Layer 2b ground-truth factory
NDVI anomaly detection is deterministic PREP work (agent/anomaly.py), so
precision/recall need no LLM; injected cases additionally run through the
live agent to prove signal propagation.
- `doy_window_baseline(dates, values, target)`: ±15-day circular-window mean/
  std, mirroring anomaly.py semantics.
- `inject_ndvi_anomaly(history, target_date, k=-2.5)`: replaces NDVI ONLY at
  row D with `mean + k·std`; raises if D lacks an exact observation (anomaly.py
  requirement). k=±2.5 gives 25% margin beyond the z≥2 band edge.
- `clean_control_truth(history, date)`: unmodified control record (false-
  positive denominator).
- `score_detection(measured_z, truth)`: detected? + recovery error |z_meas−k|.
  measured_z=None means the origin itself was data-sparse — reported
  separately, not silently counted as a miss.

### providers.py — provider plumbing + quota-free dry runs
- `provider_available(name)` / `require_provider(name)`: env-key checks.
- `UsageCapture`: LangChain callback capturing per-call token usage
  (`on_llm_start` counts calls, `on_llm_end` reads OpenAI-style
  `token_usage` OR Gemini/Groq-style `usage_metadata`); `summary()` returns
  call count + token totals feeding Layer 4.
- `build_fake_llm(tags, moisture_below_trigger, reference_action)`:
  scripted structured-output replier for `--provider fake` smoke sweeps —
  exercises graph/validator/retry plumbing at zero cost.

### harness.py — the orchestrator (produces raw records)
One immutable JSONL record per scenario: identity, timings, token usage,
bundle digest (origin, insufficient flags, staleness, ndvi_z, generic-default),
reference decision, final recommendation, attempts, rule IDs, failures.
- `RULE_RE`: extracts stable rule IDs from validator problem strings.
- `results_path` / `completed_run_ids`: where records live + resume skip-list
  (rerunning a dead sweep skips finished run_ids — mid-quota-death is cheap).
- `ChronosTimer`: context manager monkeypatching the bundle module's
  `forecast_soil_moisture` to time Chronos inference separately from LLM
  round-trips (Layer 4's latency split).
- `resolve_query_params(query, target_date, llm)`: mirrors run_pipeline's
  two-pass offline temporal resolution (no leakage between relative dates
  and backtest origin).
- `rehydrate_bundle(d)`: synthetic dict → frozen SignalBundle.
- `invoke_graph(bundle, llm, usage, query, reference_action)`: timed graph
  invocation with usage callbacks; builds a fake LLM when llm=None.
- `bundle_digest(bundle)` / `base_record` / `finalize_record`: record
  assembly incl. insufficient_flags transfer.
- `run_real(scenario, ...)`: FULL pipeline — parse_query → offline PREP
  (real parquet, real Chronos-2) → reference decision computed from the SAME
  bundle the agent saw → graph invoke.
- `check_expected(case, bundle, final)`: evaluates synthetic expectations —
  deterministic checks outright, conditional rules only when their antecedent
  holds (e.g. R4 checked iff confidence > 0.5).
- `run_synthetic(case, ...)`: PREP bypassed; fake-provider aware.
- `injection_pool(df)`: deterministic evenly-spaced injection/control dates
  over observed NDVI days (3 negative, 3 positive, 3 controls).
- `run_injection(case, ...)`: swaps fused history via `_load_offline_fused`
  patch, builds the bundle, scores detection at PREP level, then runs the agent.
- `main()`: CLI (`--provider --file --limit --stamp`). Stamp defaults to
  day-stable `sweep-YYYYMMDD` so both providers land in one folder for comparison.

### metrics/aggregate.py — pure functions over raw records
- `BOUNDARY_MARGIN = 0.02` m³/m³ (boundary-case definition).
- `load_records(path)`, `_ok(rec)` (failure filter), `_conf_bucket(c)`.
- `layer2_irrigation(records)`: 3-class agreement, collapsed 2-class, 3×3
  confusion matrix, boundary-subset agreement, confidence calibration buckets
  × agree/disagree, per-stratum agreement, UNDECIDED count, fertilization
  caveat presence rate, Spearman(apply_fertilizer, ndvi_z) explicitly labeled
  DESCRIPTIVE_ONLY.
- `injection_metrics(records)`: precision/recall of the anomaly band, control
  false-positive rate, per-sign breakdown, mean |recovery error|,
  sparse-origin exclusions listed by name.
- `layer3_agent(records)`: schema-failure rate, first-attempt grounding proxy,
  retry rate, recovery rate, unsupported-after-retry rate, conflict-flag rate,
  conflict-rule breakdown BY INDIVIDUAL RULE ID, contributing-signal coverage,
  synthetic expected-check pass counts.
- `layer4_ops(records)`: latency stats (mean/median/p95/max; Chronos split
  from LLM graph time), LLM calls per run vs theoretical ceiling of 3
  (parse + up to 2 recommends — there is NO tool loop), token totals/means,
  failure rate by type, and the real_dates soil-insufficient rate (the
  zero-forward-window detector that guards Layer 2's denominators).

### metrics/report.py — renders the deliverables
- `_md_table(d)`: dict → markdown table.
- `render_file_report(...)`: per (provider,file): Layer 2 (real_dates only),
  Layer 2b (injection file), Layers 3–4, plus a Failures section.
- `render_comparison(results_dir)`: THE confound report — Layer-3 metrics
  side-by-side across providers so Groq throughput gains are validated
  against reliability before being trusted; rule-breakdown per provider/file.
- `main([stamp])`: writes `summary_<file>.md` next to each JSONL +
  `comparison.md` at the stamp root.

---

## Performing the evaluation

```bash
# 0. optional: re-inspect selection evidence / regenerate synthetics
uv run python -m evaluation.inspect_dataset
uv run python -m evaluation.scenarios.synthetic_cases

# 1. ZERO-COST smoke (always safe): plumbing check
uv run python -m evaluation.harness --provider fake --file synthetic_edge_cases

# 2. Groq sweep (fast, ~130 calls total, fits one day)
uv run python -m evaluation.harness --provider groq --file real_dates
uv run python -m evaluation.harness --provider groq --file synthetic_edge_cases
uv run python -m evaluation.harness --provider groq --file ndvi_injection

# 3. Gemini sweep (20 req/day → trim with --limit; reruns RESUME automatically)
uv run python -m evaluation.harness --provider gemini --file real_dates --limit 10

# 4. Reports
uv run python -m evaluation.metrics.report                # latest stamp
```

Reading the outputs:
- `<stamp>/<provider>/summary_real_dates.md` → Layer 2 decision quality +
  Layer 3/4 aggregates.
- `<stamp>/<provider>/summary_synthetic_edge_cases.md` → stress-test behavior,
  expected-check passes, per-rule firing.
- `<stamp>/<provider>/summary_ndvi_injection.md` → anomaly precision/recall.
- `<stamp>/comparison.md` → Gemini-vs-Groq reliability verdict.

Interpretation anchors: agreement_3class is headline BUT read per_stratum and
boundary_cases alongside (boundary performance is what separates calibrated
from lucky agents). soil_moisture_insufficient_data_rate must be ~0 — any
nonzero rate silently shrinks Layer 2's denominators. On comparison.md: Groq
is only trusted as primary if its schema-failure rate stays within ~2 pts of
Gemini's and grounding doesn't collapse.
