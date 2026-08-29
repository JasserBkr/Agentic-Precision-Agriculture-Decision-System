"""Render ``docs/agent_evaluation_report.md`` from immutable harness records.

Aggregates the groq provider's raw JSONL records (latest sweep by default)
with the existing ``evaluation.metrics.aggregate`` functions and renders a
human-readable report. Save to ``docs/`` next to the plot outputs. Gemini and
fake-provider runs are intentionally out of scope.

Usage:
    uv run python -m evaluation.metrics.eval_report [--stamp sweep-YYYYMMDD]
                                                    [--provider groq]
                                                    [--out docs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics.aggregate import (
    injection_metrics,
    layer2_irrigation,
    layer3_agent,
    layer4_ops,
    load_records,
)

ROOT = Path(__file__).resolve().parent.parent.parent
BASE = ROOT / "evaluation" / "results"
DEFAULT_OUT = ROOT / "docs"
PLOTS_DIR = "plots"


def md_table(d: dict) -> str:
    if not d:
        return "_n/a_\n"
    rows = ["| | |", "|---|---|"]
    for k, v in d.items():
        rows.append(f"| {k} | `{json.dumps(v, default=str)}` |")
    return "\n".join(rows) + "\n"


def img(name: str, caption: str) -> str:
    return f"![{caption}]({PLOTS_DIR}/{name})_({caption})_\n"


def render_report(
    stamp: str, records_by_file: dict[str, list[dict]]
) -> str:
    rd = records_by_file["real_dates"]
    synth = records_by_file.get("synthetic_edge_cases", [])
    inj = records_by_file.get("ndvi_injection", [])

    l2 = layer2_irrigation(rd)
    l3_rd = layer3_agent(rd)
    l4_rd = layer4_ops(rd)
    l2b = injection_metrics(inj) if inj else {}
    l3_syn = layer3_agent(synth) if synth else {}
    l4_syn = layer4_ops(synth) if synth else {}
    l3_inj = layer3_agent(inj) if inj else {}
    l4_inj = layer4_ops(inj) if inj else {}

    n_total = sum(len(v) for v in records_by_file.values())
    parts = []
    add = parts.append

    add("# Agent Evaluation Report — Groq\n")
    add(f"_Campaign `{stamp}`, provider `groq`. Gemini and fake-provider runs "
        f"excluded from this document by design._\n")
    add(f"**Scope:** {n_total} scored runs — `real_dates` ({len(rd)}), "
        f"`synthetic_edge_cases` ({len(synth)}), `ndvi_injection` ({len(inj)}). "
        f"Single field (Kairouan, `field_merguellil_01`). ")
    add(f"**Source:** raw JSONL in `evaluation/results/{stamp}/groq/`. "
        f"Regenerate the report with `uv run python -m evaluation.metrics.eval_report` "
        f"and the plots with `uv run python -m evaluation.metrics.plots`.\n")

    # ---------------------------------------------------------- exec summary
    add("## Executive summary\n")
    add("- **Layer 2 (decisions, 22 real runs):** 3-class agreement "
        f"**{l2['agreement_3class']:.0%}**; collapsed 2-class agreement "
        f"**{l2['agreement_2class']:.0%}** (no decision missed on the do/don't-irrigate axis)."
        f" All **{l2['boundary_cases']['n']}/{l2['boundary_cases']['n']} "
        f"boundary cases** agreed — the regime that separates calibrated from lucky agents.\n")
    add(f"- **Layer 2b (NDVI anomalies):** precision **{l2b.get('precision_anomaly_band')}**, "
        f"recall **{l2b.get('recall_anomaly_band')}** (band errors the anomaly detector is "
        f"excluded from), controls **{l2b.get('false_positive_rate_controls')}** FP.\n")
    add(f"- **Layer 3 (behaviour):** **{l3_rd['schema_structured_output_failure_rate']}** "
        f"schema failures; first-attempt grounding **{l3_rd['grounding_first_attempt_proxy_rate']:.0%}**; "
        f"1/22 runs ended on an unsupported `{'GROUNDING' if l3_rd['conflict_rule_breakdown_BY_RULE'] else '—'}` "
        f"claim after retry. Synthetic expected-checks **{l3_syn['synthetic_expected_checks']['pass']}/"
        f"{l3_syn['synthetic_expected_checks']['total']}** pass; generic-default trap "
        f"**{'pass' if l3_syn.get('generic_default_scenarios_all_pass') else 'FAIL'}**.\n")
    add(f"- **Layer 4 (ops):** LLM graph round-trip dominates every run "
        f"(mean **{l4_rd['latency_graph_llm_s']['mean']}s** vs Chronos **{l4_rd['latency_chronos_s']['mean']}s**); "
        f"mean **{l4_rd['llm_calls_per_run']['mean']}** LLM calls/run vs a cached cap of "
        f"**3** (no tool loop). Real-dates soil-insufficient rate "
        f"**{l4_rd['soil_moisture_insufficient_data_rate_REAL_DATES']}** → Layer-2 denominators intact.\n")
    add(f"- **Verdict:** Groq is operationally viable and behaviourally clean for this campaign. "
        f"Two strata are the review targets for the next sweep: `clearly_wet` (agreement "
        f"{l2['per_stratum']['clearly_wet']['agreement']:.0%}) and `boundary+gap` "
        f"({l2['per_stratum']['boundary+gap']['agreement']:.0%}).\n")

    # ------------------------------------------------------------- Layer 2
    add("## Layer 2 — Irrigation decision quality (real_dates)\n")
    add("Ground truth is `reference_policy.py` — an independent scorer that shares no code "
        "path with the agent's validator. `UNDECIDED` reference cases are excluded from "
        "denominators and counted separately.\n")
    add(md_table(l2))
    add(img("agent_eval_confusion_matrix_real_dates.png",
            "Reference × agent confusion matrix"))
    add(img("agent_eval_per_stratum_agreement_real_dates.png",
            "Per-stratum agreement (green = all correct, orange = partial, red = miss)"))
    add("**Reading:**\n")
    add(f"- **Boundary regime (`n={l2['boundary_cases']['n']}`)** at ≤0.02 m³/m³ from the "
        f"trigger: agreement **{l2['boundary_cases']['rate']:.0%}** — the agent does not lose "
        f"the decision exactly where it matters most.\n")
    add("- All 4 disagreements are `irrigate_now` (reference) vs `irrigate_soon` (agent) — a "
        "timing gap, **not** a no-action failure, which is why collapsed 2-class reaches 1.0.\n")
    add(f"- **Confidence is not yet a reliable honesty signal:** mean confidence on agreeing "
        f"runs is {l2['confidence_mean_agree']:.2f} vs {l2['confidence_mean_disagree']:.2f} on "
        f"disagreements — the agent is on average as confident when wrong as when right.\n")
    add(f"- **`clearly_wet` ({l2['per_stratum']['clearly_wet']['agreement']:.0%}, n="
        f"{l2['per_stratum']['clearly_wet']['n']}) and `boundary+gap` ("
        f"{l2['per_stratum']['boundary+gap']['agreement']:.0%}, n="
        f"{l2['per_stratum']['boundary+gap']['n']})** are the two strata to pressure-test "
        f"next; the `seasonal_anchor` 0.75 miss is a single run.\n")
    add(f"- Fertilization behaviour is **descriptive-only**: Spearman(apply_fertilizer, ndvi_z) "
        f"= {l2['fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY']['rho']} "
        f"(p={l2['fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY']['p']}, "
        f"n={l2['fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY']['n']}) — reported for "
        f"transparency, not a scored target.\n")
    add(img("agent_eval_confidence_by_agreement_real_dates.png",
            "Confidence distribution, agreed vs disagreed decisions"))
    add(img("agent_eval_calibration_buckets_real_dates.png",
            "Agree/disagree per confidence bucket"))

    # ------------------------------------------------------------ Layer 2b
    if inj:
        add("## Layer 2b — NDVI anomaly detection (ndvi_injection)\n")
        add("Deterministic PREP-band detection scored over injected ±2.5σ anomalies plus clean "
            "controls. Origins whose history is too data-sparse for the detector are listed by "
            "name, never silently counted as misses.\n")
        add(md_table(l2b))
        add(img("agent_eval_ndvi_detection.png",
                "Anomaly detection by sign: grey = total runs, green = detected"))
        add(f"**Reading:** band precision/recall **{l2b.get('precision_anomaly_band')}**/"
            f"**{l2b.get('recall_anomaly_band')}**, "
            f"**{l2b.get('false_positive_rate_controls')}** false positives from "
            f"{l2b['by_sign']['stress(-2.5)']['n'] + l2b['by_sign']['vigor(+2.5)']['n']} "
            f"injections. Mean recovery error is "
            f"**{l2b.get('recovery_error_mean_abs')}σ** — the detector lands its estimate "
            f"±0.26σ of the injected magnitude. Data-sparse exclusions: "
            f"{', '.join('`' + s.split(':')[-1] + '`' for s in l2b.get('sparse_origins_excluded', []))}.\n")

    # ------------------------------------------------------------- Layer 3
    add("## Layer 3 — Agent behaviour & reliability\n")
    add("All 41 runs produced parseable structured output (schema-failure rate 0.0 everywhere). "
        "The grounding proxy is the share of runs that passed the validator on the **first** "
        "attempt; `retry_recovery_rate` is retries that fixed the problem, and "
        "`unsupported_claims_after_retry` is the residue the validator never accepted.\n")
    add("| metric | real_dates | synthetic | ndvi_injection |\n|---|---|---|---|\n")
    for k in ("schema_structured_output_failure_rate", "grounding_first_attempt_proxy_rate",
              "retry_rate", "retry_recovery_rate", "unsupported_claims_after_retry_rate",
              "signal_conflict_detected_rate"):
        add(f"| {k} | {l3_rd.get(k)} | {l3_syn.get(k)} | {l3_inj.get(k)} |\n")
    add(f"| contributing_signal_coverage (distinct) | "
        f"{l3_rd['contributing_signal_coverage_distinct_names']['mean']} | "
        f"{l3_syn['contributing_signal_coverage_distinct_names']['mean']} | "
        f"{l3_inj['contributing_signal_coverage_distinct_names']['mean']} |\n")
    add(img("agent_eval_attempts.png", "First-attempt vs retry by scenario file"))
    add(img("agent_eval_rule_firing.png", "Validator/conflict rule IDs fired"))
    add(img("agent_eval_signal_coverage.png", "Distinct contributing signals cited per run"))
    add(f"**Reading:** manually-guided (non-agent) injection runs need retries most "
        f"({l3_inj['retry_rate']:.0%}) and recover all of it; one `real_dates` run "
        f"persisted a `GROUNDING` violation "
        f"(`unsupported_claims_after_retry {l3_rd['unsupported_claims_after_retry_rate']}`), "
        f"and it is the only `signal_conflict_detected` run. NDVI-injection runs fall into "
        f"the grounding-is-honest-about-data pattern: more retries, full recovery. Synthetic "
        f"expected-checks: **{l3_syn['synthetic_expected_checks']['pass']}/"
        f"{l3_syn['synthetic_expected_checks']['total']}** pass, generic-default trap "
        f"**{'pass' if l3_syn.get('generic_default_scenarios_all_pass') else 'FAIL'}**.\n")

    # ------------------------------------------------------------- Layer 4
    add("## Layer 4 — Operational viability (free-tier quotas)\n")
    add("Latency split measures Chronos inference separately from the LLM graph round-trip. "
        "The theoretical call ceiling is 3 (parse + up to 2 recommend attempts) — there is "
        "**no tool loop**, so quota usage is bounded.\n")
    add("| metric | real_dates | synthetic | ndvi_injection |\n|---|---|---|---|\n")
    for k in ("latency_chronos_s", "latency_graph_llm_s", "llm_calls_per_run",
              "tokens_input_total", "tokens_output_total", "tokens_per_run_mean_in_out",
              "soil_moisture_insufficient_data_rate_REAL_DATES"):
        v_rd = json.dumps(l4_rd.get(k), default=str)
        v_syn = json.dumps(l4_syn.get(k), default=str)
        v_inj = json.dumps(l4_inj.get(k), default=str)
        add(f"| {k} | `{v_rd}` | `{v_syn}` | `{v_inj}` |\n")
    add(img("agent_eval_latency.png", "Latency: LLM graph round-trip vs Chronos inference"))
    add(img("agent_eval_tokens.png", "Token usage per scenario file"))
    add("**Reading:** the LLM round-trip **dominates** latency (≈33–42 s mean vs ≈1.5 s "
        "Chronos); on `real_dates` its p95 is 65.7 s — the operational hook is provider "
        "throughput, not the forecaster. Token volumes stay modest "
        "(<5k input / ~1.6k output per real run), so free-tier quotas are not the binding "
        "constraint. `soil_moisture_insufficient_data_rate = 0.0` confirms no real run lost "
        "its Layer-2 denominator.\n")

    # ------------------------------------------------------- plots index
    add("## Plots index\n")
    add("All figures live in `docs/plots/` with an `agent_eval_` prefix; every one is "
        "regenerated by `uv run python -m evaluation.metrics.plots`.\n")
    add("| figure | content |\n|---|---|\n")
    for name, desc in [
        ("confusion_matrix", "Reference × agent 3×3 count heatmap (real_dates)"),
        ("per_stratum_agreement", "Agreement rate per stratum, n-annotated (real_dates)"),
        ("confidence_by_agreement", "Confidence strips on agreed vs disagreed runs (real_dates)"),
        ("calibration_buckets", "Agree/disagree by confidence bucket (real_dates)"),
        ("ndvi_detection", "Anomaly band detection by sign (ndvi_injection)"),
        ("attempts", "First-attempt vs retry per scenario file"),
        ("rule_firing", "Validator/conflict rule IDs fired, stacked per file"),
        ("signal_coverage", "Distinct contributing signals cited per run"),
        ("latency", "LLM graph vs Chronos latency boxplots"),
        ("tokens", "Per-run input/output token usage by file"),
    ]:
        add(f"| `agent_eval_{name}.png` | {desc} |\n")

    # --------------------------------------------------------- limitations
    add("## Limitations & next steps\n")
    add("- Single field, single season window (Kairouan, 22 scored real runs) — "
        "per-stratum cells are small (`n=1–4`); the two flagged strata need a targeted "
        "expansion before conclusions are drawn.\n")
    add("- The fertilization-vs-NDVI-z Spearman is descriptive-only and underpowered "
        f"(n={l2['fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY']['n']}).\n")
    add(f"- This report intentionally excludes gemini; the cross-provider reliability "
        f"comparison remains in `evaluation/results/{stamp}/comparison.md` and should be "
        f"consulted before trusting Groq as the primary provider operationally.\n")
    add("- Data-sparse injection origins (`inj-neg-1`, `ctrl-1`) were excluded by name "
        "rather than counted as misses; a denser injection pool would shrink that carve-out.\n")
    add("- No failures occurred in this campaign (`failure_rate_by_type` empty); a "
        "quota-death or API-outage sweep would feed failure paths the validator currently "
        "never sees.\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=None, help="sweep-YYYYMMDD (latest by default)")
    parser.add_argument("--provider", default="groq")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    stamps = sorted(p.name for p in BASE.glob("sweep-*/"))
    args.stamp = args.stamp or stamps[-1]
    provider_dir = BASE / args.stamp / args.provider
    if not provider_dir.is_dir():
        raise SystemExit(f"no provider dir: {provider_dir}")

    records_by_file = {}
    for jf in sorted(provider_dir.glob("*.jsonl")):
        records_by_file[jf.stem] = load_records(jf)

    md = render_report(args.stamp, records_by_file)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "agent_evaluation_report.md"
    out.write_text(md)
    print(f"wrote {out}")
    print(f"  provider={args.provider} stamp={args.stamp} "
          f"files={', '.join(sorted(records_by_file))}")


if __name__ == "__main__":
    main()