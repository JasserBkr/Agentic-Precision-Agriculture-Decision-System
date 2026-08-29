"""Evaluation harness: run scenario files through the REAL agent and append
one immutable JSONL record per run. Resumable via run_id skip-lists.

Modes (per --file):
  real_dates             full pipeline: parse_query -> offline PREP -> graph
  synthetic_edge_cases   PREP bypassed (hand-built bundle) -> graph directly
  ndvi_injection         real parquet history + injected NDVI at chosen dates
                         (detection scored at PREP level; agent run optional
                         but performed for signal-propagation evidence)

Run from repo root:
  uv run python -m evaluation.harness --provider groq --file real_dates [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from agri_agent.agent.bundle import FUSED_PARQUET, build_signal_bundle, SignalBundle
from agri_agent.agent.graph import (
    _has_relative_date_expression,
    build_graph,
    get_llm,
    initial_state,
    parse_query,
    resolve_temporal_expressions,
)
from agri_agent.agent.schemas import QueryParams
from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.utils.logging_config import get_logger

from evaluation.providers import (
    UsageCapture,
    build_fake_llm,
    require_provider,
)
from evaluation.reference_policy import reference_irrigation_decision

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "evaluation" / "scenarios"
FIELD_CONFIG = ROOT / "configs" / "field.yaml"

RULE_RE = re.compile(
    r"\[(GROUNDING|R1_RAIN_OFFSET|R2_STRESS_NOACTION|"
    r"R3_FERTILIZE_THRIVING|R4_CONFIDENCE_CEILING)\]"
)

THEORETICAL_MAX_CALLS = 3  # parse(1) + recommend(up to 2); C1 correction


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def load_field_config() -> dict:
    import yaml
    with open(FIELD_CONFIG) as f:
        return yaml.safe_load(f)


def rule_ids(problems: list[str]) -> list[str]:
    out = []
    for p in problems or []:
        m = RULE_RE.search(p)
        if m:
            out.append(m.group(1))
        else:
            out.append("UNTAGGED")
    return sorted(set(out))


def results_path(stamp: str, provider: str, file_key: str) -> Path:
    d = Path("evaluation") / "results" / stamp / provider
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{file_key}.jsonl"


def completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        try:
            ids.add(json.loads(line)["run_id"])
        except Exception:  # noqa: BLE001 — tolerate a torn last line
            continue
    return ids


class ChronosTimer:
    """Times Chronos-2 inference inside bundle builds by wrapping the
    bundle module's imported symbol (bundle.py does `from ... import
    forecast_soil_moisture`, so patch THERE)."""

    def __init__(self) -> None:
        self.seconds: float | None = None

    def __enter__(self):
        import agri_agent.agent.bundle as B

        orig = B.forecast_soil_moisture
        timer = self

        def timed(*a, **k):
            t0 = time.perf_counter()
            out = orig(*a, **k)
            timer.seconds = (timer.seconds or 0.0) + (time.perf_counter() - t0)
            return out

        B.forecast_soil_moisture = timed
        self._restore = lambda: setattr(B, "forecast_soil_moisture", orig)
        return self

    def __exit__(self, *exc):
        self._restore()
        return False


def resolve_query_params(query: str, target_date, llm) -> QueryParams:
    """Mirrors scripts/run_pipeline.run_query's two-pass offline resolution."""
    qp = parse_query(query, llm=llm)
    fused = load_fused_dataset(str(FUSED_PARQUET))
    parquet_max = fused["date"].max().date()

    saved = qp.target_date
    if qp.target_date is None and _has_relative_date_expression(qp.raw_query):
        resolve_temporal_expressions(qp, parquet_max, mode="offline")

    reference = saved if saved is not None else parquet_max - pd.Timedelta(days=7).to_pytimedelta()
    resolve_temporal_expressions(qp, pd.Timestamp(reference).date(), mode="offline")
    if target_date is not None:
        qp.target_date = target_date
    return qp


def rehydrate_bundle(d: dict) -> SignalBundle:
    return SignalBundle(
        field_id=d["field_id"],
        origin_date=pd.Timestamp(d["origin_date"]),
        query_params=QueryParams(**d.get("_query_params", {})),
        vegetation=d["vegetation"],
        weather_forecast=d["weather_forecast"],
        soil_moisture_forecast=d["soil_moisture_forecast"],
        thresholds=d["thresholds"],
        load_errors=d.get("load_errors", {}),
    )


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------

def invoke_graph(bundle: SignalBundle, llm, usage: UsageCapture, query: str,
                 reference_action: str | None = None):
    if llm is None:
        # fake provider: deterministic structured replier for zero-cost smoke
        from evaluation.providers import build_fake_llm

        llm = build_fake_llm([], moisture_below_trigger=False,
                             reference_action=reference_action)
    graph = build_graph(bundle, llm=llm)
    state = initial_state(bundle, bundle.query_params, query=query)
    t0 = time.perf_counter()
    result = graph.invoke(state, config={"callbacks": [usage]})
    return result, time.perf_counter() - t0


def base_record(scenario_id: str, provider: str, file_key: str) -> dict:
    return {"run_id": f"{provider}:{file_key}:{scenario_id}", "scenario_id": scenario_id,
            "provider": provider, "file": file_key}


def finalize_record(rec: dict, result: dict, ref_decision: dict, usage: UsageCapture,
                    timings: dict) -> dict:
    final = result.get("final_output") or {}
    digest = rec.get("_bundle_digest") or {}
    rec.update({
        "final_output": final,
        "ndvi_z": digest.get("ndvi_z"),
        "recommend_attempts": result.get("recommend_attempts"),
        "validation_problems": final.get("validation_problems", []),
        "rule_ids": rule_ids(final.get("validation_problems", [])),
        "signal_conflict_detected": final.get("signal_conflict_detected"),
        "reference_decision": ref_decision,
        "usage": usage.summary(),
        "theoretical_max_calls": THEORETICAL_MAX_CALLS,
        **timings,
        "insufficient_flags": {
            "weather": bool(digest.get("weather_insufficient")),
            "soil": bool(digest.get("soil_insufficient")),
        },
    })
    rec.pop("_bundle_digest", None)
    return rec


def bundle_digest(bundle: SignalBundle) -> dict:
    return {
        "origin_date": str(bundle.origin_date.date()),
        "horizon_days": int(bundle.soil_moisture_forecast.get("horizon_days", 0) or 0),
        "weather_insufficient": bool(bundle.weather_forecast.get("insufficient_data")),
        "soil_insufficient": bool(bundle.soil_moisture_forecast.get("insufficient_data")),
        "days_since_last_scene": bundle.vegetation.get("days_since_last_scene"),
        "ndvi_z": (bundle.vegetation.get("ndvi_anomaly") or {}).get("z_score"),
        "generic_default_used": bool(bundle.thresholds.get("generic_default_used")),
        "load_errors": bundle.load_errors,
    }


def run_real(scenario: dict, provider: str, llm) -> dict:
    rec = base_record(scenario["run_id"], provider, "real_dates")
    rec["stratum"] = scenario.get("stratum")
    rec["selection_reason"] = scenario.get("selection_reason")
    usage = UsageCapture()
    timings = {}
    try:
        qp = resolve_query_params(scenario["query"], scenario["date"], llm)
        field = load_field_config()
        t0 = time.perf_counter()
        with ChronosTimer() as ct:
            bundle = build_signal_bundle(field, qp, mode="offline")
        timings = {
            "chronos_s": round(ct.seconds or 0.0, 3),
            "prep_total_s": round(time.perf_counter() - t0, 3),
        }
        rec["_bundle_digest"] = bundle_digest(bundle)
        ref = reference_irrigation_decision(
            bundle.weather_forecast, bundle.soil_moisture_forecast, bundle.thresholds
        )
        result, graph_s = invoke_graph(bundle, llm, usage, scenario["query"],
                                       reference_action=ref.get("action"))
        timings["graph_s"] = round(graph_s, 3)
        return finalize_record(rec, result, ref, usage, timings)
    except Exception as exc:  # noqa: BLE001 — record failures, keep sweeping
        rec["failure_type"] = type(exc).__name__
        rec["failure_detail"] = str(exc)[:500]
        return rec


def check_expected(case: dict, bundle: SignalBundle, final: dict) -> dict:
    exp = case.get("expected", {}).get("deterministic", {})
    checks = {}

    subs = exp.get("insufficient_sub_bundles")
    if subs is not None:
        actual = [k for k, s in (("weather_forecast", bundle.weather_forecast),
                                 ("soil_moisture_forecast", bundle.soil_moisture_forecast),
                                 ("vegetation", bundle.vegetation))
                  if s.get("insufficient_data")]
        checks["insufficient_sub_bundles"] = sorted(actual) == sorted(subs)

    if "generic_default_used" in exp:
        checks["generic_default_used"] = (
            bool(bundle.thresholds.get("generic_default_used")) == exp["generic_default_used"]
        )
    if "surfaced_days_since_last_scene" in exp:
        checks["days_since_last_scene_surfaced"] = (
            bundle.vegetation.get("days_since_last_scene")
            == exp["surfaced_days_since_last_scene"]
        )

    # Conditional rules: IF antecedent THEN rule must appear.
    cond = case.get("expected", {}).get("conditional", {})
    rules_present = set(rule_ids(final.get("validation_problems", [])))
    irr_action = (final.get("irrigation") or {}).get("action")
    fert_action = (final.get("fertilization") or {}).get("action")

    for act, need in (cond.get("if_action_then_rules") or {}).items():
        fired_for = []
        if irr_action == act:
            fired_for.append("irrigation")
        if fert_action == act:
            fired_for.append("fertilization")
        for branch in fired_for:
            checks[f"{branch}_=={act}=>rules"] = all(r in rules_present for r in need)

    if cond.get("if_confident_then_rules"):
        needed = cond["if_confident_then_rules"]
        for label in ("irrigation", "fertilization"):
            conf = (final.get(label) or {}).get("confidence")
            if conf is not None and conf > 0.5:
                checks[f"{label}_confident=>{needed[0]}"] = all(r in rules_present for r in needed)

    return {"checks": checks, "all_deterministic_pass": all(checks.values()) if checks else None}


def run_synthetic(case: dict, provider: str, llm) -> dict:
    rec = base_record(case["run_id"], provider, "synthetic_edge_cases")
    rec["name"] = case.get("name")
    rec["descriptive_tags"] = case.get("expected", {}).get("descriptive_tags", [])
    usage = UsageCapture()
    try:
        bundle = rehydrate_bundle({**case["bundle"],
                                   "_query_params": case.get("query_params", {})})
        rec["_bundle_digest"] = bundle_digest(bundle)
        ref = reference_irrigation_decision(
            bundle.weather_forecast, bundle.soil_moisture_forecast, bundle.thresholds
        )
        if provider == "fake":
            tags = case.get("expected", {}).get("descriptive_tags", [])
            below = (ref.get("inputs") or {}).get("m_min", 1.0) < (ref.get("inputs") or {}).get("trigger", 0.0)
            llm = build_fake_llm(tags, moisture_below_trigger=below)
        result, graph_s = invoke_graph(bundle, llm, usage, case.get("name", ""))
        final = result.get("final_output") or {}
        rec["expected_checks"] = check_expected(case, bundle, final)
        return finalize_record(rec, result, ref, usage, {"graph_s": round(graph_s, 3)})
    except Exception as exc:  # noqa: BLE001
        rec["failure_type"] = type(exc).__name__
        rec["failure_detail"] = str(exc)[:500]
        return rec


def injection_pool(df: pd.DataFrame, k_values=(-2.5, 2.5), n_per_sign=3, n_controls=3) -> list[dict]:
    """Choose injection dates from NDVI-observed days spanning both years,
    deterministically (evenly spaced over observed days)."""
    obs = df[df["NDVI"].notna()].reset_index(drop=True)
    obs = obs[obs["date"] <= df["date"].max() - pd.Timedelta(days=8)]
    idx = [round(i * (len(obs) - 1) / max(1, (n_per_sign * len(k_values)) - 1)) for i in range(n_per_sign * len(k_values))]
    dates = [obs.iloc[i]["date"].strftime("%Y-%m-%d") for i in idx]
    cases = [{"run_id": f"inj-{'neg' if k < 0 else 'pos'}-{j}", "date": d, "k": k}
             for k in k_values for j, d in enumerate(dates[len(dates) // 2 * (0 if k < 0 else 1):][:n_per_sign], 1)]
    ctrl_idx = [round(i * (len(obs) - 1) / max(1, n_controls - 1)) for i in range(n_controls)]
    cases += [{"run_id": f"ctrl-{j}", "date": obs.iloc[i]["date"].strftime("%Y-%m-%d"), "k": 0.0}
              for j, i in enumerate(ctrl_idx, 1)]
    return cases


def run_injection(case: dict, provider: str, llm, field: dict, df_real: pd.DataFrame) -> dict:
    from evaluation.ndvi_injection import inject_ndvi_anomaly, clean_control_truth, score_detection

    rec = base_record(case["run_id"], provider, "ndvi_injection")
    rec["injection_date"] = case["date"]
    rec["magnitude_z"] = case["k"]
    usage = UsageCapture()
    try:
        if case["k"] == 0.0:
            truth = clean_control_truth(df_real, case["date"])
            history = df_real.copy()
        else:
            history, truth = inject_ndvi_anomaly(df_real, case["date"], case["k"])

        import agri_agent.agent.bundle as B
        orig_loader = B._load_offline_fused
        B._load_offline_fused = lambda: history.copy()
        try:
            qp = QueryParams(target_date=None)
            qp.raw_query = ""
            qp.target_date = pd.Timestamp(truth["target_date"]).date()
            with ChronosTimer() as ct:
                bundle = build_signal_bundle(field, qp, mode="offline")
        finally:
            B._load_offline_fused = orig_loader

        measured_z = (bundle.vegetation.get("ndvi_anomaly") or {}).get("z_score")
        rec["detection"] = score_detection(measured_z, truth)

        ref = reference_irrigation_decision(
            bundle.weather_forecast, bundle.soil_moisture_forecast, bundle.thresholds
        )
        result, graph_s = invoke_graph(bundle, llm, usage, "Assess crop vigor and decide on fertilizer application.")
        rec = finalize_record(rec, result, ref, usage, {"graph_s": round(graph_s, 3),
                                                        "chronos_s": round(ct.seconds or 0.0, 3)})
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["failure_type"] = type(exc).__name__
        rec["failure_detail"] = str(exc)[:500]
        return rec


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluation sweep harness")
    ap.add_argument("--provider", required=True,
                    choices=["openai", "gemini", "groq", "fake"])
    ap.add_argument("--file", required=True,
                    choices=["real_dates", "synthetic_edge_cases", "ndvi_injection"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stamp", default="sweep-" + time.strftime("%Y%m%d"),
                    help="results subfolder; day-stable by default so all "
                         "providers/files of one evaluation campaign land "
                         "together for comparison")
    args = ap.parse_args(argv)

    import os
    os.environ["AGRI_LLM_PROVIDER"] = args.provider
    if args.provider != "fake":
        require_provider(args.provider)
        llm = get_llm()
        if llm is None:
            raise SystemExit(f"{args.provider} key present but get_llm() returned None")
    else:
        llm = None  # runners build deterministic fake structured LLMs

    out = results_path(args.stamp, args.provider, args.file)
    done = completed_run_ids(out)

    def was_done(rid: str) -> bool:
        # Scenario files store bare ids ("rd-001") while records store
        # prefixed ones ("groq:real_dates:rd-001"); accept either form.
        return rid in done or f"{args.provider}:{args.file}:{rid}" in done
    log.info("harness: provider=%s file=%s out=%s already_done=%d",
             args.provider, args.file, out, len(done))

    def write(rec: dict) -> None:
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        status = rec.get("failure_type") or "ok"
        log.info("[%s] %s", rec["run_id"], status)

    field = load_field_config()

    if args.file == "real_dates":
        scenarios = [json.loads(line) for line in
                     (SCENARIOS / "real_dates.jsonl").read_text().splitlines() if line.strip()]
        for s in scenarios[: args.limit]:
            if was_done(s["run_id"]):
                continue
            write(run_real(s, args.provider, llm))

    elif args.file == "synthetic_edge_cases":
        cases = [json.loads(line) for line in
                 (SCENARIOS / "synthetic_edge_cases.jsonl").read_text().splitlines() if line.strip()]
        for c in cases[: args.limit]:
            if was_done(c["run_id"]):
                continue
            write(run_synthetic(c, args.provider, llm))

    else:
        df = load_fused_dataset(str(FUSED_PARQUET)).sort_values("date").reset_index(drop=True)
        cases = injection_pool(df)
        for c in cases[: args.limit]:
            if was_done(c["run_id"]):
                continue
            write(run_injection(c, args.provider, llm, field, df))

    print(f"sweep complete -> {out}")


if __name__ == "__main__":
    main()
