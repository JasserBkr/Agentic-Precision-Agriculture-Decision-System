"""Layer 2/3/4 metric computations over harness JSONL records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

BOUNDARY_MARGIN = 0.02  # m3/m3, |m_min - trigger| (plan §8 Q2 default)


def load_records(path: str | Path) -> list[dict]:
    recs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def _ok(rec: dict) -> bool:
    return not rec.get("failure_type")


def _conf_bucket(c):
    if c is None:
        return "missing"
    if c <= 0.5:
        return "(0-0.5]"
    if c <= 0.75:
        return "(0.5-0.75]"
    return "(0.75-1]"


# ---------------------------------------------------------------- Layer 2

def layer2_irrigation(records: list[dict]) -> dict:
    scored = [r for r in records
              if _ok(r) and r.get("reference_decision", {}).get("action") != "UNDECIDED"]
    undecided = sum(1 for r in records if _ok(r)
                    and r.get("reference_decision", {}).get("action") == "UNDECIDED")

    agree = disagree = 0
    agree2 = disagree2 = 0
    confusion = defaultdict(Counter)
    boundary = {"n": 0, "agree": 0}
    calib = defaultdict(lambda: {"agree": [], "disagree": []})
    per_stratum = defaultdict(lambda: {"n": 0, "agree": 0})
    conf_agree, conf_disagree = [], []

    def collapse(a):
        return "irrigate" if a in ("irrigate_now", "irrigate_soon") else a

    for r in scored:
        ref_a = r["reference_decision"]["action"]
        agent = (r.get("final_output") or {}).get("irrigation") or {}
        agent_a = agent.get("action")
        if agent_a is None:
            continue
        confusion[ref_a][agent_a] += 1
        hit = ref_a == agent_a
        agree += hit
        disagree += (not hit)
        agree2 += collapse(ref_a) == collapse(agent_a)
        disagree2 += collapse(ref_a) != collapse(agent_a)

        st = r.get("stratum", "?")
        per_stratum[st]["n"] += 1
        per_stratum[st]["agree"] += hit

        dist = abs((r["reference_decision"].get("inputs") or {}).get("distance_to_trigger", 9e9))
        if dist <= BOUNDARY_MARGIN:
            boundary["n"] += 1
            boundary["agree"] += hit

        c = agent.get("confidence")
        bucket = _conf_bucket(c)
        calib[bucket]["agree" if hit else "disagree"].append(c)
        (conf_agree if hit else conf_disagree).append(c)

    n = agree + disagree
    n2 = agree2 + disagree2
    fert = [r for r in records if _ok(r) and (r.get("final_output") or {}).get("fertilization")]
    caveat_rate = (sum(1 for r in fert if (r["final_output"]["fertilization"].get("caveat") or "").strip())
                   / len(fert)) if fert else None
    zs, acts = [], []
    for r in fert:
        z = ((r.get("ndvi_z")))
        a = r["final_output"]["fertilization"].get("action")
        if z is not None and a is not None:
            zs.append(z)
            acts.append(1 if a == "apply_fertilizer" else 0)
    spearman = None
    if len(set(acts)) == 2 and len(acts) >= 5:
        from scipy.stats import spearmanr
        rho, p = spearmanr(zs, acts)
        spearman = {"rho": round(float(rho), 3), "p": round(float(p), 4), "n": len(acts)}

    return {
        "n_scored": n,
        "n_undecided_ref": undecided,
        "agreement_3class": round(agree / n, 3) if n else None,
        "agreement_2class": round(agree2 / n2, 3) if n2 else None,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "boundary_cases": {**boundary,
                           "rate": round(boundary["agree"] / boundary["n"], 3) if boundary["n"] else None},
        "confidence_mean_agree": round(sum(conf_agree) / len(conf_agree), 3) if conf_agree else None,
        "confidence_mean_disagree": round(sum(conf_disagree) / len(conf_disagree), 3) if conf_disagree else None,
        "calibration_buckets": {k: {"n_agree": len(v["agree"]), "n_disagree": len(v["disagree"])}
                                for k, v in calib.items()},
        "per_stratum": {k: {"n": v["n"], "agreement": round(v["agree"] / v["n"], 3)}
                        for k, v in sorted(per_stratum.items())},
        "fertilization_caveat_rate": round(caveat_rate, 3) if caveat_rate is not None else None,
        "fertilization_vs_ndvi_z_spearman_DESCRIPTIVE_ONLY": spearman,
    }


def injection_metrics(records: list[dict]) -> dict:
    """Precision/recall over ndvi_injection records. Records with
    measured_z=None are reported separately (data-sparse origins where the
    anomaly itself reports insufficient_data) — they are excluded from the
    denominators, not silently counted as misses."""
    scored, sparse = [], []
    for r in records:
        d = r.get("detection")
        if not d:
            continue
        if d.get("measured_z") is None:
            sparse.append(r["run_id"])
        else:
            scored.append(d)

    def _pr(rows, kind):
        rel = [d for d in rows if d["kind"] == kind]
        tp = sum(1 for d in rel if d["detected"])
        return {"n": len(rel), "detected": tp}

    inj_pos = [d for d in scored if d["kind"] == "injected" and d["magnitude_z"] > 0]
    inj_neg = [d for d in scored if d["kind"] == "injected" and d["magnitude_z"] < 0]
    ctrls = [d for d in scored if d["kind"] == "control"]

    tp = sum(1 for d in scored if d["kind"] == "injected" and d["detected"])
    fp = sum(1 for d in ctrls if d["detected"])
    fn = sum(1 for d in scored if d["kind"] == "injected" and not d["detected"])

    rec_errs = [d.get("recovery_error") for d in scored
                if d["kind"] == "injected" and d.get("recovery_error") is not None]

    precision = round(tp / (tp + fp), 3) if (tp + fp) else None
    recall = round(tp / (tp + fn), 3) if (tp + fn) else None
    return {
        "precision_anomaly_band": precision,
        "recall_anomaly_band": recall,
        "false_positive_rate_controls": round(fp / len(ctrls), 3) if ctrls else None,
        "by_sign": {"stress(-2.5)": _pr(scored, "injected") and {
                        "n": len(inj_neg), "detected": sum(1 for d in inj_neg if d["detected"])},
                    "vigor(+2.5)": {
                        "n": len(inj_pos), "detected": sum(1 for d in inj_pos if d["detected"])}},
        "recovery_error_mean_abs": round(sum(rec_errs) / len(rec_errs), 3) if rec_errs else None,
        "sparse_origins_excluded": sparse,
    }

# ---------------------------------------------------------------- Layer 3

def layer3_agent(records: list[dict]) -> dict:
    ok = [r for r in records if _ok(r)]
    n = len(ok)
    attempts = [r.get("recommend_attempts") or 0 for r in ok]
    retried = sum(1 for a in attempts if a and a > 1)
    recovered = sum(1 for r, a in zip(ok, attempts)
                    if a and a > 1 and not r.get("validation_problems"))
    persisted = sum(1 for r, a in zip(ok, attempts)
                    if a and a > 1 and r.get("validation_problems"))
    conflict_flagged = sum(1 for r in ok if r.get("signal_conflict_detected"))

    rule_breakdown = Counter()
    for r in ok:
        for rid in r.get("rule_ids", []):
            rule_breakdown[rid] += 1

    def coverage(r):
        final = r.get("final_output") or {}
        cited = set()
        for part in ("irrigation", "fertilization"):
            for s in (final.get(part) or {}).get("contributing_signals", []):
                cited.add(s.get("signal_name"))
        return len(cited)

    cov_values = [coverage(r) for r in ok]
    checks_pass, checks_total = 0, 0
    generic_ok = []
    for r in records:
        ec = r.get("expected_checks")
        if not ec:
            continue
        for name, passed in (ec.get("checks") or {}).items():
            checks_total += 1
            checks_pass += bool(passed)
        if ec.get("all_deterministic_pass") is not None:
            generic_ok.append(ec["all_deterministic_pass"])

    schema_failures = len(records) - n

    return {
        "n_runs": len(records),
        "schema_structured_output_failure_rate": round(schema_failures / len(records), 3) if records else None,
        "grounding_first_attempt_proxy_rate": (round((len(ok) - retried) / len(ok), 3)
                                                if records else None),
        "retry_rate": round(retried / n, 3) if n else None,
        "retry_recovery_rate": round(recovered / retried, 3) if retried else None,
        "unsupported_claims_after_retry_rate": round(persisted / retried, 3) if retried else None,
        "signal_conflict_detected_rate": round(conflict_flagged / n, 3) if n else None,
        "conflict_rule_breakdown_BY_RULE": dict(rule_breakdown),
        "contributing_signal_coverage_distinct_names":
            {"mean": round(sum(cov_values) / len(cov_values), 2) if cov_values else None},
        "synthetic_expected_checks": ({"pass": checks_pass, "total": checks_total}
                                       if checks_total else None),
        "generic_default_scenarios_all_pass": (all(generic_ok) if generic_ok else None),
    }


# ---------------------------------------------------------------- Layer 4

def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return {"mean": round(sum(xs) / n, 2), "median": xs[n // 2],
            "p95": xs[min(n - 1, int(round(0.95 * (n - 1))))], "max": xs[-1]}


def layer4_ops(records: list[dict]) -> dict:
    ok = [r for r in records if _ok(r)]
    failures = Counter(r.get("failure_type") for r in records if not _ok(r))
    llm_calls = [ (r.get("usage") or {}).get("llm_calls") for r in ok ]
    tok_in = [(r.get("usage") or {}).get("input_tokens_total") or 0 for r in ok]
    tok_out = [(r.get("usage") or {}).get("output_tokens_total") or 0 for r in ok]
    chronos = [r.get("chronos_s") for r in ok]
    graph_t = [r.get("graph_s") for r in ok]
    prep = [r.get("prep_total_s") for r in ok]

    soil_insufficient = sum(
        1 for r in records
        if r.get("file") == "real_dates"
        and (r.get("final_output") is not None)
        and r.get("insufficient_flags", {}).get("soil")
    )
    real_n = max(1, sum(1 for r in records if r.get("file") == "real_dates"))

    return {
        "latency_chronos_s": _stats(chronos),
        "latency_graph_llm_s": _stats(graph_t),
        "latency_prep_total_s": _stats(prep),
        "llm_calls_per_run": {"mean": (round(sum(llm_calls) / len(llm_calls), 2)
                                        if any(c is not None for c in llm_calls) else None),
                               "max": max((c for c in llm_calls if c is not None), default=None),
                               "theoretical_ceiling": 3},
        "tokens_input_total": int(sum(tok_in)) if tok_in else None,
        "tokens_output_total": int(sum(tok_out)) if tok_out else None,
        "tokens_per_run_mean_in_out": (round(sum(tok_in) / len(ok), 1), round(sum(tok_out) / len(ok), 1)) if ok else None,
        "failure_rate_by_type": dict(failures),
        "soil_moisture_insufficient_data_rate_REAL_DATES": (
            round(soil_insufficient / real_n, 3)),
    }
