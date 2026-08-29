"""Render markdown summaries per (provider, scenario-file) + cross-provider
comparison. Raw records are read-only inputs; outputs go to results/<stamp>/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluation.metrics.aggregate import (
    injection_metrics,
    layer2_irrigation,
    layer3_agent,
    layer4_ops,
    load_records,
)


def _md_table(d: dict) -> str:
    if not d:
        return "_n/a_\n"
    rows = ["| | |", "|---|---|"]
    for k, v in d.items():
        rows.append(f"| {k} | `{json.dumps(v, default=str)}` |")
    return "\n".join(rows) + "\n"


def render_file_report(records: list[dict], provider: str, file_key: str) -> str:
    parts = [f"# {provider} — {file_key}", ""]
    parts += ["## Layer 2 — Decision quality (real_dates only)", ""]
    parts += [_md_table(layer2_irrigation(records))] if file_key == "real_dates" \
        else ["_skipped (not real_dates)_\n"]
    if file_key == "ndvi_injection":
        parts += ["## Layer 2b — NDVI injection detection", "",
                  _md_table(injection_metrics(records))]
    parts += ["## Layer 3 — Agent quality", "", _md_table(layer3_agent(records))]
    parts += ["## Layer 4 — Operational quality", "", _md_table(layer4_ops(records))]

    fails = [r for r in records if r.get("failure_type")]
    if fails:
        parts += ["## Failures", ""]
        parts += [f"- `{r['run_id']}`: **{r['failure_type']}** — {r.get('failure_detail', '')[:200]}"
                  for r in fails]
        parts.append("")
    return "\n".join(parts)


def render_comparison(results_dir: Path) -> str:
    providers = sorted({p.name for p in results_dir.iterdir() if p.is_dir()})

    def col(provider):
        # Prefer synthetic_edge_cases; fall back to real_dates so a provider
        # that only completed one file still appears in the comparison.
        for fk in ("synthetic_edge_cases", "real_dates"):
            f = results_dir / provider / f"{fk}.jsonl"
            if f.exists():
                return fk, layer3_agent(load_records(f))
        return None, layer3_agent([])

    cols = {p: col(p) for p in providers}
    lines = ["# Provider comparison (Layer 3 focus)", "",
             "_Per-provider basis: " +
              ", ".join(f"{p}={fk}" for p, (fk, _) in cols.items() if fk) + "._",
             "",
             "| metric | " + " | ".join(providers) + " |",
             "|---|" + "---|" * len(providers)]
    for metric in ("schema_structured_output_failure_rate", "grounding_first_attempt_proxy_rate",
                    "retry_rate", "retry_recovery_rate", "unsupported_claims_after_retry_rate",
                    "signal_conflict_detected_rate"):
        lines.append(f"| {metric} | " +
                     " | ".join(str(cols[p][1].get(metric)) for p in providers) + " |")

    lines += ["", "## Rule breakdown by provider/file", ""]
    for p in providers:
        for fk in ("real_dates", "synthetic_edge_cases"):
            f = results_dir / p / f"{fk}.jsonl"
            if f.exists():
                bd = layer3_agent(load_records(f))["conflict_rule_breakdown_BY_RULE"]
                lines.append(f"- **{p}/{fk}**: `{bd}`")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    stamp = sys.argv[1] if len(sys.argv) > 1 else None
    base = Path("evaluation") / "results"
    stamps = sorted({p.parent.parent.name for p in base.glob("*/*/*.jsonl")})
    stamp = stamp or (stamps[-1] if stamps else None)
    if not stamp:
        raise SystemExit("no results found")
    results_dir = base / stamp

    for provider_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for jf in sorted(provider_dir.glob("*.jsonl")):
            recs = load_records(jf)
            md = render_file_report(recs, provider_dir.name, jf.stem)
            out = provider_dir / f"summary_{jf.stem}.md"
            out.write_text(md)
            print(f"wrote {out}")

    cmp_md = render_comparison(results_dir)
    (results_dir / "comparison.md").write_text(cmp_md)
    print(f"wrote {results_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
