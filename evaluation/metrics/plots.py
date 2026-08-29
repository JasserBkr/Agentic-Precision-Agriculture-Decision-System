"""General-purpose diagnostic plots over harness JSONL records.

Reads immutable raw records (``evaluation/results/<stamp>/<provider>/<file>.jsonl``)
and renders reusable figures to ``docs/plots/`` under an ``agent_eval_`` prefix.
Every function accepts an arbitrary ``(records, file_key)`` pair, so the same
code runs on any future provider or campaign unmodified; the CLI default targets
the most recent groq sweep.

Usage:
    uv run python -m evaluation.metrics.plots [--stamp sweep-YYYYMMDD]
                                              [--provider groq]
                                              [--out docs/plots]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics.aggregate import load_records

ACTIONS = ["irrigate_now", "irrigate_soon", "no_action_needed"]
DPI = 120
ROOT = Path(__file__).resolve().parent.parent.parent
BASE = ROOT / "evaluation" / "results"
DEFAULT_OUT = ROOT / "docs" / "plots"


# ---------------------------------------------------------------- helpers

def _save(fig, name: str, out_dir: Path) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def _agent_action(rec: dict) -> str | None:
    return ((rec.get("final_output") or {}).get("irrigation") or {}).get("action")


def _scored(records: list[dict]) -> list[dict]:
    return [r for r in records
            if not r.get("failure_type")
            and (r.get("reference_decision") or {}).get("action") != "UNDECIDED"
            and _agent_action(r)]


def _pile(records: list[dict]) -> dict[int, list]:
    cov = defaultdict(list)
    for r in records:
        if r.get("failure_type"):
            continue
        cov[len(r.get("rule_ids") or [])].append(r["run_id"])
    return dict(cov)


def _calc_agreement(rec: dict) -> bool:
    return rec["reference_decision"]["action"] == _agent_action(rec)


# ---------------------------------------------------------------- Layer 2

def plot_confusion_matrix(records: list[dict], file_key: str, out_dir: Path) -> Path:
    """Reference x agent 3x3 confusion matrix over scored Layer-2 records."""
    if file_key != "real_dates":
        return None
    scored = _scored(records)
    if not scored:
        return None
    cm = np.zeros((3, 3), dtype=int)
    for r in scored:
        ref = ACTIONS.index(r["reference_decision"]["action"])
        ag = ACTIONS.index(_agent_action(r))
        cm[ref, ag] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(ACTIONS, rotation=20, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(ACTIONS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_xlabel("Agent decision")
    ax.set_ylabel("Reference decision")
    ax.set_title(f"{file_key} — irrigation confusion matrix (n={len(scored)})",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="runs")
    fig.tight_layout()
    return _save(fig, f"agent_eval_confusion_matrix_{file_key}.png", out_dir)


def plot_per_stratum_agreement(records: list[dict], file_key: str, out_dir: Path) -> Path:
    """Agreement rate per stratum, annotated with stratum size."""
    if file_key != "real_dates":
        return None
    scored = _scored(records)
    if not scored:
        return None
    n = defaultdict(int)
    ok = defaultdict(int)
    for r in scored:
        st = r.get("stratum", "?")
        n[st] += 1
        ok[st] += _calc_agreement(r)
    strata = sorted(n)
    rates = [ok[s] / n[s] for s in strata]
    colors = ["#2ca02c" if r == 1.0 else ("#ff7f0e" if r >= 0.5 else "#d62728")
              for r in rates]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(strata)), rates, color=colors)
    ax.axhline(0.818, color="grey", ls="--", lw=1, alpha=0.7)
    ax.text(len(strata) - 0.4, 0.818, "overall 0.818", color="grey",
            ha="right", va="bottom", fontsize=9)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(strata, rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Agreement rate")
    ax.set_title(f"{file_key} — per-stratum agreement (bar colour = full/partial/miss)",
                 fontweight="bold")
    for b, s in zip(bars, strata):
        ax.annotate(f"n={n[s]}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color="#333333")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"agent_eval_per_stratum_agreement_{file_key}.png", out_dir)


def plot_confidence_by_agreement(records: list[dict], file_key: str, out_dir: Path) -> Path:
    """Irrigation confidence of agreeing vs disagreeing runs (Layer 2 calibration)."""
    if file_key != "real_dates":
        return None
    scored = _scored(records)
    if not scored:
        return None
    agree = [r["final_output"]["irrigation"]["confidence"]
             for r in scored if _calc_agreement(r)]
    disagree = [r["final_output"]["irrigation"]["confidence"]
                for r in scored if not _calc_agreement(r)]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    parts = ax.violinplot([agree, disagree], positions=[0, 1],
                          showmedians=True, widths=0.55)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_alpha(0.4)
    for y in agree:
        ax.plot(0 + np.random.uniform(-0.08, 0.08), y, "o", ms=5,
                color="#1f77b4", alpha=0.7)
    for y in disagree:
        ax.plot(1 + np.random.uniform(-0.08, 0.08), y, "o", ms=6,
                color="#d62728", alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"agree (n={len(agree)})", f"disagree (n={len(disagree)})"])
    ax.set_ylabel("Irrigation confidence")
    ax.set_title(f"{file_key} — confidence vs decision agreement", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"agent_eval_confidence_by_agreement_{file_key}.png", out_dir)


def plot_calibration_buckets(records: list[dict], file_key: str, out_dir: Path) -> Path:
    """Agree/disagree counts per confidence bucket ([[0.5-0.75], (0.75-1])."""
    if file_key != "real_dates":
        return None
    scored = _scored(records)
    if not scored:
        return None
    buckets = defaultdict(lambda: {"agree": 0, "disagree": 0})
    for r in scored:
        c = r["final_output"]["irrigation"]["confidence"]
        b = "(0.5-0.75]" if c <= 0.75 else "(0.75-1]"
        buckets[b]["agree" if _calc_agreement(r) else "disagree"] += 1
    names = sorted(buckets)
    agree = [buckets[b]["agree"] for b in names]
    disagree = [buckets[b]["disagree"] for b in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(names))
    ax.bar(x, agree, 0.5, label="agree", color="#2ca02c")
    ax.bar(x, disagree, 0.5, bottom=agree, label="disagree", color="#d62728")
    for i, (a, d) in enumerate(zip(agree, disagree)):
        ax.text(i, a + d + 0.15, f"n={a + d}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Runs")
    ax.set_xlabel("Confidence bucket")
    ax.set_title(f"{file_key} — calibration buckets", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"agent_eval_calibration_buckets_{file_key}.png", out_dir)


# ---------------------------------------------------------------- Layer 2b

def plot_ndvi_detection(records: list[dict], file_key: str, out_dir: Path) -> Path:
    """NDVI anomaly detection: detected vs total per sign (stress/vigor/control)."""
    if file_key != "ndvi_injection":
        return None
    rows = []
    for r in records:
        d = r.get("detection") or {}
        if not d:
            continue
        kind = d["kind"]
        label = {"injected": "injected(+2.5)" if d.get("magnitude_z", 0) > 0
                 else "injected(-2.5)",
                 "control": "control"}[kind]
        rows.append((label, d.get("detected"), d.get("measured_z") is not None))
    names = ["injected(-2.5)", "injected(+2.5)", "control"]
    total = [sum(1 for x in rows if x[0] == n) for n in names]
    detected = [sum(1 for x in rows if x[0] == n and x[1]) for n in names]
    sparse = [sum(1 for x in rows if x[0] == n and not x[2]) for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, total, w, label="total runs", color="#7f7f7f")
    ax.bar(x + w / 2, detected, w, label="detected", color="#2ca02c")
    for i, (t, d, s) in enumerate(zip(total, detected, sparse)):
        ax.text(i - w / 2, t + 0.1, f"{t}", ha="center", fontsize=9)
        ax.text(i + w / 2, d + 0.1, f"{d}", ha="center", fontsize=9)
        if s:
            ax.text(i, -0.75, f"{s} sparse\nexcluded", ha="center", fontsize=8,
                    color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Runs")
    ax.set_title(f"{file_key} — NDVI anomaly detection by sign", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "agent_eval_ndvi_detection.png", out_dir)


# ---------------------------------------------------------------- Layer 3

def plot_attempts(records_by_file: dict[str, list[dict]], out_dir: Path) -> Path:
    """First-attempt vs retried runs per scenario file."""
    files = sorted(records_by_file)
    first = [sum(1 for r in records_by_file[f] if not r.get("failure_type")
                 and (r.get("recommend_attempts") or 0) <= 1) for f in files]
    retry = [sum(1 for r in records_by_file[f] if not r.get("failure_type")
                 and (r.get("recommend_attempts") or 0) > 1) for f in files]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(files))
    w = 0.35
    ax.bar(x - w / 2, first, w, label="first-attempt OK", color="#2ca02c")
    ax.bar(x + w / 2, retry, w, label="needed retry", color="#ff7f0e")
    for i, a in enumerate(first):
        ax.text(i - w / 2, a + 0.1, f"{a}", ha="center", fontsize=9)
    for i, b in enumerate(retry):
        ax.text(i + w / 2, b + 0.1, f"{b}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=12)
    ax.set_ylabel("Runs")
    ax.set_title("groq — validation attempts per scenario file", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "agent_eval_attempts.png", out_dir)


def plot_rule_firing(records_by_file: dict[str, list[dict]], out_dir: Path) -> Path:
    """Frequency of fired validator rule IDs, stacked per scenario file."""
    files = sorted(records_by_file)
    counts = defaultdict(Counter)
    for f in files:
        for r in records_by_file[f]:
            if r.get("failure_type"):
                continue
            for rid in r.get("rule_ids", []):
                counts[rid][f] += 1
    rules = sorted(counts)
    if not rules:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(files))
    for rid in rules:
        vals = [counts[rid].get(f, 0) for f in files]
        ax.bar(files, vals, bottom=bottom, label=rid)
        bottom += np.array(vals)
    ax.set_ylabel("Fired (validation/conflict) rule IDs")
    ax.set_title("groq — rules fired across runs", fontweight="bold")
    ax.legend(title="rule id", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "agent_eval_rule_firing.png", out_dir)


def plot_signal_coverage(records_by_file: dict[str, list[dict]], out_dir: Path) -> Path:
    """Distribution of distinct contributing signals cited per run."""
    data = {f: [] for f in records_by_file}
    for f, recs in records_by_file.items():
        for r in recs:
            if r.get("failure_type"):
                continue
            final = r.get("final_output") or {}
            cited = set()
            for part in ("irrigation", "fertilization"):
                for s in (final.get(part) or {}).get("contributing_signals", []):
                    cited.add(s.get("signal_name"))
            data[f].append(len(cited))

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([data[f] for f in data], positions=range(len(data)),
                          showmedians=True, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
    pc.set_alpha(0.4)
    for i, f in enumerate(data):
        ys = data[f]
        ax.plot(np.full(len(ys), i) + np.random.uniform(-0.08, 0.08, len(ys)),
                ys, "o", ms=5, color="#1f77b4", alpha=0.7)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(list(data), rotation=12)
    ax.set_ylabel("Distinct contributing signals")
    ax.set_title("groq — grounding coverage (distinct signals cited)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "agent_eval_signal_coverage.png", out_dir)


# ---------------------------------------------------------------- Layer 4

def plot_latency(records_by_file: dict[str, list[dict]], out_dir: Path) -> Path:
    """Boxplots of LLM graph round-trip and Chronos inference latency."""
    files = sorted(records_by_file)
    llm = [[r.get("graph_s") for r in records_by_file[f]
            if not r.get("failure_type") and r.get("graph_s") is not None]
           for f in files]
    chrono_files = [f for f in files
                    if any(r.get("chronos_s") is not None
                           for r in records_by_file[f])]
    chrono = [[r.get("chronos_s") for r in records_by_file[f]
               if not r.get("failure_type") and r.get("chronos_s") is not None]
              for f in chrono_files]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    ax1.boxplot(llm, tick_labels=files, patch_artist=True,
                boxprops={"facecolor": "#ff7f0e", "alpha": 0.6})
    ax1.set_title("LLM graph round-trip (s)", fontweight="bold")
    ax1.set_ylabel("Seconds")
    ax1.grid(axis="y", alpha=0.3)
    if chrono_files:
        ax2.boxplot(chrono, tick_labels=chrono_files, patch_artist=True,
                    boxprops={"facecolor": "#1f77b4", "alpha": 0.6})
        ax2.set_title("Chronos inference (s)", fontweight="bold")
        ax2.set_ylabel("Seconds")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "no chronos latency recorded", ha="center", va="center")
        ax2.axis("off")
    fig.suptitle("groq — latency composition (LLM dominates)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "agent_eval_latency.png", out_dir)


def plot_tokens(records_by_file: dict[str, list[dict]], out_dir: Path) -> Path:
    """Input vs output tokens per run, grouped by scenario file."""
    files = sorted(records_by_file)
    inp = {f: [] for f in files}
    out = {f: [] for f in files}
    for f, recs in records_by_file.items():
        for r in recs:
            if r.get("failure_type"):
                continue
            u = r.get("usage") or {}
            inp[f].append(u.get("input_tokens_total") or 0)
            out[f].append(u.get("output_tokens_total") or 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(files))
    ax.bar(x - width / 2, [np.mean(inp[f]) for f in files], width,
           label="mean input", color="#1f77b4")
    ax.bar(x + width / 2, [np.mean(out[f]) for f in files], width,
           label="mean output", color="#d62728")
    ax.errorbar(x - width / 2, [np.mean(inp[f]) for f in files],
                yerr=[np.std(inp[f]) for f in files], fmt="none", ecolor="black",
                capsize=3)
    ax.errorbar(x + width / 2, [np.mean(out[f]) for f in files],
                yerr=[np.std(out[f]) for f in files], fmt="none", ecolor="black",
                capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=12)
    ax.set_ylabel("Tokens per run (mean ± std)")
    ax.set_title("groq — token usage per scenario file", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "agent_eval_tokens.png", out_dir)


# ---------------------------------------------------------------- main

def generate(records_by_file: dict[str, list[dict]], out_dir: Path) -> list[str]:
    """Run every plot over the supplied records; return saved filenames."""
    saved = []
    for fk, recs in records_by_file.items():
        for fn in (plot_confusion_matrix, plot_per_stratum_agreement,
                   plot_confidence_by_agreement, plot_calibration_buckets,
                   plot_ndvi_detection):
            p = fn(recs, fk, out_dir)
            if p:
                saved.append(p.name)
    for fn in (plot_attempts, plot_rule_firing, plot_signal_coverage,
               plot_latency, plot_tokens):
        p = fn(records_by_file, out_dir)
        if p:
            saved.append(p.name)
    return saved


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

    print(f"provider={args.provider} stamp={args.stamp}")
    for fk, recs in sorted(records_by_file.items()):
        print(f"  {fk}: {len(recs)} records")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"saving to {args.out}/")

    saved = generate(records_by_file, args.out)
    print(f"\n{len(saved)} figures written")


if __name__ == "__main__":
    main()