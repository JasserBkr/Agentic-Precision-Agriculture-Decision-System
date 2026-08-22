"""End-to-end entrypoint for the Step 3 fusion agent (rebuild).

Thin orchestrator only:
    parse query -> merge CLI overrides into the SAME QueryParams ->
    PREP (build_signal_bundle) -> build graph -> invoke -> print result.

get_llm() is called exactly once and reused everywhere. All data loading,
forecasting, and error handling lives in build_signal_bundle (agent/bundle.py)
and runs once, before the graph starts.

Usage:
    python scripts/run_pipeline.py --mode offline --target-date 2026-07-01
    python scripts/run_pipeline.py --mode live --query "irrigate in the next 2 days?"
    python scripts/run_pipeline.py --interactive          # keep typing your own queries
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml

from agri_agent.agent.bundle import (
    FORECAST_HORIZON_DAYS,
    FUSED_PARQUET,
    SignalBundle,
    build_signal_bundle,
)
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

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FIELD_CONFIG_PATH = ROOT / "configs" / "field.yaml"


def parse_date(value: str) -> date:
    """Parse a --target-date value. Raises a clear error on bad input."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--target-date must be an ISO date (YYYY-MM-DD), got '{value}'"
        )


def load_field_config(path: str | Path = FIELD_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step 3 fusion agent.")
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="offline reads the frozen parquet (point-in-time backtest); live does a fresh 2-year fetch.",
    )
    parser.add_argument(
        "--query",
        default="Recommend irrigation and fertilization for the next 7 days.",
    )
    parser.add_argument("--target-date", type=parse_date, default=None)
    parser.add_argument("--crop-type", default=None)
    parser.add_argument("--growth-stage", default=None)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read queries from stdin in a loop; type 'exit'/'quit' or Ctrl-D to leave.",
    )
    return parser.parse_args(argv)


def _warn_zero_forward_window(bundle: SignalBundle) -> None:
    """Loud, unmissable warning when the resolved run ends up with zero
    forward weather rows (e.g. origin deliberately set to the dataset's
    last date). Log lines scroll by; this goes to stderr on its own."""
    if bundle.weather_forecast.get("forecast"):
        return
    print(
        "WARNING: no forward weather window available; soil-moisture and "
        "weather-forecast tools will report insufficient data. Use "
        "--target-date <date well before the dataset's end> or --mode live.",
        file=sys.stderr,
    )


def _resolve_reference_date(mode: str, query_params: QueryParams | None = None) -> date:
    """Return the reference date for temporal expression resolution.

    This MUST match the origin date that ``build_signal_bundle`` will
    compute, so that relative words like "tomorrow" are interpreted
    against the same historical decision point the bundle uses — not
    against the parquet dataset's final date, which may be later.

    * live mode  → ``date.today()``
    * offline mode:
        - if ``query_params.target_date`` is set → that date (the
          user/explicit backtest origin)
        - else → ``parquet_max - FORECAST_HORIZON_DAYS`` (the default
          origin that ``_resolve_offline_origin_and_horizon`` uses)

    The ``query_params`` argument is needed only in offline mode to
    check whether a target_date was already resolved (e.g. from an
    explicit date in the query).  When ``query_params`` is ``None``,
    the fallback (parquet_max − 7) is used.
    """
    if mode == "live":
        return date.today()

    fused = load_fused_dataset(str(FUSED_PARQUET))
    parquet_max = fused["date"].max().date()

    if query_params is not None and query_params.target_date is not None:
        return query_params.target_date

    return parquet_max - timedelta(days=FORECAST_HORIZON_DAYS)


def run_query(
    field: dict,
    query: str,
    mode: str,
    llm,
    target_date=None,
    crop_type=None,
    growth_stage=None,
) -> dict | None:
    """Run ONE query through the full pipeline and print its final output.

    get_llm() is called once by main() and reused here; every query re-parses,
    rebuilds the bundle (origin can depend on the parsed target_date), and runs
    a fresh graph.
    """
    # One QueryParams instance from this point on; CLI flags override the
    # parsed fields only when explicitly provided — there is no second,
    # competing source for these fields.
    query_params: QueryParams = parse_query(query, llm=llm)

    # ------------------------------------------------------------------
    # Offline temporal resolution — two-pass to avoid temporal leakage.
    #
    # Problem:  build_signal_bundle computes its origin *after* we set
    # target_date, but the origin IS the reference date for "tomorrow".
    # The origin depends on target_date, and target_date depends on the
    # origin — a circular dependency.
    #
    # Solution:
    #   Pass 1 — If the raw query has a relative expression ("tomorrow")
    #            but parse_query produced NO explicit target_date (LLM
    #            returned None), do a preliminary resolution against
    #            parquet_max.  This gives us the target_date that
    #            build_signal_bundle would use as its origin.
    #   Pass 2 — Compute the actual offline origin (which now matches
    #            what build_signal_bundle will compute) and do the REAL
    #            resolve_temporal_expressions against it.  This replaces
    #            any LLM-invented date with the deterministic one.
    #
    # If the LLM already resolved an explicit date (e.g. "2026-07-01"
    # in the query text), both passes are skipped — explicit dates win.
    # ------------------------------------------------------------------
    _saved_target = query_params.target_date  # original value from parse_query

    if mode == "offline" and query_params.target_date is None:
        if _has_relative_date_expression(query_params.raw_query):
            # Pass 1: resolve against parquet_max to bootstrap target_date.
            fused_tmp = load_fused_dataset(str(FUSED_PARQUET))
            parquet_max = fused_tmp["date"].max().date()
            resolve_temporal_expressions(query_params, parquet_max, mode=mode)

    # Compute the reference date (matches build_signal_bundle's origin).
    # The origin depends on target_date:
    #   - LLM found an explicit date → origin = that date
    #   - No explicit date (Pass 1 resolved "tomorrow") → origin =
    #     parquet_max - FORECAST_HORIZON_DAYS (the default offline origin)
    # We must NOT use the Pass 1 intermediate result as the origin,
    # because that would double-increment "tomorrow".
    if mode == "offline":
        had_explicit_date = _saved_target is not None
        if had_explicit_date:
            reference_date = _saved_target
        else:
            fused_tmp = load_fused_dataset(str(FUSED_PARQUET))
            parquet_max = fused_tmp["date"].max().date()
            reference_date = parquet_max - timedelta(days=FORECAST_HORIZON_DAYS)
    else:
        reference_date = date.today()

    # Pass 2 (live: always; offline: always): real resolution against
    # the correct origin.
    resolve_temporal_expressions(query_params, reference_date, mode=mode)

    # CLI overrides always win — applied AFTER all temporal resolution.
    if target_date:
        query_params.target_date = target_date
    if crop_type:
        query_params.crop_type = crop_type
    if growth_stage:
        query_params.growth_stage = growth_stage

    log.info(
        "Running fusion agent for %s (mode=%s, query=%r)",
        field["field_id"],
        mode,
        query,
    )

    bundle = build_signal_bundle(field, query_params, mode=mode)
    _warn_zero_forward_window(bundle)
    log.info(
        "Bundle origin=%s, load_errors=%s",
        bundle.origin_date.date(),
        bundle.load_errors,
    )

    graph = build_graph(bundle, llm=llm)
    state = initial_state(bundle, query_params, query=query)
    result = graph.invoke(state)

    final = result.get("final_output") or result.get("draft_recommendation")
    if final is None:
        log.error("Graph finished without a final recommendation.")
        return None

    print(json.dumps(final, indent=2, default=str))
    return final


def interactive_main(
    field: dict, llm, mode: str, args: argparse.Namespace
) -> dict | None:
    """REPL: prompt for queries on stdin, run each one, keep going until
    'exit'/'quit' or Ctrl-D. A failed query is logged and the loop continues."""
    if llm is None:
        log.error(
            "No LLM configured (set OPENAI_API_KEY, or AGRI_LLM_PROVIDER plus "
            "its key, in .env) — interactive mode needs it."
        )
        return None

    print(
        f"Interactive mode for field {field['field_id']} (mode={mode}). "
        "Type a query, or 'exit'/'quit'/Ctrl-D to leave."
    )
    last: dict | None = None
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break
        try:
            last = run_query(
                field,
                query,
                mode,
                llm,
                target_date=args.target_date,
                crop_type=args.crop_type,
                growth_stage=args.growth_stage,
            )
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive
            log.error("Query failed: %s", exc)
    return last


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    field = load_field_config()
    llm = get_llm()  # called exactly ONCE, reused everywhere

    if args.interactive:
        return interactive_main(field, llm, mode=args.mode, args=args)

    return run_query(
        field,
        args.query,
        args.mode,
        llm,
        target_date=args.target_date,
        crop_type=args.crop_type,
        growth_stage=args.growth_stage,
    )


if __name__ == "__main__":
    main()
