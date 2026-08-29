"""parse_query (STEP 1), the LLM factory, and the LangGraph StateGraph
(STEP 4) for the rebuilt fusion agent.

Graph shape: inject_evidence (deterministic, no LLM) -> recommend
(structured output) -> validate (deterministic) -> END, with a
conditional retry edge back to recommend when validation finds problems
and retries remain.

All four evidence categories (vegetation, weather, soil-moisture
forecast, thresholds) are formatted into the message history
deterministically before the LLM ever sees them.  No tool-calling
loop is involved in evidence gathering.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agri_agent.agent.bundle import SignalBundle
from agri_agent.agent.schemas import FusionRecommendation, QueryParams
from agri_agent.agent.validator import validate_recommendation
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

MAX_RECOMMEND_ATTEMPTS = 2

RECOMMEND_SYSTEM_PROMPT = (
    "You are an agronomy advisor for precision agriculture.  All evidence "
    "has been pre-gathered and will be provided in the conversation.  "
    "Produce a structured FusionRecommendation, grounding every "
    "contributing signal in the evidence provided.\n"
    "Irrigation action must be one of: irrigate_now, irrigate_soon, "
    "no_action_needed.\n"
    "Fertilization action must be one of: apply_fertilizer, no_application."
)

RECOMMEND_INSTRUCTION = (
    "Produce the structured FusionRecommendation now, grounding every "
    "contributing signal in the evidence provided above."
)

_DATE_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")

# Common misspellings of "tomorrow" are normalised to the canonical form
# BEFORE the relative-date regex runs, so that downstream code only needs
# to handle a small set of canonical tokens.
_TOMORROW_MISSPELLINGS = re.compile(
    r"\b(tommorow|tomorow|tmrw|tommorrow|tomarrow|tomoro)\b",
    re.IGNORECASE,
)


def _normalize_temporal(text: str) -> str:
    """Normalise common temporal misspellings in *text*.

    This is purely a string-to-string transform — no side effects, no
    schema mutations.  It runs once on the raw query before any regex
    matching so that downstream code only needs canonical tokens.
    """
    return _TOMORROW_MISSPELLINGS.sub("tomorrow", text)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    bundle: SignalBundle  # set once at graph start, never mutated
    query_params: QueryParams  # set once at graph start, never mutated
    draft_recommendation: FusionRecommendation | None
    recommend_attempts: int
    validation_problems: list[str]
    signal_conflict_detected: bool
    final_output: dict | None


# ---------------------------------------------------------------------
# LLM factory — called exactly ONCE per run_pipeline.py execution
# ---------------------------------------------------------------------


def get_llm():
    """
    Build the chat LLM for the agent's reasoning steps. Provider is chosen
    by AGRI_LLM_PROVIDER (openai | gemini | groq, default openai). Returns
    None when the provider's API key is not set, so callers can fall back
    (e.g. parse_query's regex path) instead of crashing.
    """
    provider = os.environ.get("AGRI_LLM_PROVIDER", "openai").lower()
    model = os.environ.get("AGRI_LLM_MODEL")

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)

    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or "gemini-3.5-flash")

    if provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            return None
        from langchain_groq import ChatGroq

        return ChatGroq(model=model or "openai/gpt-oss-120b")

    log.warning("Unknown AGRI_LLM_PROVIDER '%s'; treating as openai.", provider)
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------
# STEP 1 — parse_query
# ---------------------------------------------------------------------


def _regex_parse(query: str) -> QueryParams:
    """
    Best-effort regex fallback when no LLM is available: parse target_date
    only. No date -> None (normal case). Multiple DIFFERENT dates -> None
    (ambiguous, never guess). Other fields stay None.
    """
    parsed_dates = set()
    for y, m, d in _DATE_RE.findall(query):
        try:
            parsed_dates.add(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    target = None
    if len(parsed_dates) == 1:
        target = parsed_dates.pop()
    return QueryParams(target_date=target)


def parse_query(query: str, llm=None) -> QueryParams:
    """
    Parse free text into structured params. Runs ONCE, before PREP. The
    returned QueryParams is the single source of truth for
    target_date/crop_type/growth_stage/focus_window/field_id from here on.

    ``raw_query`` is always set on the returned object so that
    resolve_temporal_expressions() can re-resolve relative dates later
    against the correct reference date (today in live mode,
    dataset-last-date in offline mode).

    If ``llm`` is None (no API key), fall back to a regex-based best-effort
    parse for target_date only. An unparseable/ambiguous date stays None
    rather than guessing.
    """
    if llm is None:
        result = _regex_parse(query)
        result.raw_query = query
        return result

    try:
        parser = llm.with_structured_output(QueryParams)
        result = parser.invoke(
            [
                HumanMessage(
                    content=(
                        "Extract structured parameters from this irrigation / "
                        f"fertilization query:\n{query}"
                    )
                )
            ]
        )
        if isinstance(result, dict):
            result = QueryParams(**result)
        result.raw_query = query
        return result
    except Exception as exc:  # noqa: BLE001 — fall back rather than guess
        log.warning("Structured query parsing failed (%s); using regex fallback.", exc)
        result = _regex_parse(query)
        result.raw_query = query
        return result


# ---------------------------------------------------------------------
# Temporal expression resolution (STEP 1b)
# ---------------------------------------------------------------------


_RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|yesterday|day after tomorrow)\b",
    re.IGNORECASE,
)

# Duration / horizon phrases.  "next week" maps to 7; "next N days" to N.
_HORIZON_WEEK_RE = re.compile(r"\bnext\s+week\b", re.IGNORECASE)
_HORIZON_DAYS_RE = re.compile(r"\bnext\s+(\d+)\s+days?\b", re.IGNORECASE)


def _has_relative_date_expression(raw_query: str | None) -> bool:
    """Pure check: does the raw query contain a relative temporal expression?

    Common misspellings (e.g. "tommorow") are normalised before matching
    so that callers don't need to guess every variant.

    No side effects, no logging.  Used by run_pipeline to decide whether
    an intermediate resolution pass is needed before computing the offline
    backtest origin.
    """
    normalised = _normalize_temporal(raw_query or "")
    return bool(_RELATIVE_DATE_RE.search(normalised))


def resolve_temporal_expressions(
    query_params: QueryParams,
    reference_date: date,
    *,
    mode: str = "offline",
) -> QueryParams:
    """Re-resolve relative temporal expressions in the raw query against a
    known *reference_date*.

    This is necessary because ``parse_query`` lets the LLM resolve words
    like "tomorrow" against whatever date the LLM thinks is today — which
    may be wrong (offline/backtest mode) or simply an LLM hallucination.
    By re-resolving here, we make the temporal semantics **explicit and
    deterministic**.

    Common misspellings (e.g. "tommorow") are normalised to canonical
    forms before matching so that callers don't need to guess every
    variant.

    Duration phrases are also extracted here:
      * "next week"  → horizon_days = 7
      * "next N days" → horizon_days = N

    ``reference_date`` should be:
      * ``date.today()`` in live mode
      * the dataset's last available date in offline mode

    CLI ``--target-date`` overrides are applied *after* this function
    returns (see run_query), so they always win over a parsed relative
    expression.
    """
    import logging

    _log = logging.getLogger(__name__)

    raw = query_params.raw_query or ""
    normalised = _normalize_temporal(raw)
    has_relative = bool(_RELATIVE_DATE_RE.search(normalised))

    if not has_relative:
        _log.info(
            "Temporal resolution: no relative expression in query; "
            "target_date=%s (unchanged)",
            query_params.target_date,
        )

    if has_relative:
        match = _RELATIVE_DATE_RE.search(normalised)
        token = match.group(1).lower() if match else ""

        if token == "today":
            resolved = reference_date
        elif token == "tomorrow":
            resolved = reference_date + timedelta(days=1)
        elif token == "yesterday":
            resolved = reference_date - timedelta(days=1)
        elif token == "day after tomorrow":
            resolved = reference_date + timedelta(days=2)
        else:
            resolved = None

        old_target = query_params.target_date
        query_params.target_date = resolved

        _log.info(
            "Temporal resolution: query=%r  mode=%s  reference_date=%s  "
            "token=%r  resolved_target_date=%s (was %s)",
            raw,
            mode,
            reference_date,
            token,
            resolved,
            old_target,
        )

    # --- Duration / horizon extraction -----------------------------------
    if query_params.horizon_days is None:
        week_match = _HORIZON_WEEK_RE.search(normalised)
        days_match = _HORIZON_DAYS_RE.search(normalised)
        if week_match:
            query_params.horizon_days = 7
        elif days_match:
            query_params.horizon_days = int(days_match.group(1))

    # --- Focus window (informational) ------------------------------------
    if query_params.target_date is not None and query_params.horizon_days is not None:
        end = query_params.target_date + timedelta(days=query_params.horizon_days - 1)
        query_params.focus_window = (
            f"{query_params.target_date.isoformat()} to {end.isoformat()}"
        )
    elif query_params.target_date is not None and query_params.focus_window is None:
        query_params.focus_window = query_params.target_date.isoformat()

    return query_params


# ---------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------


def initial_state(
    bundle: SignalBundle, query_params: QueryParams, query: str | None = None
) -> AgentState:
    """Wire the fixed inputs into the state. The raw user query (if given) is
    carried as a HumanMessage — some providers (e.g. Gemini) reject empty
    conversations."""
    messages = [SystemMessage(content=RECOMMEND_SYSTEM_PROMPT)]
    if query:
        messages.append(HumanMessage(content=query))
    return {
        "messages": messages,
        "bundle": bundle,
        "query_params": query_params,
        "draft_recommendation": None,
        "recommend_attempts": 0,
        "validation_problems": [],
        "signal_conflict_detected": False,
        "final_output": None,
    }


# ---------------------------------------------------------------------
# Evidence formatting (deterministic, no LLM)
# ---------------------------------------------------------------------


def _format_evidence(bundle: SignalBundle) -> str:
    """Format all four evidence categories from the bundle into a single
    structured string for the recommend node.  Replaces the ReAct
    tool-calling loop — all evidence is always present by construction."""
    sections = [
        ("## Vegetation", bundle.vegetation),
        ("## Weather Forecast", bundle.weather_forecast),
        ("## Soil Moisture Forecast", bundle.soil_moisture_forecast),
        ("## Agronomic Thresholds", bundle.thresholds),
    ]
    parts: list[str] = []
    for header, data in sections:
        parts.append(header)
        parts.append(json.dumps(data, indent=2, default=str))
    return "\n\n".join(parts)


def inject_evidence_node(state: AgentState) -> dict:
    """Deterministic evidence injection.  Formats all four sub-bundles from
    the frozen SignalBundle into a single HumanMessage.  No LLM call."""
    evidence = _format_evidence(state["bundle"])
    return {
        "messages": [
            HumanMessage(
                content=(
                    "All pre-computed evidence for this field is below.  "
                    "Use it to produce a defensible FusionRecommendation.\n\n"
                    + evidence
                )
            )
        ]
    }


# ---------------------------------------------------------------------
# STEP 4 — graph nodes
# ---------------------------------------------------------------------


def build_graph(bundle: SignalBundle, llm=None):
    """
    Wire inject_evidence -> recommend -> validate (-> END | retry recommend).
    ``bundle`` is captured once and never mutated.
    """
    if llm is None:
        raise RuntimeError(
            "No LLM configured: set OPENAI_API_KEY (or AGRI_LLM_PROVIDER plus "
            "its key) in .env to run the agent."
        )

    recommend_model = llm.with_structured_output(FusionRecommendation)

    def recommend_node(state: AgentState) -> dict:
        """Structured output over the accumulated messages.  No tools bound.
        data_sources_used is set deterministically — all four categories are
        always present by construction."""
        messages = list(state["messages"])
        problems = state.get("validation_problems") or []
        if problems:
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous recommendation had these problems — fix "
                        "them in your next output:\n"
                        + "\n".join(f"- {p}" for p in problems)
                    )
                )
            )
        messages.append(HumanMessage(content=RECOMMEND_INSTRUCTION))

        rec = recommend_model.invoke(messages)
        if isinstance(rec, dict):
            rec = FusionRecommendation(**rec)

        # Ground truth wins over anything the model printed. field_id/date
        # come from the bundle, never from generation.
        rec.field_id = bundle.field_id
        rec.date = str(bundle.origin_date.date())
        rec.data_sources_used = [
            "vegetation",
            "weather_forecast",
            "soil_moisture_forecast",
            "thresholds",
        ]
        return {
            "draft_recommendation": rec,
            "recommend_attempts": state["recommend_attempts"] + 1,
        }

    def validate_node(state: AgentState) -> dict:
        """Deterministic grounding + conflict checks. No LLM call."""
        rec = state["draft_recommendation"]
        problems, conflict = validate_recommendation(rec, bundle)
        exhausted = state["recommend_attempts"] >= MAX_RECOMMEND_ATTEMPTS
        flag = conflict or (exhausted and bool(problems))

        out = rec.model_dump(mode="json")
        out["validation_problems"] = list(problems)
        out["signal_conflict_detected"] = flag
        return {
            "validation_problems": problems,
            "signal_conflict_detected": flag,
            "final_output": out,
        }

    def should_continue(state: AgentState) -> str:
        if not state["validation_problems"]:
            return "end"
        if state["recommend_attempts"] < MAX_RECOMMEND_ATTEMPTS:
            return "retry"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("inject_evidence", inject_evidence_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "inject_evidence")
    graph.add_edge("inject_evidence", "recommend")
    graph.add_edge("recommend", "validate")
    graph.add_conditional_edges(
        "validate", should_continue, {"retry": "recommend", "end": END}
    )
    return graph.compile()
