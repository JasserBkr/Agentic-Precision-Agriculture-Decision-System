"""
Week 5 deliverable: the LangGraph state schema — the object carried
through the agent's reasoning loop (see the react_agent_loop diagram
discussed with Claude, and SOTA note Section 4.3).

Not yet fully implemented — the shape below is a starting point.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    field_id: str

    # Populated by the deterministic pipeline BEFORE the agent runs —
    # the agent does not decide whether to compute these, only reasons
    # over the result.
    ndvi_evi_ndwi: dict
    weather_forecast: dict
    forecast_bundle: dict  # Chronos-2 + baseline output, see forecasting/

    # Populated/updated during the agent's own loop.
    messages: Annotated[list, add_messages]  # LangGraph conversation history
    conflict_detected: bool
    reasoning_trace: list[str]  # append-only log for the defensibility requirement
    recommendation: dict | None
