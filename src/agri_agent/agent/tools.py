"""
Week 5 deliverable: the tool functions the agent can call from within
its reasoning loop — reserved for FOLLOW-UP checks only (re-query a
wider weather window, re-check data quality), not for the primary
retrieval/forecasting pipeline, which runs deterministically before the
agent is ever invoked (SOTA note Section 4.3).

Not yet implemented.
"""

from langchain_core.tools import tool


@tool
def requery_wider_weather_window(field_id: str, extra_days: int) -> dict:
    """
    TODO (Week 5): call data_access.weather.get_forecast again with a
    longer forecast_days window, for use when the agent's initial
    forecast horizon proves insufficient to resolve a borderline case.
    """
    raise NotImplementedError


@tool
def check_data_quality(field_id: str) -> dict:
    """
    TODO (Week 5): inspect recent satellite/IoT data for known quality
    issues (high cloud cover, sensor dropout) that might explain a
    conflict between signals, per SOTA note Section 4.3.
    """
    raise NotImplementedError


# TODO: assemble these into a list and bind them to the LLM in graph.py,
# e.g. via `llm.bind_tools([requery_wider_weather_window, check_data_quality])`.
