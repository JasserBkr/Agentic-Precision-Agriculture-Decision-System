"""Precision-agriculture decision agent.

Pipeline: parse_query -> PREP (build_signal_bundle) -> LangGraph
(inject_evidence -> recommend -> validate) -> printed recommendation.
All data loading, forecasting, and anomaly computation happens ONCE in
PREP; evidence is formatted deterministically before the LLM reasons.
"""

from agri_agent.agent.anomaly import ndvi_seasonal_anomaly
from agri_agent.agent.bundle import (
    SignalBundle,
    build_signal_bundle,
    load_agronomic_thresholds,
)
from agri_agent.agent.graph import build_graph, get_llm, initial_state, parse_query
from agri_agent.agent.schemas import (
    FertilizationRecommendation,
    FusionRecommendation,
    IrrigationRecommendation,
    QueryParams,
    SignalContribution,
)
from agri_agent.agent.validator import validate_recommendation

__all__ = [
    "SignalBundle",
    "build_signal_bundle",
    "load_agronomic_thresholds",
    "ndvi_seasonal_anomaly",
    "build_graph",
    "get_llm",
    "initial_state",
    "parse_query",
    "QueryParams",
    "SignalContribution",
    "IrrigationRecommendation",
    "FertilizationRecommendation",
    "FusionRecommendation",
    "validate_recommendation",
]
