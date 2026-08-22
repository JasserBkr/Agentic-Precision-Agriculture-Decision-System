"""Structured schemas for the Step 3 fusion agent (rebuild).

Pydantic models so they can be used directly as langchain structured-output
schemas (parse_query and the recommend node) and round-tripped to JSON for
the final printed output. Dataclasses were rejected because langchain's
``with_structured_output`` does not accept them natively.
"""

from typing import Literal, get_args

from datetime import date

from pydantic import BaseModel, Field

# The action vocabulary the recommend node is constrained to produce. The
# validator (agent/validator.py) checks against these exact strings. The
# Literal types are the single source of truth — Pydantic enforces them on
# every structured-output parse AND embeds them as an enum in the JSON
# schema shown to the LLM, so off-vocabulary actions are structurally
# impossible rather than silently passing validation.
IrrigationAction = Literal["irrigate_now", "irrigate_soon", "no_action_needed"]
FertilizationAction = Literal["apply_fertilizer", "no_application"]
IRRIGATION_ACTIONS = get_args(IrrigationAction)
FERTILIZATION_ACTIONS = get_args(FertilizationAction)


class QueryParams(BaseModel):
    """
    The single source of truth for the parsed query. Built exactly once in
    run_pipeline.py (parse_query + CLI overrides merged into this ONE
    object); nothing downstream re-parses or independently overrides these
    fields.

    ``target_date`` may be None — that is the NORMAL case, not an error
    (the recommendation then falls back to the default origin date).

    ``raw_query`` carries the original user text so that relative temporal
    expressions (e.g. "tomorrow", "today") can be resolved *after* the
    reference date is known (today in live mode, dataset-last-date in
    offline mode) — rather than letting the LLM silently invent the
    reference date during parse_query.
    """

    field_id: str | None = None
    target_date: date | None = None
    crop_type: str | None = None
    growth_stage: str | None = None
    focus_window: str | None = None
    raw_query: str | None = None
    horizon_days: int | None = None


class SignalContribution(BaseModel):
    """
    One piece of evidence the recommendation cites. ``signal_name`` must
    match a signal actually present in the SignalBundle — the validator's
    grounding check enforces this programmatically.
    """

    signal_name: str
    value: float | str | None = None
    reference: str = ""
    interpretation: str = ""


class IrrigationRecommendation(BaseModel):
    action: IrrigationAction = Field(description=f"One of: {', '.join(IRRIGATION_ACTIONS)}")
    confidence: float = Field(ge=0.0, le=1.0)
    contributing_signals: list[SignalContribution] = Field(default_factory=list)
    reasoning: str = ""


class FertilizationRecommendation(BaseModel):
    action: FertilizationAction = Field(description=f"One of: {', '.join(FERTILIZATION_ACTIONS)}")
    confidence: float = Field(ge=0.0, le=1.0)
    contributing_signals: list[SignalContribution] = Field(default_factory=list)
    reasoning: str = ""
    caveat: str = ""


class FusionRecommendation(BaseModel):
    field_id: str
    date: str
    focus_window: str
    irrigation: IrrigationRecommendation
    fertilization: FertilizationRecommendation
    # Overwritten programmatically in the graph's recommend node from the
    # actual ToolMessage history — never trusted from the model's output.
    data_sources_used: list[str] = Field(default_factory=list)
