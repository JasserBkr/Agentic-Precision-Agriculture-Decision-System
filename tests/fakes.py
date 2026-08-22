"""Shared test fakes for the fusion agent. No real LLM, no network,
no Earth Engine."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agri_agent.agent.schemas import (
    FertilizationRecommendation,
    FusionRecommendation,
    IrrigationRecommendation,
    SignalContribution,
)

# Signal names the canonical test bundle carries (see conftest.py). The
# validator's grounding check must accept exactly these.
VALID_SIGNALS = [
    "NDVI",
    "EVI",
    "NDWI",
    "ndvi_anomaly_z",
    "precipitation_total_next_7d",
    "et0_total_next_7d",
    "max_temperature_next_7d",
    "soil_moisture_p50_min",
    "soil_moisture_p50_on_last_day",
    "soil_moisture_uncertainty_width_max",
    "soil_moisture_trend",
    "soil_moisture_last_observed",
    "irrigation_trigger",
    "wilting_point",
    "field_capacity",
    "target_moisture_range",
]


def sig(name: str, value=0.0) -> SignalContribution:
    return SignalContribution(
        signal_name=name,
        value=value,
        reference=f"ref for {name}",
        interpretation=f"interpretation for {name}",
    )


def make_rec(
    irr_action: str = "no_action_needed",
    fert_action: str = "no_application",
    irr_signals: list[SignalContribution] | None = None,
    fert_signals: list[SignalContribution] | None = None,
    field_id: str = "field_merguellil_01",
    date: str = "2026-07-22",
    irr_confidence: float = 0.8,
    fert_confidence: float = 0.7,
) -> FusionRecommendation:
    if irr_signals is None:
        irr_signals = [sig("NDVI"), sig("ndvi_anomaly_z"), sig("soil_moisture_p50_min"), sig("irrigation_trigger")]
    if fert_signals is None:
        fert_signals = [sig("NDVI"), sig("ndvi_anomaly_z"), sig("soil_moisture_p50_min"), sig("irrigation_trigger")]
    return FusionRecommendation(
        field_id=field_id,
        date=date,
        focus_window="next_7",
        irrigation=IrrigationRecommendation(
            action=irr_action,
            confidence=irr_confidence,
            contributing_signals=irr_signals,
            reasoning="Test reasoning.",
        ),
        fertilization=_fertilization(fert_action, fert_signals, confidence=fert_confidence),
        data_sources_used=[],
    )


def _fertilization(
    fert_action: str, fert_signals: list[SignalContribution], confidence: float = 0.7
) -> FertilizationRecommendation:
    return FertilizationRecommendation(
        action=fert_action,
        confidence=confidence,
        contributing_signals=fert_signals,
        reasoning="Test reasoning.",
        caveat="Test caveat.",
    )


def make_synthetic_fused(n_days: int = 40, start: str = "2026-06-01") -> pd.DataFrame:
    """Daily fused-style DataFrame with every column the bundle builders
    touch. NDVI/EVI/NDWI present every day so the anomaly baseline is rich."""
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    ndvi = 0.30 + 0.02 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"date": dates, "field_id": "test_field"})
    df["NDVI"] = ndvi
    df["EVI"] = ndvi * 0.8
    df["NDWI"] = ndvi - 0.3
    df["GNDVI"] = ndvi * 1.1
    df["SAVI"] = ndvi * 0.7
    df["temperature_2m_max"] = 32.0
    df["temperature_2m_min"] = 18.0
    df["precipitation_sum"] = 0.0
    df["et0_fao_evapotranspiration"] = 6.0
    df["windspeed_10m_max"] = 12.0
    df["shortwave_radiation_sum"] = 25.0
    df["weather_soil_moisture_0_to_1cm_mean"] = 0.17
    df["weather_soil_moisture_1_to_3cm_mean"] = 0.18
    df["weather_soil_moisture_3_to_9cm_mean"] = 0.20
    df["weather_soil_temperature_0cm_mean"] = 28.0
    df["iot_soil_moisture_mean"] = 0.17 + 0.01 * np.sin(np.linspace(0, 3, n))
    df["iot_valid_hours"] = 24
    df["days_since_last_scene"] = 0
    for col in ["ndvi", "evi", "ndwi", "gndvi", "savi"]:
        df[f"is_interpolated_{col}"] = False
    return df


@dataclass
class FakeLLM:
    """Replays scripted responses for with_structured_output calls.

    Each ``invoke`` call pops the next entry from ``script``.  Entries can be
    plain return values or callables ``(messages) -> value``.

    ``bind_tools`` is a no-op (tools are no longer used in the graph).

    ``structured_script`` is an optional legacy parameter: when provided it
    is prepended to ``script`` so that existing callers that construct a
    FakeLLM with ``structured_script=[parse_query_result, make_rec()]``
    continue to work without modification.
    """

    script: list = field(default_factory=list)
    structured_script: list | None = None

    def __post_init__(self):
        if self.structured_script is not None:
            self.script = list(self.structured_script) + list(self.script)

    def bind_tools(self, tools: list):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages: list):
        if not self.script:
            raise AssertionError("FakeLLM script exhausted — graph asked for more model calls.")
        entry = self.script.pop(0)
        if callable(entry):
            return entry(messages)
        return entry
