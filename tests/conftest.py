"""Shared fixtures for the rebuilt fusion agent tests. No real LLM, no
network, no Earth Engine, no Chronos-2 model loads."""

from datetime import date

import pandas as pd
import pytest

from agri_agent.agent.bundle import SignalBundle
from agri_agent.agent.schemas import QueryParams


def bundle_sig(name: str, value=0.0) -> dict:
    """Signal in the SHAPE the real builders produce: a plain dict."""
    return {
        "signal_name": name,
        "value": value,
        "reference": f"ref for {name}",
        "interpretation": f"interpretation for {name}",
    }


def _forecast_rows(n: int = 7, precip: float = 0.0) -> list[dict]:
    return [
        {
            "date": f"2026-07-{15 + i:02d}",
            "precipitation_sum": precip,
            "et0_fao_evapotranspiration": 6.0,
            "temperature_2m_max": 32.0,
            "temperature_2m_min": 18.0,
            "windspeed_10m_max": 12.0,
            "shortwave_radiation_sum": 25.0,
        }
        for i in range(n)
    ]


def make_vegetation(ndvi_z: float = 0.0) -> dict:
    return {
        "as_of": "2026-07-22",
        "latest_scene_date": "2026-07-22",
        "days_since_last_scene": 0,
        "indices": {
            "NDVI": {"value": 0.5, "is_interpolated": False},
            "EVI": {"value": 0.4, "is_interpolated": False},
            "NDWI": {"value": 0.2, "is_interpolated": False},
        },
        "ndvi_anomaly": {
            "target_date": "2026-07-22",
            "ndvi_value": 0.5,
            "baseline_mean": 0.5,
            "baseline_std": 0.05,
            "z_score": ndvi_z,
            "percentile": 50.0,
            "n_obs": 30,
            "window_days": 30,
            "season": "summer",
            "anomaly": abs(ndvi_z) >= 2.0,
            "insufficient_data": False,
        },
        "signals": [
            bundle_sig("NDVI", 0.5),
            bundle_sig("EVI", 0.4),
            bundle_sig("NDWI", 0.2),
            bundle_sig("ndvi_anomaly_z", ndvi_z),
        ],
        "insufficient_data": False,
    }


def make_weather(precip: float = 0.0, horizon: int = 7) -> dict:
    total = round(precip * horizon, 2)
    signals = [
        bundle_sig(f"precipitation_total_next_{horizon}d", total),
        bundle_sig(f"et0_total_next_{horizon}d", round(6.0 * horizon, 2)),
    ]
    if horizon == 7:
        signals.append(bundle_sig("max_temperature_next_7d", 32.0))
    return {
        "as_of": "2026-07-22",
        "horizon_days": horizon,
        "days_available": horizon,
        "forecast": _forecast_rows(horizon, precip),
        "signals": signals,
        "insufficient_data": False,
    }


def make_soil_moisture(trend: str = "stable") -> dict:
    return {
        "as_of": "2026-07-22",
        "horizon_days": 7,
        "forecast_origin_date": "2026-07-22",
        "last_observed": {"date": "2026-07-21", "moisture": 0.17, "iot_valid_hours": 24},
        "quantiles": [],
        "uncertainty_width_max": 0.02,
        "trend": trend,
        "signals": [
            bundle_sig("soil_moisture_p50_min", 0.15),
            bundle_sig("soil_moisture_p50_on_last_day", 0.16),
            bundle_sig("soil_moisture_uncertainty_width_max", 0.02),
            bundle_sig("soil_moisture_trend", trend),
            bundle_sig("soil_moisture_last_observed", 0.17),
        ],
        "insufficient_data": False,
    }


def make_thresholds() -> dict:
    return {
        "crop_type": "wheat",
        "growth_stage": "establishment",
        "generic_default_used": False,
        "source": "FAO Paper 56 (Allen et al., 1998).",
        "soil": {"texture": "loam"},
        "field_capacity": 0.30,
        "wilting_point": 0.12,
        "target_range": [0.20, 0.30],
        "vigor_zscore_bands": {"normal": [-1.0, 1.0], "watch": [-2.0, -1.0], "anomaly": [-999.0, -2.0]},
        "mad_fraction": 0.50,
        "trigger": 0.21,
        "signals": [
            bundle_sig("irrigation_trigger", 0.21),
            bundle_sig("wilting_point", 0.12),
            bundle_sig("field_capacity", 0.30),
            bundle_sig("target_moisture_range", "[0.2, 0.3]"),
        ],
    }


def make_bundle(
    ndvi_z: float = 0.0,
    precip: float = 0.0,
    trend: str = "stable",
    field_id: str = "field_merguellil_01",
    origin_date: str = "2026-07-22",
) -> SignalBundle:
    return SignalBundle(
        field_id=field_id,
        origin_date=pd.Timestamp(origin_date),
        query_params=QueryParams(),
        vegetation=make_vegetation(ndvi_z),
        weather_forecast=make_weather(precip),
        soil_moisture_forecast=make_soil_moisture(trend),
        thresholds=make_thresholds(),
        load_errors={},
    )


@pytest.fixture
def bundle() -> SignalBundle:
    return make_bundle()


@pytest.fixture
def query_params() -> QueryParams:
    return QueryParams(target_date=date(2026, 7, 22))
