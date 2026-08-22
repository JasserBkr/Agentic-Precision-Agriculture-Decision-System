"""SignalBundle + the deterministic PREP phase (STEP 2a) and the agronomic
thresholds YAML loader (STEP 2c).

This is the ONLY place that fetches raw data, runs Chronos-2, computes the
NDVI anomaly, and loads thresholds. It runs ONCE, before the graph starts.
Everything downstream (tools, graph, validator) works over the already-
validated bundle and cannot trigger a data fetch or a forecast.

Key design rule: no sub-step is allowed to crash the bundle. Each of the
four sub-builders is wrapped individually; a failure degrades THAT field to
``{"insufficient_data": True, "signals": [], "reason": ...}`` and is
recorded in ``load_errors``. build_signal_bundle raises only on a fully
unrecoverable setup error (missing field config, zero history).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from agri_agent.agent.anomaly import ndvi_seasonal_anomaly
from agri_agent.agent.schemas import QueryParams
from agri_agent.data_access.fusion import (
    build_fused_dataset,
    load_fused_dataset,
    weather_response_to_daily_df,
)
from agri_agent.data_access.iot import _perturb_soil_moisture
from agri_agent.data_access.satellite import get_field_index_timeseries
from agri_agent.data_access.weather import (
    DEFAULT_DAILY_VARS,
    DEFAULT_HOURLY_VARS,
    get_forecast,
    get_historical_forecast,
)
from agri_agent.forecasting.chronos_model import forecast_soil_moisture
from agri_agent.forecasting.data_prep import FUTURE_KNOWN_COLS, TARGET_COL, to_chronos_context_df
from agri_agent.utils.auth import init_earth_engine
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS_PATH = ROOT / "configs" / "thresholds.yaml"
FUSED_PARQUET = ROOT / "data" / "processed" / "fused_2years.parquet"

DEFAULT_CROP = "wheat"
FORECAST_HORIZON_DAYS = 7
DEFAULT_CONTEXT_DAYS = 730
CHUNK_DAYS = 90

# Signal shape used across every sub-bundle's "signals" list:
#   {"signal_name": str, "value": float | str | None,
#    "reference": str, "interpretation": str}
_SUB_BUNDLES = ("vegetation", "weather_forecast", "soil_moisture_forecast", "thresholds")


@dataclass(frozen=True)
class SignalBundle:
    """
    Everything PREP hands the agent. Fully built and frozen BEFORE the graph
    is invoked; the graph never mutates it. Each sub-field is a dict with a
    ``signals`` list plus sub-field-specific keys, and carries
    ``insufficient_data``/``reason`` when that sub-step degraded.
    """

    field_id: str
    origin_date: pd.Timestamp
    query_params: QueryParams
    vegetation: dict
    weather_forecast: dict
    soil_moisture_forecast: dict
    thresholds: dict
    load_errors: dict[str, str]


# ---------------------------------------------------------------------
# Thresholds loader (STEP 2c) — lives here, used only by PREP
# ---------------------------------------------------------------------


def load_agronomic_thresholds(
    crop_type: str | None = None,
    growth_stage: str | None = None,
    path: Path | None = None,
) -> dict:
    """
    Load thresholds for a crop (default wheat), falling back to the default
    crop/growth stage when the requested key is missing and flagging the
    substitution via ``generic_default_used`` so the recommendation's
    explanation can surface it.

    ``trigger`` is COMPUTED here as FC - MAD*(FC-WP) rather than read from
    YAML, so the stored values and the "trigger = ..." reference can never
    silently drift apart. Returns the threshold dict WITH a ``signals`` list
    (irrigation_trigger, wilting_point, field_capacity, target_moisture_range)
    so the validator can ground against it like any other sub-bundle.
    """
    import yaml

    path = path or THRESHOLDS_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    crops = raw["crops"]
    requested_crop = crop_type or DEFAULT_CROP
    crop = crops.get(requested_crop, crops[DEFAULT_CROP])

    stages = crop["growth_stages"]
    default_stage = next(iter(stages))
    stage_key = (growth_stage or default_stage).lower().replace("-", "_")
    used_stage = stage_key if stage_key in stages else default_stage

    generic = (crop_type is not None and crop_type not in crops) or (
        growth_stage is not None and stage_key not in stages
    )

    field_capacity = float(crop["water"]["field_capacity"])
    wilting_point = float(crop["water"]["wilting_point"])
    mad_fraction = float(stages[used_stage]["mad_fraction"])
    target_range = [float(v) for v in crop["irrigation"]["target_range"]]

    trigger = field_capacity - mad_fraction * (field_capacity - wilting_point)

    signals = [
        {
            "signal_name": "irrigation_trigger",
            "value": trigger,
            "reference": f"trigger = FC - MAD*(FC-WP) = {trigger:.3f} m3/m3",
            "interpretation": "Below this, irrigation should start.",
        },
        {
            "signal_name": "wilting_point",
            "value": wilting_point,
            "reference": f"{wilting_point:.3f} m3/m3",
            "interpretation": "Below this the crop cannot extract water.",
        },
        {
            "signal_name": "field_capacity",
            "value": field_capacity,
            "reference": f"{field_capacity:.3f} m3/m3",
            "interpretation": "Above this risks over-irrigation.",
        },
        {
            "signal_name": "target_moisture_range",
            "value": json.dumps(target_range),
            "reference": f"target range {target_range[0]:.3f}..{target_range[1]:.3f} m3/m3",
            "interpretation": "Hold soil moisture in this band.",
        },
    ]
    if generic:
        effective_crop = requested_crop if requested_crop in crops else DEFAULT_CROP
        signals.append(
            {
                "signal_name": "generic_default_used",
                "value": True,
                "reference": (
                    f"requested crop/stage '{requested_crop}'/'{stage_key}' not in the "
                    f"thresholds table; substituted '{effective_crop}' stage '{used_stage}'"
                ),
                "interpretation": (
                    "Thresholds are a generic default, not specific to the requested "
                    "crop/growth stage — treat them as approximate."
                ),
            }
        )

    out = {
        "crop_type": requested_crop,
        "growth_stage": used_stage,
        "generic_default_used": bool(generic),
        "source": crop["source"],
        "soil": crop["soil"],
        "field_capacity": field_capacity,
        "wilting_point": wilting_point,
        "target_range": target_range,
        "vigor_zscore_bands": crop["vigor_zscore_bands"],
        "mad_fraction": mad_fraction,
        "trigger": trigger,
        "signals": signals,
    }
    return out


# ---------------------------------------------------------------------
# Origin / horizon resolution
# ---------------------------------------------------------------------


def _today() -> pd.Timestamp:
    return pd.Timestamp(date.today()).normalize()


def _resolve_live_origin_and_horizon(query_params: QueryParams) -> tuple[pd.Timestamp, int]:
    """
    Live mode always forecasts forward from TODAY (you cannot observe
    weather beyond the present forecast), so origin = today. A target_date
    in the future sizes the forward horizon to reach it:
    horizon = max(FORECAST_HORIZON_DAYS, target_date - today). A past
    target_date is unrecoverable in live mode — there is no forward window.
    """
    today = _today()
    target = (
        pd.Timestamp(query_params.target_date).normalize()
        if query_params.target_date is not None
        else None
    )
    if target is not None and target < today:
        raise ValueError(
            "live mode forecasts forward from today; a past target_date has no "
            "forward weather window. Use --mode offline for point-in-time questions."
        )
    origin = today
    horizon = FORECAST_HORIZON_DAYS
    if target is not None and target > today:
        horizon = max(FORECAST_HORIZON_DAYS, (target - today).days)
    return origin, horizon


def _resolve_offline_origin_and_horizon(
    query_params: QueryParams, fused: pd.DataFrame
) -> tuple[pd.Timestamp, int]:
    """Offline origin: query_params.target_date if set, else
    (last parquet date - FORECAST_HORIZON_DAYS) so the DEFAULT command
    always leaves a usable forward weather window out of the box. Horizon
    is always the default. An empty parquet is a hard setup failure, not a
    degradable sub-step. Passing --target-date explicitly (including the
    true last date, reproducing the zero-forward-window case on purpose)
    still resolves exactly as asked."""
    if len(fused) == 0:
        raise ValueError("zero history rows available — cannot build a signal bundle")
    if query_params.target_date is not None:
        origin = pd.Timestamp(query_params.target_date).normalize()
    else:
        origin = pd.Timestamp(fused["date"].max()).normalize() - pd.Timedelta(
            days=FORECAST_HORIZON_DAYS
        )
    return origin, FORECAST_HORIZON_DAYS


# ---------------------------------------------------------------------
# Data loading — the ONLY fetches/reads in the whole agent package
# ---------------------------------------------------------------------


def _chunked_historical_weather(
    field_config: dict, start_date: date, end_date: date
) -> dict:
    """
    Fetch historical weather in CHUNK_DAYS-sized windows and concatenate into
    a single response-shaped dict. Fetched ONCE; the same dict feeds both the
    fused dataset's weather columns and the IoT synthesis below (the previous
    version's biggest efficiency bug fetched the same chunks twice).
    """
    lat = field_config["centroid"]["lat"]
    lon = field_config["centroid"]["lon"]
    all_daily_time, all_daily_data = [], {v: [] for v in DEFAULT_DAILY_VARS}
    all_hourly_time, all_hourly_data = [], {v: [] for v in DEFAULT_HOURLY_VARS}

    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunk = get_historical_forecast(lat, lon, chunk_start, chunk_end)
        all_daily_time.extend(chunk["daily"]["time"])
        for v in DEFAULT_DAILY_VARS:
            all_daily_data[v].extend(chunk["daily"][v])
        all_hourly_time.extend(chunk["hourly"]["time"])
        for v in DEFAULT_HOURLY_VARS:
            all_hourly_data[v].extend(chunk["hourly"][v])
        chunk_start = chunk_end + timedelta(days=1)

    return {
        "daily": {"time": all_daily_time, **all_daily_data},
        "hourly": {"time": all_hourly_time, **all_hourly_data},
    }


def _fetch_live_history(field_config: dict, origin: pd.Timestamp) -> pd.DataFrame:
    """Fresh 2-year fetch as-of origin (mirrors backfill_history.py)."""
    init_earth_engine()
    start = (origin - pd.Timedelta(days=DEFAULT_CONTEXT_DAYS)).date()
    end = origin.date()

    index_records = get_field_index_timeseries(
        bbox=field_config["bbox"],
        start_date=start,
        end_date=end,
        max_cloud_cover_pct=field_config.get("max_cloud_cover_pct", 20),
    )

    weather = _chunked_historical_weather(field_config, start, end)
    rng = np.random.default_rng(seed=42)
    iot = _perturb_soil_moisture(weather, noise_std=0.01, dropout_prob=0.05, rng=rng)

    fused = build_fused_dataset(
        satellite_records=index_records,
        weather_json=weather,
        iot_dict=iot,
        field_id=field_config["field_id"],
        start_date=start,
        end_date=end,
    )
    return fused.sort_values("date").reset_index(drop=True)


def _fetch_live_forward(field_config: dict, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Forward daily weather from the live API, fetched once and reused for
    both the weather bundle and the Chronos-2 future_df."""
    lat = field_config["centroid"]["lat"]
    lon = field_config["centroid"]["lon"]
    weather_json = get_forecast(lat=lat, lon=lon, forecast_days=horizon + 1, past_days=0)
    daily = weather_response_to_daily_df(weather_json)
    return daily.sort_values("date").reset_index(drop=True)


def _load_offline_fused() -> pd.DataFrame:
    return load_fused_dataset(str(FUSED_PARQUET))


# ---------------------------------------------------------------------
# Sub-bundle builders
# ---------------------------------------------------------------------


def build_future_df_from_rows(
    field_id: str, rows: pd.DataFrame, horizon_days: int = FORECAST_HORIZON_DAYS
) -> pd.DataFrame:
    """Shape fused rows beyond the origin into a Chronos-2 future_df
    (id, timestamp, FUTURE_KNOWN_COLS). Pure reshaping — no fetching."""
    cols = ["date"] + FUTURE_KNOWN_COLS
    rows = rows[cols].copy().sort_values("date").head(horizon_days)
    future_df = rows.rename(columns={"date": "timestamp"}).copy()
    future_df.insert(0, "id", field_id)
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
    return future_df.reset_index(drop=True)


def _forward_rows(forward_df: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    if forward_df is None or len(forward_df) == 0:
        return pd.DataFrame(columns=["date"])
    return (
        forward_df[forward_df["date"] > origin]
        .sort_values("date")
        .head(horizon)
        .reset_index(drop=True)
    )


def _build_vegetation_bundle(fused_history: pd.DataFrame, origin: pd.Timestamp) -> dict:
    df = fused_history.sort_values("date").reset_index(drop=True)
    observed = df[df["NDVI"].notna()]
    if len(observed) == 0:
        return {
            "as_of": origin.date().isoformat(),
            "latest_scene_date": None,
            "days_since_last_scene": None,
            "indices": {},
            "ndvi_anomaly": None,
            "signals": [],
            "insufficient_data": True,
            "reason": "no NDVI observations in the history window",
        }

    row = observed.iloc[-1]
    scene_date = pd.Timestamp(row["date"]).normalize()
    days_since = (origin - scene_date).days

    indices = {}
    for col in ("NDVI", "EVI", "NDWI"):
        if col in df.columns and pd.notna(row.get(col)):
            indices[col] = {
                "value": float(row[col]),
                "is_interpolated": bool(row.get(f"is_interpolated_{col.lower()}", False)),
            }

    anomaly = ndvi_seasonal_anomaly(observed["date"], observed["NDVI"], scene_date)

    signals = []
    for col in ("NDVI", "EVI", "NDWI"):
        if col in indices:
            signals.append(
                {
                    "signal_name": col,
                    "value": indices[col]["value"],
                    "reference": (
                        f"{indices[col]['value']:.3f} on {scene_date.date().isoformat()} "
                        "(field-mean Sentinel-2)"
                    ),
                    "interpretation": (
                        f"Field-mean {col}; "
                        f"{'interpolated across a gap' if indices[col]['is_interpolated'] else 'observed'}."
                    ),
                }
            )
    if anomaly.get("z_score") is not None:
        z = anomaly["z_score"]
        band = "anomaly" if abs(z) >= 2.0 else ("watch" if abs(z) >= 1.0 else "normal")
        signals.append(
            {
                "signal_name": "ndvi_anomaly_z",
                "value": z,
                "reference": (
                    f"z={z:.2f} vs field day-of-year climatology "
                    f"(mean {anomaly['baseline_mean']:.3f}, std "
                    f"{anomaly['baseline_std']:.3f}, {anomaly['n_obs']} obs, {anomaly['season']})"
                ),
                "interpretation": f"NDVI anomaly band: {band}.",
            }
        )

    return {
        "as_of": origin.date().isoformat(),
        "latest_scene_date": scene_date.date().isoformat(),
        "days_since_last_scene": int(days_since),
        "indices": indices,
        "ndvi_anomaly": anomaly,
        "signals": signals,
        "insufficient_data": False,
    }


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _build_weather_bundle(
    forward_df: pd.DataFrame, origin: pd.Timestamp, horizon: int
) -> dict:
    rows = _forward_rows(forward_df, origin, horizon)
    if len(rows) == 0:
        return {
            "as_of": origin.date().isoformat(),
            "horizon_days": horizon,
            "forecast": [],
            "signals": [],
            "insufficient_data": True,
            "reason": "no forward weather window after the origin date",
        }

    forecast = [
        {
            "date": r["date"].date().isoformat(),
            "precipitation_sum": _num(r.get("precipitation_sum")),
            "et0_fao_evapotranspiration": _num(r.get("et0_fao_evapotranspiration")),
            "temperature_2m_max": _num(r.get("temperature_2m_max")),
            "temperature_2m_min": _num(r.get("temperature_2m_min")),
            "windspeed_10m_max": _num(r.get("windspeed_10m_max")),
            "shortwave_radiation_sum": _num(r.get("shortwave_radiation_sum")),
        }
        for _, r in rows.iterrows()
    ]

    precip = sum(p for p in (x["precipitation_sum"] for x in forecast) if p is not None)
    et0 = sum(e for e in (x["et0_fao_evapotranspiration"] for x in forecast) if e is not None)
    tmax = max(
        (x["temperature_2m_max"] for x in forecast if x["temperature_2m_max"] is not None),
        default=None,
    )

    signals = [
        {
            "signal_name": f"precipitation_total_next_{horizon}d",
            "value": round(precip, 2),
            "reference": f"{precip:.1f} mm over the next {horizon} days",
            "interpretation": "Expected rainfall; >=5mm offsets some irrigation demand.",
        },
        {
            "signal_name": f"et0_total_next_{horizon}d",
            "value": round(et0, 2),
            "reference": f"{et0:.1f} mm ET0 over the next {horizon} days",
            "interpretation": "Atmospheric water demand driving soil dry-down.",
        },
    ]
    if tmax is not None:
        signals.append(
            {
                "signal_name": f"max_temperature_next_{horizon}d",
                "value": tmax,
                "reference": f"{tmax:.1f} C max over the next {horizon} days",
                "interpretation": "High maxima raise crop water stress.",
            }
        )

    return {
        "as_of": origin.date().isoformat(),
        "horizon_days": horizon,
        "days_available": len(rows),
        "forecast": forecast,
        "signals": signals,
        "insufficient_data": False,
    }


def _build_soil_moisture_bundle(
    fused_history: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    field_id: str,
    forward_df: pd.DataFrame,
) -> dict:
    rows = _forward_rows(forward_df, origin, horizon)
    if len(rows) < horizon:
        return {
            "as_of": origin.date().isoformat(),
            "horizon_days": horizon,
            "quantiles": [],
            "signals": [],
            "insufficient_data": True,
            "reason": (
                "insufficient forward weather window for the soil-moisture "
                f"forecast: {len(rows)} of {horizon} days available"
            ),
        }

    future_df = build_future_df_from_rows(field_id, rows, horizon)
    context_df = to_chronos_context_df(fused_history)
    pred_df = forecast_soil_moisture(context_df, future_df, prediction_length=horizon)

    pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    p10, p50, p90 = pred_df["0.1"], pred_df["0.5"], pred_df["0.9"]
    quantiles = [
        {
            "date": pd.Timestamp(ts).date().isoformat(),
            "p10": float(p10.iloc[i]),
            "p50": float(p50.iloc[i]),
            "p90": float(p90.iloc[i]),
        }
        for i, ts in enumerate(pred_df["timestamp"])
    ]

    width_max = float((p90 - p10).max())
    first, last = float(p50.iloc[0]), float(p50.iloc[-1])
    eps = 0.002
    trend = "falling" if last < first - eps else ("rising" if last > first + eps else "stable")

    obs = fused_history[fused_history[TARGET_COL].notna()]
    last_observed = None
    if len(obs):
        orow = obs.iloc[-1]
        last_observed = {
            "date": pd.Timestamp(orow["date"]).date().isoformat(),
            "moisture": float(orow[TARGET_COL]),
            "iot_valid_hours": int(orow.get("iot_valid_hours", 0)),
        }

    signals = [
        {
            "signal_name": "soil_moisture_p50_min",
            "value": float(p50.min()),
            "reference": f"min daily p50 = {float(p50.min()):.3f} m3/m3 over next {horizon} days",
            "interpretation": "Best-guess driest point of the forecast window.",
        },
        {
            "signal_name": "soil_moisture_p50_on_last_day",
            "value": last,
            "reference": f"p50 day {horizon} = {last:.3f} m3/m3",
            "interpretation": "Forecast soil moisture at the end of the window.",
        },
        {
            "signal_name": "soil_moisture_uncertainty_width_max",
            "value": width_max,
            "reference": f"max (p90-p10) = {width_max:.3f} m3/m3",
            "interpretation": "Wider width => lower forecast confidence.",
        },
        {
            "signal_name": "soil_moisture_trend",
            "value": trend,
            "reference": f"p50 day1 {first:.3f} -> day{horizon} {last:.3f}",
            "interpretation": f"Forecast soil-moisture trend is {trend}.",
        },
    ]
    if last_observed:
        signals.append(
            {
                "signal_name": "soil_moisture_last_observed",
                "value": last_observed["moisture"],
                "reference": (
                    f"{last_observed['moisture']:.3f} m3/m3 on {last_observed['date']} "
                    f"({last_observed['iot_valid_hours']} valid hours)"
                ),
                "interpretation": "Latest real sensor reading; forecast is anchored here.",
            }
        )

    return {
        "as_of": origin.date().isoformat(),
        "horizon_days": horizon,
        "forecast_origin_date": origin.date().isoformat(),
        "last_observed": last_observed,
        "quantiles": quantiles,
        "uncertainty_width_max": width_max,
        "trend": trend,
        "signals": signals,
        "insufficient_data": False,
    }


def _build_thresholds_bundle(query_params: QueryParams) -> dict:
    return load_agronomic_thresholds(query_params.crop_type, query_params.growth_stage)


def _safe_subbuild(key: str, fn, load_errors: dict) -> dict:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — degrade, never crash the bundle
        load_errors[key] = str(exc)
        log.warning("Sub-bundle '%s' failed: %s", key, exc)
        return {"insufficient_data": True, "signals": [], "reason": str(exc)}


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def build_signal_bundle(
    field_config: dict,
    query_params: QueryParams,
    mode: Literal["offline", "live"],
) -> SignalBundle:
    """
    Runs ONCE, before the graph. Raises only on a fully unrecoverable setup
    error (missing field config, zero history at all); every sub-step failure
    degrades its own field and is recorded in ``load_errors``.
    """
    for key in ("field_id", "bbox", "centroid"):
        if key not in field_config:
            raise ValueError(f"field_config is missing required key '{key}'")

    if mode == "live":
        origin, horizon = _resolve_live_origin_and_horizon(query_params)
        fused_history = _fetch_live_history(field_config, origin)
        forward_df = _fetch_live_forward(field_config, origin, horizon)
    else:
        fused = _load_offline_fused()
        origin, horizon = _resolve_offline_origin_and_horizon(query_params, fused)
        fused_history = fused[fused["date"] <= origin].copy().sort_values("date")
        forward_df = fused[fused["date"] > origin].copy()

    if len(fused_history) == 0:
        raise ValueError("zero history rows available — cannot build a signal bundle")

    load_errors: dict[str, str] = {}
    vegetation = _safe_subbuild(
        "vegetation",
        lambda: _build_vegetation_bundle(fused_history, origin),
        load_errors,
    )
    weather_forecast = _safe_subbuild(
        "weather_forecast",
        lambda: _build_weather_bundle(forward_df, origin, horizon),
        load_errors,
    )
    soil_moisture_forecast = _safe_subbuild(
        "soil_moisture_forecast",
        lambda: _build_soil_moisture_bundle(
            fused_history, origin, horizon, field_config["field_id"], forward_df
        ),
        load_errors,
    )
    thresholds = _safe_subbuild(
        "thresholds",
        lambda: _build_thresholds_bundle(query_params),
        load_errors,
    )

    return SignalBundle(
        field_id=field_config["field_id"],
        origin_date=origin,
        query_params=query_params,
        vegetation=vegetation,
        weather_forecast=weather_forecast,
        soil_moisture_forecast=soil_moisture_forecast,
        thresholds=thresholds,
        load_errors=load_errors,
    )
