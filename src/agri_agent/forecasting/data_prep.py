"""
Week 3 deliverable: reshape fusion.py's daily DataFrame into the format
Chronos-2's predict_df() expects — a long-format context_df (id,
timestamp, target, covariates) plus a separate future_df of known-future
covariate values for the forecast horizon.
"""

import pandas as pd

from agri_agent.data_access.weather import get_forecast
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

TARGET_COL = "iot_soil_moisture_mean"

FUTURE_KNOWN_COLS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "weather_soil_moisture_0_to_1cm_mean",
    "weather_soil_moisture_1_to_3cm_mean",
    "weather_soil_moisture_3_to_9cm_mean",
]

PAST_ONLY_COLS = ["NDVI", "NDWI"]

ALL_COVARIATE_COLS = FUTURE_KNOWN_COLS + PAST_ONLY_COLS


def select_model_columns(fused_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["date", "field_id", TARGET_COL] + ALL_COVARIATE_COLS
    missing = [c for c in keep_cols if c not in fused_df.columns]
    if missing:
        raise KeyError(f"fused_df is missing expected columns: {missing}")
    return fused_df[keep_cols].copy()


def to_chronos_context_df(fused_df: pd.DataFrame) -> pd.DataFrame:
    df = select_model_columns(fused_df)
    context_df = df.rename(columns={
        "field_id": "id",
        "date": "timestamp",
        TARGET_COL: "target",
    })
    context_df["timestamp"] = pd.to_datetime(context_df["timestamp"])
    context_df = context_df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    log.info(
        "Built Chronos-2 context_df: %d rows, target=%s, covariates=%s",
        len(context_df), TARGET_COL, ALL_COVARIATE_COLS,
    )
    return context_df


def future_weather_to_daily_df(weather_json: dict) -> pd.DataFrame:
    from agri_agent.data_access.fusion import HOURLY_WEATHER_AGG_RULES

    daily = pd.DataFrame(weather_json["daily"])
    daily["date"] = pd.to_datetime(daily["time"]).dt.floor("D")
    daily = daily.drop(columns=["time"])

    hourly = pd.DataFrame(weather_json["hourly"])
    hourly["date"] = pd.to_datetime(hourly["time"]).dt.floor("D")
    hourly_daily = (
        hourly.groupby("date")
        .agg({col: rule for col, rule in HOURLY_WEATHER_AGG_RULES.items() if col in hourly.columns})
        .reset_index()
    )
    hourly_daily = hourly_daily.rename(
        columns={c: f"weather_{c}_mean" for c in HOURLY_WEATHER_AGG_RULES if c in hourly_daily.columns}
    )
    return daily.merge(hourly_daily, on="date", how="outer")


def to_chronos_future_df(
    field_id: str,
    lat: float,
    lon: float,
    last_context_date: pd.Timestamp,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """
    Build Chronos-2's future_df: known-future covariate values for the
    forecast horizon. Only FUTURE_KNOWN_COLS are included — NDVI/NDWI
    have no future values and must NOT appear here.

    last_context_date must be context_df's actual last timestamp. The
    future window is built strictly AFTER this date. Open-Meteo's
    forecast_days parameter counts TODAY as day 1 of the forecast, not
    tomorrow, so a naive forecast_days=N call would overlap by one day
    with a context_df that already includes today.
    """
    last_context_date = pd.Timestamp(last_context_date).normalize()

    weather_json = get_forecast(lat=lat, lon=lon, forecast_days=horizon_days + 1, past_days=0)
    daily = future_weather_to_daily_df(weather_json)

    missing = [c for c in FUTURE_KNOWN_COLS if c not in daily.columns]
    if missing:
        raise KeyError(f"Future weather response is missing expected columns: {missing}")

    daily = daily[daily["date"] > last_context_date].sort_values("date").head(horizon_days)

    if len(daily) < horizon_days:
        log.warning(
            "Requested %d-day future horizon but only %d usable days were "
            "available after %s.",
            horizon_days, len(daily), last_context_date.date(),
        )

    future_df = daily[["date"] + FUTURE_KNOWN_COLS].copy()
    future_df.insert(0, "id", field_id)
    future_df = future_df.rename(columns={"date": "timestamp"})
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
    future_df = future_df.reset_index(drop=True)

    log.info(
        "Built Chronos-2 future_df: %d rows (%d-day horizon requested), covariates=%s",
        len(future_df), horizon_days, FUTURE_KNOWN_COLS,
    )
    return future_df

def split_covariates() -> tuple[list[str], list[str]]:
    return PAST_ONLY_COLS, FUTURE_KNOWN_COLS


def check_context_future_continuity(context_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
    last_context_date = context_df["timestamp"].max()
    first_future_date = future_df["timestamp"].min()
    expected_first_future_date = last_context_date + pd.Timedelta(days=1)

    if first_future_date != expected_first_future_date:
        raise ValueError(
            f"context_df ends {last_context_date.date()}, future_df starts "
            f"{first_future_date.date()} — expected {expected_first_future_date.date()}. "
            "context_df and future_df were likely built from different "
            "pipeline runs or date ranges; rebuild both together before "
            "forecasting."
        )
