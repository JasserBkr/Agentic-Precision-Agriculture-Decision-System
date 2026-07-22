"""
Week 2 deliverable: align satellite, weather, and IoT data onto a common
daily pandas DataFrame, addressing the fragmentation problem described in
Section 3.1 of the SOTA note (docs/sota_note.pdf) — for THIS pipeline the
fragmentation is temporal + semantic, not spatial, since satellite data
already arrives field-averaged from data_access.satellite.

This module is deliberately model-agnostic: it produces one clean daily
table and knows nothing about Chronos-2 or any other forecaster. Model-
specific reshaping (context_df/future_df, covariate roles) belongs in
forecasting/data_prep.py (Week 3), not here.
"""

from datetime import date

import numpy as np
import pandas as pd

from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

# Explicit per-variable aggregation rule for resampling hourly weather to
# daily. No blanket .mean() — precipitation/ET0 must be summed, not
# averaged, or daily totals are silently wrong.
DAILY_WEATHER_AGG_RULES = {
    "temperature_2m_max": "max",
    "temperature_2m_min": "min",
    "precipitation_sum": "sum",
    "et0_fao_evapotranspiration": "sum",
    "windspeed_10m_max": "max",
    "shortwave_radiation_sum": "sum",
}

HOURLY_WEATHER_AGG_RULES = {
    "soil_moisture_0_to_1cm": "mean",
    "soil_moisture_1_to_3cm": "mean",
    "soil_moisture_3_to_9cm": "mean",
    "soil_temperature_0cm": "mean",
}

# Satellite index columns produced by data_access.satellite.
SATELLITE_INDEX_COLS = ["NDVI", "EVI", "NDWI", "GNDVI", "SAVI"]

# Days beyond which we stop interpolating and leave a real NaN instead of
# fabricating data. Tune this against your crop/field once you have more
# than one field's worth of experience.
MAX_INTERPOLATION_GAP_DAYS = 2


# ---------------------------------------------------------------------
# Per-source normalization: raw source output -> tidy daily DataFrame
# ---------------------------------------------------------------------


def satellite_records_to_daily_df(records: list[dict]) -> pd.DataFrame:
    """
    Convert data_access.satellite.get_field_index_timeseries() output
    into a daily-indexed DataFrame. Scenes sharing a date (Sentinel-2
    tile overlap) are averaged together, since they represent the same
    field on the same day, not two distinct observations.
    """
    if not records:
        return pd.DataFrame(columns=["date"] + SATELLITE_INDEX_COLS)
 
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df = df.groupby("date", as_index=False)[SATELLITE_INDEX_COLS].mean()
    log.info("Normalized %d satellite records into %d daily rows", len(records), len(df))
    return df


def weather_response_to_daily_df(weather_json: dict) -> pd.DataFrame:
    """
    Convert data_access.weather.get_forecast() raw JSON into a daily
    DataFrame: the "daily" block is used as-is (already daily), and the
    "hourly" soil variables are resampled to daily using
    HOURLY_WEATHER_AGG_RULES (mean — these are state variables, not
    accumulations).
    """
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

    merged = daily.merge(hourly_daily, on="date", how="outer")
    log.info("Normalized weather response into %d daily rows", len(merged))
    return merged


def iot_stream_to_daily_df(iot_dict: dict) -> pd.DataFrame:
    """
    Convert data_access.iot.simulate_soil_moisture_stream() output into
    a daily DataFrame: mean soil moisture per day (NaN-safe), plus
    iot_valid_hours — how many non-dropped hourly readings contributed
    to that day's mean, a direct trust signal for that day's value.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(iot_dict["time"]).floor("D"),
        "iot_soil_moisture": iot_dict["soil_moisture"],
    })
    daily = df.groupby("date").agg(
        iot_soil_moisture_mean=("iot_soil_moisture", "mean"),
        iot_valid_hours=("iot_soil_moisture", lambda s: int(s.notna().sum())),
    ).reset_index()
    log.info("Normalized IoT stream into %d daily rows", len(daily))
    return daily


# ---------------------------------------------------------------------
# Alignment: put every source on one canonical daily grid
# ---------------------------------------------------------------------


def build_daily_index(start_date: date, end_date: date) -> pd.DatetimeIndex:
    """The canonical daily grid every source gets left-joined onto."""
    return pd.date_range(start=start_date, end=end_date, freq="D", name="date")


def merge_daily_sources(
    satellite_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    iot_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Left-join all three normalized sources onto the canonical daily
    grid. Days with no satellite scene (or, unexpectedly, no
    weather/IoT data) become explicit NaN, not silently missing rows.
    """
    grid = pd.DataFrame({"date": daily_index})
    merged = (
        grid.merge(satellite_df, on="date", how="left")
        .merge(weather_df, on="date", how="left")
        .merge(iot_df, on="date", how="left")
    )
    return merged


# ---------------------------------------------------------------------
# Gap handling: the one place interpolation policy lives
# ---------------------------------------------------------------------
 
 
def add_gap_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit gap-tracking columns and apply bounded interpolation
    ONLY to the satellite index columns (the sparse source). Weather and
    IoT are expected to be complete already; if they have gaps too,
    this leaves them as real NaN rather than guessing — a genuine gap
    there is a data-quality problem worth surfacing, not papering over.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # days_since_last_scene: 0 on an observed day, counts up on gap days.
    has_scene = df["NDVI"].notna()
    scene_dates = df.loc[has_scene, "date"]
    df["days_since_last_scene"] = df["date"].apply(
        lambda d: (d - scene_dates[scene_dates <= d].max()).days
        if (scene_dates <= d).any() else np.nan
    )

    for col in SATELLITE_INDEX_COLS:
        flag_col = f"is_interpolated_{col.lower()}"
        df[flag_col] = df[col].isna()
        df[col] = df[col].astype("float64").interpolate(
            method="linear", limit=MAX_INTERPOLATION_GAP_DAYS, limit_area="inside"
        )
        # A value is only "interpolated" if it was NaN before and got
        # filled — not if it remains NaN (gap too large) or was never
        # missing to begin with.
        df[flag_col] = df[flag_col] & df[col].notna()

    return df


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def build_fused_dataset(
    satellite_records: list[dict],
    weather_json: dict,
    iot_dict: dict,
    field_id: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Produce one tidy daily DataFrame fusing all three Week 1 sources for
    a single field. This is the Week 2 deliverable — model-agnostic,
    gap-aware, one row per day.
    """
    satellite_df = satellite_records_to_daily_df(satellite_records)
    weather_df = weather_response_to_daily_df(weather_json)
    iot_df = iot_stream_to_daily_df(iot_dict)

    daily_index = build_daily_index(start_date, end_date)
    merged = merge_daily_sources(satellite_df, weather_df, iot_df, daily_index)
    merged = add_gap_metadata(merged)
    merged.insert(1, "field_id", field_id)

    log.info(
        "Built fused dataset for %s: %d rows, %d columns (%s to %s)",
        field_id, len(merged), len(merged.columns), start_date, end_date,
    )
    return merged

