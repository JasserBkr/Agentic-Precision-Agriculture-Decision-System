"""
One-time (or occasional) backfill: pull 2 years of satellite, historical
weather, and historical IoT data for the field defined in
configs/field.yaml, fuse into a single daily DataFrame, and persist to
data/processed/fused_2years.parquet.

Weather/IoT are fetched in 90-day chunks to avoid API timeouts on the
large payload, then concatenated into a single response dict that has
the exact same shape as a single call would return.

Usage:
    cd scripts && python backfill_history.py
"""

from datetime import date, timedelta

import yaml

from agri_agent.data_access.fusion import build_fused_dataset, save_fused_dataset
from agri_agent.data_access.iot import _perturb_soil_moisture
from agri_agent.data_access.satellite import get_field_index_timeseries
from agri_agent.data_access.weather import (
    DEFAULT_DAILY_VARS,
    DEFAULT_HOURLY_VARS,
    get_historical_forecast,
)
from agri_agent.utils.auth import init_earth_engine
from agri_agent.utils.logging_config import get_logger
import numpy as np

log = get_logger(__name__)

BACKFILL_DAYS = 730
CHUNK_DAYS = 90
OUTPUT_PATH = "data/processed/fused_2years.parquet"


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _chunked_historical_weather(
    lat: float, lon: float, start_date: date, end_date: date
) -> dict:
    """
    Fetch historical weather in CHUNK_DAYS-sized windows and concatenate
    into a single dict with the same shape as a single API response.
    """
    all_daily_time, all_daily_data = [], {v: [] for v in DEFAULT_DAILY_VARS}
    all_hourly_time, all_hourly_data = [], {v: [] for v in DEFAULT_HOURLY_VARS}

    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        log.info("Fetching weather chunk: %s to %s", chunk_start, chunk_end)
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


def main():
    field = load_field_config()
    log.info("Running 2-year backfill for %s", field["field_id"])

    end_date = date.today()
    start_date = end_date - timedelta(days=BACKFILL_DAYS)
    log.info("Date range: %s to %s (%d days requested)", start_date, end_date, BACKFILL_DAYS)

    # --- Satellite (Earth Engine handles any historical range natively) ---
    init_earth_engine()
    index_records = get_field_index_timeseries(
        bbox=field["bbox"],
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover_pct=field.get("max_cloud_cover_pct", 20),
    )
    if index_records:
        sat_dates = sorted(r["date"] for r in index_records)
        log.info(
            "Retrieved %d satellite scenes (%s to %s)",
            len(index_records), sat_dates[0], sat_dates[-1],
        )
    else:
        log.warning("No satellite scenes found in the 2-year window")

    # --- Historical weather (chunked to avoid API timeouts) ---
    weather = _chunked_historical_weather(
        lat=field["centroid"]["lat"],
        lon=field["centroid"]["lon"],
        start_date=start_date,
        end_date=end_date,
    )
    log.info(
        "Assembled historical weather: %d daily rows, %d hourly rows",
        len(weather["daily"]["time"]), len(weather["hourly"]["time"]),
    )

    # --- Historical IoT (chunked, same shared perturbation logic) ---
    rng = np.random.default_rng(seed=42)
    all_iot_times, all_iot_values = [], []
    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end_date)
        log.info("Fetching IoT chunk: %s to %s", chunk_start, chunk_end)
        chunk_weather = get_historical_forecast(
            lat=field["centroid"]["lat"],
            lon=field["centroid"]["lon"],
            start_date=chunk_start,
            end_date=chunk_end,
        )
        iot_chunk = _perturb_soil_moisture(chunk_weather, noise_std=0.01, dropout_prob=0.05, rng=rng)
        all_iot_times.extend(iot_chunk["time"])
        all_iot_values.extend(iot_chunk["soil_moisture"])
        chunk_start = chunk_end + timedelta(days=1)

    iot = {"time": all_iot_times, "soil_moisture": all_iot_values}
    log.info("Assembled historical IoT stream: %d points", len(iot["time"]))

    # --- Fusion (UNCHANGED call pattern — the whole point of this exercise) ---
    fused_df = build_fused_dataset(
        satellite_records=index_records,
        weather_json=weather,
        iot_dict=iot,
        field_id=field["field_id"],
        start_date=start_date,
        end_date=end_date,
    )

    log.info("Fused dataset: %d rows x %d columns", *fused_df.shape)
    log.info("Columns: %s", fused_df.columns.tolist())
    log.info("Null count per column:\n%s", fused_df.isna().sum().to_string())

    actual_start = fused_df["date"].min().date()
    actual_end = fused_df["date"].max().date()
    log.info("Actual date range: %s to %s (%d days)", actual_start, actual_end, len(fused_df))

    save_fused_dataset(fused_df, OUTPUT_PATH)
    log.info("Backfill complete: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
