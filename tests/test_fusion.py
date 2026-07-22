from datetime import date, timedelta

import numpy as np
import pandas as pd

from agri_agent.data_access.fusion import (
    build_fused_dataset,
    iot_stream_to_daily_df,
    satellite_records_to_daily_df,
    weather_response_to_daily_df,
)


def _make_satellite_records(dates, ndvi_values):
    return [
        {
            "date": d.isoformat(),
            "NDVI": v,
            "EVI": v * 0.8,
            "NDWI": v - 0.3,
            "GNDVI": v * 1.1,
            "SAVI": v * 0.7,
        }
        for d, v in zip(dates, ndvi_values)
    ]


def _make_weather_json(dates):
    time_strs = [d.isoformat() for d in dates]
    hourly_times = []
    for d in dates:
        for h in range(24):
            hourly_times.append(f"{d.isoformat()}T{h:02d}:00")
    return {
        "daily": {
            "time": time_strs,
            "temperature_2m_max": [30.0] * len(dates),
            "temperature_2m_min": [15.0] * len(dates),
            "precipitation_sum": [0.0] * len(dates),
            "et0_fao_evapotranspiration": [5.0] * len(dates),
            "windspeed_10m_max": [10.0] * len(dates),
            "shortwave_radiation_sum": [20.0] * len(dates),
        },
        "hourly": {
            "time": hourly_times,
            "soil_moisture_0_to_1cm": [0.05] * len(hourly_times),
            "soil_moisture_1_to_3cm": [0.06] * len(hourly_times),
            "soil_moisture_3_to_9cm": [0.08] * len(hourly_times),
            "soil_temperature_0cm": [25.0] * len(hourly_times),
        },
    }


def _make_iot_dict(dates):
    hourly_times = []
    hourly_vals = []
    for d in dates:
        for h in range(24):
            hourly_times.append(f"{d.isoformat()}T{h:02d}:00")
            hourly_vals.append(0.05 + np.random.default_rng(42).normal(0, 0.005))
    return {"time": hourly_times, "soil_moisture": hourly_vals}


def test_build_fused_dataset_basic():
    end = date.today()
    start = end - timedelta(days=9)
    dates = [start + timedelta(days=i) for i in range(10)]

    satellite_records = _make_satellite_records(dates[::3], [0.3, 0.31, 0.32, 0.33])
    weather_json = _make_weather_json(dates)
    iot_dict = _make_iot_dict(dates)

    fused = build_fused_dataset(
        satellite_records=satellite_records,
        weather_json=weather_json,
        iot_dict=iot_dict,
        field_id="test_field",
        start_date=start,
        end_date=end,
    )

    assert isinstance(fused, pd.DataFrame)
    assert len(fused) == 10
    assert "field_id" in fused.columns
    assert "NDVI" in fused.columns
    assert "iot_soil_moisture_mean" in fused.columns
    assert fused["field_id"].eq("test_field").all()


def test_build_fused_dataset_empty_satellite():
    end = date.today()
    start = end - timedelta(days=4)
    dates = [start + timedelta(days=i) for i in range(5)]

    weather_json = _make_weather_json(dates)
    iot_dict = _make_iot_dict(dates)

    fused = build_fused_dataset(
        satellite_records=[],
        weather_json=weather_json,
        iot_dict=iot_dict,
        field_id="test_field",
        start_date=start,
        end_date=end,
    )

    assert len(fused) == 5
    assert fused["NDVI"].isna().all()
    assert fused["iot_soil_moisture_mean"].notna().all()
