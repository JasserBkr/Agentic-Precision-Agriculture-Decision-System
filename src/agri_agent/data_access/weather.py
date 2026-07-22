"""
Weather and soil forecast access via Open-Meteo. No API key required.
Week 1 deliverable.
"""
 
import requests
 
from agri_agent.utils.logging_config import get_logger
 
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
 
log = get_logger(__name__)
 
DEFAULT_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "windspeed_10m_max",
    "shortwave_radiation_sum",
]
 
DEFAULT_HOURLY_VARS = [
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_temperature_0cm",
]
 
 
def get_forecast(
    lat: float,
    lon: float,
    daily_vars: list[str] | None = None,
    hourly_vars: list[str] | None = None,
    forecast_days: int = 7,
    past_days: int = 0,
) -> dict:
    """
    Fetch weather + soil forecast for a field's centroid.
 
    `past_days` (0-92) includes recent historical data alongside the
    forecast — needed so weather's date range can overlap with
    satellite's historical lookback window when building the fused
    dataset (fusion.py). Without this, weather/IoT default to a
    forward-looking window that shares no dates with satellite's
    backward-looking one.
 
    Returns the raw parsed JSON response — daily and hourly arrays keyed
    by variable name, aligned to a `time` array. See Open-Meteo docs for
    the full variable list: https://open-meteo.com/en/docs
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(daily_vars or DEFAULT_DAILY_VARS),
        "hourly": ",".join(hourly_vars or DEFAULT_HOURLY_VARS),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "auto",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    log.info(
        "Fetched forecast for (%.4f, %.4f): %d past days + %d forecast days",
        lat, lon, past_days, forecast_days,
    )
    return data
 
 
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def get_historical_forecast(
    lat: float,
    lon: float,
    start_date,
    end_date,
    daily_vars: list[str] | None = None,
    hourly_vars: list[str] | None = None,
) -> dict:
    """
    Fetch archived weather + soil data from Open-Meteo's Historical Forecast
    API. This endpoint archives the LIVE forecast model's output (not
    reanalysis), so its response schema — variable names, units, daily/hourly
    structure — is identical to get_forecast(). This is critical for
    fusion.py compatibility.

    `start_date` / `end_date` are date objects or YYYY-MM-DD strings.

    Returns the same shape as get_forecast() (a dict with "daily" and "hourly"
    keys), so fusion.py's weather_response_to_daily_df() consumes it without
    modification.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(daily_vars or DEFAULT_DAILY_VARS),
        "hourly": ",".join(hourly_vars or DEFAULT_HOURLY_VARS),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto",
    }
    resp = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    n_daily = len(daily_vars or DEFAULT_DAILY_VARS)
    n_hourly = len(hourly_vars or DEFAULT_HOURLY_VARS)
    log.info(
        "Fetched historical forecast for (%.4f, %.4f): %s to %s "
        "(%d daily vars, %d hourly vars)",
        lat, lon, start_date, end_date, n_daily, n_hourly,
    )
    return data
