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
 
 
# TODO (Week 1): add a function converting this raw response into a tidy
# pandas DataFrame or xarray Dataset with a proper datetime index — this
# is what data_access/fusion.py will consume in Week 2.
