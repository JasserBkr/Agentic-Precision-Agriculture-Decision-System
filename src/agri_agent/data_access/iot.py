"""
Simulated IoT soil-moisture stream, seeded from Open-Meteo's own soil
variables (an accepted approach per the project brief, since physical
sensors aren't available). Week 1 deliverable.

If real sensors ever become available, replace this module's interface
with one that reads from the real device/API — downstream code should
only depend on the output shape (timestamp, moisture value), not on how
it was produced.
"""

import numpy as np

from agri_agent.data_access.weather import get_forecast, get_historical_forecast
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)


def _perturb_soil_moisture(
    weather_json: dict,
    noise_std: float,
    dropout_prob: float,
    rng: np.random.Generator,
) -> dict:
    """
    Shared perturbation logic: takes an already-fetched weather JSON
    response (live or historical — same schema) and returns the IoT-style
    stream with Gaussian noise + random dropout applied to the
    soil_moisture_0_to_1cm hourly variable.
    """
    times = weather_json["hourly"]["time"]
    values = np.array(weather_json["hourly"]["soil_moisture_0_to_1cm"], dtype=float)

    noisy = values + rng.normal(0, noise_std, size=values.shape)
    dropout_mask = rng.random(size=values.shape) < dropout_prob
    noisy[dropout_mask] = np.nan

    log.info(
        "Simulated IoT stream: %d points, %d dropped (%.1f%%)",
        len(noisy), dropout_mask.sum(), 100 * dropout_mask.mean(),
    )
    return {"time": times, "soil_moisture": noisy.tolist()}


def simulate_soil_moisture_stream(
    lat: float,
    lon: float,
    forecast_days: int = 7,
    past_days: int = 0,
    noise_std: float = 0.01,
    dropout_prob: float = 0.05,
    seed: int | None = None,
) -> dict:
    """
    Generate a noisy, gap-containing soil-moisture stream by perturbing
    Open-Meteo's own soil_moisture_0_to_1cm hourly variable. Deliberately
    imperfect: the noise + dropout are there on purpose, since Week 3-4
    forecasting work will need to handle exactly this kind of real-world
    messiness.

    `past_days` matches weather.get_forecast's parameter — pass the same
    value used for satellite's historical window when building the
    fused dataset, or the two sources won't overlap in time.
    """
    rng = np.random.default_rng(seed)
    raw = get_forecast(lat, lon, forecast_days=forecast_days, past_days=past_days)
    return _perturb_soil_moisture(raw, noise_std, dropout_prob, rng)


def simulate_historical_soil_moisture_stream(
    lat: float,
    lon: float,
    start_date,
    end_date,
    noise_std: float = 0.01,
    dropout_prob: float = 0.05,
    seed: int | None = None,
) -> dict:
    """
    Historical equivalent of simulate_soil_moisture_stream(): fetches from
    the Historical Forecast API (same schema as the live API) and applies
    the same noise + dropout perturbation via the shared helper.
    """
    rng = np.random.default_rng(seed)
    raw = get_historical_forecast(lat, lon, start_date=start_date, end_date=end_date)
    return _perturb_soil_moisture(raw, noise_std, dropout_prob, rng)
