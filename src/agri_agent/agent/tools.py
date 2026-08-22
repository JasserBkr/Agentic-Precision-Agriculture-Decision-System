"""Trivial dict-lookup tools over an already-built SignalBundle (STEP 3).

None of these fetch data, run Chronos-2, compute anomalies, or maintain any
cache — all of that already happened exactly once in
build_signal_bundle(). A tool call here cannot raise for data-related
reasons; it returns the bundle sub-field unchanged.

The docstrings stay informative because the model still needs to know WHEN
to call each tool.
"""

from langchain_core.tools import tool

from agri_agent.agent.bundle import SignalBundle


def make_tools(bundle: SignalBundle) -> list:
    """Build the four deterministic tools bound to a SignalBundle."""

    @tool
    def get_vegetation_indices() -> dict:
        """
        Latest field-mean NDVI/EVI/NDWI with the NDVI seasonal anomaly
        z-score (vs this field's own day-of-year climatology) and staleness
        metadata (days since last scene, interpolation flags). Call once to
        assess the current vegetation state and crop vigor.
        """
        return bundle.vegetation

    @tool
    def get_weather_forecast() -> dict:
        """
        Forward daily weather over the forecast horizon: precipitation,
        ET0, min/max temperature, wind, radiation, plus rolled-up totals
        (precipitation_total, et0_total, max_temperature). Use to weigh
        rain vs irrigation demand over the window.
        """
        return bundle.weather_forecast

    @tool
    def get_soil_moisture_forecast() -> dict:
        """
        Chronos-2 quantile soil-moisture forecast (p10/p50/p90) over the
        horizon, with uncertainty width, trend, and the last observed sensor
        reading. This is the primary input for the irrigation decision.
        """
        return bundle.soil_moisture_forecast

    @tool
    def get_agronomic_thresholds() -> dict:
        """
        Static crop/growth-stage thresholds: field capacity, wilting point,
        computed irrigation trigger (FC - MAD*(FC-WP)), target moisture
        range, and vigor z-score bands. Use to judge whether the soil-
        moisture forecast warrants irrigation.
        """
        return bundle.thresholds

    return [
        get_vegetation_indices,
        get_weather_forecast,
        get_soil_moisture_forecast,
        get_agronomic_thresholds,
    ]
