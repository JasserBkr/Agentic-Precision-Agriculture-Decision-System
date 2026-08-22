"""Deterministic recommendation validator (STEP 5).

Two categories of checks, no LLM call, fully unit-testable with hand-built
objects:

1. GROUNDING — every SignalContribution the recommendation cites must name a
   signal that actually exists in the bundle. This makes traceability
   programmatically checkable: what the model claims it used must be a
   subset of what the bundle really carries.

   2. CONFLICT RULES — hand-written, deterministic checks that a sensible
      recommendation cannot contradict:
      a. irrigate_now while >= OVER_IRRIGATION_RAIN_MM rain is forecast over
         the window is wasteful and risks over-irrigation.
      b. NDVI anomaly z <= -2.0 (crop clearly below its own climatology) while
         irrigation says no_action_needed contradicts the vegetation signal.
      c. apply_fertilizer while the NDVI anomaly z >= 2.0 (vigor already far
         above this field's climatology) contradicts the vegetation signal —
         extra fertilizer on an already-thriving crop is wasteful and risks
         over-fertilization.

   3. CONFIDENCE CEILING — when either forecast tool reports
      insufficient_data, no recommendation may express confidence above
      MAX_CONFIDENCE_ON_INSUFFICIENT_FORECAST. This is deterministic: the
      model's judgment is not trusted to lower confidence on its own, the
      validator enforces it and feeds it back as a retry problem.
"""

from agri_agent.agent.bundle import SignalBundle
from agri_agent.agent.schemas import FusionRecommendation

OVER_IRRIGATION_RAIN_MM = 5.0

# Tunable ceiling: confidence above this is not allowed on a recommendation
# when weather/soil-moisture forecast data is unavailable (insufficient_data).
# Intentionally a module-level constant, not a magic number inside the check.
MAX_CONFIDENCE_ON_INSUFFICIENT_FORECAST = 0.5


def _signal_value(bundle_sub: dict, signal_name: str):
    for s in bundle_sub.get("signals", []):
        if s["signal_name"] == signal_name:
            return s["value"]
    return None


def _forecast_precip_total(weather_bundle: dict) -> float | None:
    """Sum of forecast precipitation over the window, computed from the raw
    forecast rows (not the rolled-up signal) so the check is independent of
    signal naming and can't be fooled by a malformed signal value."""
    rows = weather_bundle.get("forecast", [])
    values = [r["precipitation_sum"] for r in rows if r.get("precipitation_sum") is not None]
    if not values:
        return None
    return float(sum(values))


def _collect_real_signal_names(bundle: SignalBundle) -> set[str]:
    names = set()
    for sub in (
        bundle.vegetation,
        bundle.weather_forecast,
        bundle.soil_moisture_forecast,
        bundle.thresholds,
    ):
        names.update(s["signal_name"] for s in sub.get("signals", []))
    return names


def validate_recommendation(
    rec: FusionRecommendation,
    bundle: SignalBundle,
) -> tuple[list[str], bool]:
    """
    Deterministic checks. Returns (problems, conflict_bool).

    ``problems`` is a list of human-readable issues (empty when clean);
    ``conflict_bool`` is True when any conflict rule fired.
    """
    problems: list[str] = []
    conflict = False

    # 1. GROUNDING -------------------------------------------------------
    real = _collect_real_signal_names(bundle)
    cited = list(rec.irrigation.contributing_signals) + list(
        rec.fertilization.contributing_signals
    )
    for contrib in cited:
        if contrib.signal_name not in real:
            problems.append(
                f"Recommendation cites signal '{contrib.signal_name}' which is not "
                "present in the gathered bundle."
            )

    # 2a. Over-irrigation risk -------------------------------------------
    if rec.irrigation.action == "irrigate_now":
        precip = _forecast_precip_total(bundle.weather_forecast)
        if precip is not None and precip >= OVER_IRRIGATION_RAIN_MM:
            problems.append(
                f"Over-irrigation risk: action is 'irrigate_now' but "
                f"{precip:.1f} mm rain is forecast over the window "
                f"(>= {OVER_IRRIGATION_RAIN_MM} mm)."
            )
            conflict = True

    # 2b. Stress / no-action contradiction --------------------------------
    ndvi_z = _signal_value(bundle.vegetation, "ndvi_anomaly_z")
    if ndvi_z is not None and ndvi_z <= -2.0 and rec.irrigation.action == "no_action_needed":
        problems.append(
            "Contradiction: NDVI anomaly z-score is <= -2.0 (crop well below its "
            "own climatology) but irrigation action is 'no_action_needed'."
        )
        conflict = True

    # 2c. Fertilize on an already-thriving crop ---------------------------
    if rec.fertilization.action == "apply_fertilizer":
        if ndvi_z is not None and ndvi_z >= 2.0:
            problems.append(
                "Contradiction: action is 'apply_fertilizer' but NDVI anomaly "
                "z-score is >= 2.0 (vigor already far above this field's "
                "climatology) — extra fertilizer is wasteful."
            )
            conflict = True

    # 3. Confidence ceiling on insufficient forecast data -----------------
    forecast_unavailable = (
        bundle.weather_forecast.get("insufficient_data", False)
        or bundle.soil_moisture_forecast.get("insufficient_data", False)
    )
    if forecast_unavailable:
        for label, rec_sub in (
            ("irrigation", rec.irrigation),
            ("fertilization", rec.fertilization),
        ):
            if rec_sub.confidence > MAX_CONFIDENCE_ON_INSUFFICIENT_FORECAST:
                problems.append(
                    f"{label.capitalize()} confidence {rec_sub.confidence:.2f} exceeds "
                    f"the {MAX_CONFIDENCE_ON_INSUFFICIENT_FORECAST:.2f} ceiling while "
                    "weather/soil-moisture forecast data is unavailable — lower it."
                )

    return problems, conflict
