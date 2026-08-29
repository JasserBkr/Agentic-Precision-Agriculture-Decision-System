"""Independent deterministic irrigation reference policy (Layer 2 scoring).

DELIBERATELY shares NO code path with agri_agent.agent.validator: this is
a from-scratch implementation of the agronomic decision rules stated in
the system's own spec, used to score the agent's irrigation.action.
Constants below coincide with validator values BY SPEC, not by import —
do not "deduplicate" them against the agent package.

Inputs are read from the SignalBundle sub-dicts directly (real computed
values), never from LLM output text.

Decision semantics (first match wins):
  UNDECIDED          any required input missing (insufficient forecast)
  irrigate_now       moisture dips below the FAO-56 trigger with no
                     meaningful rain forecast to offset it
  irrigate_soon      below trigger but rain offsets urgency; OR trending
                     toward trigger within the horizon; OR below target
                     range but above trigger
  no_action_needed   otherwise
"""

from __future__ import annotations

# Independent declarations (spec-coincident, intentionally not imported).
RAIN_OFFSET_MM = 5.0

IRRIGATE_NOW = "irrigate_now"
IRRIGATE_SOON = "irrigate_soon"
NO_ACTION = "no_action_needed"
UNDECIDED = "UNDECIDED"

ACTION_CLASSES = (IRRIGATE_NOW, IRRIGATE_SOON, NO_ACTION)


def _sum_forecast(weather_sub: dict, key: str) -> float | None:
    rows = weather_sub.get("forecast") or []
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return float(sum(vals)) if vals else None


def _p50_min(soil_sub: dict) -> float | None:
    quantiles = soil_sub.get("quantiles") or []
    p50s = [q.get("p50") for q in quantiles if q.get("p50") is not None]
    return min(p50s) if p50s else None


def _horizon_crossing_days(p50_first: float, p50_last: float, trig: float, horizon: int) -> int | None:
    """Days until linear extrapolation of the p50 slope crosses `trig`.

    Returns None when the slope never reaches the trigger inside the
    horizon (or slope is ~flat / horizon invalid).
    """
    if horizon <= 1:
        return None
    slope_per_day = (p50_last - p50_first) / (horizon - 1)
    if abs(slope_per_day) < 1e-9:
        return None
    days = (trig - p50_last) / slope_per_day  # >0 means crossing lies ahead
    if 0 < days <= horizon:
        return days
    return None


def reference_irrigation_decision(
    weather_forecast: dict,
    soil_moisture_forecast: dict,
    thresholds: dict,
) -> dict:
    """Compute the reference decision + all intermediate quantities used,
    so the harness can log them and aggregation never re-derives anything.

    Returns {action, inputs: {...}, rule_fired: str|None}. action is one of
    ACTION_CLASSES or UNDECIDED.
    """
    trig = thresholds.get("trigger")
    target_range = thresholds.get("target_range") or []
    lo = target_range[0] if len(target_range) == 2 else None

    rain = _sum_forecast(weather_forecast, "precipitation_sum")
    m_min = _p50_min(soil_moisture_forecast)

    quantiles = soil_moisture_forecast.get("quantiles") or []
    trend = soil_moisture_forecast.get("trend")
    horizon = soil_moisture_forecast.get("horizon_days")
    p50_first = quantiles[0].get("p50") if quantiles else None
    p50_last = quantiles[-1].get("p50") if quantiles else None

    missing = []
    for name, v in (
        ("trigger", trig),
        ("target_range_lo", lo),
        ("rain_total", rain),
        ("m_min", m_min),
        ("trend", trend),
        ("horizon", horizon),
        ("p50_first", p50_first),
        ("p50_last", p50_last),
    ):
        if v is None:
            missing.append(name)
    insufficient = bool(
        weather_forecast.get("insufficient_data")
        or soil_moisture_forecast.get("insufficient_data")
        or thresholds.get("insufficient_data")
    )
    if insufficient or missing:
        return {
            "action": UNDECIDED,
            "inputs": {},
            "rule_fired": None,
            "abstain_reason": f"insufficient_data={insufficient} missing={missing}",
        }

    crossing = _horizon_crossing_days(p50_first, p50_last, float(trig), int(horizon))

    if m_min < trig:
        if rain >= RAIN_OFFSET_MM:
            action, rule = IRRIGATE_SOON, "below_trigger_rain_offset"
        else:
            action, rule = IRRIGATE_NOW, "below_trigger_no_rain"
    elif trend == "falling" and crossing is not None and rain < RAIN_OFFSET_MM:
        action, rule = IRRIGATE_SOON, "trend_toward_trigger"
    elif m_min < lo:
        action, rule = IRRIGATE_SOON, "below_target_range"
    else:
        action, rule = NO_ACTION, "within_target_band"

    return {
        "action": action,
        "rule_fired": rule,
        "inputs": {
            "m_min": m_min,
            "trigger": float(trig),
            "target_range_lo": float(lo),
            "rain_total_mm": rain,
            "trend": trend,
            "horizon_days": int(horizon),
            "crossing_days": crossing,
            "distance_to_trigger": float(m_min - trig),
        },
    }


def collapsed_2class(action: str) -> str:
    """Collapse now/soon into a single 'irrigate' class for the secondary
    agreement metric."""
    if action in (IRRIGATE_NOW, IRRIGATE_SOON):
        return "irrigate"
    return action
