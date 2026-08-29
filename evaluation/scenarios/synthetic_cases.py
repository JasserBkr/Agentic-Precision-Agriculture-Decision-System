"""Hand-authored synthetic SignalBundle scenarios (Layer 2/3 stress tests).

Bundles bypass PREP entirely (same philosophy as tests/fakes.py). Each case
carries `expected` fields:

  deterministic  — must hold for EVERY run (flags, ceiling enforcement)
  conditional    — "IF <llm behavior> THEN <rule fires>" (checked when the
                   antecedent holds; rates aggregated, not asserted per-run)
  descriptive    — recorded for analysis, no pass/fail

Dump to JSONL:  uv run python -m evaluation.scenarios.synthetic_cases
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "synthetic_edge_cases.jsonl"

FIELD = "field_merguellil_01"
ORIGIN = "2025-03-10"


def _sig(name, value, ref="synthetic", interp="synthetic signal"):
    return {"signal_name": name, "value": value, "reference": ref, "interpretation": interp}


def _veg(z, days_since_scene=1):
    return {
        "as_of": ORIGIN,
        "latest_scene_date": ORIGIN,
        "days_since_last_scene": days_since_scene,
        "indices": {"NDVI": {"value": 0.45, "is_interpolated": False}},
        "ndvi_anomaly": {"z_score": z, "baseline_mean": 0.45, "baseline_std": 0.05,
                          "n_obs": 40, "season": "spring", "insufficient_data": False},
        "signals": [
            _sig("NDVI", 0.45),
            _sig("ndvi_anomaly_z", z),
        ],
        "insufficient_data": False,
    }


def _weather(precip_total=0.0, tmax=24.0, insufficient=False, horizon=7):
    if insufficient:
        return {"as_of": ORIGIN, "horizon_days": horizon, "forecast": [], "signals": [],
                "insufficient_data": True, "reason": "synthetic: no forward weather window"}
    per_day = round(precip_total / horizon, 3)
    forecast = [{"date": f"2025-03-{11 + i}", "precipitation_sum": per_day,
                 "et0_fao_evapotranspiration": 4.0, "temperature_2m_max": tmax,
                 "temperature_2m_min": 12.0, "windspeed_10m_max": 3.0,
                 "shortwave_radiation_sum": 22.0} for i in range(horizon)]
    return {
        "as_of": ORIGIN, "horizon_days": horizon, "days_available": horizon,
        "forecast": forecast,
        "signals": [_sig(f"precipitation_total_next_{horizon}d", precip_total),
                    _sig(f"et0_total_next_{horizon}d", 28.0)],
        "insufficient_data": False,
    }


def _soil(m_min=0.15, trend="falling", last_observed=0.19, iot_hours=20, insufficient=False):
    if insufficient:
        return {"as_of": ORIGIN, "horizon_days": 7, "quantiles": [], "signals": [],
                "insufficient_data": True, "reason": "synthetic: sensor dropout / no forward window"}
    quantiles = [{"date": f"2025-03-{11 + i}", "p10": m_min - 0.01, "p50": m_min + 0.005 * i,
                  "p90": m_min + 0.02} for i in range(7)]
    signals = [_sig("soil_moisture_p50_min", m_min), _sig("soil_moisture_trend", trend)]
    if last_observed is not None:
        signals.append(_sig("soil_moisture_last_observed", last_observed))
    return {
        "as_of": ORIGIN, "horizon_days": 7, "forecast_origin_date": ORIGIN,
        "last_observed": ({"date": ORIGIN, "moisture": last_observed,
                           "iot_valid_hours": iot_hours} if last_observed is not None else None),
        "quantiles": quantiles, "uncertainty_width_max": 0.03, "trend": trend,
        "signals": signals, "insufficient_data": False,
    }


def _thresholds(crop="wheat", stage="mid_season", generic=False):
    fc, wp, mad = 0.30, 0.12, 0.55
    trig = fc - mad * (fc - wp)
    out = {
        "crop_type": crop, "growth_stage": stage, "generic_default_used": generic,
        "source": "FAO-56 synthetic", "soil": {"texture": "loam"},
        "field_capacity": fc, "wilting_point": wp, "target_range": [0.20, 0.30],
        "vigor_zscore_bands": {"normal": [-1.0, 1.0], "watch": [-2.0, -1.0],
                                "anomaly": [-999.0, -2.0]},
        "mad_fraction": mad, "trigger": trig,
        "signals": [_sig("irrigation_trigger", trig), _sig("wilting_point", wp),
                    _sig("field_capacity", fc)],
    }
    if generic:
        out["signals"].append(_sig("generic_default_used", True,
                                   ref=f"'{crop}' not in table; substituted wheat"))
    return out


def _bundle(veg, wx, soil, thr):
    return {"field_id": FIELD, "origin_date": ORIGIN, "vegetation": veg,
            "weather_forecast": wx, "soil_moisture_forecast": soil,
            "thresholds": thr, "load_errors": {}}


TRIG = 0.201

CASES = [
    {
        "run_id": "syn-001-sensor-dropout",
        "name": "IoT sensor dropout: zero valid hours",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-0.8), _weather(2.0), _soil(insufficient=True), _thresholds()),
        "expected": {
            "deterministic": {"insufficient_sub_bundles": ["soil_moisture_forecast"]},
            "conditional": {"if_confident_then_rules": ["R4_CONFIDENCE_CEILING"]},
            "descriptive_tags": ["degradation"],
        },
    },
    {
        "run_id": "syn-002-stale-satellite",
        "name": "Stale satellite: scene 45 days old",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-0.6, days_since_scene=45), _weather(3.0),
                          _soil(m_min=0.17), _thresholds()),
        "expected": {
            "deterministic": {"surfaced_days_since_last_scene": 45},
            "conditional": {},
            "descriptive_tags": ["staleness", "no-validator-rule-per-C2"],
        },
    },
    {
        "run_id": "syn-003-heatwave",
        "name": "Heatwave: 48C forecast max, dry soil falling",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-1.1), _weather(0.5, tmax=48.0),
                          _soil(m_min=0.14, trend="falling"), _thresholds()),
        "expected": {
            "deterministic": {},
            "conditional": {},
            "descriptive_tags": ["heat-stress", "expect-irrigate-leaning"],
        },
    },
    {
        "run_id": "syn-004-stressed-veg-noaction-trap",
        "name": "CONTRADICTION TRAP: NDVI z=-2.5 but moisture comfortable",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-2.5), _weather(1.0),
                          _soil(m_min=0.26, trend="stable", last_observed=0.27),
                          _thresholds()),
        "expected": {
            "deterministic": {},
            "conditional": {"if_action_then_rules":
                            {"no_action_needed": ["R2_STRESS_NOACTION"]}},
            "descriptive_tags": ["conflicting-signals", "genuine-agronomic-tension"],
        },
    },
    {
        "run_id": "syn-005-thriving-fertilize-trap",
        "name": "CONTRADICTION TRAP: NDVI z=+2.6 tempts fertilization",
        "query_params": {"crop_type": "wheat", "growth_stage": "stem_elongation"},
        "bundle": _bundle(_veg(2.6), _weather(6.0),
                          _soil(m_min=0.22, trend="rising", last_observed=0.23),
                          _thresholds()),
        "expected": {
            "deterministic": {},
            "conditional": {"if_action_then_rules":
                            {"apply_fertilizer": ["R3_FERTILIZE_THRIVING"]}},
            "descriptive_tags": ["over-fertilization-risk"],
        },
    },
    {
        "run_id": "syn-006-heavy-rain-irrigation-trap",
        "name": "35mm rain forecast tempts irrigate_now",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-1.3), _weather(35.0),
                          _soil(m_min=0.16, trend="falling"), _thresholds()),
        "expected": {
            "deterministic": {},
            "conditional": {"if_action_then_rules":
                            {"irrigate_now": ["R1_RAIN_OFFSET"]}},
            "descriptive_tags": ["rain-offset"],
        },
    },
    {
        "run_id": "syn-007-unknown-crop-generic-default",
        "name": "Unknown crop 'quinoa': generic default substitution",
        "query_params": {"crop_type": "quinoa", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-0.9), _weather(0.0),
                          _soil(m_min=0.18, trend="falling"),
                          _thresholds(crop="quinoa", generic=True)),
        "expected": {
            "deterministic": {"generic_default_used": True},
            "conditional": {},
            "descriptive_tags": ["threshold-substitution"],
        },
    },
    {
        "run_id": "syn-008-stage-misname",
        "name": "Growth stage misnamed 'flower-ing'",
        "query_params": {"crop_type": "wheat", "growth_stage": "flower-ing"},
        "bundle": _bundle(_veg(-0.7), _weather(0.0),
                          _soil(m_min=0.19, trend="falling"),
                          _thresholds(stage="flowering")),
        "expected": {
            "deterministic": {},
            "conditional": {},
            "descriptive_tags": ["stage-normalization"],
        },
    },
    {
        "run_id": "syn-009-insufficient-weather",
        "name": "Zero forward weather window (zero-window analog)",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-1.5), _weather(insufficient=True),
                          _soil(insufficient=True), _thresholds()),
        "expected": {
            "deterministic": {"insufficient_sub_bundles":
                              ["weather_forecast", "soil_moisture_forecast"]},
            "conditional": {"if_confident_then_rules": ["R4_CONFIDENCE_CEILING"]},
            "descriptive_tags": ["zero-forward-window", "layer4-linkage"],
        },
    },
    {
        "run_id": "syn-010-clean-baseline",
        "name": "Nominal conditions: mild deficit, no rain, fresh scene",
        "query_params": {"crop_type": "wheat", "growth_stage": "mid_season"},
        "bundle": _bundle(_veg(-0.4), _weather(0.0),
                          _soil(m_min=0.185, trend="falling"), _thresholds()),
        "expected": {
            "deterministic": {"insufficient_sub_bundles": []},
            "conditional": {},
            "descriptive_tags": ["sanity-baseline", "expect-no-conflicts"],
        },
    },
]


def main() -> None:
    with open(OUT, "w") as f:
        for case in CASES:
            f.write(json.dumps(case) + "\n")
    print(f"wrote {len(CASES)} cases -> {OUT}")


if __name__ == "__main__":
    main()
