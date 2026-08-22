"""Tests for the deterministic validator (STEP 5 spec): grounding + the three
conflict rules. No LLM involved."""

from agri_agent.agent.validator import OVER_IRRIGATION_RAIN_MM, validate_recommendation
from tests.conftest import make_bundle
from tests.fakes import make_rec, sig


class TestGrounding:
    def test_clean_recommendation_has_no_problems(self, bundle):
        problems, conflict = validate_recommendation(make_rec(), bundle)
        assert problems == []
        assert conflict is False

    def test_fabricated_signal_caught(self, bundle):
        rec = make_rec(irr_signals=[sig("NDVI"), sig("made_up_band")])
        problems, conflict = validate_recommendation(rec, bundle)
        assert any("made_up_band" in p and "not present" in p for p in problems)
        assert conflict is False

    def test_fabricated_fertilization_signal_caught(self, bundle):
        rec = make_rec(fert_signals=[sig("definitely_not_real")])
        problems, _ = validate_recommendation(rec, bundle)
        assert any("definitely_not_real" in p for p in problems)

    def test_signal_from_any_subbundle_is_valid(self, bundle):
        rec = make_rec(
            irr_signals=[sig("irrigation_trigger"), sig("soil_moisture_p50_min"), sig("NDWI")]
        )
        problems, _ = validate_recommendation(rec, bundle)
        assert problems == []


class TestRuleAOverIrrigation:
    def test_irrigate_now_with_forecast_rain_conflicts(self):
        bundle = make_bundle(precip=6.0)
        rec = make_rec(irr_action="irrigate_now")
        problems, conflict = validate_recommendation(rec, bundle)
        assert conflict is True
        assert any("Over-irrigation risk" in p for p in problems)

    def test_irrigate_now_with_low_rain_is_fine(self):
        bundle = make_bundle(precip=0.5)
        problems, conflict = validate_recommendation(make_rec(irr_action="irrigate_now"), bundle)
        assert conflict is False
        assert problems == []

    def test_boundary_rain_is_not_a_conflict(self):
        total = OVER_IRRIGATION_RAIN_MM - 0.1
        bundle = make_bundle(precip=total / 7)
        _, conflict = validate_recommendation(make_rec(irr_action="irrigate_now"), bundle)
        assert conflict is False

    def test_non_irrigate_now_actions_ignored(self):
        bundle = make_bundle(precip=10.0)
        for action in ("irrigate_soon", "no_action_needed"):
            _, conflict = validate_recommendation(make_rec(irr_action=action), bundle)
            assert conflict is False

    def test_degraded_weather_bundle_skips_rain_check(self, bundle):
        from dataclasses import replace

        b = replace(
            bundle,
            weather_forecast={
                "insufficient_data": True,
                "signals": [],
                "forecast": [],
                "reason": "no forward weather window",
            },
        )
        # Low confidence so the insufficient-data ceiling (rule 3) stays quiet
        # and only the rain-check-skip behavior is under test.
        problems, conflict = validate_recommendation(
            make_rec(irr_action="irrigate_now", irr_confidence=0.4, fert_confidence=0.4), b
        )
        assert conflict is False
        assert problems == []


class TestRuleBStressNoAction:
    def test_severe_stress_with_no_action_conflicts(self):
        bundle = make_bundle(ndvi_z=-2.5)
        rec = make_rec(irr_action="no_action_needed")
        problems, conflict = validate_recommendation(rec, bundle)
        assert conflict is True
        assert any("Contradiction" in p and "no_action_needed" in p for p in problems)

    def test_stress_with_action_is_fine(self):
        bundle = make_bundle(ndvi_z=-2.5)
        for action in ("irrigate_now", "irrigate_soon"):
            problems, conflict = validate_recommendation(make_rec(irr_action=action), bundle)
            assert conflict is False
            assert problems == []

    def test_mild_stress_with_no_action_is_fine(self):
        bundle = make_bundle(ndvi_z=-1.5)
        problems, conflict = validate_recommendation(make_rec(irr_action="no_action_needed"), bundle)
        assert conflict is False
        assert problems == []


class TestRuleCFertilizeThriving:
    def test_fertilize_when_already_thriving_conflicts(self):
        bundle = make_bundle(ndvi_z=2.5)
        rec = make_rec(fert_action="apply_fertilizer")
        problems, conflict = validate_recommendation(rec, bundle)
        assert conflict is True
        assert any("already thriving" in p.lower() or "wasteful" in p for p in problems)

    def test_no_application_when_thriving_is_fine(self):
        bundle = make_bundle(ndvi_z=2.5)
        problems, conflict = validate_recommendation(make_rec(fert_action="no_application"), bundle)
        assert conflict is False
        assert problems == []

    def test_fertilize_when_normal_is_fine(self):
        bundle = make_bundle(ndvi_z=1.0)
        problems, conflict = validate_recommendation(make_rec(fert_action="apply_fertilizer"), bundle)
        assert conflict is False
        assert problems == []


class TestConfidenceCeilingOnInsufficientForecast:
    def test_high_confidence_flagged_when_weather_insufficient(self, bundle):
        from dataclasses import replace

        b = replace(
            bundle,
            weather_forecast={"insufficient_data": True, "signals": [], "forecast": []},
        )
        problems, conflict = validate_recommendation(
            make_rec(irr_confidence=0.8, fert_confidence=0.85), b
        )
        assert conflict is False  # calibration problem, not a contradiction
        assert any("irrigation confidence" in p.lower() for p in problems)
        assert any("fertilization confidence" in p.lower() for p in problems)

    def test_high_confidence_flagged_when_soil_moisture_insufficient(self, bundle):
        from dataclasses import replace

        b = replace(
            bundle,
            soil_moisture_forecast={
                "insufficient_data": True,
                "signals": [],
                "quantiles": [],
                "reason": "no forward weather window",
            },
        )
        problems, _ = validate_recommendation(make_rec(irr_confidence=0.8), b)
        assert any("irrigation confidence" in p.lower() for p in problems)

    def test_confidence_at_ceiling_is_allowed(self, bundle):
        from dataclasses import replace

        b = replace(
            bundle,
            weather_forecast={"insufficient_data": True, "signals": [], "forecast": []},
        )
        problems, conflict = validate_recommendation(
            make_rec(irr_confidence=0.5, fert_confidence=0.5), b
        )
        assert problems == []
        assert conflict is False

    def test_low_confidence_is_fine_on_insufficient_data(self, bundle):
        from dataclasses import replace

        b = replace(
            bundle,
            weather_forecast={"insufficient_data": True, "signals": [], "forecast": []},
            soil_moisture_forecast={"insufficient_data": True, "signals": []},
        )
        # Signals narrowed to what the degraded bundle still carries
        # (vegetation only) so the confidence ceiling is the sole thing
        # under test and grounding stays quiet.
        problems, _ = validate_recommendation(
            make_rec(
                irr_signals=[sig("NDVI")],
                fert_signals=[sig("NDVI")],
                irr_confidence=0.3,
                fert_confidence=0.3,
            ),
            b,
        )
        assert problems == []

    def test_full_bundle_with_forecasts_is_unaffected(self, bundle):
        problems, conflict = validate_recommendation(make_rec(), bundle)
        assert problems == []
        assert conflict is False


class TestCombinations:
    def test_irrigate_now_and_fertilize_normal_is_clean(self):
        problems, conflict = validate_recommendation(
            make_rec(irr_action="irrigate_now", fert_action="apply_fertilizer"), make_bundle()
        )
        assert conflict is False
        assert problems == []

    def test_multiple_conflicts_all_surface(self):
        bundle = make_bundle(ndvi_z=2.5, precip=8.0)
        rec = make_rec(irr_action="irrigate_now", fert_action="apply_fertilizer")
        problems, conflict = validate_recommendation(rec, bundle)
        assert conflict is True
        assert len(problems) == 2
