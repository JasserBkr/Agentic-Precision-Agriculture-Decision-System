"""Tests for the agronomic thresholds loader (STEP 2c spec)."""

import pytest

from agri_agent.agent.bundle import load_agronomic_thresholds


class TestTrigger:
    def test_trigger_always_between_wp_and_fc(self):
        for crop, stage in (("wheat", "establishment"), ("wheat", "mid_season"), ("barley", "flowering")):
            out = load_agronomic_thresholds(crop_type=crop, growth_stage=stage)
            assert out["wilting_point"] < out["trigger"] < out["field_capacity"]

    def test_wheat_establishment_trigger_exact(self):
        out = load_agronomic_thresholds(crop_type="wheat", growth_stage="establishment")
        assert out["trigger"] == pytest.approx(0.21, abs=1e-9)
        assert out["generic_default_used"] is False
        assert out["growth_stage"] == "establishment"

    def test_barley_flowering_trigger_exact(self):
        out = load_agronomic_thresholds(crop_type="barley", growth_stage="flowering")
        assert out["trigger"] == pytest.approx(0.2035, abs=1e-9)
        assert out["generic_default_used"] is False

    def test_trigger_moves_with_mad_fraction(self):
        trigger_low_mad = load_agronomic_thresholds("wheat", "flowering")["trigger"]
        trigger_high_mad = load_agronomic_thresholds("wheat", "maturity")["trigger"]
        assert trigger_low_mad > trigger_high_mad


class TestGenericDefault:
    def test_default_crop_is_wheat(self):
        out = load_agronomic_thresholds()
        assert out["crop_type"] == "wheat"
        assert out["generic_default_used"] is False

    def test_unknown_crop_flags_generic_and_falls_back_to_wheat(self):
        out = load_agronomic_thresholds(crop_type="quinoa")
        assert out["generic_default_used"] is True
        assert out["crop_type"] == "quinoa"
        assert out["field_capacity"] == load_agronomic_thresholds("wheat")["field_capacity"]

    def test_unknown_growth_stage_flags_generic_and_uses_default_stage(self):
        out = load_agronomic_thresholds("wheat", "heading")
        assert out["generic_default_used"] is True
        assert out["growth_stage"] == "establishment"

    def test_valid_crop_and_stage_no_flag(self):
        for crop in ("wheat", "barley"):
            for stage in ("establishment", "mid_season", "flowering", "maturity"):
                out = load_agronomic_thresholds(crop_type=crop, growth_stage=stage)
                assert out["generic_default_used"] is False

    def test_hyphenated_growth_stage_normalized(self):
        assert load_agronomic_thresholds("wheat", "mid-season")["growth_stage"] == "mid_season"

    def test_never_flagged_for_defaults(self):
        for crop_type in (None, "wheat"):
            for growth_stage in (None, "establishment"):
                assert load_agronomic_thresholds(crop_type, growth_stage)["generic_default_used"] is False


class TestSignals:
    def test_thresholds_bundle_carries_groundable_signals(self):
        out = load_agronomic_thresholds("wheat", "establishment")
        names = {s["signal_name"] for s in out["signals"]}
        assert names == {
            "irrigation_trigger",
            "wilting_point",
            "field_capacity",
            "target_moisture_range",
        }

    def test_trigger_reference_echoes_computed_value(self):
        out = load_agronomic_thresholds("wheat", "establishment")
        sig = next(s for s in out["signals"] if s["signal_name"] == "irrigation_trigger")
        assert str(round(out["trigger"], 3)) in sig["reference"]

    def test_generic_default_used_carries_its_own_signal_when_flagged(self):
        out = load_agronomic_thresholds("wheat", "heading")
        assert out["generic_default_used"] is True
        sig = next(s for s in out["signals"] if s["signal_name"] == "generic_default_used")
        assert sig["value"] is True
        assert "heading" in sig["reference"]
        assert "establishment" in sig["reference"]

    def test_generic_default_signal_absent_when_not_flagged(self):
        out = load_agronomic_thresholds("wheat", "establishment")
        names = {s["signal_name"] for s in out["signals"]}
        assert "generic_default_used" not in names

    def test_generic_crop_substitution_signal_names_the_substituted_crop(self):
        out = load_agronomic_thresholds(crop_type="quinoa")
        assert out["generic_default_used"] is True
        sig = next(s for s in out["signals"] if s["signal_name"] == "generic_default_used")
        assert "quinoa" in sig["reference"]
        assert "wheat" in sig["reference"]
