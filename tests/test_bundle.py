"""Tests for build_signal_bundle / PREP (STEP 2a). No network, no Chronos-2
model loads — forecast_soil_moisture and the parquet read are faked."""

from datetime import date, timedelta

import pandas as pd
import pytest

from agri_agent.agent import bundle as bundle_mod
from agri_agent.agent.bundle import (
    FORECAST_HORIZON_DAYS,
    build_future_df_from_rows,
    build_signal_bundle,
    _resolve_live_origin_and_horizon,
    _resolve_offline_origin_and_horizon,
)
from agri_agent.agent.schemas import QueryParams
from tests.fakes import make_synthetic_fused


@pytest.fixture
def fake_parquet(monkeypatch):
    monkeypatch.setattr(
        bundle_mod, "_load_offline_fused", lambda: make_synthetic_fused(n_days=40, start="2026-06-01")
    )


@pytest.fixture
def fake_forecast(monkeypatch):
    def _fake_forecast(context_df, future_df, prediction_length):
        ts = pd.to_datetime(future_df["timestamp"]).sort_values()
        return pd.DataFrame(
            {
                "timestamp": ts,
                "0.1": 0.16,
                "0.5": 0.17,
                "0.9": 0.18,
            }
        )

    monkeypatch.setattr(bundle_mod, "forecast_soil_moisture", _fake_forecast)


FIELD_CONFIG = {
    "field_id": "test_field",
    "bbox": [9.8, 35.4, 10.1, 35.7],
    "centroid": {"lat": 35.56, "lon": 9.95},
    "max_cloud_cover_pct": 20,
}


class TestHappyPath:
    def test_offline_bundle_is_fully_built(self, fake_parquet, fake_forecast):
        qp = QueryParams(target_date=date(2026, 6, 20))
        bundle = build_signal_bundle(FIELD_CONFIG, qp, mode="offline")

        assert bundle.field_id == "test_field"
        assert bundle.origin_date == pd.Timestamp("2026-06-20").normalize()
        assert bundle.load_errors == {}

        assert bundle.vegetation["insufficient_data"] is False
        assert bundle.vegetation["latest_scene_date"] == "2026-06-20"
        assert {s["signal_name"] for s in bundle.vegetation["signals"]} >= {"NDVI", "EVI", "NDWI"}

        assert bundle.weather_forecast["insufficient_data"] is False
        assert bundle.weather_forecast["horizon_days"] == FORECAST_HORIZON_DAYS
        assert bundle.weather_forecast["days_available"] == FORECAST_HORIZON_DAYS

        assert bundle.soil_moisture_forecast["insufficient_data"] is False
        names = {s["signal_name"] for s in bundle.soil_moisture_forecast["signals"]}
        assert names >= {"soil_moisture_p50_min", "soil_moisture_p50_on_last_day", "soil_moisture_trend"}

        assert bundle.thresholds["generic_default_used"] is False
        assert {s["signal_name"] for s in bundle.thresholds["signals"]} >= {
            "irrigation_trigger",
            "wilting_point",
            "field_capacity",
            "target_moisture_range",
        }


class TestDegradation:
    def test_vegetation_failure_degrades_only_its_field(self, monkeypatch, fake_parquet, fake_forecast):
        def boom(*a, **k):
            raise RuntimeError("no satellite data")

        monkeypatch.setattr(bundle_mod, "_build_vegetation_bundle", boom)
        bundle = build_signal_bundle(FIELD_CONFIG, QueryParams(target_date=date(2026, 6, 20)), mode="offline")

        assert bundle.vegetation["insufficient_data"] is True
        assert bundle.vegetation["signals"] == []
        assert bundle.load_errors["vegetation"] == "no satellite data"
        assert bundle.weather_forecast["insufficient_data"] is False
        assert bundle.soil_moisture_forecast["insufficient_data"] is False
        assert bundle.thresholds["generic_default_used"] is False
        assert bundle.thresholds["signals"]

    def test_weather_failure_degrades_only_its_field(self, monkeypatch, fake_parquet, fake_forecast):
        def boom(*a, **k):
            raise RuntimeError("no forecast")

        monkeypatch.setattr(bundle_mod, "_build_weather_bundle", boom)
        bundle = build_signal_bundle(FIELD_CONFIG, QueryParams(target_date=date(2026, 6, 20)), mode="offline")

        assert bundle.weather_forecast["insufficient_data"] is True
        assert bundle.vegetation["insufficient_data"] is False
        assert bundle.soil_moisture_forecast["insufficient_data"] is False
        assert "no forecast" in bundle.load_errors["weather_forecast"]

    def test_soil_moisture_failure_degrades_only_its_field(self, monkeypatch, fake_parquet):
        def boom(*a, **k):
            raise RuntimeError("chronos blew up")

        monkeypatch.setattr(bundle_mod, "_build_soil_moisture_bundle", boom)
        bundle = build_signal_bundle(FIELD_CONFIG, QueryParams(target_date=date(2026, 6, 20)), mode="offline")

        assert bundle.soil_moisture_forecast["insufficient_data"] is True
        assert bundle.vegetation["insufficient_data"] is False
        assert bundle.weather_forecast["insufficient_data"] is False
        assert "chronos blew up" in bundle.load_errors["soil_moisture_forecast"]

    def test_thresholds_failure_degrades_only_its_field(self, monkeypatch, fake_parquet, fake_forecast):
        def boom(*a, **k):
            raise RuntimeError("bad yaml")

        monkeypatch.setattr(bundle_mod, "_build_thresholds_bundle", boom)
        bundle = build_signal_bundle(FIELD_CONFIG, QueryParams(target_date=date(2026, 6, 20)), mode="offline")

        assert bundle.thresholds["insufficient_data"] is True
        assert bundle.vegetation["insufficient_data"] is False
        assert bundle.weather_forecast["insufficient_data"] is False
        assert "bad yaml" in bundle.load_errors["thresholds"]

    def test_all_four_can_degrade_together(self, monkeypatch, fake_parquet):
        for name in ("_build_vegetation_bundle", "_build_weather_bundle",
                     "_build_soil_moisture_bundle", "_build_thresholds_bundle"):
            monkeypatch.setattr(bundle_mod, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"boom {name}")))

        bundle = build_signal_bundle(FIELD_CONFIG, QueryParams(target_date=date(2026, 6, 20)), mode="offline")

        for sub in ("vegetation", "weather_forecast", "soil_moisture_forecast", "thresholds"):
            assert getattr(bundle, sub)["insufficient_data"] is True
            assert getattr(bundle, sub)["signals"] == []
        assert set(bundle.load_errors) == {
            "vegetation", "weather_forecast", "soil_moisture_forecast", "thresholds",
        }


class TestUnrecoverableErrors:
    def test_missing_field_config_key_raises(self, fake_parquet):
        with pytest.raises(ValueError, match="missing required key"):
            build_signal_bundle({"field_id": "x"}, QueryParams(), mode="offline")

    def test_zero_history_raises(self, monkeypatch):
        empty = pd.DataFrame(columns=["date"])
        monkeypatch.setattr(bundle_mod, "_load_offline_fused", lambda: empty)
        with pytest.raises(ValueError, match="zero history"):
            build_signal_bundle(FIELD_CONFIG, QueryParams(), mode="offline")


class TestOriginAndHorizon:
    def test_offline_origin_is_target_date(self, fake_parquet):
        origin, horizon = _resolve_offline_origin_and_horizon(
            QueryParams(target_date=date(2026, 6, 20)), make_synthetic_fused()
        )
        assert origin == pd.Timestamp("2026-06-20").normalize()
        assert horizon == FORECAST_HORIZON_DAYS

    def test_offline_origin_defaults_to_parquet_max_minus_horizon(self, fake_parquet):
        fused = make_synthetic_fused(n_days=40, start="2026-06-01")
        origin, _ = _resolve_offline_origin_and_horizon(QueryParams(), fused)
        expected = pd.Timestamp(fused["date"].max()).normalize() - pd.Timedelta(
            days=FORECAST_HORIZON_DAYS
        )
        assert origin == expected

    def test_offline_default_leaves_nonempty_forward_window_on_real_parquet(self):
        parquet_path = bundle_mod.FUSED_PARQUET
        if not parquet_path.exists():
            pytest.skip("real fused_2years.parquet not present")
        fused = bundle_mod._load_offline_fused()
        origin, horizon = _resolve_offline_origin_and_horizon(QueryParams(), fused)
        forward = fused[fused["date"] > origin]
        assert len(forward) > 0, "default origin must leave a usable forward window"
        assert len(forward) == FORECAST_HORIZON_DAYS
        assert forward["date"].max() == fused["date"].max()

    def test_explicit_target_date_at_true_last_row_still_zero_forward(self, fake_parquet):
        fused = make_synthetic_fused(n_days=40, start="2026-06-01")
        last = pd.Timestamp(fused["date"].max()).normalize()
        origin, _ = _resolve_offline_origin_and_horizon(
            QueryParams(target_date=last.date()), fused
        )
        assert origin == last
        assert len(fused[fused["date"] > origin]) == 0

    def test_live_origin_is_today(self):
        origin, horizon = _resolve_live_origin_and_horizon(QueryParams())
        assert origin == pd.Timestamp(date.today()).normalize()
        assert horizon == FORECAST_HORIZON_DAYS

    def test_live_horizon_scales_to_future_target(self):
        target = pd.Timestamp(date.today() + timedelta(days=14)).normalize()
        _, horizon = _resolve_live_origin_and_horizon(QueryParams(target_date=target))
        assert horizon == 14

    def test_live_past_target_raises(self):
        target = pd.Timestamp(date.today() - timedelta(days=1)).normalize()
        with pytest.raises(ValueError, match="live mode forecasts forward"):
            _resolve_live_origin_and_horizon(QueryParams(target_date=target))


class TestFutureDf:
    def test_reshape_truncates_to_horizon(self):
        rows = make_synthetic_fused(n_days=10, start="2026-06-20")
        future = build_future_df_from_rows("test_field", rows, horizon_days=7)
        assert len(future) == 7
        assert list(future.columns)[0] == "id"
        assert "timestamp" in future.columns
        assert "precipitation_sum" in future.columns
        assert "et0_fao_evapotranspiration" in future.columns
        assert (future["timestamp"] == pd.to_datetime(rows["date"]).head(7).reset_index(drop=True)).all()
