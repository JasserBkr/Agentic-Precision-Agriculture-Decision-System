"""Tests for the NDVI seasonal anomaly (STEP 2b spec)."""

from datetime import date

import pandas as pd

from agri_agent.agent.anomaly import (
    _circular_distance,
    _season_for_doy,
    ndvi_seasonal_anomaly,
)


def _daily(start: str, n: int, values=None) -> tuple[pd.Series, pd.Series]:
    dates = pd.date_range(start=start, periods=n, freq="D")
    if values is None:
        values = [0.4 + 0.05 * (i % 3) for i in range(n)]
    return dates, pd.Series(values)


class TestSeasonForDoy:
    def test_season_boundaries(self):
        assert _season_for_doy(1) == "winter"
        assert _season_for_doy(90) == "winter"
        assert _season_for_doy(91) == "spring"
        assert _season_for_doy(181) == "summer"
        assert _season_for_doy(271) == "autumn"
        assert _season_for_doy(365) == "autumn"


class TestCircularDistance:
    def test_wraps_across_new_year(self):
        assert _circular_distance(354, 5) == 16
        assert _circular_distance(1, 365) == 1

    def test_same_doy_is_zero(self):
        assert _circular_distance(100, 100) == 0

    def test_plain_within_year(self):
        assert _circular_distance(100, 105) == 5


class TestNormalAnomaly:
    def test_z_score_positive_for_above_climatology(self):
        dates, values = _daily("2026-06-01", 40, values=[0.4] * 35 + [0.5] * 5)
        target = dates[-1]
        result = ndvi_seasonal_anomaly(dates, values, target)
        assert result["insufficient_data"] is False
        assert result["z_score"] is not None
        assert result["z_score"] > 0
        assert result["ndvi_value"] == 0.5
        assert result["anomaly"] is not None
        assert result["target_date"] == target.date().isoformat()
        assert "reason" not in result

    def test_z_score_negative_for_below_climatology(self):
        dates, values = _daily("2026-06-01", 40, values=[0.5] * 35 + [0.4] * 5)
        result = ndvi_seasonal_anomaly(dates, values, dates[-1])
        assert result["insufficient_data"] is False
        assert result["z_score"] < 0

    def test_na_dates_ignored(self):
        dates, values = _daily("2026-06-01", 40)
        values = values.copy()
        values.iloc[5] = None
        result = ndvi_seasonal_anomaly(dates, values, dates[-1])
        assert result["insufficient_data"] is False


class TestInsufficientData:
    def test_empty_history(self):
        result = ndvi_seasonal_anomaly(pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=float), date(2026, 7, 22))
        assert result["insufficient_data"] is True
        assert result["z_score"] is None
        assert "reason" in result

    def test_no_exact_target_date_observation(self):
        dates, values = _daily("2026-06-01", 40)
        target = dates[-1] + pd.Timedelta(days=1)
        result = ndvi_seasonal_anomaly(dates, values, target)
        assert result["insufficient_data"] is True
        assert "no observation exactly on target_date" in result["reason"]

    def test_fewer_than_min_obs_in_window(self):
        dates, values = _daily("2026-06-01", 40)
        result = ndvi_seasonal_anomaly(dates, values, dates[-1], window_days=0)
        assert result["insufficient_data"] is True
        assert "only 1 observations" in result["reason"]

    def test_all_null_values(self):
        dates, _ = _daily("2026-06-01", 40)
        values = pd.Series([None] * 40)
        result = ndvi_seasonal_anomaly(dates, values, dates[-1])
        assert result["insufficient_data"] is True


class TestZeroStdBaseline:
    def test_target_equals_mean_yields_zero_z(self):
        dates, values = _daily("2026-06-01", 40, values=[0.4] * 40)
        result = ndvi_seasonal_anomaly(dates, values, dates[-1])
        assert result["insufficient_data"] is False
        assert result["baseline_std"] == 0.0
        assert result["z_score"] == 0.0
        assert result["anomaly"] is False


class TestCircularWindow:
    def test_december_observation_included_for_january_target(self):
        dates = pd.to_datetime(["2025-12-25", "2026-01-01", "2026-01-05"])
        values = pd.Series([0.1, 0.4, 0.5])
        result = ndvi_seasonal_anomaly(dates, values, date(2026, 1, 5))
        assert result["insufficient_data"] is False
        assert result["n_obs"] == 3
        assert result["season"] == "winter"
        assert result["baseline_mean"] > 0.3
