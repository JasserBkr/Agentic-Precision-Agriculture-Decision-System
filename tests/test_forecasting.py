import numpy as np
import pandas as pd
import pytest

from agri_agent.forecasting.chronos_model import (
    MAX_COVARIATE_FILL_GAP_DAYS,
    _bounded_fill_series,
    _fill_covariate_gaps,
    forecast_soil_moisture,
)
from agri_agent.forecasting.data_prep import PAST_ONLY_COLS
from agri_agent.forecasting.evaluate import compute_metrics


def _make_df(n_rows=40, col_configs=None):
    """Build a synthetic DataFrame with covariate columns that have
    controlled NaN patterns. col_configs maps column name to a dict with
    keys 'start', 'end' (first/last valid index) and optionally
    'internal_gap' (tuple of start, end indices for an internal NaN gap).
    """
    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq="D")
    data = {"timestamp": timestamps, "id": "test_field", "target": np.arange(n_rows, dtype=float)}
    if col_configs is None:
        col_configs = {
            "precipitation_sum": {"start": 19, "end": n_rows},
            "NDVI": {"start": 10, "end": n_rows - 1},
        }
    for col, cfg in col_configs.items():
        arr = np.arange(n_rows, dtype=float)
        arr[: cfg["start"]] = np.nan
        arr[cfg["end"] :] = np.nan
        if "internal_gap" in cfg:
            ig_start, ig_end = cfg["internal_gap"]
            arr[ig_start:ig_end] = np.nan
        data[col] = arr
    return pd.DataFrame(data)


class TestBoundedFillSeries:
    def test_short_gap_interpolated(self):
        """NaN run <= max_gap with real values on both sides is filled."""
        s = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])
        result = _bounded_fill_series(s, max_gap=3)
        assert result.isna().sum() == 0
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(3.0)

    def test_long_gap_left_as_nan(self):
        """NaN run > max_gap is left untouched."""
        s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])
        result = _bounded_fill_series(s, max_gap=3)
        assert result.iloc[1:5].isna().all()

    def test_leading_gap_left_as_nan(self):
        """Leading NaN has no bounding value on the left — left as NaN."""
        s = pd.Series([np.nan, np.nan, 3.0, 4.0, 5.0])
        result = _bounded_fill_series(s, max_gap=3)
        assert result.iloc[0:2].isna().all()
        assert result.iloc[2] == 3.0

    def test_trailing_gap_left_as_nan(self):
        """Trailing NaN has no bounding value on the right — left as NaN."""
        s = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
        result = _bounded_fill_series(s, max_gap=3)
        assert result.iloc[3:5].isna().all()
        assert result.iloc[2] == 3.0

    def test_no_nan_unchanged(self):
        """Series with no NaN is returned unchanged."""
        s = pd.Series([1.0, 2.0, 3.0])
        result = _bounded_fill_series(s, max_gap=3)
        pd.testing.assert_series_equal(result, s)


class TestFillCovariateGaps:
    def test_leading_gaps_nan_passthrough(self):
        """Leading NaN blocks must be left as NaN, rows must NOT be dropped."""
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {"start": 19, "end": 40},
                "NDVI": {"start": 32, "end": 38},
            },
        )
        result = _fill_covariate_gaps(df, ["precipitation_sum", "NDVI"])
        # No rows dropped — passthrough preserves all 40 rows.
        assert len(result) == 40
        assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01")
        assert result["timestamp"].iloc[-1] == pd.Timestamp("2025-02-09")
        # Leading NaN in precipitation_sum (indices 0-18) preserved as NaN.
        assert result["precipitation_sum"].iloc[:19].isna().all()
        # NDVI leading NaN (indices 0-31) preserved as NaN.
        assert result["NDVI"].iloc[:32].isna().all()
        # Target column untouched.
        assert result["target"].notna().all()

    def test_trailing_nan_gap_passthrough(self):
        """Trailing NaN in a covariate must be left as NaN, rows kept."""
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {"start": 0, "end": 40},
                "NDVI": {"start": 0, "end": 38},
            },
        )
        result = _fill_covariate_gaps(df, ["precipitation_sum", "NDVI"])
        # All 40 rows preserved.
        assert len(result) == 40
        # NDVI trailing NaN at indices 38,39 preserved as NaN.
        assert result["NDVI"].iloc[38:].isna().all()
        # precipitation_sum fully valid.
        assert result["precipitation_sum"].notna().all()
        # Target column untouched.
        assert result["target"].notna().all()

    def test_leading_and_trailing_nan_passthrough(self):
        """Both leading and trailing NaN across columns must be left as NaN,
        rows must NOT be dropped."""
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {"start": 5, "end": 38},
                "NDVI": {"start": 12, "end": 40},
            },
        )
        result = _fill_covariate_gaps(df, ["precipitation_sum", "NDVI"])
        # All 40 rows preserved.
        assert len(result) == 40
        # Leading NaN preserved per column.
        assert result["precipitation_sum"].iloc[:5].isna().all()
        assert result["NDVI"].iloc[:12].isna().all()
        # Trailing NaN preserved for precipitation_sum.
        assert result["precipitation_sum"].iloc[38:].isna().all()
        # Target untouched.
        assert result["target"].notna().all()

    def test_wide_internal_gap_nan_passthrough(self):
        """A NaN gap strictly inside a column's valid range wider than
        MAX_COVARIATE_FILL_GAP_DAYS must be left as NaN (passthrough),
        NOT raise. Row count must be unchanged."""
        gap_start = 10
        gap_end = 10 + MAX_COVARIATE_FILL_GAP_DAYS + 1
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {
                    "start": 0,
                    "end": 40,
                    "internal_gap": (gap_start, gap_end),
                },
            },
        )
        result = _fill_covariate_gaps(df, ["precipitation_sum"])
        # Must NOT raise — internal gap passes through as NaN.
        assert len(result) == 40
        # Internal gap preserved as NaN.
        assert result["precipitation_sum"].iloc[gap_start:gap_end].isna().all()
        # Surrounding values still valid.
        assert result["precipitation_sum"].iloc[gap_start - 1] == gap_start - 1.0
        assert result["precipitation_sum"].iloc[gap_end] == gap_end

    def test_long_internal_gap_preserves_row_count(self):
        """Internal gap > MAX_COVARIATE_FILL_GAP_DAYS must not drop any rows."""
        gap_start = 15
        gap_end = 25
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {
                    "start": 0,
                    "end": 40,
                    "internal_gap": (gap_start, gap_end),
                },
                "NDVI": {"start": 0, "end": 40},
            },
        )
        result = _fill_covariate_gaps(df, ["precipitation_sum", "NDVI"])
        assert len(result) == 40
        # NDVI fully valid (no gap).
        assert result["NDVI"].notna().all()
        # precipitation_sum internal gap remains NaN.
        assert result["precipitation_sum"].iloc[gap_start:gap_end].isna().all()

    def test_fully_nan_column_raises(self):
        """An entire covariate column that is NaN must raise ValueError."""
        timestamps = pd.date_range("2025-01-01", periods=40, freq="D")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "id": "test_field",
            "target": np.arange(40, dtype=float),
            "precipitation_sum": np.full(40, np.nan),
        })
        with pytest.raises(ValueError, match="entirely NaN"):
            _fill_covariate_gaps(df, ["precipitation_sum"])

    def test_no_nan_returns_unchanged(self):
        """DataFrame with no NaN must be returned as-is."""
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {"start": 0, "end": 40},
                "NDVI": {"start": 0, "end": 40},
            },
        )
        result = _fill_covariate_gaps(df, ["precipitation_sum", "NDVI"])
        assert len(result) == 40

    def test_bounded_fill_preserves_short_gaps(self):
        """Gaps <= MAX_COVARIATE_FILL_GAP_DAYS with real values on both
        sides must be interpolated, not trimmed."""
        df = _make_df(
            n_rows=40,
            col_configs={
                "precipitation_sum": {"start": 0, "end": 40},
            },
        )
        # Insert a small internal gap
        df.loc[15: 17, "precipitation_sum"] = np.nan
        result = _fill_covariate_gaps(df, ["precipitation_sum"])
        # Gap of 3 rows (idx 15,16,17) <= MAX_COVARIATE_FILL_GAP_DAYS
        assert result["precipitation_sum"].notna().all()
        assert len(result) == 40


class TestForecastSoilMoistureContinuity:
    def test_continuity_checked_after_fill(self):
        """check_context_future_continuity must run AFTER _fill_covariate_gaps
        so it validates the post-trim timestamps, not pre-trim ones."""
        # This is a structural check: the function must not raise on
        # continuity before fill completes. We verify by checking the
        # source code ordering, since running the full pipeline requires
        # Chronos-2. A trailing gap in context_df would shift its last
        # timestamp, and a pre-fill continuity check would pass on the
        # wrong date.
        import inspect
        src = inspect.getsource(forecast_soil_moisture)
        fill_pos = src.index("_fill_covariate_gaps")
        continuity_pos = src.index("check_context_future_continuity")
        assert fill_pos < continuity_pos, (
            "check_context_future_continuity must be called AFTER "
            "_fill_covariate_gaps in forecast_soil_moisture"
        )

    def test_no_satellite_override_exists(self):
        """There must be no per-column override that loosens the fill
        bound for satellite columns (NDVI/NDWI) beyond
        MAX_COVARIATE_FILL_GAP_DAYS."""
        import inspect
        src = inspect.getsource(forecast_soil_moisture)
        assert "satellite_gap_overrides" not in src, (
            "satellite_gap_overrides must not exist — it would loosen "
            "the fill bound for NDVI/NDWI beyond MAX_COVARIATE_FILL_GAP_DAYS"
        )
        assert "column_max_gaps" not in src, (
            "forecast_soil_moisture must not pass column_max_gaps — "
            "fusion.py's tighter bound must not be contradicted"
        )


class TestForecastMetricsNotYetImplemented:
    def test_forecast_soil_moisture_not_yet_implemented(self):
        with pytest.raises(TypeError):
            forecast_soil_moisture(context_series=[])

    def test_compute_metrics_empty_input_raises(self):
        with pytest.raises(ValueError):
            compute_metrics([], [], [])
