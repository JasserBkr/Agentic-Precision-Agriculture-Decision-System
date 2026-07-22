"""
Week 3 deliverable: zero-shot forecasting with Chronos-2, using the
context_df/future_df produced by forecasting.data_prep.
 
Loads the pipeline fresh on each call — fine for this project's daily-
batch use case, not a low-latency serving scenario. If this becomes a
bottleneck later, cache the loaded pipeline instead of reloading it.
"""
 
import pandas as pd
 
from agri_agent.forecasting.data_prep import (
    ALL_COVARIATE_COLS,
    FUTURE_KNOWN_COLS,
    check_context_future_continuity,
)
from agri_agent.utils.logging_config import get_logger
 
log = get_logger(__name__)
 
MODEL_ID = "amazon/chronos-2"
DEFAULT_QUANTILE_LEVELS = [0.1, 0.5, 0.9]
 
# Same discipline as fusion.py's MAX_INTERPOLATION_GAP_DAYS: a short gap
# is safe to bridge, a long one should never be silently fabricated.
MAX_COVARIATE_FILL_GAP_DAYS = 3
 
 
def _select_device() -> str:
    """Use GPU if available, fall back to CPU otherwise."""
    import torch
 
    return "cuda" if torch.cuda.is_available() else "cpu"
 
 
def _bounded_fill_series(s: pd.Series, max_gap: int) -> pd.Series:
    """
    Fill only NaN runs of length <= max_gap, via linear interpolation
    between the real values bounding that run. Runs longer than
    max_gap, and any run missing a real value on one side (leading or
    trailing edge of the series), are left untouched as NaN.
 
    IMPORTANT: sequential ffill(limit=N).bfill(limit=N) looks like it
    bounds gaps at N days but does not — ffill fills the first N days
    of a gap from one side, then bfill fills the REMAINING days from
    the other side, silently bridging gaps up to 2N days wide. This
    function measures actual run length directly to avoid that (caught
    by a test before this ever ran against real data).
    """
    s = s.copy()
    is_na = s.isna()
    if not is_na.any():
        return s
 
    group_id = (is_na != is_na.shift()).cumsum()
    for _, group in s[is_na].groupby(group_id[is_na]):
        idx = group.index
        run_len = len(idx)
        before_pos = s.index.get_loc(idx[0]) - 1
        after_pos = s.index.get_loc(idx[-1]) + 1
        has_before = before_pos >= 0 and not pd.isna(s.iloc[before_pos])
        has_after = after_pos < len(s) and not pd.isna(s.iloc[after_pos])
 
        if run_len <= max_gap and has_before and has_after:
            before_idx = s.index[before_pos]
            after_idx = s.index[after_pos]
            s.loc[before_idx:after_idx] = s.loc[before_idx:after_idx].interpolate(method="linear")
        # else: leave as NaN — too wide, or missing a real bounding
        # value on one side (leading/trailing edge case).
    return s
 
 
def _fill_covariate_gaps(
    df: pd.DataFrame,
    covariate_cols: list[str],
    column_max_gaps: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Bounded fill for covariate NaN gaps — a gap of up to
    MAX_COVARIATE_FILL_GAP_DAYS is bridged via linear interpolation;
    anything wider, or any gap at the leading/trailing edge with no
    real value to interpolate from, is left as NaN.

    This replaces an earlier unbounded ffill/bfill version that silently
    fabricated an entire month of "history" when a real run's
    LOOKBACK_DAYS exceeded Open-Meteo's actual data horizon for a given
    variable — the bug went undetected until a manual backtest
    comparison exposed it (172 of 602 covariate cells had been filled
    in that run).

    Remaining NaN cells after the bounded fill are left as-is (NaN
    passthrough). Chronos-2 has a native observed/missing mask per
    input dimension and already treats sparse-by-design columns (e.g.
    NDVI/NDWI) this way — we apply the same treatment to weather
    covariates when a gap can't be safely interpolated. This avoids
    dropping rows that contain a good target value and healthy
    unrelated covariates.

    A hard error is raised only when an ENTIRE covariate column is NaN
    (no real value anywhere) — that indicates a broken data source, not
    ordinary missingness, and gives Chronos-2 nothing to anchor against.

    NOTE: assumes a single id/series (this project's current scope —
    one field). Revisit this if/when a second field is added.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    present_cols = [c for c in covariate_cols if c in df.columns]

    n_missing_before = int(df[present_cols].isna().sum().sum())
    if n_missing_before == 0:
        return df

    for col in present_cols:
        gap_limit = (column_max_gaps or {}).get(col, MAX_COVARIATE_FILL_GAP_DAYS)
        df[col] = _bounded_fill_series(df[col], gap_limit)
    n_filled = n_missing_before - int(df[present_cols].isna().sum().sum())

    still_missing_mask = df[present_cols].isna().any(axis=1)
    n_remaining = int(still_missing_mask.sum())

    if n_remaining:
        for col in present_cols:
            col_na_idx = df.index[df[col].isna()]
            if len(col_na_idx) == 0:
                continue
            col_valid_idx = df.index[~df[col].isna()]
            if len(col_valid_idx) == 0:
                raise ValueError(
                    f"Column '{col}' is entirely NaN — needs manual "
                    "investigation."
                )
            n_col_na = len(col_na_idx)
            log.warning(
                "Column '%s': %d cells left as NaN after bounded fill "
                "(%d-day limit), %d rows kept (NaN passthrough).",
                col, n_col_na, MAX_COVARIATE_FILL_GAP_DAYS, len(df),
            )

    n_nan_remaining = int(df[present_cols].isna().sum().sum())
    log.info(
        "Covariate gap handling: %d values bridged (<=%d day gaps), "
        "%d cells left as NaN, 0 rows dropped.",
        n_filled, MAX_COVARIATE_FILL_GAP_DAYS, n_nan_remaining,
    )
    return df
 
 
def forecast_soil_moisture(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    prediction_length: int = 7,
    quantile_levels: list[float] | None = None,
    covariate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run zero-shot forecasting with Chronos-2 on the target defined by
    forecasting.data_prep (iot_soil_moisture_mean, renamed "target").
 
    context_df / future_df must already be shaped by
    data_prep.to_chronos_context_df() / to_chronos_future_df() — same
    field, contiguous dates (checked below), matching column names.
 
    Returns Chronos-2's own output DataFrame: id, timestamp,
    target_name, predictions (median), and one column per requested
    quantile level.
    """
    from chronos import Chronos2Pipeline

    covariate_cols = covariate_cols or ALL_COVARIATE_COLS
    quantile_levels = quantile_levels or DEFAULT_QUANTILE_LEVELS

    # Only fill gaps in weather/IoT covariates — satellite columns (NDVI,
    # NDWI) are inherently sparse (Sentinel-2 ~5-day revisit) and have
    # natural internal gaps that the bounded fill cannot bridge without
    # raising. Chronos-2 handles NaN covariates natively.
    context_df = _fill_covariate_gaps(context_df, FUTURE_KNOWN_COLS)
    future_df = _fill_covariate_gaps(future_df, FUTURE_KNOWN_COLS)

    check_context_future_continuity(context_df, future_df)
 
    device = _select_device()
    log.info("Loading Chronos-2 (%s) on device=%s", MODEL_ID, device)
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)
 
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
 
    log.info(
        "Chronos-2 forecast: %d rows, horizon=%d days, quantiles=%s",
        len(pred_df), prediction_length, quantile_levels,
    )
    return pred_df

