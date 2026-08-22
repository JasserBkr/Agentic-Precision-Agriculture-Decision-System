"""
Shared reshaping for NeuralForecast's expected format: long format with
unique_id, ds, y, plus covariate columns. Written to be reusable for
N-HiTS later too (identical input shape).

Mirrors data_prep.to_chronos_context_df() but with NeuralForecast's
column-naming convention (unique_id/ds/y instead of id/timestamp/target).
"""

import pandas as pd

from agri_agent.forecasting.data_prep import (
    FUTURE_KNOWN_COLS,
    PAST_ONLY_COLS,
    TARGET_COL,
)
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

ALL_COVARIATE_COLS = FUTURE_KNOWN_COLS + PAST_ONLY_COLS

# Sentinel-2 revisit is ~5 days, so a 2-day fill limit is safe —
# consistent with fusion.py's MAX_INTERPOLATION_GAP_DAYS.
_MAX_FILL_GAP = 2

# Generous but finite ceiling for the ffill/bfill fallback that handles
# remaining gaps after bounded interpolation (long Sentinel-2 revisits,
# leading/trailing edges). NDVI/NDWI change slowly enough that a ~10-day
# flat-fill is a defensible physical approximation, but a gap wider than
# that indicates something is wrong with the data, not just "a bit
# sparse".
_SATELLITE_FALLBACK_FILL_LIMIT = 10


def _bounded_fill(s: pd.Series, max_gap: int = _MAX_FILL_GAP) -> pd.Series:
    """
    Forward-fill then backward-fill NaN runs of length <= max_gap.
    Longer runs and leading/trailing NaN runs are left as-is.
    """
    s = s.copy()
    is_na = s.isna()
    if not is_na.any():
        return s

    group_id = (is_na != is_na.shift()).cumsum()
    for _, group in s[is_na].groupby(group_id[is_na]):
        idx = group.index
        run_len = len(idx)
        loc = s.index.get_loc(idx[0])
        has_before = loc > 0 and not pd.isna(s.iloc[loc - 1])
        has_after = loc + run_len < len(s) and not pd.isna(s.iloc[loc + run_len])

        if run_len <= max_gap and has_before and has_after:
            before_idx = s.index[loc - 1]
            after_idx = s.index[loc + run_len]
            s.loc[before_idx:after_idx] = s.loc[before_idx:after_idx].interpolate(
                method="linear"
            )
    return s


def _fill_covariate_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN gaps in ALL covariate columns (PAST_ONLY + FUTURE_KNOWN)
    using bounded linear interpolation. NeuralForecast requires no NaN
    in input covariates.
    """
    cols_to_fill = [c for c in ALL_COVARIATE_COLS if c in df.columns]
    n_before = int(df[cols_to_fill].isna().sum().sum())
    if n_before == 0:
        return df

    for col in cols_to_fill:
        df[col] = _bounded_fill(df[col])

    n_after_bounded = int(df[cols_to_fill].isna().sum().sum())
    log.info(
        "Covariate NaN fill: %d -> %d remaining after bounded fill (%d-day gap limit)",
        n_before, n_after_bounded, _MAX_FILL_GAP,
    )

    # NeuralForecast hard-fails if ANY input column has NaN — no
    # passthrough like Chronos has. For remaining gaps (>2-day Sentinel-2
    # revisits or leading/trailing edges), ffill/bfill is acceptable here
    # because these are sparse exogenous covariates, not the target.
    # Unlike chronos_model.py's earlier unbounded-fill bug, we are NOT
    # fabricating future history — this is training data only, and the
    # bounded fill already handled all short safe-to-interpolate gaps.
    # The limit= parameter prevents a single real value from propagating
    # indefinitely across an arbitrarily wide gap.
    still_na = df[cols_to_fill].isna().any(axis=1).sum()
    if still_na > 0:
        for col in cols_to_fill:
            df[col] = df[col].ffill(limit=_SATELLITE_FALLBACK_FILL_LIMIT)
            df[col] = df[col].bfill(limit=_SATELLITE_FALLBACK_FILL_LIMIT)
        n_final = int(df[cols_to_fill].isna().sum().sum())
        log.info(
            "Applied bounded ffill/bfill (limit=%d) for %d remaining rows, "
            "%d NaN cells left.",
            _SATELLITE_FALLBACK_FILL_LIMIT, still_na, n_final,
        )
        if n_final > 0:
            raise ValueError(
                f"{n_final} covariate cells still NaN after bounded fill "
                f"(interpolation limit={_MAX_FILL_GAP}, fallback limit="
                f"{_SATELLITE_FALLBACK_FILL_LIMIT}). Gap too wide to "
                "safely approximate — investigate the data source."
            )
    return df


def to_neuralforecast_df(fused_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape fused_df into NeuralForecast's long format: unique_id,
    ds, y, plus ALL covariate columns (both PAST_ONLY and FUTURE_KNOWN,
    since during the historical period all are observed).

    NaN gaps in covariates are filled via bounded linear interpolation
    (max 2-day gap) since NeuralForecast requires no NaN in inputs.
    This mirrors Chronos-2's _fill_covariate_gaps() logic.
    """
    keep_cols = ["date", "field_id", TARGET_COL] + ALL_COVARIATE_COLS
    missing = [c for c in keep_cols if c not in fused_df.columns]
    if missing:
        raise KeyError(f"fused_df is missing expected columns: {missing}")

    nf_df = fused_df[keep_cols].copy()
    nf_df = nf_df.rename(columns={
        "field_id": "unique_id",
        "date": "ds",
        TARGET_COL: "y",
    })
    nf_df["ds"] = pd.to_datetime(nf_df["ds"])
    nf_df = nf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    n_y_nan = int(nf_df["y"].isna().sum())
    if n_y_nan > 0:
        log.warning("Target column has %d NaN values — dropping those rows.", n_y_nan)
        nf_df = nf_df.dropna(subset=["y"]).reset_index(drop=True)

    nf_df = _fill_covariate_nans(nf_df)

    log.info(
        "Built NeuralForecast df: %d rows, y=%s, covariates=%s",
        len(nf_df), TARGET_COL, ALL_COVARIATE_COLS,
    )
    return nf_df


def historical_slice_to_futr_df(
    test_df: pd.DataFrame, field_id: str,
) -> pd.DataFrame:
    """
    Build NeuralForecast's futr_df from ALREADY-KNOWN historical data,
    for backtesting only — mirrors evaluate.historical_slice_to_future_df()
    exactly, just with unique_id/ds column names instead of id/timestamp.

    Only FUTURE_KNOWN_COLS included (never PAST_ONLY_COLS or y) — same
    reasoning as the Chronos-2 equivalent: during a backtest "the future"
    is actually the past you're pretending not to see yet.
    """
    missing = [c for c in FUTURE_KNOWN_COLS if c not in test_df.columns]
    if missing:
        raise KeyError(f"test_df is missing expected columns: {missing}")

    futr_df = test_df[["date"] + FUTURE_KNOWN_COLS].copy()
    futr_df.insert(0, "unique_id", field_id)
    futr_df = futr_df.rename(columns={"date": "ds"})
    futr_df["ds"] = pd.to_datetime(futr_df["ds"])
    return futr_df.sort_values("ds").reset_index(drop=True)
