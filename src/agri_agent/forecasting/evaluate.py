"""
Week 3-4 deliverable: backtesting harness for Chronos-2 (and later, the
TFT/N-HiTS baseline) — the only way to know whether a forecast is any
good, since real future dates have no ground truth yet to check against.
 
Implements the temporal train/test split disciplined against data
leakage (SOTA note Section 6.2): a chronological split, never random,
since fused_df's rows are autocorrelated in time.
"""
 
import numpy as np
import pandas as pd
 
from agri_agent.forecasting.data_prep import FUTURE_KNOWN_COLS, TARGET_COL, to_chronos_context_df
from agri_agent.utils.logging_config import get_logger
 
log = get_logger(__name__)
 
 
def temporal_train_test_split(
    fused_df: pd.DataFrame, horizon_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: the LAST horizon_days rows become the held-out
    test set, everything before becomes train/context. Never a random
    shuffle split — a random split would leak future information into
    training on autocorrelated time-series data.
    """
    fused_df = fused_df.sort_values("date").reset_index(drop=True)
    if len(fused_df) <= horizon_days:
        raise ValueError(
            f"fused_df has only {len(fused_df)} rows, not enough to hold "
            f"out {horizon_days} days for testing."
        )
    train_df = fused_df.iloc[:-horizon_days].copy()
    test_df = fused_df.iloc[-horizon_days:].copy()
 
    log.info(
        "Temporal split: %d train rows (%s to %s), %d test rows (%s to %s)",
        len(train_df), train_df["date"].min().date(), train_df["date"].max().date(),
        len(test_df), test_df["date"].min().date(), test_df["date"].max().date(),
    )
    return train_df, test_df
 
 
def historical_slice_to_future_df(test_df: pd.DataFrame, field_id: str) -> pd.DataFrame:
    """
    Build a future_df-shaped table from ALREADY-KNOWN historical data,
    for backtesting only. Unlike data_prep.to_chronos_future_df (which
    fetches a live weather forecast for real production use), this
    reuses real recorded values from the held-out test period —
    legitimate here specifically because during a backtest, "the
    future" is actually the past.
    """
    missing = [c for c in FUTURE_KNOWN_COLS if c not in test_df.columns]
    if missing:
        raise KeyError(f"test_df is missing expected columns: {missing}")
 
    future_df = test_df[["date"] + FUTURE_KNOWN_COLS].copy()
    future_df.insert(0, "id", field_id)
    future_df = future_df.rename(columns={"date": "timestamp"})
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
    return future_df.sort_values("timestamp").reset_index(drop=True)
 
 
def compute_metrics(y_true, y_pred, y_train) -> dict:
    """
    RMSE and MASE (Mean Absolute Scaled Error).
 
    MASE scales the forecast's mean absolute error against a naive
    one-step-persistence baseline computed from the TRAINING data:
    MASE < 1 means the model beats naive persistence, MASE > 1 means it
    doesn't. Scale-independent and standard for comparing forecasters —
    this is what makes the Chronos-2 vs. TFT/N-HiTS comparison
    (SOTA note Section 4.2) an actual number, not just eyeballing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
 
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        raise ValueError("No overlapping non-NaN values between y_true and y_pred.")
 
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
 
    naive_errors = np.abs(np.diff(y_train[~np.isnan(y_train)]))
    if len(naive_errors) == 0 or naive_errors.mean() == 0:
        mase = float("nan")
        log.warning("Cannot compute MASE: no valid naive baseline from training data.")
    else:
        mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        mase = float(mae / naive_errors.mean())
 
    return {"rmse": rmse, "mase": mase, "n_points": int(mask.sum())}
 
 
def backtest_chronos(fused_df: pd.DataFrame, field_id: str, horizon_days: int = 7) -> dict:
    """
    Full backtest: split fused_df chronologically, forecast the held-out
    window with Chronos-2 using REAL historical values as "future"
    covariates, and score against the REAL known target values for that
    window. This is the only way to know if the forecast is any good —
    a forecast of genuinely future dates has nothing to compare against.
    """
    from agri_agent.forecasting.chronos_model import forecast_soil_moisture
 
    train_df, test_df = temporal_train_test_split(fused_df, horizon_days)
 
    context_df = to_chronos_context_df(train_df)
    future_df = historical_slice_to_future_df(test_df, field_id)
 
    pred_df = forecast_soil_moisture(context_df, future_df, prediction_length=horizon_days)
 
    y_true = test_df.sort_values("date")[TARGET_COL].to_numpy()
    y_pred = pred_df.sort_values("timestamp")["predictions"].to_numpy()
    y_train = train_df.sort_values("date")[TARGET_COL].to_numpy()
 
    metrics = compute_metrics(y_true, y_pred, y_train)
    log.info("Chronos-2 backtest (%d-day horizon): %s", horizon_days, metrics)
    return metrics

