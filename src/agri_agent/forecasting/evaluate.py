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
    mae_val = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

    naive_errors = np.abs(np.diff(y_train[~np.isnan(y_train)]))
    if len(naive_errors) == 0 or naive_errors.mean() == 0:
        mase = float("nan")
        log.warning("Cannot compute MASE: no valid naive baseline from training data.")
    else:
        mase = float(mae_val / naive_errors.mean())

    return {"rmse": rmse, "mae": mae_val, "mase": mase, "n_points": int(mask.sum())}
 
 
def backtest_tft(
    fused_df: pd.DataFrame,
    field_id: str,
    horizon_days: int = 7,
    return_predictions: bool = False,
) -> dict | tuple[dict, np.ndarray, np.ndarray]:
    """
    Full TFT backtest on a single fixed holdout at the END of the series.
    Delegates the actual training/prediction to _forecast_single_window().
    """
    train_df, test_df = temporal_train_test_split(fused_df, horizon_days)
    y_true, y_pred = _forecast_single_window(
        train_df, test_df, field_id, "tft", horizon_days,
    )
    y_train = train_df.sort_values("date")[TARGET_COL].to_numpy()
    metrics = compute_metrics(y_true, y_pred, y_train)
    log.info("TFT backtest (%d-day horizon): %s", horizon_days, metrics)
    if return_predictions:
        return metrics, y_true, y_pred
    return metrics


def backtest_nhits(
    fused_df: pd.DataFrame,
    field_id: str,
    horizon_days: int = 7,
    return_predictions: bool = False,
) -> dict | tuple[dict, np.ndarray, np.ndarray]:
    """
    Full N-HiTS backtest on a single fixed holdout at the END of the series.
    Delegates the actual training/prediction to _forecast_single_window().
    """
    train_df, test_df = temporal_train_test_split(fused_df, horizon_days)
    y_true, y_pred = _forecast_single_window(
        train_df, test_df, field_id, "nhits", horizon_days,
    )
    y_train = train_df.sort_values("date")[TARGET_COL].to_numpy()
    metrics = compute_metrics(y_true, y_pred, y_train)
    log.info("N-HiTS backtest (%d-day horizon): %s", horizon_days, metrics)
    if return_predictions:
        return metrics, y_true, y_pred
    return metrics


def _forecast_single_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    field_id: str,
    model_type: str,
    horizon_days: int = 7,
    chronos_pipeline=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an arbitrary pre-split train_df and test_df (both containing the
    full fused_df columns including date, field_id, TARGET_COL, and all
    covariates), run the specified model's training + prediction pipeline
    and return (y_true, y_pred) as numpy arrays.

    This is the single-dispatch point reused by rolling_origin_backtest()
    and by the existing backtest_tft/backtest_nhits/backtest_chronos
    wrappers. It does NOT call temporal_train_test_split() — that is the
    caller's responsibility when they need a mid-series cut.
    """
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    if model_type == "chronos":
        from agri_agent.forecasting.chronos_model import (
            _fill_covariate_gaps,
            MODEL_ID,
            _select_device,
        )
        from chronos import Chronos2Pipeline

        context_df = to_chronos_context_df(train_df)
        futr_df = historical_slice_to_future_df(test_df, field_id)
        context_df = _fill_covariate_gaps(
            context_df,
            ["precipitation_sum", "et0_fao_evapotranspiration"],
        )
        futr_df = _fill_covariate_gaps(
            futr_df,
            ["precipitation_sum", "et0_fao_evapotranspiration"],
        )
        if chronos_pipeline is not None:
            pipeline = chronos_pipeline
        else:
            device = _select_device()
            pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)

        pred_df = pipeline.predict_df(
            context_df,
            future_df=futr_df,
            prediction_length=horizon_days,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        y_true = test_df[TARGET_COL].to_numpy()
        y_pred = pred_df.sort_values("timestamp")["predictions"].to_numpy()

    elif model_type == "tft":
        from agri_agent.forecasting import nf_data_prep, tft_model

        nf_train_df = nf_data_prep.to_neuralforecast_df(train_df)
        nf = tft_model.train_tft(nf_train_df, horizon_days)
        futr_df = nf_data_prep.historical_slice_to_futr_df(test_df, field_id)
        pred_df = tft_model.predict_tft(nf, futr_df)
        pred_col = [c for c in pred_df.columns if c not in ("unique_id", "ds")][0]
        y_true = test_df[TARGET_COL].to_numpy()
        y_pred = pred_df.sort_values("ds")[pred_col].to_numpy()

    elif model_type == "nhits":
        from agri_agent.forecasting import nhits_model, nf_data_prep

        nf_train_df = nf_data_prep.to_neuralforecast_df(train_df)
        nf = nhits_model.train_nhits(nf_train_df, horizon_days)
        futr_df = nf_data_prep.historical_slice_to_futr_df(test_df, field_id)
        pred_df = nhits_model.predict_nhits(nf, futr_df)
        pred_col = [c for c in pred_df.columns if c not in ("unique_id", "ds")][0]
        y_true = test_df[TARGET_COL].to_numpy()
        y_pred = pred_df.sort_values("ds")[pred_col].to_numpy()

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'chronos', 'tft', or 'nhits'.")

    return y_true, y_pred


def rolling_origin_backtest(
    fused_df: pd.DataFrame,
    field_id: str,
    model_type: str,
    horizon_days: int = 7,
    step_days: int = 7,
    min_train_days: int = 180,
) -> pd.DataFrame:
    """
    Slides a horizon_days-length test window across fused_df, starting as
    early as min_train_days allows and stepping forward by step_days each
    time, until the window would run past the end of the dataset.

    For EACH window:
      1. train_df = fused_df[fused_df["date"] < window_start]
         test_df  = fused_df[(fused_df["date"] >= window_start)
                             & (fused_df["date"] < window_start + horizon_days)]
         (date-based slicing — does NOT use temporal_train_test_split,
          which only cuts at the end of the series)
      2. Skip any window where test_df has fewer than horizon_days rows.
      3. Run the appropriate model via _forecast_single_window().
      4. Record per-day absolute errors for ALL horizon_days in that window.

    Returns a DataFrame with one row per TEST DAY:
    columns = date, model_type, actual, predicted, abs_error.
    """
    fused_df = fused_df.sort_values("date").reset_index(drop=True)
    dates = fused_df["date"].values
    date_min = pd.Timestamp(dates[0])
    date_max = pd.Timestamp(dates[-1])

    # First possible window start: enough training data before it
    first_window_start = date_min + pd.Timedelta(days=min_train_days)
    # Last possible window start: window must fit within the dataset
    last_window_start = date_max - pd.Timedelta(days=horizon_days - 1)

    window_starts = []
    cursor = first_window_start
    while cursor <= last_window_start:
        window_starts.append(cursor)
        cursor += pd.Timedelta(days=step_days)

    total_windows = len(window_starts)
    log.info(
        "rolling_origin_backtest(%s): %d windows, min_train=%d, "
        "window range %s to %s, step=%d",
        model_type, total_windows, min_train_days,
        window_starts[0].date(), window_starts[-1].date(), step_days,
    )

    rows = []
    chronos_pipeline = None
    if model_type == "chronos":
        from agri_agent.forecasting.chronos_model import MODEL_ID, _select_device
        from chronos import Chronos2Pipeline
        device = _select_device()
        log.info("Pre-loading Chronos-2 pipeline (%s) once for all windows...", MODEL_ID)
        chronos_pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)

    for i, ws in enumerate(window_starts):
        we = ws + pd.Timedelta(days=horizon_days)
        log.info(
            "[%s] Window %d/%d: %s to %s",
            model_type, i + 1, total_windows, ws.date(), (we - pd.Timedelta(days=1)).date(),
        )

        train_df = fused_df[fused_df["date"] < ws].copy()
        test_df = fused_df[
            (fused_df["date"] >= ws) & (fused_df["date"] < we)
        ].copy()

        if len(test_df) < horizon_days:
            log.warning(
                "  Skipping window %s: only %d test rows (need %d)",
                ws.date(), len(test_df), horizon_days,
            )
            continue

        if len(train_df) < min_train_days:
            log.warning(
                "  Skipping window %s: only %d train rows (need %d)",
                ws.date(), len(train_df), min_train_days,
            )
            continue

        # ── Standard dispatch path ──────────────────────────────────────
        try:
            y_true, y_pred = _forecast_single_window(
                train_df, test_df, field_id, model_type, horizon_days,
                chronos_pipeline=chronos_pipeline,
            )
        except (ValueError, KeyError) as exc:
            log.warning(
                "  Skipping window %s: model prep failed (%s)",
                ws.date(), exc,
            )
            continue

        test_dates = test_df.sort_values("date")["date"].to_numpy()
        for d, yt, yp in zip(test_dates, y_true, y_pred):
            rows.append({
                "date": pd.Timestamp(d),
                "model_type": model_type,
                "actual": float(yt),
                "predicted": float(yp),
                "abs_error": float(abs(yt - yp)),
            })

    result_df = pd.DataFrame(rows)
    log.info(
        "rolling_origin_backtest(%s): completed %d windows, %d test days",
        model_type, total_windows, len(result_df),
    )

    return result_df


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 7,
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: two models have equal forecast accuracy.
    H1 (two-sided): they differ.

    d_t = e_a_t^2 - e_b_t^2  (positive d_bar means model B is more accurate)
    DM  = d_bar / sqrt(V_d / T)
    where V_d uses a Newey-West HAC estimator with bandwidth = horizon - 1
    to account for h-step-ahead overlapping forecast errors.

    Returns (dm_statistic, two-sided p_value).
    A significantly negative DM means model A is more accurate;
    significantly positive means model B is more accurate.
    """
    from scipy import stats as sp_stats

    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    T = len(errors_a)

    d = errors_a ** 2 - errors_b ** 2
    d_bar = d.mean()

    # Newey-West HAC variance with bandwidth = horizon - 1
    q = horizon - 1
    gamma_0 = np.var(d, ddof=1)
    V_d = gamma_0
    for k in range(1, q + 1):
        w = 1 - k / (q + 1)  # Bartlett kernel weights
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / T
        V_d += 2 * w * gamma_k

    se = np.sqrt(V_d / T)
    if se == 0:
        dm_stat = np.inf if d_bar > 0 else (-np.inf if d_bar < 0 else 0.0)
        p_value = 1.0
    else:
        dm_stat = d_bar / se
        p_value = 2 * sp_stats.t.sf(abs(dm_stat), df=T - 1)

    return float(dm_stat), float(p_value)


def backtest_chronos(fused_df: pd.DataFrame, field_id: str, horizon_days: int = 7) -> dict:
    """
    Full Chronos-2 backtest on a single fixed holdout at the END of the series.
    Delegates the actual training/prediction to _forecast_single_window().
    """
    train_df, test_df = temporal_train_test_split(fused_df, horizon_days)
    y_true, y_pred = _forecast_single_window(
        train_df, test_df, field_id, "chronos", horizon_days,
    )
    y_train = train_df.sort_values("date")[TARGET_COL].to_numpy()
    metrics = compute_metrics(y_true, y_pred, y_train)
    log.info("Chronos-2 backtest (%d-day horizon): %s", horizon_days, metrics)
    return metrics

