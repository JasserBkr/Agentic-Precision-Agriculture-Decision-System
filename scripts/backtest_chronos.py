"""
Backtest Chronos-2: chronological train/test split, forecast the held-out
window, and output evaluation metrics against ground truth (IoT soil
moisture).
"""

import numpy as np
import pandas as pd

from agri_agent.forecasting.chronos_model import forecast_soil_moisture
from agri_agent.forecasting.data_prep import (
    TARGET_COL,
    to_chronos_context_df,
    to_chronos_future_df,
)
from agri_agent.forecasting.evaluate import (
    compute_metrics,
    historical_slice_to_future_df,
    temporal_train_test_split,
)
from agri_agent.utils.logging_config import get_logger
from run_pipeline import load_field_config, main as run_week1_2_pipeline

log = get_logger(__name__)

HORIZON_DAYS = 7


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(y_pred[mask] - y_true[mask]))


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _print_metrics(
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_dates: pd.Series,
) -> None:
    print("\n" + "=" * 55)
    print("  EVALUATION METRICS")
    print("=" * 55)
    print(f"  RMSE          : {metrics['rmse']:.6f}")
    print(f"  MAE           : {_mae(y_true, y_pred):.6f}")
    print(f"  Bias          : {_bias(y_true, y_pred):+.6f}")
    print(f"  R-squared     : {_r_squared(y_true, y_pred):.4f}")
    if not np.isnan(metrics["mase"]):
        print(f"  MASE          : {metrics['mase']:.4f}")
        verdict = "beats persistence" if metrics["mase"] < 1 else "worse than persistence"
        print(f"    -> MASE < 1 means model beats naive persistence ({verdict})")
    else:
        print("  MASE          : N/A")
    print(f"  Eval points   : {metrics['n_points']}")
    print("=" * 55)

    comparison = pd.DataFrame({
        "date": test_dates.values,
        "actual": y_true,
        "predicted": y_pred,
    })
    comparison["error"] = comparison["predicted"] - comparison["actual"]
    print("\n  Day-by-day comparison:")
    print(comparison.to_string(index=False))
    print()


def main():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Running Week 1-2 pipeline for fused data...")
    fused_df = run_week1_2_pipeline()

    # --- Backtest: chronological split and evaluate ---
    log.info("Running %d-day backtest...", HORIZON_DAYS)
    train_df, test_df = temporal_train_test_split(fused_df, HORIZON_DAYS)

    context_df = to_chronos_context_df(train_df)
    future_df = historical_slice_to_future_df(test_df, field_id)

    log.info("Running Chronos-2 forecast on test window...")
    pred_df = forecast_soil_moisture(context_df, future_df, prediction_length=HORIZON_DAYS)

    y_true = test_df.sort_values("date")[TARGET_COL].to_numpy()
    y_pred = pred_df.sort_values("timestamp")["predictions"].to_numpy()
    y_train = train_df.sort_values("date")[TARGET_COL].to_numpy()

    metrics = compute_metrics(y_true, y_pred, y_train)
    test_dates = test_df.sort_values("date")["date"]
    _print_metrics(metrics, y_true, y_pred, test_dates)

    # --- Forward forecast: next 7 days (no ground truth to compare) ---
    log.info("Running forward forecast (next 7 days)...")
    full_context_df = to_chronos_context_df(fused_df)
    forward_future_df = to_chronos_future_df(
        field_id=field_id,
        lat=field["centroid"]["lat"],
        lon=field["centroid"]["lon"],
        last_context_date=full_context_df["timestamp"].max(),
        horizon_days=HORIZON_DAYS,
    )
    forward_pred_df = forecast_soil_moisture(full_context_df, forward_future_df, prediction_length=HORIZON_DAYS)

    print("  Forward forecast (next 7 days):")
    print(forward_pred_df.to_string(index=False))

    return metrics


if __name__ == "__main__":
    main()
