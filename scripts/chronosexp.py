"""
Chronos-2 backtest: target-only vs standard (with covariates) on the
2-year fused dataset.

Runs two Chronos-2 backtests on the same chronological train/test split:
  1. STANDARD  — context_df / future_df include all covariates
  2. TARGET-ONLY — only the target column, no covariates

Side-by-side metric comparison shows how much the covariates help.

Usage:
    cd scripts && python chronosexp.py
"""

import math

import numpy as np
import pandas as pd
import yaml

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.chronos_model import forecast_soil_moisture
from agri_agent.forecasting.data_prep import (
    TARGET_COL,
    to_chronos_context_df,
)
from agri_agent.forecasting.evaluate import (
    compute_metrics,
    historical_slice_to_future_df,
    temporal_train_test_split,
)
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── helpers for target-only DataFrames ──────────────────────────────────────


def _to_target_only_df(fused_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only id/date/target columns — drop every covariate."""
    df = fused_df[["date", "field_id", TARGET_COL]].copy()
    df = df.rename(columns={
        "field_id": "id",
        "date": "timestamp",
        TARGET_COL: "target",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return df


def _build_target_only_future_df(
    test_df: pd.DataFrame, field_id: str
) -> pd.DataFrame:
    """Build a future_df with only id and timestamp (no covariates)."""
    future_df = test_df[["date"]].copy()
    future_df.insert(0, "id", field_id)
    future_df = future_df.rename(columns={"date": "timestamp"})
    future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
    return future_df.sort_values("timestamp").reset_index(drop=True)


# ── metric helpers ──────────────────────────────────────────────────────────


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


def _print_single_metrics(
    label: str,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_dates: pd.Series,
) -> None:
    print("\n" + "=" * 60)
    print(f"  EVALUATION METRICS  — {label}")
    print("=" * 60)
    print(f"  RMSE          : {metrics['rmse']:.6f}")
    print(f"  MAE           : {_mae(y_true, y_pred):.6f}")
    print(f"  Bias          : {_bias(y_true, y_pred):+.6f}")
    print(f"  R-squared     : {_r_squared(y_true, y_pred):.4f}")
    if not math.isnan(metrics["mase"]):
        print(f"  MASE          : {metrics['mase']:.4f}")
        verdict = "beats persistence" if metrics["mase"] < 1 else "worse than persistence"
        print(f"    -> MASE < 1 means model beats naive persistence ({verdict})")
    else:
        print("  MASE          : N/A")
    print(f"  Eval points   : {metrics['n_points']}")
    print("=" * 60)

    comparison = pd.DataFrame({
        "date": test_dates.values,
        "actual": y_true,
        "predicted": y_pred,
    })
    comparison["error"] = comparison["predicted"] - comparison["actual"]
    print("\n  Day-by-day comparison:")
    print(comparison.to_string(index=False))
    print()


def _print_comparison(std_metrics: dict, tgt_metrics: dict) -> None:
    print("\n" + "#" * 60)
    print("  SIDE-BY-SIDE COMPARISON  (standard vs target-only)")
    print("#" * 60)
    print(f"  {'Metric':<16} {'Standard':>12} {'Target-Only':>12} {'Delta':>12}")
    print("  " + "-" * 54)

    for key, label in [("rmse", "RMSE"), ("mase", "MASE")]:
        sv = std_metrics[key]
        tv = tgt_metrics[key]
        sd = f"{sv:.6f}" if not math.isnan(sv) else "N/A"
        td = f"{tv:.6f}" if not math.isnan(tv) else "N/A"
        if not math.isnan(sv) and not math.isnan(tv):
            d = tv - sv
            dd = f"{d:+.6f}"
        else:
            dd = "N/A"
        print(f"  {label:<16} {sd:>12} {td:>12} {dd:>12}")

    if not math.isnan(std_metrics["mase"]) and not math.isnan(tgt_metrics["mase"]):
        diff = tgt_metrics["mase"] - std_metrics["mase"]
        if abs(diff) < 0.01:
            print("\n  -> Covariates make almost no difference for this split.")
        elif diff > 0:
            print(f"\n  -> Target-only is WORSE by {diff:+.4f} MASE — covariates help.")
        else:
            print(f"\n  -> Target-only is BETTER by {-diff:+.4f} MASE — covariates hurt (noise?).")
    print("#" * 60)
    print()


# ── main ────────────────────────────────────────────────────────────────────


def main():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Loading 2-year fused dataset from %s ...", FUSED_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)
    log.info("Loaded: %d rows, %d columns", *fused_df.shape)

    log.info("Running %d-day temporal train/test split ...", HORIZON_DAYS)
    train_df, test_df = temporal_train_test_split(fused_df, HORIZON_DAYS)
    y_train = train_df.sort_values("date")[TARGET_COL].to_numpy()

    # ── 1. STANDARD: with covariates ────────────────────────────────────────
    log.info("Running Chronos-2 backtest WITH covariates ...")
    std_context_df = to_chronos_context_df(train_df)
    std_future_df = historical_slice_to_future_df(test_df, field_id)
    std_pred_df = forecast_soil_moisture(
        std_context_df, std_future_df, prediction_length=HORIZON_DAYS,
    )

    y_true = test_df.sort_values("date")[TARGET_COL].to_numpy()
    y_pred_std = std_pred_df.sort_values("timestamp")["predictions"].to_numpy()
    test_dates = test_df.sort_values("date")["date"]

    std_metrics = compute_metrics(y_true, y_pred_std, y_train)
    _print_single_metrics("STANDARD (with covariates)", std_metrics, y_true, y_pred_std, test_dates)

    # ── 2. TARGET-ONLY: no covariates ───────────────────────────────────────
    log.info("Running Chronos-2 backtest TARGET-ONLY (no covariates) ...")
    tgt_context_df = _to_target_only_df(train_df)
    tgt_future_df = _build_target_only_future_df(test_df, field_id)
    tgt_pred_df = forecast_soil_moisture(
        tgt_context_df, tgt_future_df, prediction_length=HORIZON_DAYS,
    )

    y_pred_tgt = tgt_pred_df.sort_values("timestamp")["predictions"].to_numpy()
    tgt_metrics = compute_metrics(y_true, y_pred_tgt, y_train)
    _print_single_metrics("TARGET-ONLY (no covariates)", tgt_metrics, y_true, y_pred_tgt, test_dates)

    # ── 3. Comparison ───────────────────────────────────────────────────────
    _print_comparison(std_metrics, tgt_metrics)

    return std_metrics, tgt_metrics


if __name__ == "__main__":
    main()
