"""
Rolling-origin backtest + Diebold-Mariano test for Chronos-2, TFT, N-HiTS
across the full 2-year dataset.

78 non-overlapping 7-day windows (min_train=180 days, step=7 days),
covering 2025-01-18 to 2026-07-18.  Each model retrained fresh per window.
Saves per-day results to data/processed/rolling_backtest_results.parquet.

Usage:
    python scripts/rolling_backtest_dm.py
"""

import time
import yaml

import pandas as pd

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import (
    compute_metrics,
    diebold_mariano_test,
    rolling_origin_backtest,
)
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
OUT_PARQUET   = "data/processed/rolling_backtest_results.parquet"
HORIZON_DAYS  = 7
MIN_TRAIN_DAYS = 180
STEP_DAYS     = 7


def load_field_config(path="configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pooled_metrics(results_df: pd.DataFrame, full_fused_df: pd.DataFrame,
                   model_name: str) -> dict:
    """
    Compute MASE/RMSE for one model by pooling ALL test-day errors across
    all windows.  y_train is the FULL dataset's target series (used only
    for the naive-persistence MASE denominator).
    """
    sub = results_df[results_df["model_type"] == model_name].copy()
    sub = sub.sort_values("date")
    y_true = sub["actual"].to_numpy()
    y_pred = sub["predicted"].to_numpy()
    y_train = full_fused_df.sort_values("date")["iot_soil_moisture_mean"].to_numpy()
    return compute_metrics(y_true, y_pred, y_train)


def main():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Loading dataset from %s ...", FUSED_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)
    log.info("Loaded %d rows, %s to %s",
             len(fused_df), fused_df["date"].min().date(), fused_df["date"].max().date())

    all_results = []

    for model_type in ["chronos", "tft", "nhits"]:
        print(f"\n{'=' * 64}")
        print(f"  ROLLING-ORIGIN BACKTEST: {model_type.upper()}")
        print(f"  horizon={HORIZON_DAYS}  min_train={MIN_TRAIN_DAYS}  step={STEP_DAYS}")
        print(f"{'=' * 64}")
        t0 = time.perf_counter()
        results = rolling_origin_backtest(
            fused_df, field_id, model_type,
            horizon_days=HORIZON_DAYS,
            step_days=STEP_DAYS,
            min_train_days=MIN_TRAIN_DAYS,
        )
        elapsed = time.perf_counter() - t0
        n_windows = results["date"].nunique()
        print(f"  {model_type.upper():>10} completed: {n_windows} windows, "
              f"{len(results)} test days, {elapsed:.1f}s wall time")
        results["_wall_seconds"] = elapsed
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved {len(combined)} rows to {OUT_PARQUET}")

    # ---- Pooled MASE/RMSE per model ----
    print(f"\n{'=' * 64}")
    print("  POOLED MASE / RMSE  (all windows combined, single compute_metrics call)")
    print(f"{'=' * 64}")
    pooled = {}
    for model_type in ["chronos", "tft", "nhits"]:
        m = pooled_metrics(combined, fused_df, model_type)
        pooled[model_type] = m
        n_days = len(combined[combined["model_type"] == model_type])
        wall   = combined[combined["model_type"] == model_type]["_wall_seconds"].iloc[0]
        print(f"  {model_type.upper():>10}  MASE={m['mase']:.4f}  "
              f"RMSE={m['rmse']:.6f}  days={n_days}  time={wall:.1f}s")

    # ---- Pairwise DM tests ----
    print(f"\n{'=' * 64}")
    print("  DIEBOLD-MARIANO TESTS  (aligned on shared dates)")
    print(f"{'=' * 64}")

    # Build per-model date→error series, aligned to common dates
    pivot = combined.pivot_table(
        index="date", columns="model_type", values=["actual", "predicted"],
    )
    # Only keep dates where all three models have results
    common_dates = pivot.dropna().index
    print(f"  Common test dates: {len(common_dates)} (out of "
          f"{combined['date'].nunique()} total unique dates)")

    pairs = [("chronos", "tft"), ("chronos", "nhits"), ("tft", "nhits")]
    for ma, mb in pairs:
        ea = (pivot.loc[common_dates, ("actual", ma)]
                     - pivot.loc[common_dates, ("predicted", ma)]).values
        eb = (pivot.loc[common_dates, ("actual", mb)]
                     - pivot.loc[common_dates, ("predicted", mb)]).values

        dm_stat, p_value = diebold_mariano_test(ea, eb, horizon=HORIZON_DAYS)
        sig = "YES" if p_value < 0.05 else "NO"
        better = ma if dm_stat < 0 else mb
        print(f"\n  {ma.upper():>10} vs {mb.upper():>10}")
        print(f"    DM stat  = {dm_stat:+.4f}")
        print(f"    p-value  = {p_value:.4f}")
        print(f"    Significant at 5%: {sig}")
        if p_value < 0.05:
            print(f"    -> {better.upper()} is significantly better (p={p_value:.4f})")
        else:
            print(f"    -> No significant difference detected (p={p_value:.4f})")

    # ---- Final summary ----
    print(f"\n{'=' * 64}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 64}")
    ranked = sorted(pooled.items(), key=lambda x: x[1]["mase"])
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  #{i}  {name.upper():>10}  MASE={m['mase']:.4f}  RMSE={m['rmse']:.6f}")
    print(f"{'=' * 64}\n")

    return combined


if __name__ == "__main__":
    main()
