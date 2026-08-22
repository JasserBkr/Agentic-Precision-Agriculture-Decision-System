"""
TFT 2-year canonical backtest: load the backfilled fused dataset and run
TFT through the same backtest harness used for Chronos-2 — no modification
to evaluate.py (additive only), data_prep.py, or chronos_model.py required.

Usage:
    cd scripts && python backtest_tft.py
"""

import yaml

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import backtest_tft
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7

CHRONOS2_MASE = 0.211


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Loading 2-year fused dataset from %s ...", FUSED_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)
    log.info("Loaded: %d rows, %d columns", *fused_df.shape)

    log.info("Running TFT backtest on 2-year data (horizon=%d days)...", HORIZON_DAYS)
    metrics = backtest_tft(fused_df, field_id, horizon_days=HORIZON_DAYS)

    print()
    print("=" * 60)
    print("  TFT 2-YEAR BACKTEST")
    print("=" * 60)
    print(f"  RMSE      : {metrics['rmse']:.6f}")
    if not __import__("math").isnan(metrics["mase"]):
        print(f"  MASE      : {metrics['mase']:.4f}")
        verdict = "beats persistence" if metrics["mase"] < 1 else "worse than persistence"
        print(f"    -> MASE < 1 means model beats naive persistence ({verdict})")
    else:
        print("  MASE      : N/A")
    print(f"  Eval pts  : {metrics['n_points']}")
    print("=" * 60)
    print()

    print(f"  Comparison: Chronos-2 2-year backtest MASE = {CHRONOS2_MASE}")
    if not __import__("math").isnan(metrics["mase"]):
        diff = metrics["mase"] - CHRONOS2_MASE
        if abs(diff) < 0.05:
            print(f"  TFT MASE is COMPARABLE to Chronos-2 (delta={diff:+.4f})")
        elif diff < 0:
            print(f"  TFT MASE BEATS Chronos-2 (delta={diff:+.4f}, lower is better)")
        else:
            print(f"  TFT MASE is WORSE than Chronos-2 (delta={diff:+.4f}, lower is better)")
    print()

    return metrics


if __name__ == "__main__":
    main()
