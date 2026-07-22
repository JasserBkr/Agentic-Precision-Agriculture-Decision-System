"""
2-year canonical backtest: load the backfilled fused dataset and run
Chronos-2 through the same backtest harness used for the ~93-day
operational run — no modification to evaluate.py, data_prep.py, or
chronos_model.py required.

Usage:
    cd scripts && python backtest_chronos_2years.py
"""

import yaml

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import backtest_chronos
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Loading 2-year fused dataset from %s ...", FUSED_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)
    log.info("Loaded: %d rows, %d columns", *fused_df.shape)

    log.info("Running Chronos-2 backtest on 2-year data (horizon=%d days)...", HORIZON_DAYS)
    metrics = backtest_chronos(fused_df, field_id, horizon_days=HORIZON_DAYS)

    print()
    print("=" * 55)
    print("  2-YEAR CANONICAL BACKTEST")
    print("=" * 55)
    print(f"  RMSE      : {metrics['rmse']:.6f}")
    if not __import__("math").isnan(metrics["mase"]):
        print(f"  MASE      : {metrics['mase']:.4f}")
        verdict = "beats persistence" if metrics["mase"] < 1 else "worse than persistence"
        print(f"    -> MASE < 1 means model beats naive persistence ({verdict})")
    else:
        print("  MASE      : N/A")
    print(f"  Eval pts  : {metrics['n_points']}")
    print("=" * 55)
    print()
    print("  Comparison: ~93-day operational backtest MASE = 0.332")
    if not __import__("math").isnan(metrics["mase"]):
        diff = metrics["mase"] - 0.332
        if abs(diff) < 0.05:
            print("  2-year MASE is SIMILAR to the ~93-day result (delta={:+.4f})".format(diff))
        elif diff < 0:
            print("  2-year MASE IMPROVED vs ~93-day result (delta={:+.4f})".format(diff))
        else:
            print("  2-year MASE WORSENED vs ~93-day result (delta={:+.4f})".format(diff))
    print()

    return metrics


if __name__ == "__main__":
    main()
