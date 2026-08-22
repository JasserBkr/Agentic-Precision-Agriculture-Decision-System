"""
N-HiTS 2-year canonical backtest: load the backfilled fused dataset and run
N-HiTS through the same backtest harness used for Chronos-2 and TFT —
no modification to data_prep.py or nf_data_prep.py required.

Usage:
    python scripts/backtest_nhits.py
"""

import math
import time
import yaml

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import backtest_nhits
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7

# Corrected (2-covariate) numbers from prior leakage-fix task.
# Chronos-2 MASE from data_prep.py ablation comment (confirmed in
# 2-year backtest script): 0.1542.
CHRONOS2_MASE_CORRECTED = 0.1542

# TFT corrected MASE: confirmed via python scripts/backtest_tft.py run.
TFT_MASE_CORRECTED = 0.2918


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Loading 2-year fused dataset from %s ...", FUSED_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)
    log.info("Loaded: %d rows, %d columns", *fused_df.shape)

    log.info("Running N-HiTS backtest on 2-year data (horizon=%d days)...", HORIZON_DAYS)
    t0 = time.perf_counter()
    metrics = backtest_nhits(fused_df, field_id, horizon_days=HORIZON_DAYS)
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("  N-HITS 2-YEAR BACKTEST")
    print("=" * 60)
    print(f"  RMSE      : {metrics['rmse']:.6f}")
    if not math.isnan(metrics["mase"]):
        print(f"  MASE      : {metrics['mase']:.4f}")
        verdict = "beats persistence" if metrics["mase"] < 1 else "worse than persistence"
        print(f"    -> MASE < 1 means model beats naive persistence ({verdict})")
    else:
        print("  MASE      : N/A")
    print(f"  Eval pts  : {metrics['n_points']}")
    print(f"  Wall time : {elapsed:.1f}s")
    print("=" * 60)
    print()

    # ---- Three-way comparison (corrected 2-covariate feature set) ----
    print("=" * 60)
    print("  THREE-WAY COMPARISON  (corrected 2-covariate feature set)")
    print("  target: iot_soil_moisture_mean")
    print("  future-known: precipitation_sum, et0_fao_evapotranspiration")
    print("  past-only:    NDVI, NDWI")
    print("=" * 60)
    print("  Naive persistence MASE :  1.0000  (reference)")
    print(f"  Chronos-2  MASE        :  {CHRONOS2_MASE_CORRECTED:.4f}  (corrected, from prior ablation)")
    if TFT_MASE_CORRECTED is not None:
        print(f"  TFT        MASE        :  {TFT_MASE_CORRECTED:.4f}  (corrected)")
    else:
        print("  TFT        MASE        :  NEEDS CONFIRMATION  (run: python scripts/backtest_tft.py)")
    if not math.isnan(metrics["mase"]):
        print(f"  N-HiTS     MASE        :  {metrics['mase']:.4f}  (this run)")
    else:
        print("  N-HiTS     MASE        :  N/A")
    print()
    if not math.isnan(metrics["mase"]):
        print("  Lower MASE is better. All models use horizon_days=7,")
        print("  724-day training / 7-day test temporal split.")
        print("=" * 60)
    print()

    return metrics


if __name__ == "__main__":
    main()
