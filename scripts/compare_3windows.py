"""
3-window x 3-model comparative backtest: run Chronos-2, TFT, and N-HiTS
against three non-overlapping 7-day holdout windows, retraining TFT/N-HiTS
fresh for each window using only data strictly before it (no leakage).

Windows:
  W1: 2026-07-02 to 2026-07-08
  W2: 2026-07-09 to 2026-07-15
  W3: 2026-07-16 to 2026-07-22  (the original canonical window)

All models use the corrected 2-covariate feature set (post-leakage-fix).

Usage:
    python scripts/compare_3windows.py
"""

import time
import yaml

import pandas as pd

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import (
    backtest_chronos,
    backtest_nhits,
    backtest_tft,
)
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7

WINDOWS = [
    ("W1", "2026-07-02", "2026-07-08"),
    ("W2", "2026-07-09", "2026-07-15"),
    ("W3", "2026-07-16", "2026-07-22"),
]


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def slice_for_window(fused_df: pd.DataFrame, window_start: str) -> pd.DataFrame:
    """
    Return fused_df with only rows strictly BEFORE window_start.
    The backtest function will then take the last 7 rows of this slice
    as the test set — which is exactly the 7-day window we want to test.
    """
    cutoff = pd.Timestamp(window_start)
    before = fused_df[fused_df["date"] < cutoff].copy()
    return before


def run_all():
    field = load_field_config()
    field_id = field["field_id"]

    log.info("Loading 2-year fused dataset from %s ...", FUSED_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)
    log.info("Loaded: %d rows, date range %s to %s",
             len(fused_df),
             fused_df["date"].min().date(),
             fused_df["date"].max().date())

    results = {}  # {(window_label, model): {"mase": ..., "rmse": ...}}

    for win_label, win_start, win_end in WINDOWS:
        print(f"\n{'=' * 64}")
        print(f"  WINDOW {win_label}: {win_start} to {win_end}")
        print(f"{'=' * 64}")

        sliced = slice_for_window(fused_df, win_start)
        n_train = len(sliced)
        log.info("Window %s: fused_df sliced to %d rows (all before %s)",
                 win_label, n_train, win_start)

        # Chronos-2 (zero-shot, but function reloads model each call)
        print(f"  [{win_label}] Chronos-2 ...", end=" ", flush=True)
        t0 = time.perf_counter()
        m_chronos = backtest_chronos(sliced, field_id, horizon_days=HORIZON_DAYS)
        dt = time.perf_counter() - t0
        results[(win_label, "Chronos-2")] = m_chronos
        print(f"MASE={m_chronos['mase']:.4f}  RMSE={m_chronos['rmse']:.6f}  ({dt:.1f}s)")

        # TFT (retrained fresh)
        print(f"  [{win_label}] TFT       ...", end=" ", flush=True)
        t0 = time.perf_counter()
        m_tft = backtest_tft(sliced, field_id, horizon_days=HORIZON_DAYS)
        dt = time.perf_counter() - t0
        results[(win_label, "TFT")] = m_tft
        print(f"MASE={m_tft['mase']:.4f}  RMSE={m_tft['rmse']:.6f}  ({dt:.1f}s)")

        # N-HiTS (retrained fresh)
        print(f"  [{win_label}] N-HiTS    ...", end=" ", flush=True)
        t0 = time.perf_counter()
        m_nhits = backtest_nhits(sliced, field_id, horizon_days=HORIZON_DAYS)
        dt = time.perf_counter() - t0
        results[(win_label, "N-HiTS")] = m_nhits
        print(f"MASE={m_nhits['mase']:.4f}  RMSE={m_nhits['rmse']:.6f}  ({dt:.1f}s)")

    # ---- Summary table ----
    print(f"\n{'=' * 64}")
    print("  3-WINDOW x 3-MODEL MASE TABLE  (corrected 2-covariate set)")
    print(f"{'=' * 64}")
    print(f"  {'Window':<20} {'Chronos-2':>10} {'TFT':>10} {'N-HiTS':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for win_label, win_start, win_end in WINDOWS:
        c = results[(win_label, "Chronos-2")]["mase"]
        t = results[(win_label, "TFT")]["mase"]
        n = results[(win_label, "N-HiTS")]["mase"]
        tag = f"{win_label} ({win_start}..)"
        print(f"  {tag:<20} {c:>10.4f} {t:>10.4f} {n:>10.4f}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    # Averages
    avg_c = sum(results[(w, "Chronos-2")]["mase"] for w, _, _ in WINDOWS) / len(WINDOWS)
    avg_t = sum(results[(w, "TFT")]["mase"] for w, _, _ in WINDOWS) / len(WINDOWS)
    avg_n = sum(results[(w, "N-HiTS")]["mase"] for w, _, _ in WINDOWS) / len(WINDOWS)
    print(f"  {'Average':<20} {avg_c:>10.4f} {avg_t:>10.4f} {avg_n:>10.4f}")
    print()

    # ---- Ranking analysis ----
    print(f"{'=' * 64}")
    print("  RANKING ANALYSIS")
    print(f"{'=' * 64}")
    for win_label, _, _ in WINDOWS:
        scores = {
            "Chronos-2": results[(win_label, "Chronos-2")]["mase"],
            "TFT":       results[(win_label, "TFT")]["mase"],
            "N-HiTS":    results[(win_label, "N-HiTS")]["mase"],
        }
        ranked = sorted(scores.items(), key=lambda x: x[1])
        best_name, best_mase = ranked[0]
        worst_name, worst_mase = ranked[2]
        print(f"  {win_label}: {best_name} ({best_mase:.4f}) < "
              f"{ranked[1][0]} ({ranked[1][1]:.4f}) < "
              f"{worst_name} ({worst_mase:.4f})")
    print()

    chronos_wins = sum(
        1 for w, _, _ in WINDOWS
        if results[(w, "Chronos-2")]["mase"] < results[(w, "TFT")]["mase"]
        and results[(w, "Chronos-2")]["mase"] < results[(w, "N-HiTS")]["mase"]
    )
    print(f"  Chronos-2 is best in {chronos_wins}/{len(WINDOWS)} windows.")
    if chronos_wins == len(WINDOWS):
        print("  -> Chronos-2 > TFT/N-HiTS ranking holds CONSISTENTLY across all 3 windows.")
    elif chronos_wins == 0:
        print("  -> Chronos-2 is NOT the best in any window.")
    else:
        print(f"  -> Chronos-2 > TFT/N-HiTS ranking holds in {chronos_wins} of 3 windows (not universal).")
    print(f"{'=' * 64}\n")

    return results


if __name__ == "__main__":
    run_all()
