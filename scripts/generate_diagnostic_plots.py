"""
Standalone diagnostic plots for the 2-year fused dataset and TFT
leakage investigation. Produces five separate PNG files in docs/plots/.

Usage:
    cd PROJECT12
    .venv/bin/python scripts/generate_diagnostic_plots.py
"""

import sys
import pathlib

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so src.agri_agent is importable.
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import (
    backtest_tft,
    temporal_train_test_split,
)
from agri_agent.forecasting.data_prep import (
    TARGET_COL,
    PAST_ONLY_COLS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PARQUET_PATH = str(ROOT / "data" / "processed" / "fused_2years.parquet")
OUT_DIR = ROOT / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 120

HORIZON_DAYS = 7
FIELD_ID = "field_merguellil_01"

LEAKED_COVS = [
    "weather_soil_moisture_0_to_1cm_mean",
    "weather_soil_moisture_1_to_3cm_mean",
    "weather_soil_moisture_3_to_9cm_mean",
]
CORRECTED_COVS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]
CORRELATION_COLS = [
    "iot_soil_moisture_mean",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "weather_soil_moisture_0_to_1cm_mean",
    "weather_soil_moisture_1_to_3cm_mean",
    "weather_soil_moisture_3_to_9cm_mean",
    "NDVI",
    "NDWI",
]


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# PLOT 1: target vs removed covariate
# ===================================================================
def plot_target_vs_removed_covariate(df):
    print("\n[Plot 1] target_vs_removed_covariate.png")
    target = "iot_soil_moisture_mean"
    cov = "weather_soil_moisture_0_to_1cm_mean"

    r = df[target].corr(df[cov])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        df["date"], df[target],
        color="#1f77b4", linewidth=1.0, alpha=0.9,
        label="Target (simulated IoT)",
    )
    ax.plot(
        df["date"], df[cov],
        color="#d62728", linewidth=1.0, alpha=0.7, linestyle="--",
        label="Removed covariate (Open-Meteo model output)",
    )
    ax.set_title(
        f"Target vs Removed Covariate — Near-Duplicate Relationship (r = {r:.4f})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Soil Moisture (m\u00b3/m\u00b3)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    save(fig, "target_vs_removed_covariate.png")
    return r


# ===================================================================
# PLOT 2: correlation heatmap
# ===================================================================
def plot_correlation_heatmap(df):
    print("\n[Plot 2] correlation_heatmap.png")
    corr = df[CORRELATION_COLS].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ticks = range(len(CORRELATION_COLS))
    short_names = [
        "IoT SM (target)",
        "Precip",
        "ET0",
        "SM 0-1cm (removed)",
        "SM 1-3cm (removed)",
        "SM 3-9cm (removed)",
        "NDVI",
        "NDWI",
    ]
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(short_names, fontsize=9)

    for i in range(len(CORRELATION_COLS)):
        for j in range(len(CORRELATION_COLS)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title(
        "Correlation Matrix: Target, Removed Soil-Moisture Bands, and Retained Covariates",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "correlation_heatmap.png")


# ===================================================================
# PLOT 2b: correlation heatmap for ALL numeric features
# ===================================================================
def plot_correlation_heatmap_all_features(df):
    print("\n[Plot 2b] correlation_heatmap_all_features.png")
    numeric_cols = [c for c in df.columns if c not in ("date", "field_id")]
    corr = df[numeric_cols].corr()

    n = len(numeric_cols)
    fig_size = max(10, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ticks = range(n)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(numeric_cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(numeric_cols, fontsize=7)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            fs = 6 if n > 15 else 7.5
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=fs, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(
        f"Full Correlation Matrix — All {n} Numeric Features in Fused 2-Year Dataset",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "correlation_heatmap_all_features.png")


# ===================================================================
# PLOT 3: precipitation vs target (stacked subplots)
# ===================================================================
def plot_precipitation_vs_target_lag(df):
    print("\n[Plot 3] precipitation_vs_target_lag.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.5]})

    ax1.bar(
        df["date"], df["precipitation_sum"],
        color="#1f77b4", width=1.0, alpha=0.7, label="Precipitation",
    )
    ax1.set_ylabel("Precipitation (mm/day)")
    ax1.legend(loc="upper left")
    ax1.set_title("Precipitation vs Target Soil Moisture", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        df["date"], df[TARGET_COL],
        color="#d62728", linewidth=1.0, label="Target soil moisture",
    )
    ax2.set_ylabel("Soil Moisture (m\u00b3/m\u00b3)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    # ---- Annotate a rain-to-moisture lag if clearly visible ----
    # Find the single largest precipitation event and the subsequent
    # soil-moisture peak within 0-3 days.
    precip_arr = df["precipitation_sum"].to_numpy()
    sm_arr = df[TARGET_COL].to_numpy()
    dates = df["date"].to_numpy()

    best_idx = int(np.nanargmax(precip_arr))
    peak_precip_val = precip_arr[best_idx]
    if peak_precip_val > 0:
        # Look for a rise in the 3 days after this precip event
        window_end = min(best_idx + 4, len(sm_arr))
        if window_end > best_idx + 1:
            sm_before = sm_arr[best_idx]
            sm_after = sm_arr[best_idx + 1:window_end]
            max_sm_after = np.nanmax(sm_after)
            rise = max_sm_after - sm_before

            # Find the day of the peak soil-moisture rise
            peak_offset = int(np.nanargmax(sm_after)) + 1
            peak_day_idx = best_idx + peak_offset

            if rise > 0.001:
                # Annotate on both subplots
                peak_date = pd.Timestamp(dates[peak_day_idx])
                peak_date_precip = pd.Timestamp(dates[best_idx])
                ax1.annotate(
                    f"Largest rain event\n({peak_precip_val:.1f} mm)",
                    xy=(peak_date_precip, peak_precip_val),
                    xytext=(peak_date_precip + pd.Timedelta(days=12), peak_precip_val * 0.85),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
                )
                ax2.annotate(
                    f"Soil moisture rise ~{peak_offset}d later\n(+{rise:.4f} m\u00b3/m\u00b3)",
                    xy=(peak_date, sm_arr[peak_day_idx]),
                    xytext=(peak_date + pd.Timedelta(days=12), sm_arr[peak_day_idx] + rise * 0.4),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1),
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
                )
                print(f"  Annotated lag: rain on {peak_date_precip.date()}, "
                      f"SM peak {peak_offset}d later, rise = {rise:.4f}")
            else:
                print("  WARNING: No clear rain-to-moisture lag identifiable.")
                _add_no_lag_note(ax2)
        else:
            print("  WARNING: Not enough days after peak precip to assess lag.")
            _add_no_lag_note(ax2)
    else:
        _add_no_lag_note(ax2)

    save(fig, "precipitation_vs_target_lag.png")


def _add_no_lag_note(ax):
    ax.text(
        0.5, 0.5,
        "No clear rain-to-moisture lag identifiable at this zoom level;\n"
        "the target is a model output with shallow-depth dominance,\n"
        "so the response may be sub-daily or obscured by smoothing.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=9, style="italic",
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9),
    )


# ===================================================================
# PLOT 4: NDVI timeline with gaps
# ===================================================================
def plot_ndvi_timeline_with_gaps(df):
    print("\n[Plot 4] ndvi_timeline_with_gaps.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        df["date"], df["NDVI"],
        color="#2ca02c", linewidth=0.6, alpha=0.5,
        zorder=1,
    )
    ax.scatter(
        df["date"], df["NDVI"],
        color="#2ca02c", s=12, alpha=0.8, edgecolors="white", linewidths=0.3,
        zorder=2,
        label="NDVI data points",
    )

    total = len(df)
    valid_ndvi = df["NDVI"].notna().sum()
    gap_pct = 100 * (1 - valid_ndvi / total)

    ax.set_title(
        f"NDVI Timeline (2 Years) — Gaps Reflect Sentinel-2 Revisit Frequency "
        f"and {gap_pct:.0f}% Coverage-Filtered Scenes Dropped Upstream\n"
        f"({valid_ndvi}/{total} days with valid NDVI)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("NDVI")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    save(fig, "ndvi_timeline_with_gaps.png")


# ===================================================================
# PLOT 5: TFT leakage diagnosis (runs real TFT backtests)
# ===================================================================
def plot_tft_leakage_diagnosis(df):
    print("\n[Plot 5] tft_leakage_diagnosis.png")
    print("  Running TFT backtest WITHOUT leaked covariates (corrected)...")
    metrics_corrected, y_true, y_pred_corrected = backtest_tft(
        df, FIELD_ID, HORIZON_DAYS, return_predictions=True,
    )
    mase_corrected = metrics_corrected["mase"]

    # Temporarily inject the 3 removed soil-moisture columns into
    # FUTURE_KNOWN_COLS so that TFT treats them as known-future covariates.
    # Must patch ALL three modules that cached the original list at import
    # time: data_prep (source of truth), nf_data_prep (uses it for
    # reshape), and tft_model (passes it to TFT constructor).
    import agri_agent.forecasting.data_prep as _dp
    import agri_agent.forecasting.nf_data_prep as _ndp
    import agri_agent.forecasting.tft_model as _tm

    _orig_dp_cols = _dp.FUTURE_KNOWN_COLS
    _orig_ndp_cols = _ndp.FUTURE_KNOWN_COLS
    _orig_tm_cols = _tm.FUTURE_KNOWN_COLS

    _leaked_full = LEAKED_COVS + CORRECTED_COVS
    _dp.FUTURE_KNOWN_COLS = _leaked_full
    _ndp.FUTURE_KNOWN_COLS = _leaked_full
    _ndp.ALL_COVARIATE_COLS = _leaked_full + PAST_ONLY_COLS
    _tm.FUTURE_KNOWN_COLS = _leaked_full

    try:
        print("  Running TFT backtest WITH leaked covariates (old set)...")
        metrics_leaked, _, y_pred_leaked = backtest_tft(
            df, FIELD_ID, HORIZON_DAYS, return_predictions=True,
        )
        mase_leaked = metrics_leaked["mase"]
    finally:
        # Restore originals — no persistent side-effects on data_prep.py
        _dp.FUTURE_KNOWN_COLS = _orig_dp_cols
        _ndp.FUTURE_KNOWN_COLS = _orig_ndp_cols
        _ndp.ALL_COVARIATE_COLS = _orig_ndp_cols + PAST_ONLY_COLS
        _tm.FUTURE_KNOWN_COLS = _orig_tm_cols

    # Build test dates
    _, test_df = temporal_train_test_split(df, HORIZON_DAYS)
    test_dates = test_df.sort_values("date")["date"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        test_dates, y_true,
        color="black", linewidth=2, marker="o", markersize=6,
        label="Actual (ground truth)",
    )
    ax.plot(
        test_dates, y_pred_leaked,
        color="#d62728", linewidth=1.5, marker="s", markersize=5,
        linestyle="--",
        label=f"TFT with leaked covariates (MASE={mase_leaked:.4f})",
    )
    ax.plot(
        test_dates, y_pred_corrected,
        color="#1f77b4", linewidth=1.5, marker="^", markersize=5,
        linestyle="--",
        label=f"TFT corrected (MASE={mase_corrected:.4f})",
    )

    ax.set_title(
        f"TFT Leakage Diagnosis — 7-Day Backtest\n"
        f"MASE with leaked covariates: {mase_leaked:.4f}  |  "
        f"MASE corrected (no soil-moisture covariates): {mase_corrected:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Test Date")
    ax.set_ylabel("Soil Moisture (m\u00b3/m\u00b3)")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    save(fig, "tft_leakage_diagnosis.png")

    return mase_leaked, mase_corrected


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("  Diagnostic Plots for 2-Year Fused Dataset")
    print("=" * 60)

    print(f"\nLoading dataset: {PARQUET_PATH}")
    df = load_fused_dataset(PARQUET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Plot 1
    r_actual = plot_target_vs_removed_covariate(df)

    # Plot 2
    plot_correlation_heatmap(df)

    # Plot 2b
    plot_correlation_heatmap_all_features(df)

    # Plot 3
    plot_precipitation_vs_target_lag(df)

    # Plot 4
    plot_ndvi_timeline_with_gaps(df)

    # Plot 5
    mase_leaked, mase_corrected = plot_tft_leakage_diagnosis(df)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Plot 1 correlation: r = {r_actual:.6f} (target vs removed covariate)")
    print(f"  Plot 5 MASE (leaked):     {mase_leaked:.4f}")
    print(f"  Plot 5 MASE (corrected):  {mase_corrected:.4f}")
    print(f"  MASE improvement:         {mase_leaked - mase_corrected:.4f}")
    print(f"\n  All plots saved to: {OUT_DIR}/")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"    {p.name}")
    print()


if __name__ == "__main__":
    main()
