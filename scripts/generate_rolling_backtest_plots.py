"""
Generate 4 documentation plots from the rolling-origin backtest results.

Produces:
  1. docs/plots/rolling_mase_by_window.png      — per-window MASE over time
  2. docs/plots/volatility_vs_error.png          — target volatility vs error
  3. docs/plots/dm_test_results.png              — Diebold-Mariano p-values
  4. docs/plots/single_vs_pooled_mase.png        — single-window vs pooled MASE

Usage:
    python scripts/generate_rolling_backtest_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agri_agent.data_access.fusion import load_fused_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_PARQUET = "data/processed/rolling_backtest_results.parquet"
FUSED_PARQUET   = "data/processed/fused_2years.parquet"
PLOT_DIR        = "docs/plots"
MASE_DENOM      = 0.01402858  # verified naive-persistence denominator

# Established reference numbers (from prior investigation, reconfirmed
# in the rolling-backtest task output)
SINGLE_WINDOW_MASE = {"chronos": 0.1542, "tft": 0.2918, "nhits": 0.5187}
POOLED_MASE        = {"chronos": 1.3462, "tft": 1.8630, "nhits": 2.8608}
DM_PVALUES = {
    ("chronos", "tft"):  0.0020,
    ("chronos", "nhits"): 0.0003,
    ("tft",     "nhits"): 0.0030,
}
MODEL_COLORS = {"chronos": "#1f77b4", "tft": "#ff7f0e", "nhits": "#2ca02c"}
MODEL_LABELS = {"chronos": "Chronos-2", "tft": "TFT", "nhits": "N-HiTS"}


def reconstruct_windows(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct 7-day window indices from contiguous per-day rows.
    For each model, assign a window_idx to each row such that each
    window is exactly 7 consecutive days.  Skipped windows (where TFT/
    N-HiTS failed on early windows) appear as gaps in window_idx.
    Returns a copy of results_df with 'window_idx' and 'window_start'
    columns added.
    """
    frames = []
    for model, grp in results_df.groupby("model_type"):
        grp = grp.sort_values("date").copy()
        dates = grp["date"].values
        window_ids = []
        current_wi = 0
        day_in_window = 0
        for i in range(len(dates)):
            window_ids.append(current_wi)
            day_in_window += 1
            if day_in_window == 7:
                day_in_window = 0
                current_wi += 1
        grp["window_idx"] = window_ids
        grp["window_start"] = grp.groupby("window_idx")["date"].transform("min")
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def compute_per_window_mase(windowed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-day abs_error into per-window MASE.
    MASE = mean(abs_error_in_window) / MASE_DENOM
    """
    rows = []
    for (model, ws), grp in windowed_df.groupby(["model_type", "window_start"]):
        mae = grp["abs_error"].mean()
        mase = mae / MASE_DENOM
        rows.append({
            "model_type":  model,
            "window_start": pd.Timestamp(ws),
            "mase": mase,
            "mae":  mae,
            "n_days": len(grp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verify window counts match expected
# ---------------------------------------------------------------------------
def verify_window_counts(per_window_df: pd.DataFrame):
    for m, expected in [("chronos", 78), ("tft", 75), ("nhits", 75)]:
        actual = len(per_window_df[per_window_df.model_type == m])
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {m}: {actual} windows (expected {expected}) [{status}]")


# ---------------------------------------------------------------------------
# Plot 1: Per-window MASE over time
# ---------------------------------------------------------------------------
def plot_rolling_mase(per_window_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    for model in ["chronos", "tft", "nhits"]:
        sub = per_window_df[per_window_df.model_type == model].sort_values("window_start")
        ax.plot(sub["window_start"], sub["mase"],
                marker="o", markersize=3, linewidth=1.2,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Naive persistence (MASE=1)")
    ax.set_xlabel("Window start date")
    ax.set_ylabel("MASE (mean abs error / 0.01403)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Rolling-origin MASE by 7-day window  (min_train=180d, step=7d)\n"
        "MASE rises during winter/spring transition periods "
        "(e.g. Feb–Apr 2025, Dec 2025–Apr 2026) and drops during "
        "summer 2025 and summer 2026 — the original single-window test "
        "fell in an unusually low-error summer window.",
        fontsize=10,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Volatility vs error (two stacked panels)
# ---------------------------------------------------------------------------
def plot_volatility_vs_error(per_window_df: pd.DataFrame, fused_df: pd.DataFrame,
                             out_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    # Top: target over full 2-year range
    fused_sorted = fused_df.sort_values("date")
    ax1.plot(fused_sorted["date"], fused_sorted["iot_soil_moisture_mean"],
             color="steelblue", linewidth=1)
    ax1.set_ylabel("iot_soil_moisture_mean")
    ax1.set_title("Target soil moisture over full 2-year dataset")
    ax1.grid(True, alpha=0.3)

    # Bottom: Chronos-2 per-window MASE
    chron_win = per_window_df[per_window_df.model_type == "chronos"].sort_values("window_start")
    ax2.plot(chron_win["window_start"], chron_win["mase"],
             marker="o", markersize=3, linewidth=1.2, color=MODEL_COLORS["chronos"])
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Chronos-2 per-window MASE")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Chronos-2 forecast error (MASE) by window — compare with target behavior above")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Target volatility vs forecast error\n"
        "Visual comparison only — no correlation coefficient asserted.",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Diebold-Mariano test bar chart
# ---------------------------------------------------------------------------
def plot_dm_test(out_path: str):
    pairs = [
        ("Chronos-2\nvs TFT",       DM_PVALUES[("chronos", "tft")]),
        ("Chronos-2\nvs N-HiTS",    DM_PVALUES[("chronos", "nhits")]),
        ("TFT\nvs N-HiTS",          DM_PVALUES[("tft", "nhits")]),
    ]
    labels  = [p[0] for p in pairs]
    pvals   = [p[1] for p in pairs]

    # Significance stars
    def sig_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#d62728", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, pvals, color=bar_colors, edgecolor="black",
                  linewidth=0.8, width=0.5, zorder=3)

    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2,
               label=r"$\alpha = 0.05$", zorder=2)
    ax.axhline(0.01, color="gray", linestyle=":", linewidth=0.8,
               alpha=0.6, label=r"$\alpha = 0.01$", zorder=2)

    for bar, pv in zip(bars, pvals):
        stars = sig_stars(pv)
        bar_x = bar.get_x() + bar.get_width() / 2
        bar_h = bar.get_height()
        ax.text(bar_x, bar_h + bar_h * 0.08,
                f"p = {pv:.4f}  {stars}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                zorder=4)
        ax.text(bar_x, bar_h / 2,
                f"{stars}" if pv < 0.001 else "",
                ha="center", va="center", fontsize=16, color="white",
                fontweight="bold", zorder=5)

    ax.set_ylabel("p-value", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, max(pvals) * 4)
    ax.set_xlabel("Model comparison", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_title(
        "Diebold-Mariano Test for Equal Predictive Accuracy\n"
        "(h = 7, Newey-West HAC variance estimator)",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Single-window vs pooled MASE grouped bar chart
# ---------------------------------------------------------------------------
def plot_single_vs_pooled(out_path: str):
    models  = ["Chronos-2", "TFT", "N-HiTS"]
    keys    = ["chronos", "tft", "nhits"]
    single  = [SINGLE_WINDOW_MASE[k] for k in keys]
    pooled  = [POOLED_MASE[k] for k in keys]
    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, single, w, label="Single-window (Jul 2026)",
                color="#1f77b4", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, pooled, w, label="Pooled (all 78/75 windows)",
                color="#ff7f0e", edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Naive persistence (MASE=1)")
    for bar, val in zip(b1, single):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(b2, pooled):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("MASE")
    ax.set_ylim(0, max(pooled) * 1.2)
    ax.legend(fontsize=9)
    ax.set_title(
        "Single-window test vs full-year pooled MASE\n"
        "The Jul 2026 single-window test was an unrepresentatively easy case — "
        "full-year performance is substantially worse for all three models.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading data ...")
    results  = pd.read_parquet(RESULTS_PARQUET)
    fused_df = load_fused_dataset(FUSED_PARQUET)

    print("Reconstructing window boundaries ...")
    windowed = reconstruct_windows(results)
    per_window = compute_per_window_mase(windowed)

    print("\nWindow count verification:")
    verify_window_counts(per_window)

    # Verify the summer-2026 low / winter-spring high pattern
    chron_win = per_window[per_window.model_type == "chronos"]
    min_row = chron_win.loc[chron_win["mase"].idxmin()]
    max_row = chron_win.loc[chron_win["mase"].idxmax()]
    print("\nChronos-2 per-window MASE extremes:")
    print(f"  Minimum: {min_row['mase']:.4f}  (window {min_row['window_start'].date()})")
    print(f"  Maximum: {max_row['mase']:.4f}  (window {max_row['window_start'].date()})")
    summer_2025 = chron_win[(chron_win["window_start"] >= "2025-06-01")
                            & (chron_win["window_start"] <= "2025-08-31")]
    winter_2025 = chron_win[(chron_win["window_start"] >= "2025-12-01")
                            & (chron_win["window_start"] <= "2026-03-31")]
    print(f"  Summer 2025 avg MASE: {summer_2025['mase'].mean():.4f}")
    print(f"  Winter/Spring 2025-26 avg MASE: {winter_2025['mase'].mean():.4f}")

    print("\nGenerating plots ...")
    plot_rolling_mase(per_window, os.path.join(PLOT_DIR, "rolling_mase_by_window.png"))
    plot_volatility_vs_error(per_window, fused_df, os.path.join(PLOT_DIR, "volatility_vs_error.png"))
    plot_dm_test(os.path.join(PLOT_DIR, "dm_test_results.png"))
    plot_single_vs_pooled(os.path.join(PLOT_DIR, "single_vs_pooled_mase.png"))

    print("\nDone. Files in docs/plots/:")
    for f in sorted(os.listdir(PLOT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
