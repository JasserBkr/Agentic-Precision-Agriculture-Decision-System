"""Read-only inspection of fused_2years.parquet for manual date curation.

Prints (never writes):
  1. monthly aggregates — seasonality eyeballing
  2. per-stratum candidate dates for real_dates.jsonl selection

Run:  uv run python -m evaluation.inspect_dataset
"""

from __future__ import annotations

import pandas as pd

from agri_agent.agent.bundle import FUSED_PARQUET
from agri_agent.data_access.fusion import load_fused_dataset

WHEAT_TRIGGER = 0.30 - 0.55 * (0.30 - 0.12)  # mid-season trigger, from thresholds.yaml math


def _rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["precip_7d"] = df["precipitation_sum"].rolling(7).sum()
    df["et0_7d"] = df["et0_fao_evapotranspiration"].rolling(7).sum()
    df["moisture_14d_mean"] = df["iot_soil_moisture_mean"].rolling(14).mean()
    df["moisture_14d_slope"] = df["moisture_14d_mean"].diff(7)
    df["tmax_month_pctl"] = df.groupby(df["date"].dt.month)["temperature_2m_max"].rank(pct=True)
    return df


def print_monthly(df: pd.DataFrame) -> None:
    m = (
        df.set_index("date")
        .resample("MS")
        .agg(
            {
                "precipitation_sum": "sum",
                "et0_fao_evapotranspiration": "sum",
                "temperature_2m_max": "mean",
                "iot_soil_moisture_mean": "mean",
                "NDVI": "mean",
            }
        )
        .round(2)
    )
    print("=== MONTHLY AGGREGATES ===")
    print(m.to_string())


def print_candidates(df: pd.DataFrame) -> None:
    print(f"\n=== CANDIDATES (trigger={WHEAT_TRIGGER:.3f} m3/m3) ===")

    def show(label: str, mask: pd.Series, cols: list[str], n: int = 12) -> None:
        sub = df.loc[mask, ["date"] + cols].head(n)
        print(f"\n-- {label} ({int(mask.sum())} total) --")
        if len(sub):
            print(sub.to_string(index=False))
        else:
            print("(none)")

    base_cols = [
        "precip_7d", "et0_7d", "iot_soil_moisture_mean", "moisture_14d_slope",
        "days_since_last_scene", "is_interpolated_ndvi", "NDVI",
        "temperature_2m_max", "tmax_month_pctl",
    ]

    # Seasonal anchors: one near each season midpoint, both years.
    anchors = [
        "2024-11-15", "2025-01-15", "2025-04-15", "2025-07-15",
        "2025-11-15", "2026-01-15", "2026-04-10",
    ]
    print("\n-- seasonal anchor windows (--7/+7 days around each) --")
    for a in anchors:
        w = df[(df["date"] >= a) & (df["date"] <= pd.Timestamp(a) + pd.Timedelta(days=7))]
        if len(w):
            print(w[["date", "precip_7d", "iot_soil_moisture_mean", "NDVI", "days_since_last_scene"]].head(2).to_string(index=False))

    dry = (df["precip_7d"] <= 1.0) & (
        df["iot_soil_moisture_mean"]
        < df.groupby(df["date"].dt.month)["iot_soil_moisture_mean"].transform(lambda s: s.quantile(0.25))
    )
    wet = (df["precip_7d"] >= 20.0)
    transition = df["date"].dt.month.isin([3, 4, 10, 11]) & (df["moisture_14d_slope"].abs() > 0.005)
    boundary = df["iot_soil_moisture_mean"].sub(WHEAT_TRIGGER).abs().le(0.02)
    gaps = (df["is_interpolated_ndvi"]) | (df["days_since_last_scene"] >= 10)
    heat = df["tmax_month_pctl"] >= 0.95

    for label, mask in (
        ("CLEARLY DRY (precip_7d<=1mm & month-q25 moisture)", dry),
        ("CLEARLY WET (precip_7d>=20mm)", wet),
        ("TRANSITION (Mar/Apr/Oct/Nov, |14d slope|>0.005)", transition),
        ("BOUNDARY (|moisture-trigger|<=0.02)", boundary),
        ("GAP-ADJACENT (interpolated NDVI or scene>=10d stale)", gaps),
        ("HEATWAVE (month p95 tmax)", heat),
    ):
        show(label, mask.fillna(False), base_cols)


def main() -> None:
    df = load_fused_dataset(str(FUSED_PARQUET))
    df = _rolling_stats(df)
    print(f"rows={len(df)} range={df['date'].min().date()} -> {df['date'].max().date()}")
    print_monthly(df)
    print_candidates(df)


if __name__ == "__main__":
    main()
