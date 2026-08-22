"""NDVI seasonal anomaly against a field's own day-of-year climatology.

Rebuilt to the STEP 2b spec. The field's NDVI is judged against ITS OWN
historical distribution for the same season, never against absolute
textbook cutoffs — a generic "NDVI > 0.5 is healthy" threshold would
false-alarm on fields whose seasonal peak NDVI is naturally lower.

Never raises on sparse/empty history; it always returns a dict with
``insufficient_data=True`` and ``z_score=None`` instead.
"""

from datetime import date

import numpy as np
import pandas as pd

WINDOW_DAYS = 30  # +/- around the target day-of-year
MIN_OBS = 3  # minimum window observations to compute a baseline


def _season_for_doy(doy: int) -> str:
    """Map a day-of-year to a season. winter: 1-90, spring: 91-180,
    summer: 181-270, autumn: 271-365."""
    if doy <= 90:
        return "winter"
    if doy <= 180:
        return "spring"
    if doy <= 270:
        return "summer"
    return "autumn"


def _circular_distance(doy_a: int, doy_b: int) -> int:
    """Shortest distance between two day-of-years, wrapping across New Year
    (e.g. Dec 20 and Jan 5 are close)."""
    d = abs(doy_a - doy_b)
    return min(d, 365 - d)


def ndvi_seasonal_anomaly(
    dates: pd.Series | pd.DatetimeIndex,
    values: pd.Series,
    target_date: date | str | pd.Timestamp,
    window_days: int = WINDOW_DAYS,
    min_obs: int = MIN_OBS,
) -> dict:
    """
    Z-score anomaly for the NDVI observed at ``target_date`` against the
    field's own climatology. Baseline = mean/std of every historical
    observation whose day-of-year falls within +/- ``window_days/2`` of the
    target's day-of-year, using CIRCULAR distance.

    Returns a dict with ``insufficient_data=True`` and ``z_score=None`` when:
    - there are no historical observations at all;
    - there is no observation exactly on ``target_date`` (the caller is
      expected to have already resolved which date/value it's asking about);
    - fewer than ``min_obs`` observations fall inside the window.

    ``baseline_std == 0`` (or non-finite) is handled explicitly: if the
    target value equals the mean, ``z_score=0.0``; otherwise ``z_score=None``
    rather than dividing by zero.
    """
    if len(dates) == 0:
        return _insufficient("no historical observations at all", target_date)

    df = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True),
        "value": pd.to_numeric(pd.Series(list(values)), errors="coerce").reset_index(drop=True),
    }).dropna(subset=["value"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        return _insufficient("no non-null NDVI observations", target_date)

    target = pd.Timestamp(target_date).normalize()
    target_mask = df["date"].dt.normalize() == target
    if not target_mask.any():
        return _insufficient(
            f"no observation exactly on target_date {target.date().isoformat()}", target_date
        )

    target_value = float(df.loc[target_mask, "value"].iloc[0])
    target_doy = int(target.dayofyear)

    half = window_days / 2
    in_window = df["date"].apply(lambda d: _circular_distance(int(d.dayofyear), target_doy) <= half)
    window = df[in_window]

    if len(window) < min_obs:
        return _insufficient(
            f"only {len(window)} observations within +/-{window_days // 2} days of "
            f"day-of-year {target_doy} (need >= {min_obs})",
            target_date,
        )

    baseline_mean = float(window["value"].mean())
    baseline_std = float(window["value"].std(ddof=0))

    if not np.isfinite(baseline_std) or baseline_std == 0.0:
        if target_value == baseline_mean:
            z_score = 0.0
        else:
            z_score = None
    else:
        z_score = (target_value - baseline_mean) / baseline_std

    percentile = float((window["value"] <= target_value).mean() * 100)

    return {
        "target_date": target.date().isoformat(),
        "ndvi_value": target_value,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "z_score": z_score,
        "percentile": percentile,
        "n_obs": int(len(window)),
        "window_days": window_days,
        "season": _season_for_doy(target_doy),
        "anomaly": abs(z_score) >= 2.0 if z_score is not None else None,
        "insufficient_data": False,
    }


def _insufficient(reason: str, target_date) -> dict:
    target = pd.Timestamp(target_date)
    return {
        "target_date": target.date().isoformat() if not pd.isna(target) else None,
        "ndvi_value": None,
        "baseline_mean": None,
        "baseline_std": None,
        "z_score": None,
        "percentile": None,
        "n_obs": 0,
        "window_days": WINDOW_DAYS,
        "season": _season_for_doy(int(target.dayofyear)) if not pd.isna(target) else None,
        "anomaly": None,
        "insufficient_data": True,
        "reason": reason,
    }
