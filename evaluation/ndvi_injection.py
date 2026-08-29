"""Inject a known-magnitude NDVI anomaly into real fused history.

Gives Layer 2's NDVI-anomaly check a true label: after injection, whether
the pipeline flags the date as anomalous (abs(z) >= 2.0) is scored against
the injected magnitude k. Detection itself is deterministic PREP work
(agent/anomaly.py), so precision/recall are computed WITHOUT any LLM call;
the same scenarios can additionally be pushed through the live agent to
verify the surfaced signal reaches the recommendation.

Injection rule: v' = baseline_mean(D) + k * baseline_std(D), applied ONLY
to row D, where the baseline uses the SAME day-of-year window semantics as
agent/anomaly.py (+/-15 days, circular wrap). k = +/-2.5 by default: 25%
beyond the z >= 2.0 band edge so detection isn't knife-edge.
"""

from __future__ import annotations

import pandas as pd

from agri_agent.agent.anomaly import _circular_distance, WINDOW_DAYS


def doy_window_baseline(
    dates: pd.Series, values: pd.Series, target: pd.Timestamp
) -> dict | None:
    """Mean/std of all observations within the anomaly.py-style DOY window."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True),
            "value": pd.to_numeric(pd.Series(list(values)), errors="coerce").reset_index(drop=True),
        }
    ).dropna()
    if len(df) == 0:
        return None
    half = WINDOW_DAYS / 2
    target_doy = int(pd.Timestamp(target).dayofyear)
    mask = df["date"].apply(lambda d: _circular_distance(int(d.dayofyear), target_doy) <= half)
    window = df[mask]
    if len(window) < 3:  # mirrors anomaly.MIN_OBS
        return None
    return {"mean": float(window["value"].mean()), "std": float(window["value"].std(ddof=0)), "n": len(window)}


def inject_ndvi_anomaly(
    fused_history: pd.DataFrame,
    target_date: str,
    magnitude_z: float = -2.5,
) -> tuple[pd.DataFrame, dict]:
    """Return (modified_history_copy, truth).

    Raises ValueError when no NDVI observation exists exactly on
    target_date (anomaly.py requires an exact-date observation) or when
    the DOY-window baseline is underdetermined (n < 3 or zero std).
    """
    df = fused_history.copy().sort_values("date").reset_index(drop=True)
    ts = pd.Timestamp(target_date).normalize()

    mask = (df["date"].dt.normalize() == ts) & df["NDVI"].notna()
    if not mask.any():
        raise ValueError(f"no real NDVI observation on {ts.date()} — pick another date")

    baseline = doy_window_baseline(df.loc[~mask, "date"], df.loc[~mask, "NDVI"], ts)
    # Baseline should include ALL observations except the one being replaced,
    # matching how the scorer sees the modified series' other rows; including
    # the original D value shifts mean/std negligibly but excluding it keeps
    # the intended z closer to k.
    if baseline is None or baseline["std"] <= 0:
        raise ValueError(f"DOY-window baseline underdetermined on {ts.date()}: {baseline}")

    original = float(df.loc[mask, "NDVI"].iloc[0])
    injected_value = baseline["mean"] + magnitude_z * baseline["std"]

    df.loc[mask, "NDVI"] = injected_value

    truth = {
        "target_date": ts.date().isoformat(),
        "magnitude_z": float(magnitude_z),
        "original_ndvi": original,
        "injected_ndvi": float(injected_value),
        "baseline_mean": baseline["mean"],
        "baseline_std": baseline["std"],
        "baseline_n": baseline["n"],
        "expected_anomaly_band": abs(magnitude_z) >= 2.0,
    }
    return df, truth


def clean_control_truth(fused_history: pd.DataFrame, target_date: str) -> dict:
    """Truth record for an unmodified control date (expected NOT anomalous
    unless nature says otherwise — recorded honestly either way, since a
    naturally anomalous control date must not be counted as false positive
    without noting it)."""
    ts = pd.Timestamp(target_date).normalize()
    row = fused_history[fused_history["date"].dt.normalize() == ts]
    if row.empty or pd.isna(row.iloc[0]["NDVI"]):
        raise ValueError(f"no real NDVI observation on {ts.date()}")
    return {
        "target_date": ts.date().isoformat(),
        "magnitude_z": 0.0,
        "original_ndvi": float(row.iloc[0]["NDVI"]),
        "injected_ndvi": float(row.iloc[0]["NDVI"]),
        "expected_anomaly_band": False,
        "note": "clean control; verify measured z < 2.0",
    }


def score_detection(measured_z: float | None, truth: dict) -> dict:
    """Compare the pipeline's measured z against injected truth."""
    expected = truth["expected_anomaly_band"]
    detected = measured_z is not None and abs(measured_z) >= 2.0
    return {
        "target_date": truth["target_date"],
        "kind": "injected" if truth["magnitude_z"] != 0.0 else "control",
        "magnitude_z": truth["magnitude_z"],
        "measured_z": measured_z,
        "recovery_error": (abs(measured_z - truth["magnitude_z"]) if measured_z is not None else None),
        "expected_detected": expected,
        "detected": bool(detected),
    }
