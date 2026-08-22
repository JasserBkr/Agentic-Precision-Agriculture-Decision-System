"""
N-HiTS ablation experiments: test the pooling schedule hypothesis and
the covariate-usage hypothesis with a single-window 724/7 backtest.

CHECK 1 — Re-run with n_pool_kernel_size=[1,1,1] and
           n_freq_downsample=[1,1,1] (no multi-rate pooling).
CHECK 2 — Re-run with hist_exog_list=[] and futr_exog_list=[]
           (target-only, no covariates).

Compares both against the original [2,2,1]/[4,2,1] with-covariates result.

Usage:
    python scripts/backtest_nhits_ablation.py
"""

import time
import yaml
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.forecasting.evaluate import (
    temporal_train_test_split,
)
from agri_agent.forecasting.nf_data_prep import (
    to_neuralforecast_df,
    historical_slice_to_futr_df,
)
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

FUSED_PARQUET = "data/processed/fused_2years.parquet"
HORIZON_DAYS = 7


def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single_backtest(
    fused_df: pd.DataFrame,
    field_id: str,
    label: str,
    pool_kernel_size: list[int] | None = None,
    freq_downsample: list[int] | None = None,
    use_covariates: bool = True,
) -> dict:
    """
    Run a single N-HiTS 724/7 backtest with overridable pooling params
    and covariate usage.

    Returns the metrics dict from compute_metrics().
    """
    train_df, test_df = temporal_train_test_split(fused_df, HORIZON_DAYS)
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    nf_train_df = to_neuralforecast_df(train_df)

    # Build NHITS model with caller's overrides
    hparams = dict(
        stack_types=["identity", "identity", "identity"],
        n_blocks=[1, 1, 1],
        mlp_units=[[64, 64], [64, 64], [64, 64]],
        n_pool_kernel_size=pool_kernel_size or [2, 2, 1],
        n_freq_downsample=freq_downsample or [4, 2, 1],
        pooling_mode="MaxPool1d",
        dropout_prob_theta=0.15,
        loss=MAE(),
        max_steps=500,
        early_stop_patience_steps=30,
        learning_rate=1e-3,
        batch_size=32,
        random_seed=42,
        scaler_type="robust",
    )

    hist_exog = [] if not use_covariates else ["NDVI", "NDWI"]
    futr_exog = [] if not use_covariates else ["precipitation_sum", "et0_fao_evapotranspiration"]

    model = NHITS(
        h=HORIZON_DAYS,
        input_size=28,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        **hparams,
    )

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=nf_train_df, val_size=14)

    futr_df = historical_slice_to_futr_df(test_df, field_id)
    pred_df = nf.predict(futr_df=futr_df)
    pred_col = [c for c in pred_df.columns if c not in ("unique_id", "ds")][0]

    y_true = test_df["iot_soil_moisture_mean"].to_numpy()
    y_pred = pred_df.sort_values("ds")[pred_col].to_numpy()
    y_train = train_df["iot_soil_moisture_mean"].to_numpy()

    from agri_agent.forecasting.evaluate import compute_metrics
    metrics = compute_metrics(y_true, y_pred, y_train)
    print(f"  {label}: RMSE={metrics['rmse']:.6f}  MASE={metrics['mase']:.4f}  n={metrics['n_points']}")
    return metrics


def main():
    field = load_field_config()
    field_id = field["field_id"]

    print("Loading 2-year fused dataset ...")
    fused_df = load_fused_dataset(FUSED_PARQUET)
    print(f"  Loaded: {fused_df.shape[0]} rows, {fused_df.shape[1]} columns")

    results = {}

    # --- Original config (with covariates, default pooling) ---
    print("\n--- Original (with covariates, pool=[2,2,1], downsample=[4,2,1]) ---")
    t0 = time.perf_counter()
    orig = run_single_backtest(fused_df, field_id, label="Original")
    results["original"] = orig
    print(f"  Wall time: {time.perf_counter() - t0:.1f}s")

    # --- CHECK 1: Disable pooling ---
    print("\n--- CHECK 1: Disable pooling (pool=[1,1,1], downsample=[1,1,1]) ---")
    t0 = time.perf_counter()
    c1 = run_single_backtest(
        fused_df, field_id, label="No pooling",
        pool_kernel_size=[1, 1, 1],
        freq_downsample=[1, 1, 1],
        use_covariates=True,
    )
    results["no_pooling"] = c1
    print(f"  Wall time: {time.perf_counter() - t0:.1f}s")

    # --- CHECK 2: Target-only (no covariates) ---
    print("\n--- CHECK 2: Target-only (no covariates) ---")
    t0 = time.perf_counter()
    c2 = run_single_backtest(
        fused_df, field_id, label="Target-only",
        use_covariates=False,
    )
    results["target_only"] = c2
    print(f"  Wall time: {time.perf_counter() - t0:.1f}s")

    # --- CHECK 3: No pooling + target-only ---
    print("\n--- CHECK 3: No pooling + target-only ---")
    t0 = time.perf_counter()
    c3 = run_single_backtest(
        fused_df, field_id, label="No pooling + target-only",
        pool_kernel_size=[1, 1, 1],
        freq_downsample=[1, 1, 1],
        use_covariates=False,
    )
    results["no_pooling_target_only"] = c3
    print(f"  Wall time: {time.perf_counter() - t0:.1f}s")

    # ---- Comparison table ----
    print()
    print("=" * 65)
    print("  N-HiTS ABLATION RESULTS  (724/7 single-window backtest)")
    print("=" * 65)
    print(f"  {'Experiment':<25} {'RMSE':>10} {'MASE':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'Original [2,2,1]/[4,2,1]':<25} {results['original']['rmse']:>10.6f} {results['original']['mase']:>10.4f}")
    print(f"  {'CHECK 1: No pooling':<25} {results['no_pooling']['rmse']:>10.6f} {results['no_pooling']['mase']:>10.4f}")
    print(f"  {'CHECK 2: Target-only':<25} {results['target_only']['rmse']:>10.6f} {results['target_only']['mase']:>10.4f}")
    print(f"  {'CHECK 3: No pool + target':<25} {results['no_pooling_target_only']['rmse']:>10.6f} {results['no_pooling_target_only']['mase']:>10.4f}")
    print("=" * 65)

    mase_orig = results["original"]["mase"]
    mase_c1 = results["no_pooling"]["mase"]
    mase_c2 = results["target_only"]["mase"]
    mase_c3 = results["no_pooling_target_only"]["mase"]

    print()
    print("  INTERPRETATION")
    print("  -------------")
    delta_c1 = mase_c1 - mase_orig
    if delta_c1 <= 0:
        print(f"  CHECK 1: Disabling pooling {'improves' if delta_c1 < 0 else 'matches'} MASE "
              f"(delta={delta_c1:+.4f}). This supports the hypothesis that the")
        print("           default pooling schedule was not tuned to this series.")
    else:
        print(f"  CHECK 1: Disabling pooling *hurts* MASE (delta={delta_c1:+.4f}). The")
        print("           pooling schedule was not the problem — the hierarchical")
        print("           decomposition is working as designed for this series.")

    delta_c2 = mase_c2 - mase_orig
    if delta_c2 > 0.01:
        print(f"  CHECK 2: Target-only MASE ({mase_c2:.4f}) is notably worse than")
        print(f"           with-covariates ({mase_orig:.4f}) (delta={delta_c2:+.4f}).")
        print("           N-HiTS IS extracting meaningful signal from covariates;")
        print("           the pooling schedule is the more likely culprit.")
    elif delta_c2 < -0.01:
        print(f"  CHECK 2: Target-only MASE ({mase_c2:.4f}) is notably BETTER than")
        print(f"           with-covariates ({mase_orig:.4f}) (delta={delta_c2:+.4f}).")
        print("           N-HiTS is NOT extracting useful signal from")
        print("           precipitation/ET0 — covariates actively hurt.")
    else:
        print(f"  CHECK 2: Target-only MASE ({mase_c2:.4f}) is approximately on par")
        print(f"           with with-covariates ({mase_orig:.4f}) (delta={delta_c2:+.4f}).")
        print("           N-HiTS is NOT extracting useful signal from covariates.")

    delta_c3 = mase_c3 - mase_orig
    print(f"  CHECK 3: No pooling + target-only MASE ({mase_c3:.4f}): "
          f"delta={delta_c3:+.4f} vs original, delta={mase_c3 - min(mase_c1, mase_c2):+.4f} "
          f"vs best single-ablation ({min(mase_c1, mase_c2):.4f}).")
    if mase_c3 < min(mase_c1, mase_c2):
        print("           Combined ablation beats either alone — the two effects are additive.")
    else:
        print("           Combined ablation does not beat the best single ablation.")

    print()
    return results


if __name__ == "__main__":
    main()
