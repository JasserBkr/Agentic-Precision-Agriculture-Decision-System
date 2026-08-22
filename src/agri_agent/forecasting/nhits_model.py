"""
N-HiTS (Neural Hierarchical Interpolation for Time Series) training
and prediction via NeuralForecast (Nixtla). Uses a fixed, manually-specified
small configuration — no AutoNHITS / hyperparameter search.

Ablation-tuned configuration (from backtest_nhits_ablation.py results):
  - n_pool_kernel_size = [1, 1, 1]  (no multi-rate pooling)
  - n_freq_downsample  = [1, 1, 1]  (full resolution in all stacks)
  - hist_exog_list = [], futr_exog_list = []  (target-only — covariates
    were found to add noise for this series)
Both changes improved MASE individually (by ~12% and ~10%) and were
additive when combined, indicating the default pooling schedule was
mismatched to this series and N-HiTS was not extracting useful signal
from precipitation/ET0 covariates.

Other fixed hyperparameters (per project constraints):
  - h = 7 (forecast horizon, matching Chronos-2 / TFT)
  - input_size = 28 (lookback window, ~4 weeks, same as TFT)
  - 3 identity stacks
  - mlp_units = [[64, 64], [64, 64], [64, 64]] (small — library default
    is [[512,512],[512,512],[512,512]], which overfits on 724 training
    days from one series; 64 matches the spirit of TFT's hidden_size=12
    small-model discipline)
  - loss = MAE (point-forecast loss, matching TFT's; must produce
    comparable predictions for compute_metrics())
  - dropout_prob_theta = 0.15
  - max_steps = 500
  - early_stop_patience_steps = 30
"""

import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS

from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

# Ablation-tuned configuration: no pooling, target-only.
_FIXED_HPARAMS = dict(
    stack_types=["identity", "identity", "identity"],
    n_blocks=[1, 1, 1],
    mlp_units=[[64, 64], [64, 64], [64, 64]],
    n_pool_kernel_size=[1, 1, 1],
    n_freq_downsample=[1, 1, 1],
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


def train_nhits(
    train_df: pd.DataFrame,
    horizon_days: int = 7,
    val_size_days: int = 14,
) -> NeuralForecast:
    """
    Train an N-HiTS model on NeuralForecast-formatted data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must already be in NeuralForecast long format (unique_id, ds, y,
        + covariate columns) — produced by to_neuralforecast_df().
    horizon_days : int
        Forecast horizon h (default 7).
    val_size_days : int
        Number of days carved from the END of the training portion for
        chronological validation / early stopping. This is AFTER
        temporal_train_test_split() has already removed the final
        horizon_days as the held-out test set — so validation never
        overlaps with or comes after the test slice.

    Returns
    -------
    NeuralForecast
        The fitted NeuralForecast object (call .predict() on it).
    """
    model = NHITS(
        h=horizon_days,
        input_size=28,
        hist_exog_list=None,
        futr_exog_list=None,
        stat_exog_list=None,
        **_FIXED_HPARAMS,
    )

    nf = NeuralForecast(
        models=[model],
        freq="D",
    )

    log.info(
        "Training N-HiTS: h=%d, input_size=28, val_size=%d, "
        "stacks=3, mlp_units=[64,64], pool=[1,1,1], downsample=[1,1,1], "
        "target-only (no covariates), "
        "dropout_theta=0.15, max_steps=500, early_stop_patience=30",
        horizon_days, val_size_days,
    )
    nf.fit(df=train_df, val_size=val_size_days)
    log.info("N-HiTS training complete.")
    return nf


def predict_nhits(nf: NeuralForecast, futr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions from a fitted NeuralForecast object.

    Parameters
    ----------
    nf : NeuralForecast
        Fitted NeuralForecast object from train_nhits().
    futr_df : pd.DataFrame
        Future data in NeuralForecast format — accepted for API
        compatibility with the shared backtest harness even though
        the target-only N-HiTS ignores exogenous features.

    Returns
    -------
    pd.DataFrame
        Raw prediction DataFrame from nf.predict(). Column name for
        the point forecast is the model's alias (default "NHITS").
    """
    pred_df = nf.predict(futr_df=futr_df)
    log.info("N-HiTS predict: %d rows, columns=%s", len(pred_df), list(pred_df.columns))
    return pred_df
