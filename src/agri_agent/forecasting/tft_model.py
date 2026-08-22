"""
TFT (Temporal Fusion Transformer) training and prediction via
NeuralForecast (Nixtla). Uses a fixed, manually-specified small
configuration — no AutoTFT / hyperparameter search.

Fixed hyperparameters (per project constraints):
  - hidden_size = 12 (small — library default is 128, which overfits
    on 724 training days on a single series)
  - n_head = 1 (attention heads)
  - dropout = 0.2
  - attn_dropout = 0.2
  - loss = MAE (point-forecast loss, matching Chronos-2's predictions
    column so compute_metrics() applies to both without modification)
  - input_size = 28 (lookback window, ~4 weeks)
  - max_steps = 500 (early stopping via val_size will typically cut
    this short)
  - early_stop_patience_steps = 30 (stop if val loss doesn't improve
    for 30 steps)
"""

import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import TFT

from agri_agent.forecasting.data_prep import FUTURE_KNOWN_COLS, PAST_ONLY_COLS
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

# Fixed model configuration — deliberately small for this dataset size.
_FIXED_HPARAMS = dict(
    hidden_size=12,
    n_head=1,
    dropout=0.2,
    attn_dropout=0.2,
    loss=MAE(),
    max_steps=500,
    early_stop_patience_steps=30,
    learning_rate=1e-3,
    batch_size=32,
    random_seed=42,
    scaler_type="robust",
)


def train_tft(
    train_df: pd.DataFrame,
    horizon_days: int = 7,
    val_size_days: int = 14,
) -> NeuralForecast:
    """
    Train a TFT model on NeuralForecast-formatted data.

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
    model = TFT(
        h=horizon_days,
        input_size=28,
        hist_exog_list=PAST_ONLY_COLS,
        futr_exog_list=FUTURE_KNOWN_COLS,
        **_FIXED_HPARAMS,
    )

    nf = NeuralForecast(
        models=[model],
        freq="D",
    )

    log.info(
        "Training TFT: h=%d, input_size=28, val_size=%d, "
        "hidden_size=12, n_head=1, max_steps=500, early_stop_patience=30",
        horizon_days, val_size_days,
    )
    nf.fit(df=train_df, val_size=val_size_days)
    log.info("TFT training complete.")
    return nf


def predict_tft(nf: NeuralForecast, futr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions from a fitted NeuralForecast object.

    Parameters
    ----------
    nf : NeuralForecast
        Fitted NeuralForecast object from train_tft().
    futr_df : pd.DataFrame
        Future covariates in NeuralForecast format (unique_id, ds,
        + FUTURE_KNOWN_COLS) — produced by historical_slice_to_futr_df()
        during backtesting.

    Returns
    -------
    pd.DataFrame
        Raw prediction DataFrame from nf.predict(). Column name for
        the point forecast is the model's alias (default "TFT").
    """
    pred_df = nf.predict(futr_df=futr_df)
    log.info("TFT predict: %d rows, columns=%s", len(pred_df), list(pred_df.columns))
    return pred_df
