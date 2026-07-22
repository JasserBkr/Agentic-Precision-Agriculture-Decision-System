"""
Week 3-4 deliverable: TFT / N-HiTS supervised baseline, trained on
whatever field history is available. Used strictly as a validation
reference against the Chronos-2 zero-shot forecast (SOTA note Section
4.2) — the gap between the two is itself a diagnostic of how well the
field is represented in Chronos-2's pretraining distribution.

Not yet implemented.
"""


def train_baseline(train_data, model_type: str = "tft"):
    """
    TODO (Week 3-4):
    1. Pick a library (e.g. pytorch-forecasting for TFT, or
       neuralforecast for N-HiTS) and confirm current install/API.
    2. Implement strict temporal (chronological) train/test splitting —
       see SOTA note Section 6.2. No random shuffling of time-ordered
       data.
    3. Train on the fused dataset's history for the sample field.
    """
    raise NotImplementedError("Build this in Week 3-4, per SOTA note Section 4.2.")


def predict_baseline(model, context_series, horizon_days: int = 5):
    """TODO: run inference with the trained baseline model."""
    raise NotImplementedError
