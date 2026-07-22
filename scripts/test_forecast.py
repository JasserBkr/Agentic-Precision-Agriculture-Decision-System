"""
Week 3 smoke test: run the real Week 1-2 pipeline, reshape the result
for Chronos-2, and run an actual zero-shot forecast.
"""

from agri_agent.forecasting.chronos_model import forecast_soil_moisture
from agri_agent.forecasting.data_prep import to_chronos_context_df, to_chronos_future_df
from agri_agent.utils.logging_config import get_logger
from run_pipeline import load_field_config, main as run_week1_2_pipeline

log = get_logger(__name__)


def main():
    field = load_field_config()

    log.info("Running Week 1-2 pipeline for fused data...")
    fused_df = run_week1_2_pipeline()

    log.info("Building Chronos-2 context_df...")
    context_df = to_chronos_context_df(fused_df)

    log.info("Building Chronos-2 future_df (7-day horizon)...")
    future_df = to_chronos_future_df(
        field_id=field["field_id"],
        lat=field["centroid"]["lat"],
        lon=field["centroid"]["lon"],
        last_context_date=context_df["timestamp"].max(),
        horizon_days=7,
    )

    log.info("Running Chronos-2 forecast...")
    pred_df = forecast_soil_moisture(context_df, future_df, prediction_length=7)

    log.info("Forecast result:\\n%s", pred_df.to_string())
    return pred_df


if __name__ == "__main__":
    main()
