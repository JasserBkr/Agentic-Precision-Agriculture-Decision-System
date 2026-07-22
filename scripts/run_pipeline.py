
"""
End-to-end entrypoint: retrieve -> align -> forecast -> agent ->
recommendation, for the field defined in configs/field.yaml.
 
Now wires up through Week 2 (fusion). Extend this further as Weeks 3+
modules become real (forecasting, agent) rather than raising
NotImplementedError.
"""
 
from datetime import date, timedelta
 
import yaml
 
from agri_agent.data_access.fusion import build_fused_dataset
from agri_agent.data_access.iot import simulate_soil_moisture_stream
from agri_agent.data_access.satellite import get_field_index_timeseries
from agri_agent.data_access.weather import get_forecast
from agri_agent.utils.auth import init_earth_engine
from agri_agent.utils.logging_config import get_logger
 
log = get_logger(__name__)
 
# Shared historical window for all three sources. All three MUST use the
# same start/end range, or fusion.build_fused_dataset() will just produce
# a table full of NaN where the sources don't overlap in time.
LOOKBACK_DAYS = 92
 
 
def load_field_config(path: str = "configs/field.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
 
 
def main():
    field = load_field_config()
    log.info("Running pipeline for %s", field["field_id"])
 
    start_date = date.today() - timedelta(days=LOOKBACK_DAYS)
    end_date = date.today()
 
    # --- Week 1: retrieval + index computation ---
    init_earth_engine()
 
    index_records = get_field_index_timeseries(
        bbox=field["bbox"],
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover_pct=field.get("max_cloud_cover_pct", 20),
    )
    if index_records:
        log.info("Retrieved %d scenes with computed indices", len(index_records))
    else:
        log.warning(
            "No scenes found in the last %d days — try widening the date "
            "range or raising max_cloud_cover_pct in configs/field.yaml.",
            LOOKBACK_DAYS,
        )
 
    # past_days=LOOKBACK_DAYS makes weather/IoT's window overlap
    # satellite's — otherwise weather defaults to a forward-looking
    # forecast that shares no dates with satellite's backward-looking
    # history.
    weather = get_forecast(
        lat=field["centroid"]["lat"],
        lon=field["centroid"]["lon"],
        past_days=LOOKBACK_DAYS,
        forecast_days=1,
    )
    log.info("Retrieved weather forecast with keys: %s", list(weather.keys()))
 
    iot = simulate_soil_moisture_stream(
        lat=field["centroid"]["lat"],
        lon=field["centroid"]["lon"],
        past_days=LOOKBACK_DAYS,
        forecast_days=1,
    )
    log.info("Simulated IoT stream with %d points", len(iot["time"]))
 
    # --- Week 2: fusion ---
    fused_df = build_fused_dataset(
        satellite_records=index_records,
        weather_json=weather,
        iot_dict=iot,
        field_id=field["field_id"],
        start_date=start_date,
        end_date=end_date,
    )
 
    log.info("Fused dataset: %d rows x %d columns", *fused_df.shape)
    log.info("Columns: %s", fused_df.columns.tolist())
    log.info("Null count per column:\n%s", fused_df.isna().sum().to_string())
    log.info("First 5 rows:\n%s", fused_df.head().to_string())
    log.info("Last 5 rows:\n%s", fused_df.tail().to_string())
 
    # TODO (Week 3+): forecasting.chronos_model / baseline_model on
    # fused_df, then agent.graph.build_graph().invoke(...)
 
    return fused_df
 
 
if __name__ == "__main__":
    main()
