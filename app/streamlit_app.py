"""
Field Merguellil dashboard. Thin UI over the existing PREP -> graph
pipeline — no business logic here. All decision/forecast logic stays in
agri_agent.agent.*; this file only wires inputs to that pipeline and
renders the FusionRecommendation.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit adds only the entrypoint script's own directory (app/) to
# sys.path, so the repo ROOT (parent of app/) would not be importable and
# `from app.components...` would fail. Put the repo root on the path before
# any project imports so `app` and `agri_agent` resolve regardless of the
# CWD or how Streamlit was launched.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
import streamlit as st

from datetime import date, timedelta

from agri_agent.agent.bundle import (
    FORECAST_HORIZON_DAYS,
    FUSED_PARQUET,
    build_signal_bundle,
)
from agri_agent.agent.graph import (
    _has_relative_date_expression,
    build_graph,
    get_llm,
    initial_state,
    parse_query,
    resolve_temporal_expressions,
)
from agri_agent.data_access.fusion import load_fused_dataset
from app.components.debug_panel import render_debug_panel
from app.components.field_map import render_field_map
from app.components.forecast_chart import render_forecast_chart
from app.components.input_form import render_input_form
from app.components.recommendation_card import render_recommendation_cards
from app.style import inject_base_css

st.set_page_config(page_title="Field Merguellil — irrigation advisor", layout="centered")


@st.cache_resource
def _cached_llm():
    """One LLM client per session, not one per click."""
    return get_llm()


def _load_field_config() -> dict:
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "field.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    inject_base_css()

    st.markdown(
        "<p style='font-size:12px;font-weight:500;text-transform:uppercase;"
        "letter-spacing:0.6px;color:#7A756A;margin-bottom:0;'>Irrigation advisor</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='font-size:24px;font-weight:500;margin-top:2px;'>Field Merguellil</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Irrigation and fertilization advisor")

    inputs = render_input_form()

    if not inputs["submitted"]:
        st.info("Set your inputs above and click **Get recommendation**.")
        return

    field = _load_field_config()
    llm = _cached_llm()

    with st.spinner("Parsing your request..."):
        params = parse_query(inputs["query"], llm=llm)

        # Offline temporal resolution — two-pass, mirroring run_pipeline.py.
        # Relative expressions ("tomorrow", "next week") must resolve against
        # the dataset's own origin, never the LLM's guess at real-world today.
        _saved_target = params.target_date
        if inputs["mode"] == "offline" and params.target_date is None:
            if _has_relative_date_expression(params.raw_query):
                fused_tmp = load_fused_dataset(str(FUSED_PARQUET))
                parquet_max = fused_tmp["date"].max().date()
                resolve_temporal_expressions(params, parquet_max, mode="offline")

        if inputs["mode"] == "offline":
            had_explicit_date = _saved_target is not None
            if had_explicit_date:
                reference_date = _saved_target
            else:
                fused_tmp = load_fused_dataset(str(FUSED_PARQUET))
                parquet_max = fused_tmp["date"].max().date()
                reference_date = parquet_max - timedelta(days=FORECAST_HORIZON_DAYS)
        else:
            reference_date = date.today()

        resolve_temporal_expressions(params, reference_date, mode=inputs["mode"])

        # Form overrides always win — applied AFTER all temporal resolution,
        # matching the CLI where --target-date/--crop-type/--growth-stage are
        # applied last. There is exactly one source of truth; overrides never
        # compete, they supersede.
        if inputs["target_date"]:
            params.target_date = inputs["target_date"]
        if inputs["crop_type"]:
            params.crop_type = inputs["crop_type"]
        if inputs["growth_stage"]:
            params.growth_stage = inputs["growth_stage"]

    with st.spinner("Gathering signals and running the soil-moisture forecast..."):
        try:
            bundle = build_signal_bundle(field, params, mode=inputs["mode"])
        except Exception:
            st.error("Couldn't build the data context for this request. Try again.")
            return

    if bundle.load_errors:
        with st.expander("Some data sources had issues", icon=":material/warning:"):
            for source, msg in bundle.load_errors.items():
                st.write(f"**{source}**: {msg}")

    with st.spinner("Reasoning over the signals..."):
        try:
            graph = build_graph(bundle, llm=llm)
            state = initial_state(bundle, params, query=inputs["query"])
            result = graph.invoke(state)
        except Exception:
            st.error(
                "The recommendation engine hit an error and couldn't finish. "
                "Try again in a moment."
            )
            return

    final = result.get("final_output")
    if final is None:
        st.error("No recommendation was produced. Try a different date.")
        return

    render_recommendation_cards(final)
    render_forecast_chart(bundle)
    render_field_map(field, bundle)
    render_debug_panel(final, result)


if __name__ == "__main__":
    main()
