"""Top-of-page input row: field mode, date, crop, optional free-text query."""
from datetime import date

import streamlit as st

DEFAULT_QUERY = "Recommend irrigation and fertilization for the next 7 days."


def render_input_form() -> dict:
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.selectbox(
                "Mode",
                ["offline", "live"],
                index=0,
                help=(
                    "offline = historical dataset (point-in-time backtest); "
                    "live = fresh 2-year fetch + forward weather."
                ),
            )
        with col2:
            target_date = st.date_input("Date", value=None)
        with col3:
            crop_type = st.text_input("Crop", placeholder="wheat")

        growth_stage = st.text_input("Growth stage (optional)", placeholder="mid_season")
        query = st.text_input(
            "Ask a question (optional)",
            placeholder="Should I irrigate in the next 2 days?",
        )

        submitted = st.button("Get recommendation", type="primary")

    # st.date_input defaults to today even when the user never touched it.
    # Only pass a date through when it differs from today, and let
    # build_signal_bundle's own default-origin logic decide otherwise —
    # matching the CLI's documented behaviour.
    chosen_date = target_date if target_date != date.today() else None

    return {
        "mode": mode,
        "target_date": chosen_date,
        "crop_type": crop_type or None,
        "growth_stage": growth_stage or None,
        "query": query or DEFAULT_QUERY,
        "submitted": submitted,
    }
