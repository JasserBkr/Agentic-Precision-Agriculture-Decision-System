"""Chronos-2 p10/p50/p90 soil-moisture forecast band, plus the irrigation
trigger line so the user can see the two numbers the reasoning compares."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_forecast_chart(bundle) -> None:
    sm = bundle.soil_moisture_forecast
    quantiles = sm.get("quantiles")
    if not quantiles:
        st.info("No soil-moisture forecast available for this run.")
        return

    df = pd.DataFrame(quantiles)
    trigger = bundle.thresholds.get("trigger")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df["date"]) + list(df["date"])[::-1],
        y=list(df["p90"]) + list(df["p10"])[::-1],
        fill="toself",
        fillcolor="rgba(31,119,180,0.15)",
        line=dict(width=0),
        name="p10-p90 range",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["p50"],
        mode="lines+markers",
        name="p50 (median)",
        line=dict(color="rgb(31,119,180)"),
    ))
    if trigger is not None:
        fig.add_hline(
            y=trigger,
            line_dash="dash",
            line_color="firebrick",
            annotation_text="irrigation trigger",
        )

    horizon = sm.get("horizon_days", 7)
    fig.update_layout(
        title=f"{horizon}-day soil-moisture forecast",
        yaxis_title="m³/m³",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)
