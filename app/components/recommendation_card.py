"""Irrigation/fertilization recommendation cards with badge, reasoning,
and an expandable 'why' section grounded in contributing_signals."""
from __future__ import annotations

import streamlit as st

from app.style import action_badge, icon_chip

# Human labels for every action the pipeline can actually produce
# (agri_agent.agent.schemas / style.py COLORS). Keep in sync with COLORS.
_LABELS: dict[str, str] = {
    "irrigate_now": "Irrigate now",
    "irrigate_soon": "Irrigate soon",
    "no_action_needed": "No action needed",
    "apply_fertilizer": "Apply fertilizer",
    "no_application": "No application",
}

_CAVEAT_CSS = (
    "display:flex;align-items:center;gap:8px;background:#FAEEDA;color:#633806;"
    "border-radius:12px;padding:8px 12px;font-size:13px;line-height:1.5;"
)


def _caveat_banner(text: str) -> str:
    return (
        f'<div style="{_CAVEAT_CSS}">'
        f'<span style="flex:0 0 auto;">⚠️</span>'
        f'<span>{text}</span></div>'
    )


def _card(title: str, kind: str, rec: dict, caveat: str | None = None) -> None:
    action = rec["action"]
    label = _LABELS.get(action, action)

    with st.container(border=True):
        header = (
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'margin-bottom:6px;">{icon_chip(kind)}'
            f'<span style="font-size:12px;font-weight:500;color:#7A756A;'
            f'text-transform:uppercase;letter-spacing:0.4px;">{title}</span></div>'
        )
        st.markdown(header, unsafe_allow_html=True)
        st.markdown(action_badge(action, label), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:14px;line-height:1.5;margin-top:10px;">'
            f'{rec["reasoning"]}</div>',
            unsafe_allow_html=True,
        )
        if caveat:
            st.markdown(_caveat_banner(caveat), unsafe_allow_html=True)
        st.caption(f"Confidence {rec['confidence']:.0%}")
        with st.expander("Why this recommendation?"):
            for sig in rec["contributing_signals"]:
                st.markdown(
                    f"<span style='font-weight:600;'>{sig['signal_name']}</span>"
                    f" — {sig['reference']}",
                    unsafe_allow_html=True,
                )
                st.caption(sig["interpretation"])


def render_recommendation_cards(final: dict) -> None:
    if final.get("signal_conflict_detected"):
        st.warning(
            "Some signals conflict for this recommendation — treat it as a "
            "starting point and check the details below.",
            icon=":material/report:",
        )

    col1, col2 = st.columns(2)
    with col1:
        _card("Irrigation", "irrigation", final["irrigation"])
    with col2:
        _card(
            "Fertilization",
            "fertilization",
            final["fertilization"],
            caveat=final["fertilization"].get("caveat"),
        )
