"""Design tokens for the Field Merguellil dashboard.

Central place for colour mapping and small reusable HTML helpers.  Every
action value in this module matches the *real* vocabulary the pipeline can
produce (see agri_agent.agent.schemas): the LLM is constrained to these
strings, and the validator checks against the same set.  Using any other
vocabulary here (or in components/recommendation_card.py) would silently
drop colouring for actions that actually occur.

Colours are deliberately desaturated / muted (coral, amber, sage family)
rather than pure red/green, and each pair's text colour is a dark shade
pulled from the same family as the background for contrast and cohesion.
"""

from __future__ import annotations

import streamlit as st

# action -> (background_hex, text_hex)
COLORS: dict[str, tuple[str, str]] = {
    # --- Irrigation actions ---
    "irrigate_now": ("#F09595", "#501313"),  # coral — act now
    "irrigate_soon": ("#FAC775", "#412402"),  # amber — act within days
    "no_action_needed": ("#C0DD97", "#173404"),  # sage — hold / no action
    # --- Fertilization actions ---
    "apply_fertilizer": ("#FAC775", "#412402"),  # amber — apply
    "no_application": ("#C0DD97", "#173404"),  # sage — no application
}

# kind -> (tabler icon class, light bg, dark icon colour)
ICON_CHIP: dict[str, tuple[str, str, str]] = {
    "irrigation": ("ti-droplet", "#FAECE7", "#993C1D"),
    "fertilization": ("ti-leaf", "#EAF3DE", "#3B6D11"),
}

_FALLBACK = ("#E5E0D3", "#1A2418")


def _colors_for(action: str) -> tuple[str, str]:
    return COLORS.get(action, _FALLBACK)


def action_badge(action: str, label: str) -> str:
    """HTML pill for a recommendation action. background/text from COLORS."""
    bg, fg = _colors_for(action)
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'border-radius:20px;padding:4px 12px;font-size:12px;font-weight:500;'
        f'line-height:1.4;">{label}</span>'
    )


def icon_chip(kind: str) -> str:
    """Small 28x28 rounded chip with a centred Tabler icon."""
    icon, bg, fg = ICON_CHIP.get(kind, ("ti-circle", "#F4F1E8", "#1A2418"))
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:28px;height:28px;border-radius:8px;background:{bg};color:{fg};'
        f'flex:0 0 auto;">'
        f'<i class="{icon}" style="font-size:15px;line-height:1;"></i></span>'
    )


_BASE_CSS = """
<style>
/* Tabler icons webfont */
@import url('https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.47.0/iconfont/tabler-icons.min.css');

/* Inter from Google Fonts, applied globally */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* Bordered containers -> rounded cards with a warm, light border */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
    border-color: #E5E0D3;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    box-shadow: none;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background-color: #2D5A3D;
    border-color: #2D5A3D;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background-color: #244A32;
    border-color: #244A32;
}

/* Inputs / selects / date inputs */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox [data-baseweb="select"] > div {
    border-radius: 8px;
}

/* Tighter vertical rhythm */
div[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}
</style>
"""


def inject_base_css() -> None:
    """Inject the global stylesheet. Call ONCE at the very top of main()."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
