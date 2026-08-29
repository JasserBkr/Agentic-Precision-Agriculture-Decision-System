"""Small field-location map. Minimal — one marker, no heavy layers."""
import folium
import streamlit as st
from streamlit_folium import st_folium


def render_field_map(field: dict, bundle) -> None:
    lat, lon = field["centroid"]["lat"], field["centroid"]["lon"]

    veg = bundle.vegetation
    ndvi = next(
        (s["value"] for s in veg.get("signals", []) if s["signal_name"] == "NDVI"),
        None,
    )

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")
    popup = f"{field['field_id']}" + (f"<br>NDVI: {ndvi:.3f}" if ndvi is not None else "")
    folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color="green")).add_to(m)

    with st.expander("Field location", expanded=False):
        st_folium(m, height=280, use_container_width=True)
