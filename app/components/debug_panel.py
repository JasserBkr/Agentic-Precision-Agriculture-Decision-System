"""Developer/evaluator view: raw trace, validation state, tool calls.
Hidden behind a sidebar toggle so the default view stays clean."""
import streamlit as st


def render_debug_panel(final: dict, result: dict) -> None:
    with st.sidebar:
        show_debug = st.checkbox("Show debug info", value=False)

    if not show_debug:
        return

    st.divider()
    st.subheader("Debug")
    st.write("Data sources used:", final.get("data_sources_used"))
    st.write("Validation passed:", not final.get("validation_problems"))
    st.write("Signal conflict detected:", final.get("signal_conflict_detected"))
    problems = final.get("validation_problems") or []
    if problems:
        st.write("Validation problems:")
        for p in problems:
            st.code(p)
    with st.expander("Raw final_output JSON"):
        st.json(final)
