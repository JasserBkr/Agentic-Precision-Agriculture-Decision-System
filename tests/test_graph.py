"""Tests for the StateGraph using the scripted FakeLLM.

The graph flow is:
  inject_evidence (deterministic, no LLM) -> recommend (structured output)
  -> validate (deterministic) -> END | retry recommend.

FakeLLM replays scripted responses for the structured-output calls
(parse_query + recommend_node).  No tool-calling script is needed.
"""

import pytest

from agri_agent.agent.graph import (
    MAX_RECOMMEND_ATTEMPTS,
    build_graph,
    initial_state,
)
from tests.fakes import FakeLLM, make_rec, sig


def _invoke(fake_llm, bundle, qp):
    graph = build_graph(bundle, llm=fake_llm)
    return graph.invoke(initial_state(bundle, qp))


class TestPassThrough:
    def test_clean_run_reaches_end_without_problems(self, bundle, query_params):
        fake = FakeLLM(structured_script=[make_rec()])
        state = _invoke(fake, bundle, query_params)

        assert state["final_output"]["validation_problems"] == []
        assert state["final_output"]["signal_conflict_detected"] is False
        assert state["recommend_attempts"] == 1
        assert state["final_output"]["data_sources_used"] == [
            "vegetation",
            "weather_forecast",
            "soil_moisture_forecast",
            "thresholds",
        ]

    def test_data_sources_used_is_always_complete(self, bundle, query_params):
        fake = FakeLLM(structured_script=[make_rec()])
        state = _invoke(fake, bundle, query_params)
        assert len(state["final_output"]["data_sources_used"]) == 4

    def test_graph_completes_in_exactly_one_recommend_call(self, bundle, query_params):
        fake = FakeLLM(structured_script=[make_rec()])
        state = _invoke(fake, bundle, query_params)
        assert state["recommend_attempts"] == 1

    def test_no_tool_calls_in_message_history(self, bundle, query_params):
        from langchain_core.messages import ToolMessage

        fake = FakeLLM(structured_script=[make_rec()])
        state = _invoke(fake, bundle, query_params)
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 0

    def test_evidence_is_deterministic_all_four_present(self, bundle, query_params):
        fake = FakeLLM(structured_script=[make_rec()])
        state = _invoke(fake, bundle, query_params)
        evidence_msgs = [
            m for m in state["messages"]
            if hasattr(m, "content") and "## Vegetation" in m.content
        ]
        assert len(evidence_msgs) >= 1
        content = evidence_msgs[0].content
        assert "## Vegetation" in content
        assert "## Weather Forecast" in content
        assert "## Soil Moisture Forecast" in content
        assert "## Agronomic Thresholds" in content


class TestRetry:
    def test_retry_then_pass(self, bundle, query_params):
        fake = FakeLLM(
            structured_script=[
                make_rec(irr_signals=[sig("NOT_A_REAL_SIGNAL")]),
                make_rec(),
            ]
        )
        state = _invoke(fake, bundle, query_params)

        assert state["recommend_attempts"] == 2
        assert state["final_output"]["validation_problems"] == []
        assert state["final_output"]["signal_conflict_detected"] is False

    def test_retry_exhausted_ships_with_conflict_flagged(self, bundle, query_params):
        fake = FakeLLM(
            structured_script=[
                make_rec(irr_signals=[sig("NOT_A_REAL_SIGNAL")]),
                make_rec(irr_signals=[sig("ALSO_NOT_REAL")]),
            ]
        )
        state = _invoke(fake, bundle, query_params)

        assert state["recommend_attempts"] == MAX_RECOMMEND_ATTEMPTS
        assert state["final_output"]["validation_problems"]
        assert state["final_output"]["signal_conflict_detected"] is True

    def test_conflict_rule_survives_retry_and_flags(self, bundle, query_params):
        from tests.conftest import make_bundle

        stressed = make_bundle(ndvi_z=-2.5)
        fake = FakeLLM(
            structured_script=[
                make_rec(irr_action="no_action_needed"),
                make_rec(irr_action="no_action_needed"),
            ]
        )
        state = _invoke(fake, stressed, query_params)
        assert state["recommend_attempts"] == MAX_RECOMMEND_ATTEMPTS
        assert state["final_output"]["validation_problems"]
        assert state["final_output"]["signal_conflict_detected"] is True


class TestGroundTruthOverrides:
    def test_field_id_and_date_never_trusted_from_model(self, bundle, query_params):
        fake = FakeLLM(
            structured_script=[make_rec(field_id="wheat_field_01", date="1999-01-01")]
        )
        state = _invoke(fake, bundle, query_params)

        assert state["final_output"]["field_id"] == bundle.field_id
        assert state["final_output"]["field_id"] == "field_merguellil_01"
        assert state["final_output"]["date"] == str(bundle.origin_date.date())
        assert state["final_output"]["date"] == "2026-07-22"
        assert state["final_output"]["field_id"] != "wheat_field_01"

    def test_ground_truth_override_survives_retry(self, bundle, query_params):
        fake = FakeLLM(
            structured_script=[
                make_rec(field_id="WRONG", date="WRONG", irr_signals=[sig("NOT_A_REAL_SIGNAL")]),
                make_rec(field_id="WRONG_AGAIN", date="WRONG_AGAIN"),
            ]
        )
        state = _invoke(fake, bundle, query_params)
        assert state["recommend_attempts"] == 2
        assert state["final_output"]["field_id"] == bundle.field_id
        assert state["final_output"]["date"] == str(bundle.origin_date.date())


class TestGenericDefaultSurfacing:
    def test_generic_default_signal_groundable_and_in_final_output(self, bundle, query_params):
        from dataclasses import replace

        from agri_agent.agent.bundle import load_agronomic_thresholds

        substituted = replace(bundle, thresholds=load_agronomic_thresholds("wheat", "heading"))
        rec = make_rec(
            fert_signals=[sig("NDVI"), sig("generic_default_used")],
            fert_action="apply_fertilizer",
        )
        fake = FakeLLM(structured_script=[rec])
        state = _invoke(fake, substituted, query_params)

        assert state["final_output"]["validation_problems"] == []
        fert_signals = state["final_output"]["fertilization"]["contributing_signals"]
        names = [s["signal_name"] for s in fert_signals]
        assert "generic_default_used" in names

    def test_thresholds_bundle_exposes_generic_default_signal(self, bundle, query_params):
        from dataclasses import replace

        from agri_agent.agent.bundle import load_agronomic_thresholds

        substituted = replace(bundle, thresholds=load_agronomic_thresholds(crop_type="quinoa"))
        names = {s["signal_name"] for s in substituted.thresholds["signals"]}
        assert "generic_default_used" in names
        assert substituted.thresholds["generic_default_used"] is True


class TestGraphConstruction:
    def test_llm_none_raises(self, bundle):
        with pytest.raises(RuntimeError, match="No LLM configured"):
            build_graph(bundle, llm=None)

    def test_initial_state_carries_the_user_query(self, bundle, query_params):
        state = initial_state(bundle, query_params, query="irrigate in the next 2 days?")
        texts = [m.content for m in state["messages"]]
        assert any("irrigate in the next 2 days" in t for t in texts)

    def test_initial_state_without_query_is_system_prompt_only(self, bundle, query_params):
        state = initial_state(bundle, query_params)
        assert len(state["messages"]) == 1
        assert state["recommend_attempts"] == 0
