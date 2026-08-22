"""Tests for the thin run_pipeline orchestrator (STEP 6 spec): get_llm called
exactly once, CLI overrides merged into the ONE QueryParams, no re-parsing."""

from datetime import date

import pandas as pd
import pytest

import scripts.run_pipeline as run_pipeline
from agri_agent.agent.bundle import FORECAST_HORIZON_DAYS
from agri_agent.agent.schemas import QueryParams
from tests.fakes import FakeLLM, make_rec

# The frozen parquet dataset spans this range.
DATASET_FIRST_DATE = date(2024, 7, 22)
DATASET_LAST_DATE = date(2026, 7, 22)

# build_signal_bundle's default offline origin (when no explicit target_date):
#   origin = parquet_max - FORECAST_HORIZON_DAYS
#           = 2026-07-22 - 7
#           = 2026-07-15
EXPECTED_DEFAULT_ORIGIN = DATASET_LAST_DATE - pd.Timedelta(days=FORECAST_HORIZON_DAYS)


def _counting_llm(fake):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return fake

    return factory, calls


def _default_fake():
    return FakeLLM(
        structured_script=[{"target_date": "2026-07-01"}, make_rec()],
    )


def _capturing_build(captured, bundle):
    def fake_build(field_config, query_params, mode):
        captured["qp"] = query_params
        return bundle

    return fake_build


def _fake_fused_df():
    """Return a fused DataFrame spanning the full dataset range.

    The date range is 2024-07-22 → 2026-07-22.  The default offline
    origin (parquet_max − 7) is 2026-07-15.
    """
    dates = pd.date_range(start=DATASET_FIRST_DATE, end=DATASET_LAST_DATE, freq="D")
    return pd.DataFrame({"date": dates})


def _mock_load_fused(monkeypatch):
    monkeypatch.setattr(
        run_pipeline, "load_fused_dataset", lambda _path: _fake_fused_df()
    )


class TestOrchestration:
    def test_query_params_merged_once_and_cli_wins(self, monkeypatch, bundle, capsys):
        fake = _default_fake()
        factory, calls = _counting_llm(fake)
        captured = {}
        _mock_load_fused(monkeypatch)

        monkeypatch.setattr(run_pipeline, "get_llm", factory)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        result = run_pipeline.main(
            [
                "--mode",
                "offline",
                "--query",
                "look at the 2026-07-01 situation please",
                "--target-date",
                "2026-07-15",
                "--crop-type",
                "barley",
                "--growth-stage",
                "flowering",
            ]
        )

        assert calls["n"] == 1
        assert isinstance(captured["qp"], QueryParams)
        assert captured["qp"].target_date == date(2026, 7, 15)
        assert captured["qp"].crop_type == "barley"
        assert captured["qp"].growth_stage == "flowering"
        assert result is not None
        assert result["field_id"] == "field_merguellil_01"
        assert '"field_id"' in capsys.readouterr().out

    def test_cli_target_date_overrides_parsed_date(self, monkeypatch, bundle):
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: _default_fake())
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(
            ["--query", "when 2026-07-01 rolls around", "--target-date", "2026-08-01"]
        )
        assert captured["qp"].target_date == date(2026, 8, 1)

    def test_query_params_passed_to_build_are_the_same_object(
        self, monkeypatch, bundle
    ):
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: _default_fake())
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )
        run_pipeline.main(["--target-date", "2026-07-20"])
        assert captured["qp"].target_date == date(2026, 7, 20)

    def test_regex_fallback_when_llm_unavailable(self, monkeypatch, bundle):
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: None)
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        with pytest.raises(RuntimeError, match="No LLM configured"):
            run_pipeline.main(["--query", "check 2026-07-10 for me"])
        assert captured["qp"].target_date == date(2026, 7, 10)

    def test_print_default_str_handles_non_json_types(
        self, monkeypatch, bundle, capsys
    ):
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: _default_fake())
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "build_signal_bundle", lambda *a, **k: bundle)
        run_pipeline.main([])
        assert capsys.readouterr().out.strip()  # printed without raising


class TestZeroForwardWindowWarning:
    def test_warning_fires_when_forecast_empty(self, bundle, capsys):
        from dataclasses import replace

        empty = replace(
            bundle,
            weather_forecast={"forecast": [], "signals": [], "insufficient_data": True},
        )
        run_pipeline._warn_zero_forward_window(empty)
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "no forward weather window" in err
        assert "--target-date" in err

    def test_no_warning_when_forecast_present(self, bundle, capsys):
        run_pipeline._warn_zero_forward_window(bundle)
        assert capsys.readouterr().err == ""

    def test_warning_wired_into_run_query(self, monkeypatch, bundle, capsys):
        from dataclasses import replace

        # With a zero-forward-window bundle the confidence ceiling fires, so
        # the retry loop needs two recommend outputs (second one also above
        # the ceiling -> attempts exhausted, run still ships).
        fake = FakeLLM(
            structured_script=[{"target_date": "2026-07-01"}, make_rec(), make_rec()],
        )
        zero_forward = replace(
            bundle,
            weather_forecast={"forecast": [], "signals": [], "insufficient_data": True},
        )
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", lambda *a, **k: zero_forward
        )

        result = run_pipeline.main([])
        assert result is not None
        assert "no forward weather window" in capsys.readouterr().err


# -----------------------------------------------------------------------
# Unit tests for resolve_temporal_expressions (graph.py)
# -----------------------------------------------------------------------


class TestResolveTemporalExpressions:
    def test_tomorrow_offline_resolves_relative_to_reference_date(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="give me fertilization advice tomorrow")
        ref = date(2026, 7, 15)  # the computed offline origin
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 7, 16)

    def test_tomorrow_live_resolves_relative_to_today(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="irrigate tomorrow")
        today = date.today()
        result = resolve_temporal_expressions(qp, today, mode="live")
        assert result.target_date == today + pd.Timedelta(days=1)

    def test_today_offline_resolves_to_reference_date(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="what about today")
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 7, 15)

    def test_today_live_resolves_to_actual_today(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="check today please")
        today = date.today()
        result = resolve_temporal_expressions(qp, today, mode="live")
        assert result.target_date == today

    def test_explicit_date_not_overwritten(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(
            target_date=date(2026, 7, 1), raw_query="check 2026-07-01 please"
        )
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 7, 1)

    def test_no_temporal_expression_leaves_target_unchanged(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(
            target_date=date(2026, 8, 1), raw_query="irrigate in the next 7 days"
        )
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 8, 1)

    def test_no_raw_query_leaves_target_unchanged(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(target_date=date(2026, 8, 1))
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 8, 1)

    def test_yesterday_offline(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="what happened yesterday")
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 7, 14)

    def test_day_after_tomorrow_offline(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="forecast for day after tomorrow")
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 7, 17)

    def test_next_2_days_is_not_a_target_date(self):
        """'next 2 days' describes a window, not a target — should not set one."""
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="irrigate in the next 2 days")
        ref = date(2026, 7, 15)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date is None


# -----------------------------------------------------------------------
# Unit tests for _has_relative_date_expression (graph.py)
# -----------------------------------------------------------------------


class TestHasRelativeDateExpression:
    def test_tomorrow_detected(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("give me advice tomorrow") is True

    def test_today_detected(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("what about today") is True

    def test_yesterday_detected(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("what happened yesterday") is True

    def test_day_after_tomorrow_detected(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("forecast for day after tomorrow") is True

    def test_explicit_date_not_detected(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("check 2026-07-01") is False

    def test_no_temporal_not_detected(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("irrigate in the next 7 days") is False

    def test_none_input(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression(None) is False

    def test_empty_string(self):
        from agri_agent.agent.graph import _has_relative_date_expression

        assert _has_relative_date_expression("") is False


# -----------------------------------------------------------------------
# Integration tests: run_query resolves temporal expressions correctly
# -----------------------------------------------------------------------


class TestTemporalResolutionInRunQuery:
    def test_tomorrow_offline_uses_default_origin_not_parquet_max(
        self, monkeypatch, bundle
    ):
        """THE KEY BUG SCENARIO:
        Dataset range: 2024-07-22 → 2026-07-22
        Default offline origin: 2026-07-15 (= parquet_max - 7)
        Query: "fertilization tomorrow"
        Expected: target_date = 2026-07-16 (= origin + 1)
        NOT 2026-07-23 (= parquet_max + 1)
        """
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)  # parquet max = 2026-07-22
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(
            ["--query", "ok just give me the fertilization advice tomorrow"]
        )
        # origin = 2026-07-15, "tomorrow" = 2026-07-16
        assert captured["qp"].target_date == date(2026, 7, 16)

    def test_data_after_origin_does_not_influence_resolution(self, monkeypatch, bundle):
        """Prove that data in the parquet AFTER the origin (2026-07-16
        through 2026-07-22) does NOT influence what 'tomorrow' resolves to.
        The resolution uses the origin (2026-07-15), not the parquet tail."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "fertilization tomorrow"])

        # Must be origin(2026-07-15) + 1 = 2026-07-16
        # NOT parquet_max(2026-07-22) + 1 = 2026-07-23
        assert captured["qp"].target_date == date(2026, 7, 16)
        assert captured["qp"].target_date != date(2026, 7, 23)

    def test_tomorrow_with_explicit_cli_target_uses_cli_date(self, monkeypatch, bundle):
        """CLI --target-date overrides the resolved temporal expression."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(
            [
                "--query",
                "give me advice for tomorrow",
                "--target-date",
                "2026-09-01",
            ]
        )
        assert captured["qp"].target_date == date(2026, 9, 1)

    def test_tomorrow_with_explicit_origin_uses_that_origin(self, monkeypatch, bundle):
        """When the LLM resolves an explicit date in the query (e.g.
        '2026-07-10'), the explicit date becomes the origin and 'tomorrow'
        resolves relative to THAT origin."""
        fake = FakeLLM(
            structured_script=[{"target_date": "2026-07-10"}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "check 2026-07-10 and give advice for tomorrow"])
        # LLM resolved target_date=2026-07-10 (the explicit date).
        # Origin = 2026-07-10.  "tomorrow" resolves relative to origin
        # → 2026-07-11.
        assert captured["qp"].target_date == date(2026, 7, 11)

    def test_tomorrow_live_uses_actual_today(self, monkeypatch, bundle):
        """'tomorrow' in live mode resolves relative to the real current date."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        # Don't mock load_fused_dataset — live mode calls date.today() directly
        run_pipeline.main(["--mode", "live", "--query", "irrigate tomorrow"])
        assert captured["qp"].target_date == date.today() + pd.Timedelta(days=1)

    def test_explicit_date_in_query_not_changed(self, monkeypatch, bundle):
        """An explicit date in the query is not overridden by temporal resolution."""
        fake = FakeLLM(
            structured_script=[{"target_date": "2026-07-01"}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "check 2026-07-01 please"])
        assert captured["qp"].target_date == date(2026, 7, 1)

    def test_raw_query_is_set_by_parse_query(self, monkeypatch, bundle):
        """parse_query always populates raw_query on the returned QueryParams."""
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: _default_fake())
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "hello world"])
        assert captured["qp"].raw_query == "hello world"

    def test_today_offline_resolves_to_default_origin(self, monkeypatch, bundle):
        """'today' in offline mode resolves to the default origin (parquet_max - 7),
        NOT the parquet's last date."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "what about today"])
        # "today" = origin = 2026-07-15 (NOT 2026-07-22)
        assert captured["qp"].target_date == EXPECTED_DEFAULT_ORIGIN

    def test_no_temporal_query_uses_default_origin(self, monkeypatch, bundle):
        """A query with no relative temporal expression and no explicit date
        gets the default offline origin (parquet_max - 7)."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "give me irrigation advice"])
        # No temporal expression, no explicit date → target_date stays None
        assert captured["qp"].target_date is None


# -----------------------------------------------------------------------
# Tests for misspelling normalization + horizon extraction
# -----------------------------------------------------------------------


class TestTemporalNormalization:
    """Tests that common misspellings of 'tomorrow' are normalised to the
    canonical form before regex matching, and that horizon/duration phrases
    are correctly extracted."""

    def test_tomorrow_resolves(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="give me advice tomorrow")
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="live")
        assert result.target_date == date(2026, 8, 21)

    def test_tommorow_misspelling_resolves(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="give me the recommandations for tommorow")
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="live")
        assert result.target_date == date(2026, 8, 21)

    def test_tomorow_misspelling_resolves(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="advice for tomorow")
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="live")
        assert result.target_date == date(2026, 8, 21)

    def test_tmrw_misspelling_resolves(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="what about tmrw")
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="live")
        assert result.target_date == date(2026, 8, 21)

    def test_next_week_beginning_from_tomorrow(self):
        """'next week beginning from tomorrow' → start=tomorrow, horizon=7."""
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(
            raw_query="give me the recommandations for next week beginning from tommorow"
        )
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="live")
        assert result.target_date == date(2026, 8, 21)
        assert result.horizon_days == 7
        assert result.focus_window == "2026-08-21 to 2026-08-27"

    def test_next_week_beginning_from_tomorow(self):
        """Variant with different misspelling."""
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(
            raw_query="recommendations for next week beginning from tomorow"
        )
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="live")
        assert result.target_date == date(2026, 8, 21)
        assert result.horizon_days == 7
        assert result.focus_window == "2026-08-21 to 2026-08-27"

    def test_next_2_days_extracts_horizon(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="irrigate in the next 2 days")
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date is None
        assert result.horizon_days == 2

    def test_next_7_days_extracts_horizon(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(raw_query="advice for the next 7 days")
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.horizon_days == 7

    def test_explicit_date_with_horizon(self):
        from agri_agent.agent.graph import resolve_temporal_expressions

        qp = QueryParams(
            target_date=date(2026, 8, 25),
            raw_query="check 2026-08-25 for the next 5 days",
        )
        ref = date(2026, 8, 20)
        result = resolve_temporal_expressions(qp, ref, mode="offline")
        assert result.target_date == date(2026, 8, 25)
        assert result.horizon_days == 5
        assert result.focus_window == "2026-08-25 to 2026-08-29"

    def test_cli_target_date_override(self, monkeypatch, bundle):
        """CLI --target-date overrides everything."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(
            [
                "--query",
                "give me the recommandations for next week beginning from tommorow",
                "--target-date",
                "2026-09-01",
            ]
        )
        assert captured["qp"].target_date == date(2026, 9, 1)

    def test_live_temporal_resolution(self, monkeypatch, bundle):
        """'tomorrow' in live mode resolves to actual today + 1."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--mode", "live", "--query", "tommorow irrigation advice"])
        assert captured["qp"].target_date == date.today() + pd.Timedelta(days=1)

    def test_offline_temporal_resolution(self, monkeypatch, bundle):
        """'tomorrow' in offline mode resolves against the default origin."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "tomorrow irrigation"])
        # origin = parquet_max - 7 = 2026-07-15; tomorrow = 2026-07-16
        assert captured["qp"].target_date == date(2026, 7, 16)

    def test_data_after_origin_cannot_influence_resolution(self, monkeypatch, bundle):
        """Prove that data in the parquet AFTER the origin does NOT
        influence what 'tomorrow' resolves to."""
        fake = FakeLLM(
            structured_script=[{}, make_rec()],
        )
        captured = {}
        _mock_load_fused(monkeypatch)
        monkeypatch.setattr(run_pipeline, "get_llm", lambda: fake)
        monkeypatch.setattr(
            run_pipeline, "build_signal_bundle", _capturing_build(captured, bundle)
        )

        run_pipeline.main(["--query", "tommorow fertilization"])

        # Must be origin(2026-07-15) + 1 = 2026-07-16
        # NOT parquet_max(2026-07-22) + 1 = 2026-07-23
        assert captured["qp"].target_date == date(2026, 7, 16)
        assert captured["qp"].target_date != date(2026, 7, 23)
