"""Provider selection, usage capture, and a zero-cost fake for smoke runs."""

from __future__ import annotations

import os
from typing import Any

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def provider_available(name: str) -> bool:
    if name == "fake":
        return True
    key = PROVIDER_ENV_KEYS.get(name)
    return bool(key and os.environ.get(key))


def require_provider(name: str) -> None:
    if not provider_available(name):
        raise SystemExit(
            f"provider '{name}' unavailable: set {PROVIDER_ENV_KEYS.get(name, '?')} "
            "in .env (and AGRI_LLM_PROVIDER is overridden per-sweep by the harness)"
        )


class UsageCapture:
    """LangChain callback handler capturing per-call token usage + call count.

    Handles OpenAI-style llm_output['token_usage'] AND Gemini/Groq-style
    usage_metadata on the response message's response_metadata.
    """

    def __init__(self) -> None:
        from langchain_core.callbacks import BaseCallbackHandler

        self._base = BaseCallbackHandler()
        self.calls: list[dict[str, Any]] = []
        self._n = 0

    def __getattr__(self, name):  # delegate ignore_chain etc. to the base
        return getattr(self._base, name)

    # LangChain callback protocol -----------------------------------------
    def on_llm_start(self, *args, **kwargs) -> None:  # noqa: D102
        self._n += 1
        self.calls.append({"n": self._n, "input_tokens": None, "output_tokens": None})

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D102
        if not self.calls:
            return
        rec = self.calls[-1]
        usage = None
        try:
            llm_out = getattr(response, "llm_output", None) or {}
            usage = llm_out.get("token_usage")
            if usage is None:
                gen0 = response.generations[0][0]
                meta = getattr(getattr(gen0, "message", None), "response_metadata", {}) or {}
                usage = meta.get("usage_metadata") or meta.get("token_usage")
        except Exception:  # noqa: BLE001 — usage capture must never break a run
            usage = None
        if isinstance(usage, dict):
            rec["input_tokens"] = usage.get("input_tokens", usage.get("prompt_tokens"))
            rec["output_tokens"] = usage.get("output_tokens", usage.get("completion_tokens"))

    # Aggregation ----------------------------------------------------------
    def summary(self) -> dict:
        ins = [c["input_tokens"] for c in self.calls if c["input_tokens"] is not None]
        outs = [c["output_tokens"] for c in self.calls if c["output_tokens"] is not None]
        return {
            "llm_calls": len(self.calls),
            "input_tokens_total": int(sum(ins)) if ins else None,
            "output_tokens_total": int(sum(outs)) if outs else None,
        }


def build_fake_llm(case_or_tags: list[str], moisture_below_trigger: bool,
                   reference_action: str | None = None):
    """Deterministic scripted LLM for zero-cost pipeline smoke tests.

    Action choice derives from scenario tags (or the reference decision when
    no tags exist, e.g. real-date smoke runs) so the fake exercises BOTH
    conflict-firing and clean paths across a sweep.
    """
    from tests.fakes import make_rec

    if reference_action is not None:
        irr = reference_action
    elif moisture_below_trigger and "rain-offset" not in case_or_tags:
        irr = "irrigate_now"
    elif "rain-offset" in case_or_tags or "heat-stress" in case_or_tags:
        irr = "irrigate_now"  # deliberately triggers R1 / leans hard
    else:
        irr = "no_action_needed"
    fert = "apply_fertilizer" if "over-fertilization-risk" in case_or_tags else "no_application"
    return type("FakeStructured", (), {
        "with_structured_output": lambda self, schema: self,
        "invoke": lambda self, messages: make_rec(irr_action=irr, fert_action=fert),
    })()
