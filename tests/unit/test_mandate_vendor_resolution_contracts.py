"""Contract tests for mandate vendor auto-resolution.

These guard against the *broadcast cascade* bug: if an agent calls
``set_model_preference`` with just a bare model id (no vendor), the mandate
used to persist as ``{vendor: None, model: "<id>", route: None}``. The next
request would then attempt that bare id across every configured provider in
priority order — anthropic → openai → openrouter — eventually landing on
whichever backend happened to serve something with a matching id (in one
observed case, OpenRouter serving a Gemini model for ``gpt-5-mini``). The
mandate must name a vendor; if the caller omitted one, we resolve it from
the discovery catalog or refuse.

Regression for issue observed in host.log at
2026-04-23T14:58+ (see epic #688).
"""

from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.service import LLMService


def _cached(models):
    """Return a mock SharedModelCache whose ``get_any()`` yields ``models``."""
    cache = MagicMock()
    cache.get_any = MagicMock(return_value=models)
    return cache


def _make_service():
    """Build a bare LLMService instance wired up enough for set/get preference."""
    svc = LLMService.__new__(LLMService)
    svc.providers = []
    svc._mandate_preference = {"vendor": None, "model": None, "route": None}
    svc._preference_persistence_callback = None
    svc._preference_persistence_tasks = set()
    return svc


def _mk_model(id_, vendor):
    return ModelInfo(
        id=id_,
        provider=vendor,
        display_name=id_,
        category=ModelCategory.CHAT,
        supports_tools=True,
    )


class TestBareModelResolution:
    """set_model_preference(model, vendor=None) auto-resolves from discovery."""

    def test_unambiguous_bare_model_auto_resolves_vendor(self):
        svc = _make_service()
        cache = _cached([_mk_model("claude-opus-4-7", "anthropic")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("claude-opus-4-7")

        pref = svc.get_model_preference()
        assert pref["vendor"] == "anthropic"
        assert pref["model"] == "claude-opus-4-7"
        assert pref["route"] is None

    def test_ambiguous_bare_model_refuses_with_vendor_list(self):
        """A model id served by >1 vendor (e.g. openrouter mirrors) must refuse."""
        svc = _make_service()
        cache = _cached([
            _mk_model("gpt-5-mini", "openai"),
            _mk_model("gpt-5-mini", "openrouter"),
        ])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError) as exc:
                svc.set_model_preference("gpt-5-mini")

        # Error must name the ambiguous vendors so the caller can choose.
        msg = str(exc.value)
        assert "gpt-5-mini" in msg
        assert "openai" in msg
        assert "openrouter" in msg

        # Mandate was NOT persisted (the key invariant — no broadcast state
        # gets left behind on a refusal).
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_unknown_bare_model_refuses(self):
        """A model not in discovery must refuse — this was the LLM-hallucinated-id path."""
        svc = _make_service()
        cache = _cached([_mk_model("claude-opus-4-7", "anthropic")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError) as exc:
                # Hypothetical: LLM tool call with a hallucinated id.
                svc.set_model_preference("gpt-9000-ultramax")

        assert "gpt-9000-ultramax" in str(exc.value)
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_empty_catalog_refuses_bare_model(self):
        """Pre-discovery / empty cache: refuse rather than silently broadcast."""
        svc = _make_service()
        cache = _cached(None)
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError):
                svc.set_model_preference("claude-opus-4-7")

        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_explicit_vendor_bypasses_resolution(self):
        """When caller supplies a vendor, we trust it — no catalog lookup."""
        svc = _make_service()
        cache = _cached([])  # Empty catalog would refuse a bare call.
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("claude-opus-4-7", vendor="anthropic", route="plan")

        pref = svc.get_model_preference()
        assert pref == {
            "vendor": "anthropic",
            "model": "claude-opus-4-7",
            "route": "plan",
        }

    def test_auto_string_still_ignored(self):
        """``auto`` short-circuits before resolution (means default routing)."""
        svc = _make_service()
        cache = _cached([])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("auto")  # no-op, no exception

        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}


class TestBroadcastBugRegression:
    """The exact path that produced the "answered as Gemini" bug in host.log."""

    def test_llm_tool_call_with_bare_model_refuses_rather_than_broadcast(self):
        """Replay the LLM tool-call sequence that caused the live failure.

        Host log sequence (2026-04-23T14:58):
          1. mandate = anthropic:plan/claude-opus-4-20250514  (correct)
          2. agent streams; the response includes a tool call to set_model
             with a bare model id
          3. mandate got silently rewritten to {vendor: None, model:
             "gpt-5-mini"}
          4. next request broadcasts: anthropic:plan → openai:plan →
             openrouter:api — OpenRouter served it, agent reported "gemini"

        After this fix, step 3 raises ValueError and step 2 never succeeds in
        overwriting the mandate. The mandate stays anthropic:plan/claude-opus.
        """
        svc = _make_service()
        svc._mandate_preference = {
            "vendor": "anthropic",
            "model": "claude-opus-4-20250514",
            "route": "plan",
        }
        cache = _cached([_mk_model("claude-opus-4-20250514", "anthropic")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError):
                svc.set_model_preference("gpt-5-mini")

        # Mandate is untouched.
        assert svc.get_model_preference() == {
            "vendor": "anthropic",
            "model": "claude-opus-4-20250514",
            "route": "plan",
        }
