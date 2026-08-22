"""Regression suite for #734 — remote-GPU shortcut must honor the mandate.

Before the vendor/route refactor the remote GPU backend ran before
:meth:`LLMService.resolve_provider_routing`, so a user-pinned
``{vendor, route, model}`` mandate or vendor-prefixed ``model_override``
could be silently ignored while the answer came from the remote pod.

The gate lives in :meth:`LLMService._remote_first_allowed`.  These tests
pin its contract (when it returns False, the remote-first branch skips).
The five call sites all enter ``_remote_route_attempt``; that one routing
boundary applies this mandate gate before exposing a host-only lease route:

- ``service.generate``                 (generate text)
- ``service.generate_with_messages``   (multi-turn tool calling)
- ``streaming.generate_stream``        (text streaming)
- ``streaming.stream_with_messages``   (streaming after tool exec)
- ``streaming.stream_with_tool_detection`` (streaming + tool detection)
"""

from unittest.mock import MagicMock

from kestrel_sovereign.llm.remote_backend import BackendType


def _service_with_remote_active():
    """Return an LLMService-shaped stand-in with the remote backend primed.

    We don't instantiate the real LLMService here because its dependency
    graph is huge and the helper under test only touches three attributes:
    ``_backend`` and ``_mandate_preference``.  A minimal
    stand-in keeps the tests tight and the failure surface clear.
    """
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._backend = BackendType.REMOTE_GPU
    svc._remote_adapter = MagicMock()
    svc._mandate_preference = {"model": None, "vendor": None, "route": None}
    svc._last_remote_error = None
    return svc


# ---------------------------------------------------------------------------
# _remote_first_allowed contract
# ---------------------------------------------------------------------------


def test_remote_first_allowed_with_no_mandate_and_no_override():
    svc = _service_with_remote_active()
    assert svc._remote_first_allowed(None) is True, (
        "With nothing pinning routing, remote-GPU fast path is the "
        "established behavior — keep it."
    )


def test_remote_first_allowed_with_bare_model_override():
    """A bare model name with no vendor prefix does NOT pin a backend —
    the remote route may try it with its own configured model."""
    svc = _service_with_remote_active()
    assert svc._remote_first_allowed("gpt-5-mini") is True


def test_remote_first_allowed_rejects_vendor_prefixed_override():
    """`model_override='anthropic/claude-sonnet-4-6'` explicitly pins the
    Anthropic vendor — remote-GPU must not hijack it."""
    svc = _service_with_remote_active()
    assert svc._remote_first_allowed("anthropic/claude-sonnet-4-6") is False


def test_remote_first_allowed_rejects_vendor_route_prefixed_override():
    """`vendor:route/model` (the canonical pinned form) must disable the
    shortcut."""
    svc = _service_with_remote_active()
    assert svc._remote_first_allowed("anthropic:plan/claude-opus-4-6") is False


def test_remote_first_allowed_rejects_vendor_selector_without_slash():
    """`anthropic:plan` on its own (no model) is still a pin — respect it."""
    svc = _service_with_remote_active()
    assert svc._remote_first_allowed("anthropic:plan") is False


def test_remote_first_allowed_rejects_mandate_with_vendor():
    svc = _service_with_remote_active()
    svc._mandate_preference = {
        "model": "claude-sonnet-4-6",
        "vendor": "anthropic",
        "route": None,
    }
    assert svc._remote_first_allowed(None) is False


def test_remote_first_allowed_rejects_mandate_with_route():
    """A mandate that narrows by route (even without an explicit vendor)
    is still a pin that the shortcut must not bypass."""
    svc = _service_with_remote_active()
    svc._mandate_preference = {
        "model": "claude-sonnet-4-6",
        "vendor": None,
        "route": "plan",
    }
    assert svc._remote_first_allowed(None) is False


def test_remote_first_allowed_with_model_only_mandate():
    """Mandate with just a model name (no vendor / no route) doesn't
    narrow routing to a specific backend — remote-GPU may still run."""
    svc = _service_with_remote_active()
    svc._mandate_preference = {
        "model": "claude-sonnet-4-6",
        "vendor": None,
        "route": None,
    }
    assert svc._remote_first_allowed(None) is True


def test_remote_first_allowed_override_beats_mandate():
    """Even if the mandate doesn't pin, a vendor-prefixed override does."""
    svc = _service_with_remote_active()
    svc._mandate_preference = {"model": None, "vendor": None, "route": None}
    assert svc._remote_first_allowed("anthropic/claude-sonnet-4-6") is False


# ---------------------------------------------------------------------------
# Static verification: every remote-first branch uses the gate
# ---------------------------------------------------------------------------


def test_all_remote_gpu_call_sites_are_gated():
    """Every remote call enters the one lease/mandate gate."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    files_expected = {
        "kestrel_sovereign/llm/service.py": {
            "route_attempts": 2,  # generate(), generate_with_messages()
        },
        "kestrel_sovereign/llm/streaming.py": {
            "route_attempts": 3,
        },
    }

    for rel, info in files_expected.items():
        text = (repo / rel).read_text(encoding="utf-8")
        count = text.count("self._remote_route_attempt(")
        assert count == info["route_attempts"], (
            f"{rel}: expected {info['route_attempts']} managed route calls, "
            f"found {count}; all remote paths must use the lease boundary"
        )

    boundary = (repo / "kestrel_sovereign/llm/remote_backend.py").read_text(
        encoding="utf-8"
    )
    assert "self._remote_first_allowed(model_override)" in boundary
