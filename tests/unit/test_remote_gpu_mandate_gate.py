"""Regression suite for #734 — remote-GPU shortcut must honor the mandate.

Before the vendor/route refactor the remote GPU backend ran before
:meth:`LLMService.resolve_provider_routing`, so a user-pinned
``{vendor, route, model}`` mandate or vendor-prefixed ``model_override``
could be silently ignored while the answer came from the remote pod.

The gate lives in :meth:`LLMService._remote_first_allowed`.  These tests
pin its contract (when it returns False, the remote-first branch skips).
Visual inspection of the five call sites confirms all of them now wrap
the ``_remote_client`` check with this guard:

- ``service.generate``                 (generate text)
- ``service.generate_with_messages``   (multi-turn tool calling)
- ``streaming.generate_stream``        (text streaming)
- ``streaming.stream_with_messages``   (streaming after tool exec)
- ``streaming.stream_with_tool_detection`` (streaming + tool detection)
"""

from unittest.mock import MagicMock

from kestrel_sovereign.llm.remote_backend import BackendType, RemoteGPUConfig


def _service_with_remote_active():
    """Return an LLMService-shaped stand-in with the remote backend primed.

    We don't instantiate the real LLMService here because its dependency
    graph is huge and the helper under test only touches three attributes:
    ``_backend``, ``_remote_client``, ``_mandate_preference``.  A minimal
    stand-in keeps the tests tight and the failure surface clear.
    """
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    svc._backend = BackendType.REMOTE_GPU
    svc._remote_client = MagicMock()
    svc._remote_config = RemoteGPUConfig(base_url="http://pod", model="local-model")
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
    """Every ``BackendType.REMOTE_GPU`` check that also tests
    ``self._remote_client`` is a remote-first shortcut site, and each one
    must be paired with a call to ``self._remote_first_allowed(...)``.

    If someone adds a sixth remote-first branch without the guard, this
    test fails with a clear pointer to the file.  Defining the helper
    itself adds one extra reference in service.py (the ``def`` line) —
    we allow for that.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    files_expected = {
        "kestrel_sovereign/llm/service.py": {
            "trigger_expected": 2,  # generate(), generate_with_messages()
            "helper_defined_here": True,  # def _remote_first_allowed lives here
        },
        "kestrel_sovereign/llm/streaming.py": {
            "trigger_expected": 3,  # generate_stream, stream_with_messages, stream_with_tool_detection
            "helper_defined_here": False,
        },
    }

    for rel, info in files_expected.items():
        text = (repo / rel).read_text(encoding="utf-8")
        # Count the remote-first branch signature: a REMOTE_GPU check that
        # gates on _remote_client in the same condition.  Fuzzy on whitespace
        # so reflowing with a linter doesn't break the assertion.
        trigger_pattern = re.compile(
            r"BackendType\.REMOTE_GPU\b[^{}]{0,200}?self\._remote_client",
            re.DOTALL,
        )
        trigger_count = len(trigger_pattern.findall(text))

        # Subtract the helper definition's own reference (not a call site).
        guard_calls = text.count("self._remote_first_allowed(")
        assert trigger_count == info["trigger_expected"], (
            f"{rel}: expected {info['trigger_expected']} remote-first branches, "
            f"found {trigger_count}. Layout changed — update this test."
        )
        assert guard_calls == trigger_count, (
            f"{rel}: {trigger_count} remote-first branches but only "
            f"{guard_calls} _remote_first_allowed guards. A branch was "
            f"added without the #734 gate."
        )
