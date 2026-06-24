"""Streaming fallback must not rotate providers on codex idle-timeout (#1429).

Three streaming entry points in ``llm/streaming.py`` each iterate a provider
fallback chain with a broad ``except Exception``. Before #1429, that broad
catch consumed ``CodexAppServerError`` and silently fell through to the next
provider — so a transient codex/ChatGPT-Plus stall answered the user from
the wrong model (anthropic, openai:api, …).

This test module pins the new behavior: harness-owned transport errors
(``CodexAppServerError`` and its subclasses) bypass the fallback chain and
surface as ``LLMStreamingError`` immediately. Non-codex exceptions keep
the existing rotate-to-next-provider semantics.

Mirrors openclaw commit ``3a64dc7623`` ("keep turn timeouts inside Codex").
"""
from typing import Any, AsyncIterator, List, Optional
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerConnectionClosed,
    CodexAppServerError,
    CodexAppServerTransportError,
)
from kestrel_sovereign.llm.streaming import (
    LLMStreamingError,
    StreamingMixin,
    _is_harness_owned_transport_error,
)


# ---------------------------------------------------------------------------
# Helper-level
# ---------------------------------------------------------------------------


def test_is_harness_owned_for_codex_transport_error():
    assert _is_harness_owned_transport_error(
        CodexAppServerTransportError("codex turn idle for 300s with no completion")
    )


def test_is_harness_owned_for_codex_app_server_connection_closed():
    """Subclass case: app-server process gone is also a transport
    failure (its parent class is now ``CodexAppServerTransportError``).
    """
    assert _is_harness_owned_transport_error(
        CodexAppServerConnectionClosed("codex app-server closed mid-turn")
    )


def test_is_harness_owned_false_for_non_transport_codex_error():
    """Caller-config / protocol-level / codex-reported-turn-failure
    errors are raised as plain ``CodexAppServerError`` (NOT the
    transport subclass). Those should retain normal fallback behavior —
    falling back to anthropic when codex says "tool_executor required"
    or "codex turn failed: content policy" is the correct user-facing
    behavior. Codex review of the v1 patch caught this.
    """
    assert not _is_harness_owned_transport_error(
        CodexAppServerError(
            "openai:plan (codex app-server) requires a tool_executor callback"
        )
    )
    assert not _is_harness_owned_transport_error(
        CodexAppServerError("codex turn failed: content_policy_violation")
    )
    assert not _is_harness_owned_transport_error(
        CodexAppServerError("thread/start returned no thread id: None")
    )


def test_is_harness_owned_false_for_generic_exceptions():
    """Generic provider errors must NOT be flagged — they should keep
    their existing rotate-to-next-provider behavior.
    """
    assert not _is_harness_owned_transport_error(RuntimeError("oops"))
    assert not _is_harness_owned_transport_error(ConnectionError("network"))
    assert not _is_harness_owned_transport_error(TimeoutError("slow"))
    assert not _is_harness_owned_transport_error(ValueError("bad arg"))


# ---------------------------------------------------------------------------
# Streaming-loop integration: build a minimal StreamingMixin host class
# and drive each of the three entry points through a 2-provider chain
# where the FIRST provider fails. Assert no rotation when the error is
# harness-owned; assert rotation works for ordinary exceptions.
# ---------------------------------------------------------------------------


class _RecordingHost(StreamingMixin):
    """StreamingMixin host with just enough surface for the fallback
    loops to run end-to-end against in-memory mocks.
    """

    def __init__(self, providers: List[dict]):
        self.providers = providers
        self.disabled = False
        self._disabled_routes = {}
        # Tracks the provider names that were actually attempted, so
        # tests can assert "second provider was never tried."
        self.attempted: List[str] = []
        # Tracks routes that ``_maybe_disable_route`` was invoked on.
        # Harness-owned errors must NOT cause this to fire — disabling
        # the route would skip the harness on every later turn.
        self.disabled_attempts: List[str] = []
        self._mandate_preference = {}
        self._mandate_fallbacks: List[dict] = []

    def _check_policy(self) -> None:
        return None

    def _available_providers(self):
        # Mirror LLMService: drop session-disabled routes. Tests don't
        # exercise disabled routes here, so this is just the full list.
        return [
            p for p in self.providers
            if p.get("name") not in self._disabled_routes
        ]

    # ``resolve_routing_meta`` and ``_match_selector`` are inherited from
    # ``StreamingMixin`` — tests exercise the SAME code path production does.

    def _filter_providers_by_selector(self, providers, selector):
        # Mirror LLMService's selector filter via the StreamingMixin helper so
        # the stub resolver below matches production resolution semantics.
        return self._match_selector(providers, selector)

    def resolve_provider_routing(
        self,
        *,
        model_override: Optional[str] = None,
        force_local_only: bool = False,
    ):
        # Honor an explicit vendor-prefixed override / mandate the way the
        # real resolver does, so tests can drive a genuine single-route
        # explicit selection through the streaming loops.
        providers = self._available_providers()
        target_model = None
        selector = None
        if model_override and ("/" in model_override or ":" in model_override):
            selector = (
                model_override.split("/", 1)[0]
                if "/" in model_override else model_override
            )
            if "/" in model_override:
                target_model = model_override.split("/", 1)[1]
        else:
            pref = self._mandate_preference or {}
            if pref.get("model") and pref.get("vendor"):
                pref_route = pref.get("route")
                selector = (
                    f"{pref['vendor']}:{pref_route}"
                    if pref_route else pref["vendor"]
                )
                target_model = pref["model"]
        if selector:
            matched = self._filter_providers_by_selector(providers, selector)
            if matched:
                return matched, target_model
            # Mandate with declared fallbacks: build the fallback chain.
            if self._mandate_fallbacks:
                fb_providers = []
                for fb in self._mandate_fallbacks:
                    fb_vendor = fb.get("vendor") or fb.get("provider")
                    fb_match = self._filter_providers_by_selector(
                        providers, fb_vendor
                    ) if fb_vendor else []
                    if fb_match:
                        fb_providers.append(fb_match[0])
                if fb_providers:
                    return fb_providers, None
        return list(providers), target_model

    def _resolve_concrete_model(self, target_model, provider):
        return provider.get("model", "test-model")

    def _maybe_disable_route(self, provider, exc):
        self.disabled_attempts.append(provider["name"])

    def _check_model_tool_support(self, providers, tools, model_override=None):
        return tools

    def _stamp_response_identity(self, response, *, model, provider):
        """Stub for #1370: StreamingMixin now calls this method."""
        pass


def _provider(name: str, adapter: Any) -> dict:
    return {
        "name": name,
        "adapter": adapter,
        "client": MagicMock(),
        "model": "claude-haiku-4-5",
        "vendor": name.split(":", 1)[0],
        "route": name.split(":", 1)[1] if ":" in name else "api",
        "is_cloud": True,
    }


class _RaisingAdapter:
    """First-provider stand-in: raises a configured exception on first
    chunk request.
    """

    def __init__(self, exc: BaseException, host: _RecordingHost, name: str):
        self._exc = exc
        self._host = host
        self._name = name

    async def get_streaming_response(self, **kwargs) -> AsyncIterator[str]:
        self._host.attempted.append(self._name)
        raise self._exc
        yield  # pragma: no cover (make this a generator)

    async def get_streaming_response_with_tools(self, **kwargs) -> AsyncIterator[Any]:
        self._host.attempted.append(self._name)
        raise self._exc
        yield  # pragma: no cover


class _OkAdapter:
    """Second-provider stand-in: yields a fixed text chunk. Used to
    prove that rotation happens for non-harness errors.
    """

    def __init__(self, host: _RecordingHost, name: str, text: str = "ok"):
        self._host = host
        self._name = name
        self._text = text

    async def get_streaming_response(self, **kwargs) -> AsyncIterator[str]:
        self._host.attempted.append(self._name)
        yield self._text

    async def get_streaming_response_with_tools(self, **kwargs) -> AsyncIterator[Any]:
        self._host.attempted.append(self._name)
        yield self._text


def _build_host(first_exc: BaseException) -> _RecordingHost:
    host = _RecordingHost([])
    host.providers = [
        _provider("openai:plan", _RaisingAdapter(first_exc, host, "openai:plan")),
        _provider("anthropic:api", _OkAdapter(host, "anthropic:api")),
    ]
    return host


# ---------------------------------------------------------------------------
# Entry point 1: get_streaming_response (assistant-stage idle timeout)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_streaming_response_no_rotation_on_codex_idle_timeout():
    host = _build_host(
        CodexAppServerTransportError(
            "codex turn idle for 300s with no completion — ..."
        )
    )
    chunks: List[str] = []
    with pytest.raises(LLMStreamingError) as ei:
        async for chunk in host.get_streaming_response(
            system_prompt="sys",
            user_prompt="hi",
        ):
            chunks.append(chunk)
    assert host.attempted == ["openai:plan"], (
        "Anthropic provider must NOT be tried after a codex stall — "
        f"got {host.attempted}"
    )
    assert "Harness-owned route" in str(ei.value)
    assert ei.value.provider == "openai:plan"
    assert isinstance(ei.value.underlying, CodexAppServerTransportError)
    assert chunks == []


@pytest.mark.asyncio
async def test_get_streaming_response_no_rotation_on_codex_prompt_timeout():
    """Prompt-stage timeout (turn/start RPC hang) is also a transport
    failure — same surface-error behavior.
    """
    host = _build_host(
        CodexAppServerTransportError("turn/start timed out after 60s")
    )
    with pytest.raises(LLMStreamingError):
        async for _ in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
        ):
            pass
    assert host.attempted == ["openai:plan"]


@pytest.mark.asyncio
async def test_get_streaming_response_rotates_on_non_transport_codex_error():
    """Caller-config or codex-reported-turn-failure errors are raised
    as plain ``CodexAppServerError`` (not the transport subclass) and
    should still fall through to the next provider — falling back to
    anthropic on "tool_executor required" or "content_policy_violation"
    gives the user a useful response. Codex review of v1 caught this.
    """
    host = _build_host(
        CodexAppServerError(
            "openai:plan (codex app-server) requires a tool_executor callback"
        )
    )
    chunks: List[str] = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "anthropic:api"], (
        "Caller-config codex error should fall through to next provider — "
        f"got {host.attempted}"
    )
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_get_streaming_response_rotates_on_generic_exception():
    """Existing fallback behavior preserved for non-codex errors."""
    host = _build_host(ConnectionError("network blip"))
    chunks: List[str] = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "anthropic:api"]
    assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# Entry point 2: stream_with_messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_with_messages_no_rotation_on_codex_idle_timeout():
    host = _build_host(
        CodexAppServerTransportError("codex turn idle for 300s with no completion")
    )
    # stream_with_messages also probes the remote-GPU backend before the
    # provider chain — disable that path so the test exercises the chain.
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None

    with pytest.raises(LLMStreamingError):
        async for _ in host.stream_with_messages(
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
    assert host.attempted == ["openai:plan"]


@pytest.mark.asyncio
async def test_stream_with_messages_rotates_on_generic_exception():
    host = _build_host(RuntimeError("transient"))
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None

    chunks: List[str] = []
    async for chunk in host.stream_with_messages(
        messages=[{"role": "user", "content": "hi"}],
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "anthropic:api"]
    assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# Entry point 3: stream_with_tool_detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_with_tool_detection_no_rotation_on_codex_idle_timeout():
    host = _build_host(
        CodexAppServerTransportError("codex turn idle for 300s with no completion")
    )
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None

    with pytest.raises(LLMStreamingError):
        async for _ in host.stream_with_tool_detection(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        ):
            pass
    assert host.attempted == ["openai:plan"]


@pytest.mark.asyncio
async def test_codex_auth_shaped_error_does_not_disable_route():
    """Regression: a codex error whose message text happens to look like
    an auth failure (``"authentication failed"``, ``"unauthorized"``)
    must NOT trigger ``_maybe_disable_route``. ``_is_permanent_auth_error``
    substring-matches ``"authentication"`` and would otherwise mark
    ``openai:plan`` permanently disabled for the rest of the process —
    the user's next turn would skip codex even after they fix the
    upstream issue. Codex review of #1429 caught this.
    """
    host = _build_host(
        CodexAppServerTransportError(
            "codex turn idle for 300s with no completion — "
            "(downstream authentication failed)"
        )
    )
    with pytest.raises(LLMStreamingError):
        async for _ in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
        ):
            pass
    # The harness-owned check runs BEFORE _maybe_disable_route now —
    # so the route should never have been considered for disabling.
    assert host.disabled_attempts == [], (
        "_maybe_disable_route must not run for harness-owned errors — "
        f"got {host.disabled_attempts}"
    )


@pytest.mark.asyncio
async def test_generic_auth_error_still_disables_route():
    """The disable-route path must still fire for non-codex 401/403-style
    errors — that's the whole point of the auth-disable mechanism.
    """
    host = _build_host(
        PermissionError("invalid_api_key: please rotate")  # auth-shaped
    )
    chunks: List[str] = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    assert host.disabled_attempts == ["openai:plan"], (
        "Non-harness auth-shaped error should still disable the route — "
        f"got {host.disabled_attempts}"
    )
    assert host.attempted == ["openai:plan", "anthropic:api"]


@pytest.mark.asyncio
async def test_stream_with_tool_detection_rotates_on_generic_exception():
    host = _build_host(ValueError("transient parser error"))
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None

    chunks: List[Any] = []
    async for chunk in host.stream_with_tool_detection(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "anthropic:api"]
    assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# No silent cross-vendor fallback away from the operator's configured route.
#
# Regression for the blind-fallback bug: an agent configured with
# ``route_priority = ["openai:plan", "openai:api"]`` (openai vendor only) but
# whose host also has ANTHROPIC_API_KEY set ends up with an auto-appended
# ``anthropic:*`` route in ``self.providers``. When the operator's openai
# routes fail at generation, the chain must NOT silently answer from anthropic
# — it must surface a loud, diagnostic error (feedback_no_blind_fallbacks).
# ---------------------------------------------------------------------------


def _build_configured_host(
    providers_spec,
    *,
    route_priority,
):
    """Host whose ``config`` carries an operator ``route_priority``.

    ``providers_spec`` is a list of ``(name, "raise"|"ok", exc_or_none)``.
    """
    host = _RecordingHost([])
    host.config = {"route_priority": list(route_priority)}
    built = []
    for name, kind, exc in providers_spec:
        if kind == "raise":
            adapter = _RaisingAdapter(exc, host, name)
        else:
            adapter = _OkAdapter(host, name)
        built.append(_provider(name, adapter))
    host.providers = built
    return host


@pytest.mark.asyncio
async def test_no_silent_swap_to_unconfigured_vendor_get_streaming_response():
    """openai vendor configured; anthropic auto-appended from env. Both openai
    routes fail -> must raise LLMStreamingError, never stream from anthropic.
    """
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("openai:api", "raise", RuntimeError("openai 500")),
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    with pytest.raises(LLMStreamingError) as ei:
        async for chunk in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
        ):
            chunks.append(chunk)
    # anthropic must NEVER have been attempted.
    assert host.attempted == ["openai:plan", "openai:api"], host.attempted
    assert chunks == []
    msg = str(ei.value)
    assert "refusing to silently swap vendors" in msg
    assert "unconfigured vendors" in msg
    assert ei.value.provider == "openai:api"


@pytest.mark.asyncio
async def test_same_vendor_fallback_still_allowed():
    """openai:plan -> openai:api is the operator's own vendor; allowed."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("openai:api", "ok", None),
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "openai:api"]
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_configured_cross_vendor_fallback_allowed():
    """If the operator EXPLICITLY listed anthropic in route_priority, falling
    over to it is their configured choice — not a blind swap.
    """
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "anthropic:api"],
    )
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "anthropic:api"]
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_no_silent_swap_stream_with_messages():
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None
    with pytest.raises(LLMStreamingError) as ei:
        async for _ in host.stream_with_messages(
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
    assert host.attempted == ["openai:plan"], host.attempted
    assert "refusing to silently swap vendors" in str(ei.value)


@pytest.mark.asyncio
async def test_no_silent_swap_stream_with_tool_detection():
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None
    with pytest.raises(LLMStreamingError) as ei:
        async for _ in host.stream_with_tool_detection(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        ):
            pass
    assert host.attempted == ["openai:plan"], host.attempted
    assert "refusing to silently swap vendors" in str(ei.value)


# ---------------------------------------------------------------------------
# Codex P2 regression: an UNCONFIGURED vendor positioned BEFORE a later
# CONFIGURED route in the chain. The old "no configured route remains" guard
# returned False here (a configured route still lay ahead) and the loop tried
# the unconfigured vendor FIRST — exactly the blind cross-vendor answer the
# guard exists to prevent. The fix skips unconfigured candidates before
# dispatch, so they are NEVER attempted.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unconfigured_vendor_before_configured_is_skipped_get_streaming_response():
    """priority=[openai:plan, openai:api]; chain orders the auto-appended
    anthropic route BEFORE openai:api. openai:plan fails -> anthropic must be
    SKIPPED (never attempted) and openai:api tried instead."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("anthropic:api", "ok", None),   # unconfigured, positioned first
            ("openai:api", "ok", None),       # configured, positioned later
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    # anthropic must NEVER be attempted even though it sits ahead of openai:api.
    assert host.attempted == ["openai:plan", "openai:api"], host.attempted
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_unconfigured_vendor_before_only_unconfigured_left_raises():
    """priority=[openai:plan]; chain is [openai:plan, anthropic:api, gemini:api]
    where the only routes after the failed configured one are unconfigured. The
    loop must raise loudly (naming openai:plan + the underlying error), never
    answer from anthropic or gemini."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", RuntimeError("openai 500")),
            ("anthropic:api", "ok", None),
            ("gemini:api", "ok", None),
        ],
        route_priority=["openai:plan"],
    )
    chunks = []
    with pytest.raises(LLMStreamingError) as ei:
        async for chunk in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
        ):
            chunks.append(chunk)
    assert host.attempted == ["openai:plan"], host.attempted
    assert chunks == []
    msg = str(ei.value)
    assert "refusing to silently swap vendors" in msg
    assert "openai:plan" in msg
    assert "openai 500" in msg
    assert ei.value.provider == "openai:plan"


@pytest.mark.asyncio
async def test_unconfigured_vendor_before_configured_skipped_stream_with_tool_detection():
    """Same P2 scenario through stream_with_tool_detection."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", CodexAppServerError("codex turn failed")),
            ("anthropic:api", "ok", None),
            ("openai:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None
    chunks = []
    async for chunk in host.stream_with_tool_detection(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "openai:api"], host.attempted
    assert chunks == ["ok"]


def test_skip_unconfigured_route_helper():
    """``_skip_unconfigured_route`` decides per-candidate (before dispatch)
    whether a route's vendor is one the operator configured."""
    host = _RecordingHost([])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    assert host._skip_unconfigured_route({"name": "openai:plan", "vendor": "openai"}) is False
    assert host._skip_unconfigured_route({"name": "openai:api", "vendor": "openai"}) is False
    # anthropic is auto-appended (credentialed) but unchosen -> skip it.
    assert host._skip_unconfigured_route({"name": "anthropic:api", "vendor": "anthropic"}) is True
    # No route_priority configured -> never skip (legacy default chain).
    host.config = {}
    assert host._skip_unconfigured_route({"name": "anthropic:api", "vendor": "anthropic"}) is False


def test_configured_routes_exhausted_helper():
    """``_configured_routes_exhausted`` raises (True) only when a configured
    route failed and NO remaining route is from a configured vendor — even if
    an unconfigured route sits between the failed one and the chain end."""
    host = _RecordingHost([])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    providers = [
        {"name": "openai:plan", "vendor": "openai"},
        {"name": "openai:api", "vendor": "openai"},
        {"name": "anthropic:api", "vendor": "anthropic"},
    ]
    # openai:plan failed, openai:api still ahead -> legitimate same-vendor.
    assert host._configured_routes_exhausted(providers, 0) is False
    # openai:api failed, only anthropic (unconfigured) left -> exhausted.
    assert host._configured_routes_exhausted(providers, 1) is True
    # No route_priority configured -> never raise (legacy behavior).
    host.config = {}
    assert host._configured_routes_exhausted(providers, 1) is False


def test_configured_routes_exhausted_unconfigured_before_configured():
    """Codex P2 scenario at the helper level: an unconfigured vendor sits
    BEFORE the next configured route. The helper must NOT report exhaustion
    yet (a legitimate configured candidate still lies ahead), but the
    unconfigured candidate must be skipped at the loop's top so it's never
    attempted first."""
    host = _RecordingHost([])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    # priority=[openai:plan, openai:api]; chain orders anthropic BEFORE openai:api.
    providers = [
        {"name": "openai:plan", "vendor": "openai"},
        {"name": "anthropic:api", "vendor": "anthropic"},  # unconfigured, ahead
        {"name": "openai:api", "vendor": "openai"},         # configured, later
    ]
    # openai:plan (idx 0) failed: a configured route (openai:api) still lies
    # ahead, so this is NOT exhaustion — the loop should keep going (skipping
    # anthropic) and reach openai:api.
    assert host._configured_routes_exhausted(providers, 0) is False
    # The intervening anthropic route must be skipped before dispatch.
    assert host._skip_unconfigured_route(providers[1]) is True
    assert host._skip_unconfigured_route(providers[2]) is False


# ---------------------------------------------------------------------------
# Codex re-review P1: a SINGLETON incidental chain is NOT an explicit
# selection. The operator configured route_priority=[openai:*] but every
# openai route failed to initialize / is disabled, leaving exactly ONE
# incidental credentialed route (anthropic, never asked for). Before the fix
# ``len(providers) == 1`` set ``mandate_restricted=True`` and BYPASSED
# ``_skip_unconfigured_route`` — so the request still went to the unconfigured
# vendor. The fix gates the bypass on a REAL explicit-selection signal, not
# list length, so the incidental singleton is still skipped/raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_singleton_incidental_unconfigured_route_is_skipped_not_attempted():
    """route_priority names only openai, but the only surviving route is the
    incidental anthropic one. It must be SKIPPED (never attempted), raising
    the no-routes diagnostic — not silently answered from anthropic."""
    host = _build_configured_host(
        [
            ("anthropic:api", "ok", None),  # incidental, sole survivor
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    with pytest.raises(LLMStreamingError):
        async for chunk in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
        ):
            chunks.append(chunk)
    # anthropic is the only route but unconfigured -> never attempted.
    assert host.attempted == [], host.attempted
    assert chunks == []


@pytest.mark.asyncio
async def test_singleton_incidental_unconfigured_route_skipped_stream_with_tool_detection():
    """Same P1 scenario through stream_with_tool_detection."""
    host = _build_configured_host(
        [
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    from kestrel_sovereign.llm.remote_backend import BackendType
    host._backend = BackendType.LOCAL
    host._remote_client = None
    chunks = []
    with pytest.raises(LLMStreamingError):
        async for chunk in host.stream_with_tool_detection(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        ):
            chunks.append(chunk)
    assert host.attempted == [], host.attempted


@pytest.mark.asyncio
async def test_genuine_explicit_single_route_selection_still_attempts_its_route():
    """A genuine explicit selection (vendor-prefixed override) of the SOLE
    route must still be ATTEMPTED — even when its vendor is outside
    route_priority. The operator/user explicitly asked for THIS route."""
    host = _build_configured_host(
        [
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
        model_override="anthropic/claude-haiku-4-5",
    ):
        chunks.append(chunk)
    # Explicitly selected -> attempted despite not being in route_priority.
    assert host.attempted == ["anthropic:api"], host.attempted
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_explicit_single_route_failure_raises_loudly_not_silent_fallthrough():
    """A genuine explicit single-route selection that FAILS must raise the
    specific 'Selected route ... failed' error — never fall through to the
    incidental default chain."""
    host = _build_configured_host(
        [
            ("anthropic:api", "raise", RuntimeError("anthropic 500")),
            ("openai:api", "ok", None),  # incidental survivor; must not be hit
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    with pytest.raises(LLMStreamingError) as ei:
        async for chunk in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
            model_override="anthropic/claude-haiku-4-5",
        ):
            chunks.append(chunk)
    assert host.attempted == ["anthropic:api"], host.attempted
    assert "Selected route" in str(ei.value)
    assert ei.value.provider == "anthropic:api"


# ---------------------------------------------------------------------------
# Codex re-review P2: the authorized-vendor set must include operator-declared
# _mandate_fallbacks, even when those vendors are NOT in route_priority.
# Before the fix ``_configured_route_vendors`` only read static route_priority,
# so a mandate fallback to a vendor outside it was wrongly skipped ->
# "All providers failed". The fix unions route_priority + the selected route's
# vendor + the mandate-fallback vendors.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mandate_fallback_vendor_outside_route_priority_is_attempted():
    """Mandate pins openai but it's unavailable; the operator declared an
    anthropic fallback. anthropic is NOT in route_priority, yet because it's
    an explicitly-declared mandate fallback it must be ATTEMPTED, not skipped."""
    host = _build_configured_host(
        [
            ("anthropic:api", "ok", None),  # the declared mandate fallback
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    # Mandate pins openai (no openai route present -> resolver uses fallbacks).
    host._mandate_preference = {
        "vendor": "openai", "route": None, "model": "gpt-5-mini",
    }
    host._mandate_fallbacks = [{"vendor": "anthropic", "model": "claude-haiku-4-5"}]
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    assert host.attempted == ["anthropic:api"], host.attempted
    assert chunks == ["ok"]


def test_resolve_routing_meta_unions_mandate_fallback_vendors():
    """Helper-level: configured_vendors includes route_priority + mandate
    vendor + fallback vendors; a bare singleton chain is NOT explicit."""
    host = _RecordingHost([
        _provider("anthropic:api", _OkAdapter(_RecordingHost([]), "anthropic:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    host._mandate_preference = {
        "vendor": "openai", "route": None, "model": "gpt-5-mini",
    }
    host._mandate_fallbacks = [{"vendor": "anthropic", "model": "claude-haiku-4-5"}]
    explicit, vendors = host.resolve_routing_meta()
    # openai (route_priority + mandate vendor) and anthropic (fallback) both in.
    assert "openai" in vendors
    assert "anthropic" in vendors
    # Mandate openai didn't match any available route (only anthropic present)
    # -> fell to the fallback CHAIN -> NOT an explicit single-route selection.
    assert explicit is False


def test_resolve_routing_meta_explicit_for_vendor_prefixed_override():
    """A vendor-prefixed model_override that matches a real route is an
    explicit selection."""
    host = _RecordingHost([
        _provider("anthropic:api", _OkAdapter(_RecordingHost([]), "anthropic:api")),
    ])
    host.config = {"route_priority": ["openai:plan"]}
    explicit, vendors = host.resolve_routing_meta(
        model_override="anthropic/claude-haiku-4-5",
    )
    assert explicit is True
    assert "anthropic" in vendors  # selected vendor folded in
    assert "openai" in vendors     # route_priority retained


def test_resolve_routing_meta_singleton_incidental_is_not_explicit():
    """A lone incidental credentialed route (no override, no matching mandate)
    is NOT an explicit selection — list length must not imply selection."""
    host = _RecordingHost([
        _provider("anthropic:api", _OkAdapter(_RecordingHost([]), "anthropic:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    explicit, vendors = host.resolve_routing_meta()
    assert explicit is False
    assert vendors == {"openai"}
