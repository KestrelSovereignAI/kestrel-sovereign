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
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Optional
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerConnectionClosed,
    CodexAppServerError,
    CodexAppServerFrameTooLarge,
    CodexAppServerTransportError,
)
from kestrel_sovereign.llm.streaming import (
    LLMStreamingError,
    RoutingResolution,
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
    assert not _is_harness_owned_transport_error(
        CodexAppServerFrameTooLarge(
            "codex app-server JSON-RPC frame exceeded the 64 MiB bridge limit"
        )
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

    @asynccontextmanager
    async def _remote_route_attempt(self, **_kwargs):
        """This provider-fallback harness has no managed private route."""
        yield None

    def _available_providers(self):
        # Mirror LLMService: drop session-disabled routes. Tests don't
        # exercise disabled routes here, so this is just the full list.
        return [
            p for p in self.providers
            if p.get("name") not in self._disabled_routes
        ]

    # ``_compute_route_authorization`` and ``_match_selector`` are inherited
    # from ``StreamingMixin`` — tests exercise the SAME code path production
    # does. ``resolve_provider_routing`` below returns a ``RoutingResolution``
    # whose ``.meta`` is computed by that shared helper, exactly as the real
    # ``LLMService.resolve_provider_routing`` does.

    def _filter_providers_by_selector(self, providers, selector):
        # Mirror LLMService's selector filter via the StreamingMixin helper so
        # the stub resolver below matches production resolution semantics.
        return self._match_selector(providers, selector)

    def resolve_provider_routing(
        self,
        *,
        model_override: Optional[str] = None,
        force_local_only: bool = False,
    ) -> RoutingResolution:
        # Honor an explicit vendor-prefixed override / mandate the way the
        # real resolver does, so tests can drive a genuine single-route
        # explicit selection through the streaming loops.
        providers = self._available_providers()
        target_model = None
        selector = None
        chosen = None
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
                chosen = (matched, target_model)
            elif self._mandate_fallbacks:
                # Mandate with declared fallbacks: build the fallback chain.
                fb_providers = []
                for fb in self._mandate_fallbacks:
                    fb_vendor = fb.get("vendor") or fb.get("provider")
                    fb_match = self._filter_providers_by_selector(
                        providers, fb_vendor
                    ) if fb_vendor else []
                    if fb_match:
                        fb_providers.append(fb_match[0])
                if fb_providers:
                    chosen = (fb_providers, None)
        if chosen is None:
            chosen = (list(providers), target_model)
        meta = self._compute_route_authorization(
            model_override=model_override,
            force_local_only=force_local_only,
        )
        return RoutingResolution(chosen[0], chosen[1], meta)

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
    allow_paid_fallback=True,
):
    """Host whose ``config`` carries an operator ``route_priority``.

    ``providers_spec`` is a list of ``(name, "raise"|"ok", exc_or_none)``.
    ``allow_paid_fallback`` defaults True (historical behavior); pass False to
    exercise cost-over-availability strict mode (refuse plan->paid downgrade).
    """
    host = _RecordingHost([])
    host.config = {
        "route_priority": list(route_priority),
        "allow_paid_fallback": allow_paid_fallback,
    }
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
# allow_paid_fallback = false (cost-over-availability): a plan/free route
# failure (e.g. a 429 throttle) must NOT silently downgrade to a metered
# :api route. It may still reach another plan/free route, or fail loudly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_mode_refuses_plan_to_paid_downgrade():
    """openai:plan fails; with allow_paid_fallback=False the same-vendor paid
    route openai:api is refused (not billed) and the chain fails loudly."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", RuntimeError("429 rate limit exceeded")),
            ("openai:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
        allow_paid_fallback=False,
    )
    with pytest.raises(LLMStreamingError):
        async for _ in host.get_streaming_response(system_prompt="s", user_prompt="hi"):
            pass
    assert host.attempted == ["openai:plan"], host.attempted


@pytest.mark.asyncio
async def test_strict_mode_still_allows_plan_to_plan_fallback():
    """Strict mode only refuses paid downgrades — a plan->plan fallback (both
    subscription routes) is still allowed."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", RuntimeError("429 rate limit exceeded")),
            ("anthropic:plan", "ok", None),
        ],
        route_priority=["openai:plan", "anthropic:plan"],
        allow_paid_fallback=False,
    )
    chunks = []
    async for chunk in host.get_streaming_response(system_prompt="s", user_prompt="hi"):
        chunks.append(chunk)
    assert host.attempted == ["openai:plan", "anthropic:plan"], host.attempted
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_strict_mode_leaves_all_paid_config_unaffected():
    """A deliberately metered config (no plan route) is unaffected — there is
    no preferred plan/free route to downgrade away from, so paid->paid
    fallback still works even with allow_paid_fallback=False."""
    host = _build_configured_host(
        [
            ("openai:api", "raise", RuntimeError("500 server error")),
            ("anthropic:api", "ok", None),
        ],
        route_priority=["openai:api", "anthropic:api"],
        allow_paid_fallback=False,
    )
    chunks = []
    async for chunk in host.get_streaming_response(system_prompt="s", user_prompt="hi"):
        chunks.append(chunk)
    assert host.attempted == ["openai:api", "anthropic:api"], host.attempted
    assert chunks == ["ok"]


def test_route_is_paid_classifies_metered_cloud_routes():
    """#2074/P2: _route_is_paid must treat a metered cloud route like
    google:vertex as PAID (bills GCP per token), not only literal ':api'. Plan
    (subscription) and local routes stay non-paid."""
    from kestrel_sovereign.llm.streaming import StreamingMixin as S
    # Metered cloud routes.
    assert S._route_is_paid({"route": "api", "is_cloud": True}) is True
    assert S._route_is_paid({"route": "vertex", "is_cloud": True}) is True
    assert S._route_is_paid({"name": "google:vertex", "is_cloud": True}) is True
    # Subscription plan and local — never per-token metered.
    assert S._route_is_paid({"route": "plan", "is_cloud": True}) is False
    assert S._route_is_paid({"route": "local", "is_local": True}) is False
    assert S._route_is_paid({"name": "ollama:local", "is_local": True}) is False


@pytest.mark.asyncio
async def test_strict_mode_refuses_plan_to_vertex_downgrade():
    """#2074/P2: with allow_paid_fallback=False, a plan throttle must NOT
    silently fall through to the GCP-metered google:vertex route (route name
    'vertex', not 'api') — it's refused, and the chain fails loudly."""
    host = _build_configured_host(
        [
            ("anthropic:plan", "raise", RuntimeError("429 rate limit exceeded")),
            ("google:vertex", "ok", None),
        ],
        route_priority=["anthropic:plan", "google:vertex"],
        allow_paid_fallback=False,
    )
    with pytest.raises(LLMStreamingError):
        async for _ in host.get_streaming_response(system_prompt="s", user_prompt="hi"):
            pass
    # vertex was never billed — the plan throttle didn't downgrade to metered.
    assert host.attempted == ["anthropic:plan"], host.attempted


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
    explicit, vendors = host._compute_route_authorization()
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
    explicit, vendors = host._compute_route_authorization(
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
    explicit, vendors = host._compute_route_authorization()
    assert explicit is False
    assert vendors == {"openai"}


# ---------------------------------------------------------------------------
# Codex P2 (round 4): a VENDOR-WIDE selection that resolves to MULTIPLE routes
# must NOT be marked explicit — same-vendor fallback among the matched routes
# must still work. ``explicit_selection`` is True only when the selection pins
# EXACTLY ONE route, OR when a route-qualified ``vendor:route`` selector was
# asked for. A bare vendor selector ("openai") matching both openai:plan AND
# openai:api is a multi-route vendor-wide selection -> NOT explicit, so the
# first route's transient failure falls through to the next same-vendor route.
# ---------------------------------------------------------------------------


def test_resolve_routing_meta_vendor_wide_override_multi_route_not_explicit():
    """``model_override="openai"`` with both openai:plan and openai:api
    available matches 2 routes -> NOT explicit (same-vendor fallback intact)."""
    host = _RecordingHost([
        _provider("openai:plan", _OkAdapter(_RecordingHost([]), "openai:plan")),
        _provider("openai:api", _OkAdapter(_RecordingHost([]), "openai:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    explicit, vendors = host._compute_route_authorization(model_override="openai")
    assert explicit is False
    assert "openai" in vendors


def test_resolve_routing_meta_route_qualified_override_is_explicit():
    """A concrete ``vendor:route`` override pins ONE route -> explicit even
    though other same-vendor routes exist."""
    host = _RecordingHost([
        _provider("openai:plan", _OkAdapter(_RecordingHost([]), "openai:plan")),
        _provider("openai:api", _OkAdapter(_RecordingHost([]), "openai:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    explicit, vendors = host._compute_route_authorization(
        model_override="openai:api/gpt-5-mini",
    )
    assert explicit is True
    assert "openai" in vendors


def test_resolve_routing_meta_vendor_wide_mandate_multi_route_not_explicit():
    """A vendor-only mandate ("openai", no route) matching 2 routes is NOT
    explicit; a route-qualified mandate would be."""
    host = _RecordingHost([
        _provider("openai:plan", _OkAdapter(_RecordingHost([]), "openai:plan")),
        _provider("openai:api", _OkAdapter(_RecordingHost([]), "openai:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    host._mandate_preference = {
        "vendor": "openai", "route": None, "model": "gpt-5-mini",
    }
    explicit, vendors = host._compute_route_authorization()
    assert explicit is False
    assert "openai" in vendors


def test_resolve_routing_meta_route_qualified_mandate_is_explicit():
    """A mandate that pins ``vendor:route`` (openai:api) is a single concrete
    route -> explicit even with a sibling same-vendor route available."""
    host = _RecordingHost([
        _provider("openai:plan", _OkAdapter(_RecordingHost([]), "openai:plan")),
        _provider("openai:api", _OkAdapter(_RecordingHost([]), "openai:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    host._mandate_preference = {
        "vendor": "openai", "route": "api", "model": "gpt-5-mini",
    }
    explicit, vendors = host._compute_route_authorization()
    assert explicit is True
    assert "openai" in vendors


@pytest.mark.asyncio
async def test_vendor_wide_selection_falls_through_to_same_vendor_route():
    """End-to-end: model_override="openai" with both openai:plan and openai:api
    available. openai:plan transiently fails -> the loop must FALL THROUGH to
    openai:api (same vendor), NOT raise. Regresses the P2 where a vendor-wide
    selection was wrongly marked explicit and raised on the first failure."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", RuntimeError("openai:plan 503 transient")),
            ("openai:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
        model_override="openai",
    ):
        chunks.append(chunk)
    # Vendor-wide -> fell through to the second same-vendor route.
    assert host.attempted == ["openai:plan", "openai:api"], host.attempted
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_route_qualified_selection_is_explicit_loud_fail_no_fallthrough():
    """End-to-end: a concrete ``openai:api`` route-qualified selection is
    explicit. Its transient failure must raise loudly (Selected route ...
    failed) and NEVER fall through to the sibling openai:plan route."""
    host = _build_configured_host(
        [
            ("openai:api", "raise", RuntimeError("openai:api 500")),
            ("openai:plan", "ok", None),  # sibling; must NOT be attempted
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    chunks = []
    with pytest.raises(LLMStreamingError) as ei:
        async for chunk in host.get_streaming_response(
            system_prompt="sys", user_prompt="hi",
            model_override="openai:api/gpt-5-mini",
        ):
            chunks.append(chunk)
    # Only the pinned route was attempted; no fall-through to openai:plan.
    assert host.attempted == ["openai:api"], host.attempted
    assert chunks == []
    assert "Selected route" in str(ei.value)
    assert ei.value.provider == "openai:api"


# ---------------------------------------------------------------------------
# Codex re-review P2 (round 5a): a BARE ``model_override`` while an UNRELATED
# persisted mandate exists. ``resolve_provider_routing`` gives the override
# precedence (its ``if model_override`` branch) and does NOT consult / filter
# to the mandate vendor. The OLD ``resolve_routing_meta`` still folded the
# mandate's vendor (and its fallbacks) into the authorized set, so
# ``_skip_unconfigured_route`` could DROP the very route serving the override
# whenever the mandate vendor differed. The single-source helper now returns
# from the override branch before touching the mandate, so the override's
# vendor is authorized and the mandate's is NOT.
# ---------------------------------------------------------------------------


def test_bare_override_does_not_authorize_unrelated_stored_mandate_vendor():
    """Helper-level P2a: a bare ``vendor/model`` override authorizes the
    override's vendor only; an unrelated stored mandate's vendor (and its
    fallbacks) must NOT enter the authorized set."""
    host = _RecordingHost([
        _provider("openai:api", _OkAdapter(_RecordingHost([]), "openai:api")),
        _provider("anthropic:api", _OkAdapter(_RecordingHost([]), "anthropic:api")),
    ])
    host.config = {"route_priority": []}  # no static route_priority
    # Stored mandate pins anthropic with an unrelated fallback chain.
    host._mandate_preference = {
        "vendor": "anthropic", "route": None, "model": "claude-haiku-4-5",
    }
    host._mandate_fallbacks = [{"vendor": "vertex_ai", "model": "gemini"}]
    # Bare override selects openai — override precedence.
    explicit, vendors = host._compute_route_authorization(
        model_override="openai/gpt-5-mini",
    )
    assert "openai" in vendors                  # the override's vendor
    assert "anthropic" not in vendors           # P2a: stale mandate vendor excluded
    assert "vertex_ai" not in vendors           # P2a: stale mandate fallbacks excluded
    assert explicit is True                      # bare override matched exactly one route


@pytest.mark.asyncio
async def test_bare_override_route_served_not_skipped_despite_stored_mandate():
    """End-to-end P2a: with a stored anthropic mandate, a bare ``openai/...``
    override must be SERVED by the openai route — never skipped by the
    unconfigured-route guard authorizing the unrelated mandate vendor."""
    host = _build_configured_host(
        [
            ("openai:api", "ok", None),       # the override target — must run
            ("anthropic:api", "ok", None),    # mandate vendor — must NOT be reached
        ],
        route_priority=[],
    )
    host._mandate_preference = {
        "vendor": "anthropic", "route": None, "model": "claude-haiku-4-5",
    }
    host._mandate_fallbacks = [{"vendor": "vertex_ai", "model": "gemini"}]
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
        model_override="openai/gpt-5-mini",
    ):
        chunks.append(chunk)
    assert host.attempted == ["openai:api"], host.attempted
    assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# Codex re-review P2 (round 5b): a STALE ``_mandate_fallbacks`` list with NO
# active mandate (default routing). ``resolve_provider_routing`` only consults
# ``_mandate_fallbacks`` when an ACTIVE mandate's preferred route is unmatched;
# on a plain default-routing request it never touches them. The OLD
# ``resolve_routing_meta`` folded fallback vendors in UNCONDITIONALLY, so a
# leftover fallback list wrongly authorized (and, via the skip guard, could
# wrongly drop) default providers. The single-source helper engages fallbacks
# only inside the active-and-unmatched mandate branch.
# ---------------------------------------------------------------------------


def test_stale_mandate_fallbacks_ignored_without_active_mandate():
    """Helper-level P2b: with no active mandate, a stale ``_mandate_fallbacks``
    list must NOT contribute vendors to the authorized set — only
    ``route_priority`` does."""
    host = _RecordingHost([
        _provider("openai:plan", _OkAdapter(_RecordingHost([]), "openai:plan")),
        _provider("openai:api", _OkAdapter(_RecordingHost([]), "openai:api")),
    ])
    host.config = {"route_priority": ["openai:plan", "openai:api"]}
    host._mandate_preference = {}  # NO active mandate
    # Leftover fallbacks from a previously-cleared mandate.
    host._mandate_fallbacks = [{"vendor": "anthropic", "model": "claude-haiku-4-5"}]
    explicit, vendors = host._compute_route_authorization()
    assert vendors == {"openai"}            # only route_priority — not the stale fallback
    assert "anthropic" not in vendors        # P2b: stale fallback vendor excluded
    assert explicit is False


@pytest.mark.asyncio
async def test_stale_mandate_fallbacks_do_not_drop_default_providers():
    """End-to-end P2b: default routing with a stale ``_mandate_fallbacks`` list.
    The configured openai routes must still be tried (the stale fallback must
    neither authorize an unconfigured vendor nor cause a configured route to be
    skipped). openai:plan transiently fails -> falls through to openai:api."""
    host = _build_configured_host(
        [
            ("openai:plan", "raise", RuntimeError("openai:plan 503 transient")),
            ("openai:api", "ok", None),
        ],
        route_priority=["openai:plan", "openai:api"],
    )
    host._mandate_preference = {}  # NO active mandate
    host._mandate_fallbacks = [{"vendor": "anthropic", "model": "claude-haiku-4-5"}]
    chunks = []
    async for chunk in host.get_streaming_response(
        system_prompt="sys", user_prompt="hi",
    ):
        chunks.append(chunk)
    # Default same-vendor chain intact; stale fallback neither authorized a new
    # vendor nor dropped a configured route.
    assert host.attempted == ["openai:plan", "openai:api"], host.attempted
    assert chunks == ["ok"]
