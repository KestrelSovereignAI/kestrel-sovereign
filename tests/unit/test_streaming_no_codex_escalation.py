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

    def _check_policy(self) -> None:
        return None

    def resolve_provider_routing(
        self,
        *,
        model_override: Optional[str] = None,
        force_local_only: bool = False,
    ):
        return list(self.providers), None

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
