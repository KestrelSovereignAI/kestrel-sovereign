"""One-shot retry on codex idle-timeout under cap (#1411).

A transient codex/ChatGPT-Plus stall (the failure mode #1410 made
diagnostic and #1429 stopped escalating to a different provider) gets
one chance to recover before the error surfaces. The retry gate is
narrow on purpose:

- Exception must be ``CodexAppServerTransportError`` with the
  assistant-stage idle marker text (``"idle for"`` + ``"no completion"``).
- The exception's ``exceeds_cap`` attribute must be ``False`` — over-cap
  stalls can't be recovered by retrying.
- Zero events yielded so far — retrying after observable output would
  duplicate.
- ``tools`` must be None — tool-bearing turns are outside the safe
  replay envelope and bypass the wrapper entirely.

If the second attempt also fails, the resulting exception is augmented
with retry context and keeps the transport classification + ``exceeds_cap``
attribute so the streaming.py harness-owned check (#1429) still kicks in.
"""
import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.llm import codex_adapter as codex_adapter_module
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerConnectionClosed,
    CodexAppServerError,
    CodexAppServerTransportError,
)


@pytest.fixture(autouse=True)
def _no_real_wait(monkeypatch):
    """Replace the production 5-7s jittered wait with 0s so the retry
    tests run instantly. Tests assert the wait happens via the
    call-count seam on ``_codex_retry_wait_seconds``.
    """
    monkeypatch.setattr(
        codex_adapter_module, "_codex_retry_wait_seconds", lambda: 0.0
    )


def _stub_adapter() -> CodexAdapter:
    """Bare ``CodexAdapter`` without spawning the app-server.

    ``_session_threads`` is the session→thread cache that
    ``_run_turn_with_retry`` invalidates on retry; initialize it here
    so tests that don't explicitly set it still hit the same code path.
    """
    a = CodexAdapter.__new__(CodexAdapter)
    a._session_threads = {}
    return a


def _transport_error(
    msg: str = "codex turn idle for 300s with no completion",
    *,
    exceeds_cap: bool = False,
) -> CodexAppServerTransportError:
    e = CodexAppServerTransportError(msg)
    e.exceeds_cap = exceeds_cap
    return e


async def _collect(agen) -> List[dict]:
    out = []
    async for ev in agen:
        out.append(ev)
    return out


class _RuntimeFakeApp:
    """Small app-server double for behavioral ``_run_turn`` tests."""

    def __init__(self):
        self.requests = []
        self.closed = []

    async def ensure_started(self):
        return None

    async def request(self, method, params=None, *, timeout=120):
        self.requests.append((method, params))
        return {}

    def open_turn_sink(self, thread_id):
        return thread_id

    def close_turn_sink(self, thread_id):
        self.closed.append(thread_id)


def _runtime_adapter() -> tuple[CodexAdapter, _RuntimeFakeApp]:
    """Build the minimum real-turn harness without starting Codex."""
    adapter = _stub_adapter()
    adapter._thread_locks = {}
    adapter.contribute_system_prompt = lambda model, instructions: instructions
    adapter._effective_model_param = lambda model: None
    adapter._resolve_thread_cwd = lambda: "/tmp"
    adapter._ensure_codex_approval_bridge = lambda app: None
    app = _RuntimeFakeApp()
    adapter._app_server = lambda: app
    return adapter, app


# ---------------------------------------------------------------------------
# Retry fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_after_first_idle_timeout_under_cap():
    """Happy retry path: attempt 1 raises idle-timeout under cap with
    zero events; attempt 2 yields a real result. Caller receives
    only the attempt-2 events.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(
        model, messages, tools, session_id, tool_executor,
        cancel_token=None, keep_trailing_system=False,
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _transport_error()
        yield {"text": "hello"}
        yield {"final": ("hello", None, {"input_tokens": 5, "output_tokens": 1})}

    adapter._run_turn = fake_run_turn

    events = await _collect(adapter._run_turn_with_retry(
        "gpt-5", [], None, "sess-1", None,
    ))
    assert call_count["n"] == 2, "second attempt must fire"
    assert events == [
        {"text": "hello"},
        {"final": ("hello", None, {"input_tokens": 5, "output_tokens": 1})},
    ]


@pytest.mark.asyncio
async def test_retry_waits_via_seam_before_second_attempt(monkeypatch):
    """The configured wait fires between attempts; tests don't burn
    real wall-clock seconds because the autouse fixture replaces the
    wait with 0s.
    """
    adapter = _stub_adapter()
    waits = []

    def fake_wait():
        waits.append("called")
        return 0.0

    monkeypatch.setattr(
        codex_adapter_module, "_codex_retry_wait_seconds", fake_wait
    )
    call_count = {"n": 0}

    async def fake_run_turn(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _transport_error()
        yield {"final": ("ok", None, {})}

    adapter._run_turn = fake_run_turn
    await _collect(adapter._run_turn_with_retry("m", [], None, None, None))
    assert waits == ["called"], (
        "wait seam must be invoked exactly once before the retry"
    )


# ---------------------------------------------------------------------------
# Retry gates: do NOT retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_when_events_already_yielded():
    """A retry after even one observable event would duplicate output.
    Better to fail clean than yield "hello hello" because the second
    half stalled.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(*args, **kwargs):
        call_count["n"] += 1
        yield {"text": "hello"}
        raise _transport_error()

    adapter._run_turn = fake_run_turn

    collected: List[dict] = []
    with pytest.raises(CodexAppServerTransportError):
        async for ev in adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ):
            collected.append(ev)
    assert call_count["n"] == 1, "no second attempt after events yielded"
    assert collected == [{"text": "hello"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_factory", "expected_type"),
    [
        pytest.param(
            lambda: _transport_error(
                "codex turn idle for 300s with no completion — EXCEEDS cap",
                exceeds_cap=True,
            ),
            CodexAppServerTransportError,
            id="payload-exceeds-cap",
        ),
        pytest.param(
            lambda: CodexAppServerTransportError(
                "codex turn idle for 300s with no completion"
            ),
            CodexAppServerTransportError,
            id="cap-classification-missing",
        ),
        pytest.param(
            lambda: _transport_error(
                "turn/start timed out after 60s", exceeds_cap=False
            ),
            CodexAppServerTransportError,
            id="not-an-idle-timeout",
        ),
        pytest.param(
            lambda: CodexAppServerConnectionClosed(
                "codex app-server closed mid-turn"
            ),
            CodexAppServerConnectionClosed,
            id="connection-closed",
        ),
    ],
)
async def test_ineligible_failure_is_not_retried(error_factory, expected_type):
    """Failures outside the narrow replay envelope surface immediately."""
    adapter = _stub_adapter()
    call_count = 0

    async def fake_run_turn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise error_factory()
        yield  # pragma: no cover - preserves async-generator shape

    adapter._run_turn = fake_run_turn

    with pytest.raises(expected_type):
        await _collect(adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ))
    assert call_count == 1


@pytest.mark.asyncio
async def test_no_retry_when_tools_are_present():
    """Tool-bearing turns bypass the wrapper entirely — the safe-replay
    analysis for inline tool execution is deferred per #1411 scope.
    The wrapper must delegate straight through with no retry attempt.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}
    received_tools = []
    received_tool_executor = []

    async def fake_run_turn(
        model, messages, tools, session_id, tool_executor,
        cancel_token=None, keep_trailing_system=False,
    ):
        call_count["n"] += 1
        received_tools.append(tools)
        received_tool_executor.append(tool_executor)
        raise _transport_error()
        yield  # pragma: no cover

    adapter._run_turn = fake_run_turn

    fake_executor = MagicMock()
    with pytest.raises(CodexAppServerTransportError):
        await _collect(adapter._run_turn_with_retry(
            "m", [], [{"name": "noop"}], "sess", fake_executor,
        ))
    assert call_count["n"] == 1, "no retry on tool-bearing turn"
    # Tools + executor passed through unchanged on the single attempt.
    assert received_tools == [[{"name": "noop"}]]
    assert received_tool_executor == [fake_executor]


# ---------------------------------------------------------------------------
# Retry exhaustion: both attempts fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_invalidates_cached_thread_for_session():
    """Codex review P1: without thread invalidation, attempt 2 would
    reuse the codex thread that the first turn is still hanging on,
    risking stale-event bleed and ``turn/start`` conflicts. The
    wrapper must pop the session's cached thread before the retry so
    ``_get_or_start_thread`` calls ``thread/start`` again and gets a
    fresh thread.
    """
    adapter = _stub_adapter()
    adapter._session_threads = {"sess-X": ("thread-1", "fingerprint-A")}

    call_count = {"n": 0}
    cache_at_start_of_attempt: List[dict] = []

    async def fake_run_turn(
        model, messages, tools, session_id, tool_executor,
        cancel_token=None, keep_trailing_system=False,
    ):
        call_count["n"] += 1
        # Snapshot the cache as seen at the start of each attempt — the
        # wrapper's invariant is "cache popped between attempts."
        cache_at_start_of_attempt.append(dict(adapter._session_threads))
        if call_count["n"] == 1:
            raise _transport_error()
        yield {"final": ("ok", None, {})}

    adapter._run_turn = fake_run_turn

    await _collect(adapter._run_turn_with_retry(
        "m", [], None, "sess-X", None,
    ))
    assert call_count["n"] == 2
    # Attempt 1 saw the prior cache entry.
    assert cache_at_start_of_attempt[0] == {"sess-X": ("thread-1", "fingerprint-A")}
    # Attempt 2 saw the cleared cache — fresh thread guaranteed.
    assert cache_at_start_of_attempt[1] == {}


@pytest.mark.asyncio
async def test_queued_same_session_turn_re_resolves_after_idle_timeout():
    """A queued turn must not reuse the thread invalidated by its predecessor.

    This drives the actual lock/cache race: both calls resolve the same cached
    thread, the first idle-times out while holding its lock, and the second must
    observe the invalidation after it acquires that lock and start a fresh
    thread. It replaces source-string assertions that could not prove ordering.
    """
    adapter, _app = _runtime_adapter()
    adapter._session_threads = {"sess-X": ("thread-old", "fingerprint")}
    adapter._forget_thread_usage = lambda session_id: None
    ensure_results = []

    async def ensure_thread(app, session_id, model, instructions, tools):
        cached = adapter._session_threads.get(session_id)
        if cached:
            result = (cached[0], False)
        else:
            result = ("thread-new", True)
            adapter._session_threads[session_id] = (
                result[0], "new-fingerprint"
            )
        ensure_results.append(result[0])
        return result

    adapter._ensure_thread = ensure_thread
    old_turn_started = asyncio.Event()
    release_old_turn = asyncio.Event()
    iterated_threads = []

    async def iter_events(
        app, sink, est_payload_tokens, *, thread_id=None, cancel_token=None
    ):
        iterated_threads.append(thread_id)
        if thread_id == "thread-old":
            old_turn_started.set()
            await release_old_turn.wait()
            raise _transport_error()
        yield {
            "method": "turn/completed",
            "params": {"turn": {"status": "completed"}},
        }

    adapter._iter_with_overflow_hint = iter_events
    args = ("m", [{"role": "user", "content": "hi"}], None, "sess-X", None)
    first = asyncio.create_task(_collect(adapter._run_turn(*args)))
    await old_turn_started.wait()
    second = asyncio.create_task(_collect(adapter._run_turn(*args)))
    await asyncio.sleep(0)  # let the second call queue on thread-old's lock
    release_old_turn.set()

    with pytest.raises(CodexAppServerTransportError):
        await first
    second_events = await asyncio.wait_for(second, timeout=1)

    assert ensure_results == ["thread-old", "thread-old", "thread-new"]
    assert iterated_threads == ["thread-old", "thread-new"]
    assert second_events[-1]["final"] == (None, None, {})
    assert adapter._session_threads["sess-X"][0] == "thread-new"


@pytest.mark.asyncio
async def test_retry_fires_with_empty_tool_list():
    """Codex review round 4 P3: ``tools=[]`` is semantically equivalent
    to "no tools." Callers that normalize the absence of tools to an
    empty list must still get the one-shot retry. Gating on
    ``tools is not None`` (the v1) would deny them retry coverage.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(
        model, messages, tools, session_id, tool_executor,
        cancel_token=None, keep_trailing_system=False,
    ):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _transport_error()
        yield {"final": ("ok", None, {})}

    adapter._run_turn = fake_run_turn

    events = await _collect(adapter._run_turn_with_retry(
        "m", [], [], None, None,  # tools=[]
    ))
    assert call_count["n"] == 2, "retry must fire for empty tool list"
    assert events == [{"final": ("ok", None, {})}]


@pytest.mark.asyncio
async def test_no_retry_when_exceeds_cap_unset_for_unknown_cap():
    """Codex review round 4 P2: when the route-cap lookup returns
    ``None`` (missing ``[route_context_caps]`` entry), the hint-rewrite
    must NOT set ``exceeds_cap=False`` — leaving the attribute unset
    is what flags "we don't know," and the retry wrapper's safety
    default kicks in to skip retry.

    Pinned via the regression test on ``_iter_with_overflow_hint``:
    when ``get_route_context_cap`` returns ``None``, the raised
    exception must not carry an ``exceeds_cap`` attribute.
    """
    adapter = _stub_adapter()

    class _FakeApp:
        async def iter_turn_events(self, sink, *, thread_id=None, cancel_token=None):
            if False:
                yield  # pragma: no cover
            raise CodexAppServerTransportError(
                "codex turn idle for 300s with no completion"
            )

    from kestrel_sovereign.llm import model_catalog

    class _UnknownCapCatalog:
        def get_route_context_cap(self, route):
            return None  # cap not configured

    real = model_catalog.get_catalog_service
    model_catalog.get_catalog_service = lambda: _UnknownCapCatalog()
    try:
        with pytest.raises(CodexAppServerTransportError) as ei:
            async for _ in adapter._iter_with_overflow_hint(
                _FakeApp(), MagicMock(), est_payload_tokens=10_000,
            ):
                pass
    finally:
        model_catalog.get_catalog_service = real

    # Attribute should be absent — wrapper's getattr default of True
    # will then skip the retry.
    assert not hasattr(ei.value, "exceeds_cap"), (
        "exceeds_cap must NOT be set when cap is unknown; otherwise the "
        "retry wrapper would silently retry on an over-cap stall"
    )


@pytest.mark.asyncio
async def test_empty_session_id_completes_without_thread_re_resolution():
    """An empty session id is sessionless and must not enter the cache loop."""
    adapter, _app = _runtime_adapter()
    ensure_calls = 0

    async def ensure_thread(app, session_id, model, instructions, tools):
        nonlocal ensure_calls
        ensure_calls += 1
        return "thread-sessionless", True

    adapter._ensure_thread = ensure_thread

    async def iter_events(
        app, sink, est_payload_tokens, *, thread_id=None, cancel_token=None
    ):
        yield {
            "method": "turn/completed",
            "params": {"turn": {"status": "completed"}},
        }

    adapter._iter_with_overflow_hint = iter_events
    events = await asyncio.wait_for(
        _collect(adapter._run_turn(
            "m", [{"role": "user", "content": "hi"}], None, "", None
        )),
        timeout=1,
    )

    assert ensure_calls == 1
    assert events[-1]["final"] == (None, None, {})


@pytest.mark.asyncio
async def test_retry_does_not_touch_cache_for_sessionless_calls():
    """``session_id=None`` means there's no cache to manage —
    ``_get_or_start_thread`` already always starts a fresh thread.
    The wrapper must not blindly call pop on a None key.
    """
    adapter = _stub_adapter()
    adapter._session_threads = {"other-session": ("thread-other", "fp")}

    call_count = {"n": 0}

    async def fake_run_turn(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _transport_error()
        yield {"final": ("ok", None, {})}

    adapter._run_turn = fake_run_turn

    await _collect(adapter._run_turn_with_retry(
        "m", [], None, None, None,
    ))
    assert call_count["n"] == 2
    # Other sessions' threads untouched.
    assert adapter._session_threads == {"other-session": ("thread-other", "fp")}


@pytest.mark.asyncio
async def test_retry_exhaustion_preserves_transport_and_exceeds_cap():
    """When both attempts hit idle-timeout under cap, the surfaced
    exception:
      * Is still ``CodexAppServerTransportError`` (so streaming.py's
        harness-owned check #1429 still recognizes it and prevents
        wrong-provider escalation).
      * Carries the original ``exceeds_cap`` attribute.
      * Has a message annotated with the retry context so operators
        can see the retry happened.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(*args, **kwargs):
        call_count["n"] += 1
        raise _transport_error()
        yield  # pragma: no cover

    adapter._run_turn = fake_run_turn

    with pytest.raises(CodexAppServerTransportError) as ei:
        await _collect(adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ))
    assert call_count["n"] == 2
    final = ei.value
    assert isinstance(final, CodexAppServerTransportError)
    assert getattr(final, "exceeds_cap", None) is False
    assert "retried" in str(final).lower()
    assert "second attempt" in str(final).lower()


# ---------------------------------------------------------------------------
# Entry-point routing: text-only public APIs retry; tool streaming does not.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["get_response", "get_streaming_response"])
async def test_text_only_public_entry_point_retries_idle_timeout(entry_point):
    """The public text APIs expose the wrapper's observable retry behavior."""
    adapter = _stub_adapter()
    call_count = 0

    async def fake_run_turn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _transport_error()
        yield {"text": "ok"}
        yield {"final": ("ok", None, {})}

    adapter._run_turn = fake_run_turn
    method = getattr(adapter, entry_point)
    call = method(
        client=None,
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    if entry_point == "get_response":
        response = await call
        assert response.content == "ok"
    else:
        assert await _collect(call) == ["ok"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_tool_streaming_public_entry_point_does_not_retry():
    """A tool-bearing public stream surfaces its first transport failure."""
    adapter = _stub_adapter()
    call_count = 0

    async def fake_run_turn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _transport_error()
        yield  # pragma: no cover - preserves async-generator shape

    adapter._run_turn = fake_run_turn
    with pytest.raises(CodexAppServerTransportError):
        await _collect(adapter.get_streaming_response_with_tools(
            client=None,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "noop"}}],
            tool_executor=MagicMock(),
        ))
    assert call_count == 1


# ---------------------------------------------------------------------------
# Wait-seconds seam
# ---------------------------------------------------------------------------


def test_codex_retry_wait_constants_within_documented_band():
    """The production wait must stay in the documented 5-7s band so a
    silent regression to "wait 0s" (which would hammer the app-server)
    or "wait 5 minutes" (which would visibly hang the agent) shows up
    in CI rather than in production.

    We pin the constants rather than calling the function — the
    autouse fixture replaces the function so other tests run instantly,
    and re-importing past it would require fragile module-reload
    gymnastics. The function body is ``base + uniform(0, jitter)``;
    pinning ``base`` and ``jitter`` is equivalent.
    """
    from kestrel_sovereign.llm.codex_adapter import (
        _CODEX_RETRY_BASE_SECONDS,
        _CODEX_RETRY_JITTER_SECONDS,
    )
    assert _CODEX_RETRY_BASE_SECONDS == 5.0
    assert _CODEX_RETRY_JITTER_SECONDS == 2.0
    # Documented band: [base, base + jitter] = [5.0, 7.0]
    assert 5.0 <= _CODEX_RETRY_BASE_SECONDS <= 6.0
    assert 1.0 <= _CODEX_RETRY_JITTER_SECONDS <= 3.0


# ---------------------------------------------------------------------------
# exceeds_cap attribute is set on the rewrite path (#1410 + #1411 contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overflow_hint_rewrite_sets_exceeds_cap_attribute():
    """The ``_iter_with_overflow_hint`` rewrite path must attach an
    ``exceeds_cap`` attribute to the raised ``CodexAppServerTransportError``
    so ``_run_turn_with_retry`` can read it without re-parsing the
    hint string.
    """
    adapter = _stub_adapter()

    class _FakeApp:
        def __init__(self):
            self._raised = False

        async def iter_turn_events(self, sink, *, thread_id=None, cancel_token=None):
            if False:
                yield  # pragma: no cover (make this a generator)
            raise CodexAppServerTransportError(
                "codex turn idle for 300s with no completion"
            )

    fake_app = _FakeApp()
    fake_sink: Any = MagicMock()

    # Stub the route-cap lookup so the hint-rewrite branch runs
    # deterministically without touching real catalog config.
    from kestrel_sovereign.llm import model_catalog

    class _FakeCatalog:
        def get_route_context_cap(self, route):
            return 100_000  # well above the est payload below

    real = model_catalog.get_catalog_service
    model_catalog.get_catalog_service = lambda: _FakeCatalog()
    try:
        with pytest.raises(CodexAppServerTransportError) as ei:
            async for _ in adapter._iter_with_overflow_hint(
                fake_app, fake_sink, est_payload_tokens=10_000,
            ):
                pass
    finally:
        model_catalog.get_catalog_service = real

    # est_payload (10k) < cap (100k) → exceeds_cap=False
    assert getattr(ei.value, "exceeds_cap", None) is False
    # Sanity check the hint text shape (operator-facing).
    assert "within the per-turn cap" in str(ei.value)
