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

    async def fake_run_turn(model, messages, tools, session_id, tool_executor):
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
async def test_retry_waits_via_seam_before_second_attempt():
    """The configured wait fires between attempts; tests don't burn
    real wall-clock seconds because the autouse fixture replaces the
    wait with 0s.
    """
    adapter = _stub_adapter()
    waits = []

    def fake_wait():
        waits.append("called")
        return 0.0

    # Override the module fixture for this test to count invocations.
    import kestrel_sovereign.llm.codex_adapter as m
    real = m._codex_retry_wait_seconds
    m._codex_retry_wait_seconds = fake_wait
    try:
        call_count = {"n": 0}

        async def fake_run_turn(*args):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _transport_error()
            yield {"final": ("ok", None, {})}

        adapter._run_turn = fake_run_turn
        await _collect(adapter._run_turn_with_retry("m", [], None, None, None))
        assert waits == ["called"], (
            "wait seam must be invoked exactly once before the retry"
        )
    finally:
        m._codex_retry_wait_seconds = real


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

    async def fake_run_turn(*args):
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
async def test_no_retry_when_exceeds_cap():
    """Over-cap stalls are caused by ChatGPT-Plus's per-turn payload
    limit; retrying without compaction can't help. Surface the error
    immediately so the operator sees the cap-exceeded hint.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(*args):
        call_count["n"] += 1
        raise _transport_error(
            "codex turn idle for 300s with no completion — EXCEEDS cap",
            exceeds_cap=True,
        )
        yield  # pragma: no cover (make this an async generator, not a coroutine)

    adapter._run_turn = fake_run_turn

    with pytest.raises(CodexAppServerTransportError):
        await _collect(adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ))
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_no_retry_when_exceeds_cap_attribute_missing():
    """Safety default: an exception without an ``exceeds_cap`` attribute
    skips retry. The attribute is only set by ``_iter_with_overflow_hint``
    when it determined cap-vs-payload; absence means "we don't know,"
    and retrying an unknown stall is worse than surfacing it.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(*args):
        call_count["n"] += 1
        # Construct directly without going through the hint rewrite,
        # so no exceeds_cap attribute is set.
        raise CodexAppServerTransportError(
            "codex turn idle for 300s with no completion"
        )
        yield  # pragma: no cover

    adapter._run_turn = fake_run_turn

    with pytest.raises(CodexAppServerTransportError):
        await _collect(adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ))
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_no_retry_when_error_is_not_idle_timeout():
    """A different transport error — RPC timeout from ``_request_unguarded``,
    say — is not an idle stall and won't be helped by waiting 5s. Pass
    through unchanged.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(*args):
        call_count["n"] += 1
        # Has exceeds_cap=False but the message doesn't carry the
        # idle-timeout markers — different transport failure class.
        raise _transport_error(
            "turn/start timed out after 60s", exceeds_cap=False,
        )
        yield  # pragma: no cover

    adapter._run_turn = fake_run_turn

    with pytest.raises(CodexAppServerTransportError):
        await _collect(adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ))
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_no_retry_on_connection_closed():
    """Connection-closed is the app-server process going away. Retrying
    doesn't bring it back; we'd just get the same closure again.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(*args):
        call_count["n"] += 1
        raise CodexAppServerConnectionClosed("codex app-server closed mid-turn")
        yield  # pragma: no cover

    adapter._run_turn = fake_run_turn

    with pytest.raises(CodexAppServerConnectionClosed):
        await _collect(adapter._run_turn_with_retry(
            "m", [], None, None, None,
        ))
    assert call_count["n"] == 1


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

    async def fake_run_turn(model, messages, tools, session_id, tool_executor):
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

    async def fake_run_turn(model, messages, tools, session_id, tool_executor):
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


def test_run_turn_rechecks_session_cache_after_acquiring_thread_lock():
    """Codex review round 3: a same-session call queued on
    ``_thread_locks[thread_id]`` already has the now-poisoned
    ``thread_id`` in a local variable. Popping ``_session_threads``
    from another concurrent failure doesn't help that queued call —
    it would still call ``turn/start`` on the hung thread when it
    unblocks. The fix: re-check ``_session_threads`` after acquiring
    the thread lock; if the cache no longer points at our
    ``thread_id``, re-resolve via ``_ensure_thread`` to get a fresh
    thread.

    Pinned source-level (behavioral test would require choreographing
    two concurrent ``_run_turn`` invocations through real asyncio with
    a poisoned-thread injection — feasible but heavy for what's
    effectively a "did the author add the re-check" assertion).
    """
    from kestrel_sovereign.llm import codex_adapter as m
    import inspect

    src = inspect.getsource(m.CodexAdapter._run_turn)
    # The re-check must come after the initial lock acquire.
    initial_acquire_idx = src.find("await lock.acquire()")
    assert initial_acquire_idx > 0
    # The re-check loop pattern.
    recheck_idx = src.find("self._session_threads.get(session_id)")
    assert recheck_idx > initial_acquire_idx, (
        "session-cache re-check must appear AFTER the initial thread "
        "lock acquire so the cache state at lock-acquisition-time is "
        "what gates the re-resolve."
    )
    # The re-resolve path must release the stale lock and re-call
    # ``_ensure_thread`` rather than just looping with the same lock
    # and thread_id (which would re-resolve to itself if the cache
    # was racy-popped after our lookup).
    assert "lock.release()" in src[recheck_idx:src.find("\n        try:")], (
        "the re-check arm must release the stale lock before "
        "re-resolving via _ensure_thread"
    )
    assert "_ensure_thread" in src[recheck_idx:src.find("\n        try:")]


def test_run_turn_invalidates_session_cache_before_lock_release():
    """Codex review round 2: the cache invalidation must happen BEFORE
    ``_run_turn``'s ``finally`` releases the per-thread lock, otherwise
    a same-session call queued on ``_session_locks`` can unblock and
    grab the still-hung thread before the retry wrapper pops the
    cache. Pin the source-level intent: ``_run_turn`` itself must
    catch ``CodexAppServerTransportError`` and pop the cache for
    idle-timeout cases.

    Source-level rather than behavioral because reproducing the
    inter-task timing window in a unit test requires substantial
    asyncio orchestration; the source pin catches the regression
    that matters (someone refactors ``_run_turn`` and accidentally
    drops the in-method invalidation, leaving only the wrapper's
    post-lock-release pop).
    """
    from kestrel_sovereign.llm import codex_adapter as m
    import inspect

    src = inspect.getsource(m.CodexAdapter._run_turn)
    assert "except CodexAppServerTransportError" in src, (
        "_run_turn must catch the transport error inside its try-block "
        "so the cache invalidation runs BEFORE the finally releases "
        "the thread lock."
    )
    assert "_session_threads.pop(session_id" in src, (
        "_run_turn's transport-error arm must pop _session_threads so "
        "concurrent same-session callers don't reuse the hung thread."
    )
    # The order check: the except arm must appear BEFORE the finally
    # in source order. (Python's runtime order is also except-then-
    # finally regardless, but the source order is what readers reason
    # about.)
    except_idx = src.find("except CodexAppServerTransportError")
    finally_idx = src.find("\n        finally:")
    assert 0 <= except_idx < finally_idx, (
        "except arm must precede finally in _run_turn's source"
    )


@pytest.mark.asyncio
async def test_retry_fires_with_empty_tool_list():
    """Codex review round 4 P3: ``tools=[]`` is semantically equivalent
    to "no tools." Callers that normalize the absence of tools to an
    empty list must still get the one-shot retry. Gating on
    ``tools is not None`` (the v1) would deny them retry coverage.
    """
    adapter = _stub_adapter()
    call_count = {"n": 0}

    async def fake_run_turn(model, messages, tools, session_id, tool_executor):
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


def test_run_turn_uses_truthy_session_id_guard_for_recheck_loop():
    """Codex review round 4 P2: the post-acquire re-check loop must
    use ``while session_id:`` (truthy), not ``while session_id is not None:``.
    ``_ensure_thread`` treats empty-string session_ids as sessionless
    via its own truthy check, so the cache is never written for them.
    With ``is not None``, the re-check loop would spin forever opening
    fresh threads because ``cached`` stays ``None`` indefinitely.
    """
    from kestrel_sovereign.llm import codex_adapter as m
    import inspect

    src = inspect.getsource(m.CodexAdapter._run_turn)
    assert "while session_id:" in src, (
        "_run_turn's re-check loop must gate on truthy session_id to "
        "match _ensure_thread's semantics and avoid infinite-loop on ''"
    )
    assert "while session_id is not None:" not in src, (
        "stale guard would spin forever on empty-string session ids"
    )


@pytest.mark.asyncio
async def test_retry_does_not_touch_cache_for_sessionless_calls():
    """``session_id=None`` means there's no cache to manage —
    ``_get_or_start_thread`` already always starts a fresh thread.
    The wrapper must not blindly call pop on a None key.
    """
    adapter = _stub_adapter()
    adapter._session_threads = {"other-session": ("thread-other", "fp")}

    call_count = {"n": 0}

    async def fake_run_turn(*args):
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

    async def fake_run_turn(*args):
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
# Entry-point routing: get_response + get_streaming_response use the
# wrapper; get_streaming_response_with_tools does NOT.
# ---------------------------------------------------------------------------


def test_get_response_calls_run_turn_with_retry():
    """Source-level guarantee — both code paths must funnel through the
    retry wrapper rather than calling ``_run_turn`` directly.
    """
    from kestrel_sovereign.llm import codex_adapter as m
    import inspect

    src = inspect.getsource(m.CodexAdapter.get_response)
    assert "_run_turn_with_retry" in src
    assert "self._run_turn(" not in src


def test_get_streaming_response_calls_run_turn_with_retry():
    from kestrel_sovereign.llm import codex_adapter as m
    import inspect

    src = inspect.getsource(m.CodexAdapter.get_streaming_response)
    assert "_run_turn_with_retry" in src
    assert "self._run_turn(" not in src


def test_get_streaming_response_with_tools_does_not_use_retry():
    """The tool-bearing streaming path must NOT use the retry wrapper.
    Inline tool execution + replay is the deferred-design case the
    ticket explicitly carves out. If a future change accidentally
    points this entry point at the wrapper, the wrapper would still
    delegate straight through (it has a ``tools is not None`` guard),
    but pinning the source-level intent is cheap insurance.
    """
    from kestrel_sovereign.llm import codex_adapter as m
    import inspect

    src = inspect.getsource(m.CodexAdapter.get_streaming_response_with_tools)
    assert "_run_turn_with_retry" not in src
    assert "self._run_turn(" in src


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
