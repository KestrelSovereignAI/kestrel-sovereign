"""#4114 (#4097, PR 1 of 3): the provider's rate-limit response headers are
recorded on every Anthropic stream open.

Ruling rv-4097-1 as amended by rv-4097-1a. One helper copies a response's
headers at both points in ``_anthropic_stream_with_retry``'s ``_enter``: an
opened stream, and a failed open's exception response. A header's value is
kept only if its name matches the rate-limit pattern and the value has a kept
shape; otherwise a matched value leaves a descriptor and every other header
leaves its name only.

* An opened stream's matched headers ride on the response to the service and
  land on the ``llm.usage`` line and in the durable row's metadata.
* A failed open's matched headers are one ``llm.attempt_headers:`` line, and
  the provider's error is re-raised as the same object.
* The set of response header names is one names-only line per route per
  process, written again only when it changes.

Every header name and value here is made up. None names a real provider
rate-limit header (P8), apart from ``retry-after``, which the ruling names.
"""

from __future__ import annotations

import gc
import json
import logging
import weakref
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import AsyncMock

import anthropic
import httpx2
import pytest

from kestrel_sovereign.a2a.stores import ObservabilityStore
from kestrel_sovereign.llm import anthropic_adapter, rate_limit_headers, retry
from kestrel_sovereign.llm.anthropic_adapter import (
    AnthropicAdapter,
    ResponseHeaderSlot,
    _anthropic_stream_with_retry,
)
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
from kestrel_sovereign.llm.anthropic_oauth import _ENV_CREDENTIALS_FILE, _ENV_DELEGATE
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.rate_limit_headers import (
    MAX_MATCHED_HEADERS,
    RATE_LIMIT_HEADERS_KEY,
    HeaderNameLedger,
    has_kept_shape,
    is_rate_limit_header,
    read_response_headers,
    response_rate_limit_headers,
    withheld_value_descriptor,
)
from kestrel_sovereign.llm.retry import AdvisedWaitExceedsRetryBudget
from kestrel_sovereign.storage.db import SQLiteBackend
from tests.shared.environment_guard import unset_environment
from tests.utils.anthropic_client import anthropic_model_record

PLAN = "anthropic:plan"
API = "anthropic:api"
MODEL = "claude-opus-5"
USER = [{"role": "user", "content": "Summarise the four design questions."}]
CONTEXT = LLMInvocationContext(companion_id="companion-1", user_id="user-1")

# Made-up header names. The first three match the pattern; none is a real
# provider header.
COUNT = ("x-example-ratelimit-units-left", "12")
RESET = ("x-example-rate-limit-window-end", "2026-11-20T08:00:00Z")
COOLDOWN = ("x-example-rate_limit-cooldown", "1h30m")
RETRY_NOW = ("retry-after", "0")
RETRY_LATER = ("retry-after", "999999")
# Matched names whose values are not a kept shape.
IDENTIFIER_LIKE = ("x-example-ratelimit-bucket", "bkt_7f3a9c21e5d4")
WORD = ("x-example-rate-limit-tier", "quokkaflare")
# Headers that are not matched: their values are never kept.
TRACE = ("x-example-trace-tag", "zebra-unmatched-value-7")
SET_COOKIE = ("set-cookie", "session=made-up-cookie-value-1")
REQUEST_ID = ("request-id", "req_made_up_000111")
ORGANIZATION = ("x-example-organization-id", "org-made-up-222333")
UNMATCHED = (TRACE, SET_COOKIE, REQUEST_ID, ORGANIZATION)

#: The error text a refusal carries; it must never reach an attempt line.
REFUSAL = "made-up refusal text that names no header"

CAPTURE_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def names_ledger(monkeypatch) -> HeaderNameLedger:
    """A fresh per-process ledger, so each test sees its own first lines."""
    ledger = HeaderNameLedger()
    monkeypatch.setattr(anthropic_adapter, "RESPONSE_HEADER_NAMES", ledger)
    return ledger


@pytest.fixture
def capture_clock(monkeypatch) -> List[str]:
    """A capture clock that advances a minute on every reading."""
    readings: List[str] = []

    def advancing() -> str:
        reading = (CAPTURE_TIME + timedelta(minutes=len(readings))).isoformat()
        readings.append(reading)
        return reading

    monkeypatch.setattr(rate_limit_headers, "observation_time", advancing)
    return readings


@pytest.fixture
def no_jitter(monkeypatch) -> None:
    """A retried throttle advised to come back now waits no jitter either."""
    monkeypatch.setattr(retry, "random", SimpleNamespace(uniform=lambda low, high: 0.0))


def _lines(caplog, prefix: str) -> List[Dict[str, Any]]:
    return [
        json.loads(record.getMessage().removeprefix(prefix))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def _attempt_lines(caplog) -> List[Dict[str, Any]]:
    return _lines(caplog, "llm.attempt_headers: ")


def _names_lines(caplog) -> List[Dict[str, Any]]:
    return _lines(caplog, "llm.response_header_names: ")


def _usage_lines(caplog) -> List[Dict[str, Any]]:
    return _lines(caplog, "llm.usage: ")


def _all_log_text(caplog) -> str:
    return "\n".join(
        record.getMessage() + (record.exc_text or "") for record in caplog.records
    )


# ---------------------------------------------------------------------------
# The filter (P2, P2a, P3, P4, P10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, matched", [
    (COUNT[0], True),
    (RESET[0], True),
    (COOLDOWN[0], True),
    ("X-Example-RATELIMIT-Units-Left", True),
    ("retry-after", True),
    ("Retry-After-Ms", True),
    ("x-retry-after", False),
    ("retry-after-seconds", False),
    ("x-example-rate-limiter", True),
    ("x-example-rate-limits", True),
    (TRACE[0], False),
    (ORGANIZATION[0], False),
    ("set-cookie", False),
    ("request-id", False),
    ("x-example-rate.limit", False),
])
def test_names_are_matched_without_case_on_the_ruled_pattern(name, matched):
    assert is_rate_limit_header(name) is matched


@pytest.mark.parametrize("value", [
    "0", "12", "-3", "+7", "1.5", ".5", "7.", "0012",
    "2026-11-20T08:00:00Z", "2026-11-20t08:00:00z", "2026-11-20T08:00:00.123456+05:30",
    "2026-12-31T23:59:60-01:00",
    "250ms", "3s", "15m", "2h", "1h30m", "1.5s", "1h0m0s",
    "1" * 64,
])
def test_short_numbers_dates_and_durations_are_kept(value):
    assert has_kept_shape(value)


@pytest.mark.parametrize("value", [
    "", "1" * 65, "1,000", "1e3", "12 ", " 12", "--1", "1.2.3",
    "2026-13-01T00:00:00Z", "2026-02-30T00:00:00Z", "2026-11-20T08:00:00",
    "2026-11-20 08:00:00Z", "2026-11-20T08:00:00+24:00",
    "Fri, 20 Nov 2026 08:00:00 GMT",
    "5d", "5 s", "s", "ms", "1h 30m", "5MS",
    IDENTIFIER_LIKE[1], WORD[1], "١٢",
])
def test_every_other_value_is_withheld(value):
    assert not has_kept_shape(value)


def test_a_withheld_value_leaves_only_its_shape():
    assert withheld_value_descriptor("Ab_9 é-x") == {
        "withheld": True,
        "length": 8,
        "classes": ["digit", "lower", "upper", "space", "non-ascii"],
        "punctuation": "-_",
    }
    assert withheld_value_descriptor(IDENTIFIER_LIKE[1]) == {
        "withheld": True, "length": 16, "classes": ["digit", "lower"],
        "punctuation": "_",
    }
    assert withheld_value_descriptor(WORD[1]) == {
        "withheld": True, "length": 11, "classes": ["lower"], "punctuation": "",
    }


def _response(*headers: Tuple[str, str], status: int = 200, request=None) -> httpx2.Response:
    return httpx2.Response(status, headers=list(headers), request=request)


def test_matched_values_are_kept_and_every_other_value_is_not(capture_clock):
    captured = read_response_headers(_response(
        COUNT, *UNMATCHED, RESET, COOLDOWN, IDENTIFIER_LIKE, WORD,
    ))
    record = captured.record()
    assert record == {
        "observed_at": CAPTURE_TIME.isoformat(),
        "headers": [
            list(COUNT), list(RESET), list(COOLDOWN),
            [IDENTIFIER_LIKE[0], withheld_value_descriptor(IDENTIFIER_LIKE[1])],
            [WORD[0], withheld_value_descriptor(WORD[1])],
        ],
    }
    assert captured.names == tuple(sorted({
        COUNT[0], RESET[0], COOLDOWN[0], IDENTIFIER_LIKE[0], WORD[0],
        *(name for name, _ in UNMATCHED),
    }))
    text = json.dumps(record)
    for _, value in (*UNMATCHED, IDENTIFIER_LIKE, WORD):
        assert value not in text


def test_a_response_with_no_matched_header_records_nothing():
    captured = read_response_headers(_response(*UNMATCHED))
    assert captured.record() is None
    assert captured.names == tuple(sorted(name for name, _ in UNMATCHED))


def test_no_response_and_no_headers_record_nothing():
    assert read_response_headers(None) is None
    assert read_response_headers(SimpleNamespace()) is None


def test_request_headers_are_never_read():
    request = httpx2.Request(
        "POST", "https://example.invalid/v1/messages",
        headers={"x-example-ratelimit-request-side": "31337"},
    )
    captured = read_response_headers(_response(COUNT, request=request))
    assert "x-example-ratelimit-request-side" not in captured.names
    assert "31337" not in json.dumps(captured.record())


def test_values_are_strings_as_received_and_repeats_are_kept():
    captured = read_response_headers(_response(COUNT, ("X-Example-Ratelimit-Units-Left", "+011")))
    assert captured.record()["headers"] == [list(COUNT), [COUNT[0], "+011"]]


def test_at_most_32_matched_headers_are_kept_and_the_rest_counted():
    headers = [(f"x-example-ratelimit-slot-{index:02d}", str(index)) for index in range(40)]
    record = read_response_headers(_response(*headers)).record()
    assert len(record["headers"]) == MAX_MATCHED_HEADERS == 32
    assert record["headers"][-1] == ["x-example-ratelimit-slot-31", "31"]
    assert record["omitted_headers"] == 8


def test_the_copy_holds_no_reference_to_the_response():
    response = _response(COUNT, *UNMATCHED)
    reference = weakref.ref(response)
    captured = read_response_headers(response)
    del response
    gc.collect()
    assert reference() is None

    def plain(value: Any) -> bool:
        if isinstance(value, (str, int, bool)):
            return True
        if isinstance(value, (list, tuple)):
            return all(plain(item) for item in value)
        if isinstance(value, dict):
            return all(isinstance(key, str) and plain(item) for key, item in value.items())
        return False

    assert plain(captured.names) and plain(captured.matched)
    assert plain(captured.observed_at)


def test_observed_at_is_utc_iso_8601_with_its_offset():
    before = datetime.now(timezone.utc)
    observed = datetime.fromisoformat(read_response_headers(_response(COUNT)).observed_at)
    assert observed.utcoffset() == timedelta(0)
    assert before <= observed <= datetime.now(timezone.utc)


def test_the_ledger_reports_a_routes_names_when_new_or_changed():
    ledger = HeaderNameLedger()
    assert ledger.changed(PLAN, ("a", "b"))
    assert not ledger.changed(PLAN, ("a", "b"))
    assert ledger.changed(API, ("a", "b"))
    assert ledger.changed(PLAN, ("a", "b", "c"))
    assert ledger.changed(PLAN, ("a", "b"))
    assert not ledger.changed(PLAN, ())


# ---------------------------------------------------------------------------
# The seam: _enter in _anthropic_stream_with_retry (P1, P5, P7)
# ---------------------------------------------------------------------------


class _FailingOpen:
    """A stream context whose open raises one exact error object."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aenter__(self):
        raise self.error

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _OpenedStream:
    """A stream context whose open succeeds with ``headers`` on its response."""

    def __init__(self, *headers: Tuple[str, str]) -> None:
        self.response = SimpleNamespace(headers=httpx2.Headers(list(headers)))
        self.exited = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        self.exited += 1
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _scripted_client(*opens: Any) -> SimpleNamespace:
    queue = list(opens)
    return SimpleNamespace(messages=SimpleNamespace(stream=lambda **_: queue.pop(0)))


def _refusal(error_class, status: int, *headers: Tuple[str, str]) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")
    return error_class(
        REFUSAL,
        response=httpx2.Response(status, headers=list(headers), request=request),
        body={"type": "error", "error": {"type": "made_up_error", "message": REFUSAL}},
    )


@pytest.fixture
def enter_boundary(monkeypatch) -> List[BaseException]:
    """Every error ``_enter`` raised, as ``with_retry`` (an open that retries)
    or ``await_execution_work`` (``retry_open=False``) received it."""
    from kestrel_sovereign import execution_custody

    raised: List[BaseException] = []

    def observing(operation):
        async def observed():
            try:
                return await operation()
            except BaseException as error:
                raised.append(error)
                raise
        return observed

    real_with_retry = anthropic_adapter.with_retry
    real_await = execution_custody.await_execution_work

    async def with_retry(func, *args, **kwargs):
        return await real_with_retry(observing(func), *args, **kwargs)

    async def await_execution_work(owner, operation, **kwargs):
        return await real_await(owner, observing(operation), **kwargs)

    monkeypatch.setattr(anthropic_adapter, "with_retry", with_retry)
    monkeypatch.setattr(execution_custody, "await_execution_work", await_execution_work)
    return raised


async def _open(client, *, retry_open: bool = True, route: Optional[str] = PLAN,
                slot: Optional[ResponseHeaderSlot] = None) -> None:
    async with _anthropic_stream_with_retry(
        client, {"model": MODEL, "max_tokens": 1, "messages": USER},
        route=route, retry_open=retry_open, response_headers=slot,
    ) as stream:
        async for _ in stream:
            pass


def _assert_one_attempt_line(caplog, status: int, *matched: Tuple[str, str]) -> None:
    lines = _attempt_lines(caplog)
    assert len(lines) == 1, lines
    assert lines[0]["route"] == PLAN
    assert lines[0]["status"] == status
    assert lines[0]["headers"] == [list(header) for header in matched]
    datetime.fromisoformat(lines[0]["observed_at"])
    text = _all_log_text(caplog)
    assert REFUSAL not in text.split("llm.attempt_headers: ", 1)[1].split("\n", 1)[0]
    for _, value in UNMATCHED:
        assert value not in text


@pytest.mark.asyncio
async def test_a_retried_throttle_records_its_headers_once_and_the_open_proceeds(
    caplog, enter_boundary, no_jitter,
):
    caplog.set_level(logging.INFO)
    throttle = _refusal(anthropic.RateLimitError, 429, COUNT, RETRY_NOW, *UNMATCHED)
    opened = _OpenedStream(RESET)
    slot = ResponseHeaderSlot()

    await _open(_scripted_client(_FailingOpen(throttle), opened), slot=slot)

    assert enter_boundary == [throttle]
    assert enter_boundary[0] is throttle
    _assert_one_attempt_line(caplog, 429, COUNT, RETRY_NOW)
    assert slot.captured.record()["headers"] == [list(RESET)]
    assert opened.exited == 1


@pytest.mark.asyncio
async def test_a_declined_throttle_records_its_headers_and_keeps_the_same_error(
    caplog, enter_boundary,
):
    caplog.set_level(logging.INFO)
    throttle = _refusal(anthropic.RateLimitError, 429, COUNT, RETRY_LATER, *UNMATCHED)

    with pytest.raises(AdvisedWaitExceedsRetryBudget) as declined:
        await _open(_scripted_client(_FailingOpen(throttle)))

    assert declined.value.__cause__ is throttle
    assert enter_boundary == [throttle]
    assert enter_boundary[0] is throttle
    _assert_one_attempt_line(caplog, 429, COUNT, RETRY_LATER)


@pytest.mark.asyncio
async def test_a_non_retryable_refusal_records_its_headers_and_raises_the_same_error(
    caplog, enter_boundary,
):
    caplog.set_level(logging.INFO)
    refusal = _refusal(anthropic.PermissionDeniedError, 403, COUNT, *UNMATCHED)

    with pytest.raises(anthropic.PermissionDeniedError) as raised:
        await _open(_scripted_client(_FailingOpen(refusal)))

    assert raised.value is refusal
    assert enter_boundary == [refusal]
    _assert_one_attempt_line(caplog, 403, COUNT)


@pytest.mark.asyncio
async def test_a_failed_open_without_retry_records_its_headers_and_raises_the_same_error(
    caplog, enter_boundary,
):
    caplog.set_level(logging.INFO)
    throttle = _refusal(anthropic.RateLimitError, 429, COUNT, RETRY_NOW, *UNMATCHED)

    with pytest.raises(anthropic.RateLimitError) as raised:
        await _open(_scripted_client(_FailingOpen(throttle)), retry_open=False)

    assert raised.value is throttle
    assert enter_boundary == [throttle]
    assert enter_boundary[0] is throttle
    _assert_one_attempt_line(caplog, 429, COUNT, RETRY_NOW)


@pytest.mark.asyncio
async def test_a_failed_open_with_no_response_writes_no_attempt_line(caplog, enter_boundary):
    caplog.set_level(logging.INFO)
    failure = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://example.invalid/v1/messages"),
    )

    with pytest.raises(anthropic.APIConnectionError) as raised:
        await _open(_scripted_client(_FailingOpen(failure)), retry_open=False)

    assert raised.value is failure
    assert enter_boundary == [failure]
    assert _attempt_lines(caplog) == []
    assert _names_lines(caplog) == []


@pytest.mark.asyncio
async def test_a_failed_open_with_no_matched_header_writes_no_attempt_line(
    caplog, enter_boundary,
):
    caplog.set_level(logging.INFO)
    refusal = _refusal(anthropic.PermissionDeniedError, 403, *UNMATCHED)

    with pytest.raises(anthropic.PermissionDeniedError) as raised:
        await _open(_scripted_client(_FailingOpen(refusal)))

    assert raised.value is refusal
    assert enter_boundary == [refusal]
    assert _attempt_lines(caplog) == []
    assert _names_lines(caplog) == [
        {"route": PLAN, "names": sorted(name for name, _ in UNMATCHED)},
    ]


@pytest.mark.asyncio
async def test_a_matched_value_of_another_shape_leaves_only_its_descriptor_on_the_attempt_line(
    caplog,
):
    caplog.set_level(logging.INFO)
    refusal = _refusal(anthropic.PermissionDeniedError, 403, IDENTIFIER_LIKE, WORD)

    with pytest.raises(anthropic.PermissionDeniedError):
        await _open(_scripted_client(_FailingOpen(refusal)))

    [line] = _attempt_lines(caplog)
    assert line["headers"] == [
        [IDENTIFIER_LIKE[0], withheld_value_descriptor(IDENTIFIER_LIKE[1])],
        [WORD[0], withheld_value_descriptor(WORD[1])],
    ]
    text = _all_log_text(caplog)
    assert IDENTIFIER_LIKE[1] not in text
    assert WORD[1] not in text


def _broken_helper(*args: Any, **kwargs: Any):
    raise RuntimeError("header capture broke")


@pytest.mark.asyncio
async def test_a_broken_helper_never_replaces_the_providers_error(
    monkeypatch, caplog, enter_boundary,
):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(anthropic_adapter, "_capture_response_headers", _broken_helper)
    refusal = _refusal(anthropic.PermissionDeniedError, 403, COUNT)

    with pytest.raises(anthropic.PermissionDeniedError) as raised:
        await _open(_scripted_client(_FailingOpen(refusal)))

    assert raised.value is refusal
    assert enter_boundary == [refusal]
    assert _attempt_lines(caplog) == []


@pytest.mark.asyncio
async def test_a_broken_helper_never_fails_an_open_stream(monkeypatch, caplog, enter_boundary):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(anthropic_adapter, "_capture_response_headers", _broken_helper)
    opened = _OpenedStream(COUNT)
    slot = ResponseHeaderSlot()

    await _open(_scripted_client(opened), slot=slot)

    assert enter_boundary == []
    assert slot.captured is None
    assert opened.exited == 1


@pytest.mark.asyncio
async def test_the_names_line_is_written_per_route_when_new_and_when_changed(caplog):
    caplog.set_level(logging.INFO)
    first = (COUNT, TRACE)

    await _open(_scripted_client(_OpenedStream(*first)))
    await _open(_scripted_client(_OpenedStream(*first)))
    await _open(_scripted_client(_OpenedStream(*first)), route=API)
    await _open(_scripted_client(_OpenedStream(*first, SET_COOKIE)))

    assert _names_lines(caplog) == [
        {"route": PLAN, "names": sorted([COUNT[0], TRACE[0]])},
        {"route": API, "names": sorted([COUNT[0], TRACE[0]])},
        {"route": PLAN, "names": sorted([COUNT[0], TRACE[0], SET_COOKIE[0]])},
    ]


# ---------------------------------------------------------------------------
# Through the real SDK, the adapter and the service (P5, P6, P9)
# ---------------------------------------------------------------------------


def _sse(events: Sequence[Dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


ANSWER = "Question 1: yes"


def _wire_events() -> List[Dict[str, Any]]:
    return [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": MODEL, "content": [], "stop_reason": None,
            "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 1},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": ANSWER}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]


def _sdk_client(*replies: Tuple[int, Sequence[Tuple[str, str]]]):
    """A real ``anthropic.AsyncAnthropic`` whose HTTP is served locally: the
    Models API record for :data:`MODEL`, then one scripted reply per
    message request, each with its status and response headers."""
    script = list(replies)
    posts: List[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET" and request.url.path == f"/v1/models/{MODEL}":
            return httpx2.Response(200, json=anthropic_model_record(
                MODEL, max_tokens=128_000,
            ).model_dump(mode="json"))
        assert request.method == "POST" and request.url.path == "/v1/messages"
        posts.append(request)
        status, headers = script.pop(0)
        if status == 200:
            return httpx2.Response(
                200, content=_sse(_wire_events()),
                headers=[("content-type", "text/event-stream"), *headers],
            )
        return httpx2.Response(status, headers=list(headers), json={
            "type": "error", "error": {"type": "made_up_error", "message": REFUSAL},
        })

    client = anthropic.AsyncAnthropic(
        api_key="made-up-test-key",
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return client, posts


@pytest.fixture
async def store(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "observability.db"))
    await backend.connect()
    observability = ObservabilityStore(backend)
    await observability.initialize()
    try:
        yield observability
    finally:
        await observability.close()


@pytest.fixture
def service(monkeypatch, store):
    from kestrel_sovereign.llm.service import LLMService

    llm = LLMService()
    monkeypatch.setattr(llm, "_ensure_models_discovered", AsyncMock())
    monkeypatch.setattr(llm, "_mandate_preference", {})
    monkeypatch.setattr(llm, "_record_model_usage", AsyncMock())
    llm._observability_store = store

    def install(adapter: Any, client: Any, route: str = PLAN) -> None:
        vendor, route_name = route.split(":")
        monkeypatch.setattr(llm, "config", {"route_priority": [route]})
        monkeypatch.setattr(llm, "providers", [{
            "name": route, "vendor": vendor, "route": route_name,
            "client": client, "adapter": adapter, "model": MODEL,
            "is_cloud": True, "is_local": False, "base_url": None,
            "selection_hints": [],
        }])

    llm.install_route = install
    return llm


#: The service's two ways into the adapter: a complete response
#: (``get_response`` over a stream) and the tool-detecting stream.
ENTRIES = ("generate_with_messages", "stream_with_tool_detection")


async def _call(llm, entry: str) -> str:
    if entry == "generate_with_messages":
        response = await llm.generate_with_messages(
            messages=USER, invocation_context=CONTEXT,
        )
        return response if isinstance(response, str) else response.content
    text = []
    async for item in llm.stream_with_tool_detection(
        messages=USER, invocation_context=CONTEXT,
    ):
        if isinstance(item, str):
            text.append(item)
    return "".join(text)


async def _only_row(store):
    calls = await store.query_llm_calls()
    assert len(calls) == 1, [(call.provider, call.success) for call in calls]
    return calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("adapter_class, route", [
    (ClaudeMaxAdapter, PLAN), (AnthropicAdapter, API),
])
async def test_a_matched_header_is_on_the_usage_line_and_in_the_row(
    service, store, caplog, capture_clock, entry, adapter_class, route,
):
    caplog.set_level(logging.INFO)
    client, _ = _sdk_client((200, [COUNT, RESET, COOLDOWN]))
    service.install_route(adapter_class(route=route), client, route)

    assert await _call(service, entry) == ANSWER

    expected = {
        "observed_at": capture_clock[0],
        "headers": [list(COUNT), list(RESET), list(COOLDOWN)],
    }
    [usage] = _usage_lines(caplog)
    assert usage["provider"] == route
    assert usage[RATE_LIMIT_HEADERS_KEY] == expected
    row = await _only_row(store)
    assert row.success is True
    assert row.metadata[RATE_LIMIT_HEADERS_KEY] == expected
    names = [line for line in _names_lines(caplog) if line["route"] == route]
    assert len(names) == 1
    assert {COUNT[0], RESET[0], COOLDOWN[0]} <= set(names[0]["names"])


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_unmatched_values_reach_no_log_record_and_no_row_and_their_names_are_listed(
    service, store, caplog, entry,
):
    caplog.set_level(logging.INFO)
    client, _ = _sdk_client((200, [COUNT, *UNMATCHED]))
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)

    await _call(service, entry)

    [usage] = _usage_lines(caplog)
    assert usage[RATE_LIMIT_HEADERS_KEY]["headers"] == [list(COUNT)]
    row = await _only_row(store)
    row_text = row.model_dump_json()
    log_text = _all_log_text(caplog)
    for _, value in UNMATCHED:
        assert value not in log_text
        assert value not in row_text
    [names] = [line for line in _names_lines(caplog) if line["route"] == PLAN]
    for name, _ in UNMATCHED:
        assert name in names["names"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_no_matched_header_means_no_key_and_an_unaffected_call(
    service, store, caplog, entry,
):
    caplog.set_level(logging.INFO)
    client, _ = _sdk_client((200, list(UNMATCHED)))
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)

    assert await _call(service, entry) == ANSWER

    [usage] = _usage_lines(caplog)
    assert RATE_LIMIT_HEADERS_KEY not in usage
    assert usage["success"] is True
    row = await _only_row(store)
    assert row.success is True
    assert RATE_LIMIT_HEADERS_KEY not in row.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_broken_helper_leaves_the_call_its_normal_result(
    service, store, caplog, monkeypatch, entry,
):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(anthropic_adapter, "_capture_response_headers", _broken_helper)
    client, _ = _sdk_client((200, [COUNT]))
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)

    assert await _call(service, entry) == ANSWER

    [usage] = _usage_lines(caplog)
    assert usage["success"] is True
    assert RATE_LIMIT_HEADERS_KEY not in usage
    assert (await _only_row(store)).success is True


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_matched_values_of_another_shape_leave_only_their_descriptors(
    service, store, caplog, entry,
):
    caplog.set_level(logging.INFO)
    client, _ = _sdk_client((200, [IDENTIFIER_LIKE, WORD]))
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)

    await _call(service, entry)

    descriptors = [
        [IDENTIFIER_LIKE[0], withheld_value_descriptor(IDENTIFIER_LIKE[1])],
        [WORD[0], withheld_value_descriptor(WORD[1])],
    ]
    [usage] = _usage_lines(caplog)
    assert usage[RATE_LIMIT_HEADERS_KEY]["headers"] == descriptors
    row = await _only_row(store)
    assert row.metadata[RATE_LIMIT_HEADERS_KEY]["headers"] == descriptors
    for value in (IDENTIFIER_LIKE[1], WORD[1]):
        assert value not in _all_log_text(caplog)
        assert value not in row.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_observed_at_is_when_the_headers_were_copied_not_when_the_record_was_written(
    service, store, caplog, capture_clock, entry,
):
    caplog.set_level(logging.INFO)
    client, _ = _sdk_client((200, [COUNT]))
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)
    await _call(service, entry)
    # The clock moves on before anything else asks it.
    later = rate_limit_headers.observation_time()

    [usage] = _usage_lines(caplog)
    assert capture_clock[0] == CAPTURE_TIME.isoformat() < later
    assert usage[RATE_LIMIT_HEADERS_KEY]["observed_at"] == capture_clock[0]
    row = await _only_row(store)
    assert row.metadata[RATE_LIMIT_HEADERS_KEY]["observed_at"] == capture_clock[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ENTRIES)
async def test_a_call_retried_once_writes_one_usage_line_and_one_attempt_line(
    service, store, caplog, no_jitter, entry,
):
    caplog.set_level(logging.INFO)
    client, posts = _sdk_client(
        (429, [COUNT, RETRY_NOW, *UNMATCHED]),
        (200, [RESET, *UNMATCHED]),
    )
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)

    assert await _call(service, entry) == ANSWER

    assert len(posts) == 2
    [attempt] = _attempt_lines(caplog)
    assert attempt["route"] == PLAN
    assert attempt["status"] == 429
    assert attempt["headers"] == [list(COUNT), list(RETRY_NOW)]
    assert set(attempt) == {"route", "observed_at", "status", "headers"}
    [usage] = _usage_lines(caplog)
    assert usage[RATE_LIMIT_HEADERS_KEY]["headers"] == [list(RESET)]
    row = await _only_row(store)
    assert row.metadata[RATE_LIMIT_HEADERS_KEY]["headers"] == [list(RESET)]
    log_text = _all_log_text(caplog)
    for _, value in UNMATCHED:
        assert value not in log_text
        assert value not in row.model_dump_json()


@pytest.mark.asyncio
async def test_a_declined_throttle_through_the_real_sdk_is_on_the_attempt_line_only(
    service, store, caplog,
):
    caplog.set_level(logging.INFO)
    client, _ = _sdk_client((429, [COUNT, RETRY_LATER, *UNMATCHED]))
    service.install_route(ClaudeMaxAdapter(route=PLAN), client)

    with pytest.raises(Exception):
        await _call(service, "stream_with_tool_detection")

    [attempt] = _attempt_lines(caplog)
    assert attempt["headers"] == [list(COUNT), list(RETRY_LATER)]
    assert REFUSAL not in json.dumps(attempt)
    row = await _only_row(store)
    assert row.success is False
    assert RATE_LIMIT_HEADERS_KEY not in row.metadata
    for _, value in UNMATCHED:
        assert value not in _all_log_text(caplog)
        assert value not in row.model_dump_json()


# ---------------------------------------------------------------------------
# The route label comes from the registry's configuration (P6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_class, route_name", [
    (ClaudeMaxAdapter, "plan"), (AnthropicAdapter, "api"),
])
def test_the_registry_names_each_anthropic_adapters_route(
    monkeypatch, adapter_class, route_name,
):
    # Only the inline made-up secret below may be used: no credentials file,
    # no delegated credential store.
    unset_environment(monkeypatch, _ENV_CREDENTIALS_FILE)
    monkeypatch.setenv(_ENV_DELEGATE, "0")
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.config = {}
    route_cfg = (
        {"auth_token": "made-up-oauth-token"} if adapter_class is ClaudeMaxAdapter
        else {"api_key": "made-up-api-key"}
    )
    _, adapter = registry._build_client_and_adapter(
        vendor="anthropic", route=route_name, adapter_cls=adapter_class,
        vendor_cfg={}, route_cfg=route_cfg,
    )
    assert adapter._route == f"anthropic:{route_name}"


def test_an_adapter_built_outside_the_registry_has_no_route():
    assert AnthropicAdapter()._route is None
    assert ClaudeMaxAdapter()._route is None


def test_response_rate_limit_headers_ignores_anything_but_a_copy():
    assert response_rate_limit_headers(SimpleNamespace()) is None
    assert response_rate_limit_headers(
        SimpleNamespace(_kestrel_rate_limit_headers={"headers": []})
    ) is None
