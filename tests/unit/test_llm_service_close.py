"""``LLMService.close()`` reports every handle it could not close (#3559).

A close that raised or timed out has not released its handle. These drive the
real ``LLMService.close()`` over provider routes whose adapters and clients a
test controls, and check that such a close raises, keeps the handle, and that
a later close retries it.
"""

import asyncio

import pytest

from kestrel_sovereign.llm import service as service_module
from kestrel_sovereign.llm.service import LLMService, LLMServiceCloseError


class _Closeable:
    """An adapter or client whose close a test can make fail or hang."""

    def __init__(self):
        self.open = True
        self.closes = 0
        self.failures: list[BaseException] = []
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def _close(self):
        self.closes += 1
        self.entered.set()
        if self.failures:
            raise self.failures.pop(0)
        if self.gate is not None:
            await self.gate.wait()
        self.open = False


class _Adapter(_Closeable):
    async def aclose(self):
        await self._close()


class _Client(_Closeable):
    async def close(self):
        await self._close()


def _route(name, *, adapter=None, client=None) -> dict:
    return {"name": name, "adapter": adapter, "client": client, "model": "m"}


@pytest.fixture
async def service(tmp_path):
    svc = LLMService(agent_data_dir=tmp_path)
    configured = svc.providers
    svc.providers = []
    yield svc
    svc.providers = configured
    for _label, handle, _close in svc._handle_closer.unconfirmed():
        if isinstance(handle, _Closeable):
            handle.failures.clear()
            if handle.gate is not None:
                handle.gate.set()
    await svc.close()


def _unconfirmed(svc: LLMService) -> list[str]:
    return [label for label, _handle, _close in svc._handle_closer.unconfirmed()]


@pytest.mark.asyncio
async def test_every_handle_is_closed_when_one_fails_and_each_failure_is_named(
    service,
):
    failing_adapter, failing_client = _Adapter(), _Client()
    failing_adapter.failures.append(OSError("adapter close failed"))
    failing_client.failures.append(ConnectionError("client close failed"))
    healthy_adapter, healthy_client = _Adapter(), _Client()
    service.providers = [
        _route("a:api", adapter=failing_adapter, client=healthy_client),
        _route("b:api", adapter=healthy_adapter, client=failing_client),
    ]

    with pytest.raises(LLMServiceCloseError) as raised:
        await service.close()

    assert [label for label, _ in raised.value.failures] == [
        "a:api adapter",
        "b:api client",
    ]
    assert isinstance(raised.value.__cause__, OSError)
    assert not healthy_adapter.open and not healthy_client.open
    assert failing_adapter.open and failing_client.open
    assert _unconfirmed(service) == ["a:api adapter", "b:api client"]

    await service.close()

    assert not failing_adapter.open and not failing_client.open
    assert _unconfirmed(service) == []


@pytest.mark.asyncio
async def test_a_timed_out_adapter_close_is_a_failure_a_retry_waits_for(
    service, monkeypatch
):
    monkeypatch.setattr(service_module, "CLIENT_CLOSE_TIMEOUT", 0.05)
    adapter = _Adapter()
    adapter.gate = asyncio.Event()
    service.providers = [_route("slow:api", adapter=adapter)]

    with pytest.raises(LLMServiceCloseError, match="slow:api adapter") as raised:
        await service.close()

    assert isinstance(raised.value.__cause__, TimeoutError)
    assert adapter.open
    assert _unconfirmed(service) == ["slow:api adapter"]

    adapter.gate.set()
    await service.close()

    assert adapter.closes == 1, "the retry waited for the close still running"
    assert not adapter.open
    assert _unconfirmed(service) == []


@pytest.mark.asyncio
async def test_a_caller_deadline_is_not_reported_as_a_completed_close(service):
    """A close cancelled by its caller's deadline stops; it does not return.

    The old close swallowed the cancellation and carried on, so a deadline
    such as the agent's shutdown budget could not tell its close was cut
    short.
    """
    adapter, later_client = _Adapter(), _Client()
    adapter.gate = asyncio.Event()
    service.providers = [
        _route("slow:api", adapter=adapter),
        _route("later:api", client=later_client),
    ]

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(service.close(), timeout=0.05)

    assert adapter.open
    assert later_client.open, "the sweep stopped at the cancellation"
    assert _unconfirmed(service) == ["slow:api adapter"]

    adapter.gate.set()
    await service.close()

    assert adapter.closes == 1
    assert not adapter.open and not later_client.open


@pytest.mark.asyncio
async def test_a_handle_shared_by_two_routes_is_closed_once_per_close(service):
    adapter = _Adapter()
    service.providers = [
        _route("a:api", adapter=adapter),
        _route("a:plan", adapter=adapter),
    ]

    await service.close()

    assert adapter.closes == 1


@pytest.mark.asyncio
async def test_a_synchronous_client_close_is_accepted(service):
    closed = []

    class _SyncClient:
        def close(self):
            closed.append(True)

    service.providers = [_route("sync:api", client=_SyncClient())]

    await service.close()

    assert closed == [True]


@pytest.mark.asyncio
async def test_the_close_error_names_handles_without_their_error_text(service):
    client = _Client()
    client.failures.append(ConnectionError("https://user:route-secret@host"))
    service.providers = [_route("leaky:api", client=client)]

    with pytest.raises(LLMServiceCloseError) as raised:
        await service.close()

    assert str(raised.value) == (
        "LLMService could not close leaky:api client (ConnectionError)"
    )


@pytest.mark.asyncio
async def test_a_failed_usage_database_close_keeps_it_for_a_retry(service):
    await service._ensure_db_initialized()
    db = service._usage_db
    assert db is not None and db.backend.is_connected

    real_close = db.close
    calls = []

    async def failing_close():
        calls.append("close")
        raise OSError("usage database close failed")

    db.close = failing_close
    try:
        with pytest.raises(LLMServiceCloseError, match="usage database"):
            await service.close()
    finally:
        db.close = real_close

    assert calls == ["close"]
    assert service._usage_db is db, "kept for a retry"
    assert db.backend.is_connected

    await service.close()

    assert service._usage_db is None
    assert not db.backend.is_connected


# ---------------------------------------------------------------------------
# A google-genai client holds two transports
#
# ``Client.close()`` closes only its synchronous HTTP client; the Google and
# Vertex adapters call through ``client.aio``, whose HTTP client only
# ``client.aio.aclose()`` closes. These use the real SDK client, offline.
# ---------------------------------------------------------------------------


def _genai_client():
    from google import genai

    return genai.Client(api_key="test-key-never-sent")


def _transports(client):
    """The synchronous and asynchronous HTTP clients of a google-genai client."""
    api = client._api_client
    sync, async_ = api._httpx_client, api._async_httpx_client
    assert not sync.is_closed and not async_.is_closed
    return sync, async_


def _fail_next_close(monkeypatch, transport, method: str, error: Exception) -> None:
    """Make ``transport``'s next ``method`` raise ``error`` before closing anything."""
    real = getattr(transport, method)
    failures = [error]

    if method == "aclose":

        async def close():
            if failures:
                raise failures.pop(0)
            await real()

    else:

        def close():
            if failures:
                raise failures.pop(0)
            real()

    monkeypatch.setattr(transport, method, close)


@pytest.mark.asyncio
async def test_both_transports_of_a_google_client_are_closed(service):
    from kestrel_sovereign.llm.google_adapter import GoogleAdapter

    client = _genai_client()
    sync, async_ = _transports(client)
    service.providers = [_route("google:api", adapter=GoogleAdapter(), client=client)]

    await service.close()

    assert sync.is_closed
    assert async_.is_closed, "the transport the adapter calls through"


@pytest.mark.asyncio
async def test_a_failed_google_async_transport_close_is_kept_for_a_retry(
    service, monkeypatch
):
    from kestrel_sovereign.llm.google_adapter import GoogleAdapter

    client = _genai_client()
    sync, async_ = _transports(client)
    _fail_next_close(monkeypatch, async_, "aclose", OSError("async close failed"))
    service.providers = [_route("google:api", adapter=GoogleAdapter(), client=client)]

    with pytest.raises(LLMServiceCloseError) as raised:
        await service.close()

    assert [label for label, _ in raised.value.failures] == [
        "google:api client async transport"
    ]
    assert sync.is_closed
    assert not async_.is_closed
    assert _unconfirmed(service) == ["google:api client async transport"]

    await service.close()

    assert async_.is_closed
    assert _unconfirmed(service) == []


@pytest.mark.asyncio
async def test_the_client_a_vertex_adapter_built_is_closed_and_kept_until_it_is(
    service, monkeypatch
):
    """Model discovery makes the Vertex adapter build a client of its own."""
    from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-never-sent")
    adapter = VertexAIAdapter(project_id="test-project")
    owned = adapter._get_client()
    sync, async_ = _transports(owned)
    _fail_next_close(monkeypatch, sync, "close", OSError("sync close failed"))
    service.providers = [_route("vertex:api", adapter=adapter)]

    with pytest.raises(LLMServiceCloseError, match="vertex:api adapter") as raised:
        await service.close()

    assert isinstance(raised.value.__cause__, OSError)
    assert async_.is_closed, "the other transport is closed regardless"
    assert not sync.is_closed
    assert adapter._client is owned, "kept for a retry"
    assert _unconfirmed(service) == ["vertex:api adapter"]

    await service.close()

    assert sync.is_closed
    assert adapter._client is None
    assert _unconfirmed(service) == []
