"""Owned local SDK/SQLite/PG fixtures; not Agent READY or paid-provider acceptance."""

import asyncio
import json
import secrets
from dataclasses import dataclass

import openai
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.execution_custody import (
    is_execution_control_error,
)
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
from kestrel_sovereign.storage.memory_system import MemorySystem
from kestrel_sovereign.storage.saved_items_store import SavedItemsStore
from kestrel_sovereign.storage.sqla import make_session_factory
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_storage_backend_parity import _incepted_assertion_identity


@dataclass
class NativeMemoryOwners:
    backend: object
    observer: object
    schema: object
    pg_storage: object
    pg_memory: object
    pg_saved: object
    sqlite_consolidator: object
    sqlite_memory: object
    sqlite_saved: object
    sqlite_factory: object
    service: object
    requests: object
    catalog_requests: object
    lease: object
    did: object
    retirement_withdrawals: list


@pytest.fixture
async def native_memory_owners(native_pg, tmp_path, monkeypatch):
    from kestrel_sovereign.llm import service as service_module

    backend, observer, schema = native_pg
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(service_module, "load_section", lambda name: {})
    did, identity = await _incepted_assertion_identity(tmp_path, "embedding-native")
    storage = AsyncStorage(
        backend=backend,
        agent_id=did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
            did, identity
        ),
    )
    await storage.initialize()
    service = LLMService(usage_db=storage.db, agent_data_dir=tmp_path / "llm")
    server = client = local_db = None
    handlers = set()
    retirement_withdrawals = []
    try:
        requests, catalog_requests, handler_failures, handlers = [], [], [], set()

        async def serve(reader, writer):
            owner = asyncio.current_task()
            handlers.add(owner)
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                if header.startswith(b"GET /v1/models "):
                    catalog_requests.append("/v1/models")
                    payload = json.dumps(
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": model,
                                    "object": "model",
                                    "created": 1,
                                    "owned_by": "local-fixture",
                                }
                                for model in ("local-test-embedding", "gpt-4o")
                            ],
                        }
                    ).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\n\r\n"
                        + payload
                    )
                    await writer.drain()
                    return
                size = next(
                    int(line.split(b":", 1)[1])
                    for line in header.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                body = json.loads(await reader.readexactly(size))
                requests.append(body)
                if header.startswith(b"POST /v1/chat/completions "):
                    if not body.get("stream"):
                        payload = json.dumps(
                            {
                                "id": "local-native-chat",
                                "object": "chat.completion",
                                "created": 1,
                                "model": body["model"],
                                "choices": [
                                    {
                                        "index": 0,
                                        "finish_reason": "stop",
                                        "message": {
                                            "role": "assistant",
                                            "content": "local summary",
                                        },
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 3,
                                    "completion_tokens": 1,
                                    "total_tokens": 4,
                                },
                            }
                        ).encode()
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                            + str(len(payload)).encode()
                            + b"\r\n\r\n"
                            + payload
                        )
                        await writer.drain()
                        return
                    assert body["stream"] is True
                    events = [
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "local"},
                                    "finish_reason": None,
                                }
                            ]
                        },
                        {
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ]
                        },
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 1,
                                "total_tokens": 4,
                            },
                        },
                    ]
                    payload = (
                        b"".join(
                            b"data: " + json.dumps(event).encode() + b"\n\n"
                            for event in events
                        )
                        + b"data: [DONE]\n\n"
                    )
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: "
                        + str(len(payload)).encode()
                        + b"\r\n\r\n"
                        + payload
                    )
                    await writer.drain()
                    return
                assert header.startswith(b"POST /v1/embeddings ")
                payload = json.dumps(
                    {
                        "object": "list",
                        "model": "local-test-embedding",
                        "data": [
                            {
                                "object": "embedding",
                                "index": index,
                                "embedding": [1.0, 0.0, 0.0],
                            }
                            for index, _ in enumerate(
                                body["input"]
                                if isinstance(body["input"], list)
                                else [body["input"]]
                            )
                        ],
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                    }
                ).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                    + str(len(payload)).encode()
                    + b"\r\n\r\n"
                    + payload
                )
                await writer.drain()
            except Exception as error:
                # Async server callbacks otherwise merely log their failure,
                # allowing a defensive discovery fallback to hide a broken
                # local fixture. The fixture owner must fail its own teardown.
                handler_failures.append(error)
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(owner)

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        client = openai.AsyncOpenAI(
            api_key="local-inert-not-a-provider-key",
            base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1",
            max_retries=0,
            timeout=5,
        )
        provider = {
            "name": "openai:local-fixture",
            "vendor": "openai",
            "route": "local-fixture",
            "client": client,
            "adapter": OpenAIAdapter(
                embedding_model="local-test-embedding", embedding_dim=3
            ),
            "is_local": True,
            "is_cloud": False,
            "capabilities": {
                "supports_embeddings": True,
                "embedding_model": "local-test-embedding",
                "embedding_dim": 3,
            },
        }
        service.providers = [provider]
        service._embedding_route = provider["name"]
        assert service.get_embedding_service() is not None
        storage.conversation._llm_service = service
        memory = MemorySystem(storage, did)
        await memory.initialize()
        local_db = await AsyncDatabase.sqlite(str(tmp_path / "episodes.db"))
        saved = SavedItemsStore(storage.db, did, llm_service=service)
        local_storage = AsyncStorage(backend=local_db.backend, agent_id=did)
        await local_storage.initialize()
        # AsyncStorage owns the canonical initialized adapter. Reuse that
        # exact object, including its SQLAlchemy cache/retirement inventory.
        local_db = local_storage.db
        consolidator = MemoryConsolidator(local_db, did, llm_service=service)
        local_storage.conversation._llm_service = service
        local_memory = MemorySystem(local_storage, did)
        await local_memory.initialize()
        local_memory.consolidator._llm_service = service
        local_saved = SavedItemsStore(local_db, did, llm_service=service)
        factory = make_session_factory(local_db)
        # Real original physical PostgreSQL lease, normally unlocked/retired.
        async with backend.advisory_locks([(3569, secrets.randbelow(2**30))]) as lease:
            lease.require_live()
        assert lease._state == "retired"
        yield NativeMemoryOwners(
            backend,
            observer,
            schema,
            storage,
            memory,
            saved,
            consolidator,
            local_memory,
            local_saved,
            factory,
            service,
            requests,
            catalog_requests,
            lease,
            did,
            retirement_withdrawals,
        )
    finally:
        backend._execution_custody = service._execution_custody = None
        unexpected = []

        async def retire(operation, *, retained_control=False):
            try:
                await operation()
            except BaseException as error:
                # A lifecycle observer's actual cancellation can remain on a
                # completed service's receipt. Tests explicitly register ONLY
                # the SAME observed object; do not clear it or accept arbitrary
                # cancellations/failed physical retirement as cleanup success.
                expected_withdrawal = (
                    retained_control and isinstance(error, asyncio.CancelledError)
                    and any(error is original for original in retirement_withdrawals)
                )
                if not (expected_withdrawal or retained_control and is_execution_control_error(error)):
                    unexpected.append(error)

        await retire(service.close, retained_control=True)
        await retire(service.wait_for_close_completion, retained_control=True)
        if client is not None:
            await retire(client.close)
        if local_db is not None:
            await retire(local_db.close)
        await retire(storage.close)
        if server is not None:
            server.close()
            await retire(server.wait_closed)
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        unexpected.extend(handler_failures)
        # Failed native release ACKs deliberately retain debt/control even
        # after their exact Task has finished. Do not falsify that receipt by
        # clearing the reaper or demanding an empty semantic debt inventory.
        # The fixture independently retires its actual borrowed DB pool.
        closing = vars(service).get("_llm_close_tasks", {})
        if any(task is not None and not task.done() for task in closing.values()):
            unexpected.append(
                AssertionError("fixture left an original native close Task running")
            )
        if unexpected:
            raise BaseExceptionGroup("native memory fixture retirement", unexpected)


@pytest.fixture
def native_vector_consumers(native_memory_owners):
    owner = native_memory_owners
    return (
        owner.sqlite_memory,
        owner.sqlite_saved,
        owner.sqlite_factory,
        owner.service,
        owner.requests,
        owner.lease,
        owner.did,
    )


def actual_native_loss(lease, observed):
    try:
        lease.require_live()
    except BaseException as error:
        observed.append(error)
        raise
