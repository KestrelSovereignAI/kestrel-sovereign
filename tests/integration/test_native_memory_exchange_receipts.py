"""Actual inline memory enrichment must target its original exchange rows.

This exercises a booted agent and native storage/tagging, not inbound MCP,
provider-native installation or later reflection/consolidation acceptance.
"""
import asyncio
from contextvars import ContextVar
from types import SimpleNamespace
from uuid import uuid4

import asyncpg
import openai
import pytest

from kestrel_sovereign.agent.memory_manager import MemoryManager
from kestrel_sovereign.bootstrap import BootstrapState
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    _explicit_control_chain,
    bind_execution_runtime,
    is_execution_control_error,
)
from kestrel_sovereign.features.privacy.feature import PrivacyAgent
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.associative_linker import LinkedConcept
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.db.interface import DatabaseError
from kestrel_sovereign.storage.turn_provenance import opened_by_person
from kestrel_sovereign.testing.native_shutdown_budget import shutdown_fixture_agent
from tests.integration.native_memory_original_control import (
    native_memory_owners as _native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_native_assistant_write_receipts import (
    assistant_agent as _assistant_agent,
)

native_memory_owners = _native_memory_owners
native_pg = _native_pg
assistant_agent = _assistant_agent


@pytest.mark.asyncio
async def test_memory_enrichment_keeps_original_exchange_identity(assistant_agent):
    agent = assistant_agent
    session = str(uuid4())
    user, assistant = "I am proud and happy today", "That is wonderful news"
    original_marker, later_marker = "owned-original-unprocessed", "owned-later-unprocessed"
    user_id = await agent.privacy_agent.add_conversation("user", user, session_id=session,
        metadata={"emotional_tag_version": original_marker})
    assistant_id = await agent._persist_assistant_conversation(assistant, session_id=session,
        metadata={"emotional_tag_version": original_marker})
    later_user = await agent.privacy_agent.add_conversation("user", user, session_id=session,
        metadata={"emotional_tag_version": later_marker})
    later_assistant = await agent._persist_assistant_conversation(assistant, session_id=session,
        metadata={"emotional_tag_version": later_marker})
    ids = [user_id, assistant_id, later_user, later_assistant]
    assert len(set(ids)) == 4 and all(type(row) is int and row > 0 for row in ids)
    previous_tasks = set(agent._background_tasks)
    await agent._post_response_pipeline(user_message_id=user_id, assistant_message_id=assistant_id)
    tasks = tuple(set(agent._background_tasks) - previous_tasks)
    if tasks:
        await asyncio.gather(*tasks)
    rows = {row["id"]: row for row in await agent._raw_storage.conversation.get_messages_by_ids(ids)}
    assert all(rows[row]["metadata"]["emotional_tag_version"] != original_marker
        for row in [user_id, assistant_id]), "Memory enrichment selected a later matching pair instead of its original write receipts"
    assert all(rows[row]["metadata"]["emotional_tag_version"] == later_marker
        for row in [later_user, later_assistant]), "Memory enrichment changed another exchange's rows"


@pytest.fixture
async def native_pipeline_owner(native_memory_owners, monkeypatch):
    n = native_memory_owners
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    user_id = await n.pg_storage.add_conversation("user", "Happy original user")
    assistant_id = await n.pg_storage.add_conversation("assistant", "Original reply")
    manager = MemoryManager(storage=n.pg_storage, agent_id=n.did)
    # The real pipeline is driven with its native dependencies. No service boot,
    # model-generation or full-host acceptance is claimed by this component case.
    owner = KestrelAgent.__new__(KestrelAgent)
    owner.did = n.did
    owner._raw_storage = n.pg_storage
    owner.memory_system = n.pg_memory
    owner.context_manager = SimpleNamespace(memory_manager=manager)
    owner.privacy_agent = PrivacyAgent(n.pg_storage, PrivacyMode.NORMAL)
    owner._background_tasks = set()
    return SimpleNamespace(n=n, owner=owner, manager=manager, user_id=user_id, assistant_id=assistant_id)


def observe_original_pg_release(monkeypatch, target, name, *, fail_reset=True):
    """Forward a selected real operation and observe its original native lease."""
    active = ContextVar("owned-native-memory-release", default=False)
    failed_connections, sources = [], []
    error = RuntimeError("Original native memory lease RESET acknowledgement refused")
    operation = getattr(target, name)
    reset = asyncpg.connection.Connection.reset

    async def original_operation(*args, **kwargs):
        token = active.set(True)
        try:
            return await operation(*args, **kwargs)
        except BaseException as original:
            sources.append(original)
            raise
        finally:
            active.reset(token)

    async def original_reset(connection, *args, **kwargs):
        if fail_reset and active.get() and not failed_connections:
            failed_connections.append(connection)
            raise error
        return await reset(connection, *args, **kwargs)

    monkeypatch.setattr(target, name, original_operation)
    monkeypatch.setattr(asyncpg.connection.Connection, "reset", original_reset)
    return SimpleNamespace(error=error, sources=sources, failed_connections=failed_connections)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["memory-system", "memory-manager", "post-pipeline"])
@pytest.mark.parametrize("kind", ["native-reset", "ordinary-constraint"])
async def test_memory_tagging_retains_the_original_native_metadata_lease_control(
    native_pipeline_owner, monkeypatch, door, kind,
):
    """Native component ownership; not Agent READY or a provider UI install."""
    n, owner, manager = native_pipeline_owner.n, native_pipeline_owner.owner, native_pipeline_owner.manager
    user_id, assistant_id = native_pipeline_owner.user_id, native_pipeline_owner.assistant_id
    constraint_name = "owned_metadata_tag_refusal"
    if kind == "ordinary-constraint":
        await n.pg_storage.db.execute(
            f"ALTER TABLE conversation_history ADD CONSTRAINT {constraint_name} "
            "CHECK (metadata::jsonb -> 'emotional_tag_version' IS NULL)"
        )
    fault = observe_original_pg_release(monkeypatch, n.pg_storage.conversation,
        "update_message_metadata", fail_reset=kind == "native-reset")
    observed, failed_connections, original = fault.sources, fault.failed_connections, fault.error
    background_admissions = []
    track = owner._track_background_task

    def original_background_admission(coroutine, **kwargs):
        background_admissions.append(coroutine)
        return track(coroutine, **kwargs)

    monkeypatch.setattr(owner, "_track_background_task", original_background_admission)
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(n.did))
    caught = None
    with bind_execution_runtime(owner):
        try:
            if door == "memory-system":
                await n.pg_memory.tag_message(user_id, "Happy original user")
            elif door == "memory-manager":
                await manager.tag_exchange("Happy original user", "Original reply", user_id, assistant_id, n.pg_memory)
            else:
                await owner._post_response_pipeline(user_message_id=user_id, assistant_message_id=assistant_id)
        except (DatabaseError, ExecutionAuthorityError, BaseExceptionGroup, asyncio.CancelledError) as error:
            caught = error
        finally:
            tasks = tuple(owner._background_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in tasks:
                    owner._finish_background_task(task)
    if kind == "ordinary-constraint":
        assert observed and caught is None and not failed_connections
        assert all(not is_execution_control_error(error) for error in observed)
        assert all(any(getattr(source, "constraint_name", None) == constraint_name
            for source in _explicit_control_chain(error)) for error in observed), (
            "Ordinary metadata failures must come from the actual PostgreSQL constraint"
        )
        return
    assert not background_admissions, "Post-response memory attempted background admission after its native control"
    assert observed and is_execution_control_error(observed[0]) and original in tuple(_explicit_control_chain(observed[0])), (
        "The original metadata lease producer did not supply classified native execution control"
    )
    assert len(failed_connections) == 1 and failed_connections[0].is_closed(), (
        "The original metadata operation did not reach and retire its failed physical PostgreSQL lease"
    )
    assert caught is not None and is_execution_control_error(caught) and original in tuple(_explicit_control_chain(caught)), (
        f"Original metadata lease control was discarded at {door}: {caught!r}"
    )


def observe_original_conversation_receipts(agent, monkeypatch, marker):
    """Observe actual native write results without replacing persistence."""
    write = agent.privacy_agent.add_conversation
    receipts = []

    async def original_write(*args, **kwargs):
        row_id = await write(*args, **kwargs)
        assert type(row_id) is int and row_id > 0, "The original conversation writer supplied no native receipt"
        receipts.append((args[0], row_id))
        await agent._raw_storage.conversation.update_message_metadata(row_id, {"emotional_tag_version": marker})
        return row_id

    monkeypatch.setattr(agent.privacy_agent, "add_conversation", original_write)
    return receipts


@pytest.mark.asyncio
async def test_streaming_persistence_returns_the_original_native_task_receipt(assistant_agent):
    agent = assistant_agent
    row_id = await agent._persist_assistant_turn_safely("Owned streamed response", require_success=True)
    assert type(row_id) is int and row_id > 0, "Streaming persistence discarded its original native task receipt"
    rows = await agent._raw_storage.conversation.get_messages_by_ids([row_id])
    assert len(rows) == 1 and rows[0]["id"] == row_id and rows[0]["role"] == "assistant"
    assert rows[0]["content"] == "Owned streamed response"


@pytest.mark.asyncio
@pytest.mark.parametrize("assistant_agent", [False], indirect=True)
async def test_actual_bootstrap_cognition_enriches_its_own_write_receipts(assistant_agent, monkeypatch):
    agent = assistant_agent
    await agent.bootstrap_service.set_bootstrap_state(BootstrapState.PENDING)
    marker = "owned-bootstrap-unprocessed"
    receipts = observe_original_conversation_receipts(agent, monkeypatch, marker)
    response = await agent.process_input("I am happy to meet you", session_id=str(uuid4()))
    assert response.startswith("Hey! I'm OwnedAssistantReceipts.")
    assert [role for role, _row in receipts] == ["user", "assistant"]
    ids = [row for _role, row in receipts]
    assert all(type(row) is int and row > 0 for row in ids) and len(set(ids)) == 2
    rows = await agent._raw_storage.conversation.get_messages_by_ids(ids)
    assert len(rows) == 2 and all(row["metadata"]["emotional_tag_version"] != marker for row in rows), (
        "Bootstrap cognition dropped its exact row receipts before canonical memory enrichment"
    )


@pytest.mark.asyncio
async def test_anonymous_memory_enrichment_uses_only_stored_privacy_projections(assistant_agent, monkeypatch):
    agent = assistant_agent
    await agent.set_privacy_mode(PrivacyMode.ANONYMOUS)
    assert agent.privacy_mode == PrivacyMode.ANONYMOUS
    private_email = "owned-contact@example.test"
    user_id = await agent.privacy_agent.add_conversation("user", f"I am happy; email {private_email}")
    assistant_id = await agent._persist_assistant_conversation(f"Reply to {private_email}")
    observed = []
    analyze = agent.memory_system.tagger.analyze
    link = agent.memory_system.link_and_route_message

    async def original_analysis(content, *args, **kwargs):
        observed.append(content)
        return await analyze(content, *args, **kwargs)

    async def original_link(*args, **kwargs):
        observed.append(kwargs["content"])
        return await link(*args, **kwargs)

    monkeypatch.setattr(agent.memory_system.tagger, "analyze", original_analysis)
    monkeypatch.setattr(agent.memory_system, "link_and_route_message", original_link)
    previous = set(agent._background_tasks)
    await agent._post_response_pipeline(user_message_id=user_id, assistant_message_id=assistant_id)
    tasks = tuple(set(agent._background_tasks) - previous)
    if tasks:
        await asyncio.gather(*tasks)
    rows = await agent._raw_storage.conversation.get_messages_by_ids([user_id, assistant_id])
    assert len(observed) >= 3 and all(private_email not in text for text in observed), (
        "Derived memory received raw input instead of the canonical stored privacy projection"
    )
    assert all(private_email not in row["content"] for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["temporal", "graph"])
async def test_background_memory_retains_original_native_lease_control(native_pipeline_owner, monkeypatch, door):
    context = native_pipeline_owner
    owner, n = context.owner, context.n
    graph_calls = []
    link = n.pg_memory.link_and_route_message

    async def original_link(*args, **kwargs):
        graph_calls.append(True)
        return await link(*args, **kwargs)

    monkeypatch.setattr(n.pg_memory, "link_and_route_message", original_link)
    target, name = ((n.pg_storage.conversation, "get_full_history_with_ids") if door == "temporal"
        else (n.pg_memory, "link_and_route_message"))
    fault = observe_original_pg_release(monkeypatch, target, name)
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(n.did))
    with bind_execution_runtime(owner):
        await owner._post_response_pipeline(user_message_id=context.user_id,
            assistant_message_id=context.assistant_id)
        tasks = tuple(owner._background_tasks)
        assert len(tasks) == 1
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            owner._finish_background_task(task)
    assert len(fault.failed_connections) == 1 and fault.failed_connections[0].is_closed()
    assert fault.sources and is_execution_control_error(fault.sources[0]) and fault.error in tuple(_explicit_control_chain(fault.sources[0])), (
        "Background failure producer did not supply its classified original native lease control"
    )
    if door == "temporal":
        assert not graph_calls, "Temporal native control admitted a later graph operation"
    terminal = getattr(owner, "_shutdown_control", None)
    assert terminal is not None and is_execution_control_error(terminal) and fault.error in tuple(_explicit_control_chain(terminal)), (
        f"Background memory discarded its original native {door} control instead of retaining the actual task receipt"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_actual_model_cognition_enriches_original_receipts(
    native_memory_owners, assistant_agent, monkeypatch, streaming,
):
    """Real SDK requests to the already-owned local HTTP model, no paid model."""
    n, agent = native_memory_owners, assistant_agent
    service = agent.llm_service
    original_providers = service.providers
    configured = n.service.providers[0]
    async with openai.AsyncOpenAI(api_key="owned-local-memory-model",
            base_url=str(configured["client"].base_url), max_retries=0, timeout=5) as client:
        try:
            service.providers = [{**configured, "client": client, "model": "gpt-4o",
                "capabilities": dict(configured["capabilities"])}]
            service._embedding_route = configured["name"]
            service.set_model_preference("gpt-4o", vendor="openai", route="local-fixture", validate=False)
            marker = "owned-model-exchange-unprocessed"
            receipts = observe_original_conversation_receipts(agent, monkeypatch, marker)
            previous = set(agent._background_tasks)
            if streaming:
                response = "".join([chunk async for chunk in agent.process_input_streaming(
                    "I am happy to share original news", session_id=str(uuid4()))])
            else:
                response = await agent.process_input("I am happy to share original news",
                    session_id=str(uuid4()), include_memories=False)
            tasks = tuple(set(agent._background_tasks) - previous)
            if tasks:
                await asyncio.gather(*tasks)
            assert "local" in response
            assert any(request.get("messages") and request.get("model") == "gpt-4o" for request in n.requests)
            ids = [row for _role, row in receipts]
            rows = await agent._raw_storage.conversation.get_messages_by_ids(ids)
            assert len(set(ids)) == len(rows) == len(receipts)
            human = [row for row in rows if opened_by_person(row["role"], row["metadata"])]
            assistant = [row for row in rows if row["role"] == "assistant"]
            assert len(human) == len(assistant) == 1
            exchange_ids = {human[0]["id"], assistant[0]["id"]}
            assert all(row["metadata"]["emotional_tag_version"] != marker
                for row in rows if row["id"] in exchange_ids), (
                "Model cognition dropped its original write receipts before canonical memory enrichment"
            )
            assert all(row["metadata"]["emotional_tag_version"] == marker
                for row in rows if row["id"] not in exchange_ids), (
                "Model cognition enriched an operator notice instead of its original exchange"
            )
        finally:
            # Retire the actual agent/service owners before the borrowed HTTP
            # client is closed; the shared fixture's final retirement is idempotent.
            service.providers = original_providers
            await shutdown_fixture_agent(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["actions", "decisions", "interactions"])
async def test_schema_routing_retains_its_original_native_lease_control(native_pipeline_owner, monkeypatch, lane):
    context = native_pipeline_owner
    owner, n = context.owner, context.n
    router = n.pg_memory.router
    await n.pg_memory.link_and_route_message(message_id=context.user_id,
        content="Owned original message", role="user")
    concepts = []
    if lane == "actions":
        content, operation = "I need to ship the feature.", "_persist_action_items"
    elif lane == "decisions":
        content, operation = "I've decided to use Postgres.", "_persist_decisions"
    else:
        person_id = f"concept:{n.did}:owned-mom"
        await router.graph.add_node(GraphNode(node_id=person_id, node_type="concept", label="mom",
            properties={"category": "person", "agent_id": n.did}))
        concepts = [LinkedConcept(node_id=person_id, label="mom", category="person")]
        content, operation = "I love when mom calls on Sundays.", "_enrich_person_interactions"
    fault = observe_original_pg_release(monkeypatch, router, operation)
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(n.did))
    caught = None
    with bind_execution_runtime(owner):
        try:
            await router.route(message_id=str(context.user_id), content=content, concepts=concepts)
        except (DatabaseError, ExecutionAuthorityError, BaseExceptionGroup, asyncio.CancelledError) as error:
            caught = error
    assert len(fault.failed_connections) == 1 and fault.failed_connections[0].is_closed()
    assert fault.sources and is_execution_control_error(fault.sources[0]) and fault.error in tuple(_explicit_control_chain(fault.sources[0])), (
        "Schema lane did not reach its classified original physical PostgreSQL lease"
    )
    assert caught is not None and is_execution_control_error(caught) and fault.error in tuple(_explicit_control_chain(caught)), (
        f"Schema routing discarded original native {lane} control as advisory enrichment"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["boolean", "negative", "duplicate", "wrong-role", "missing", "missing-user", "missing-assistant"])
async def test_exchange_receipts_require_explicit_valid_native_identity(native_pipeline_owner, monkeypatch, kind):
    context = native_pipeline_owner
    user_id, assistant_id = context.user_id, context.assistant_id
    if kind == "boolean":
        user_id = True
    elif kind == "negative":
        user_id = -user_id
    elif kind == "duplicate":
        assistant_id = user_id
    elif kind == "wrong-role":
        user_id, assistant_id = assistant_id, user_id
    elif kind == "missing":
        assistant_id = max(user_id, assistant_id) + 1
        assert await context.n.pg_storage.get_conversation_message(assistant_id) is None
    elif kind == "missing-user":
        user_id = None
    else:
        assistant_id = None
    reads = []
    read = context.n.pg_storage.conversation.get_messages_by_ids

    async def original_read(*args, **kwargs):
        reads.append(args)
        return await read(*args, **kwargs)

    monkeypatch.setattr(context.n.pg_storage.conversation, "get_messages_by_ids", original_read)
    caught = None
    try:
        await context.owner._post_response_pipeline(user_message_id=user_id, assistant_message_id=assistant_id)
    except (ValueError, DatabaseError) as error:
        caught = error
    finally:
        tasks = tuple(context.owner._background_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                context.owner._finish_background_task(task)
    if kind.startswith("missing-"):
        assert caught is None and not reads and not context.owner._background_tasks, (
            "An incomplete exchange must not infer rows or schedule derived memory"
        )
    elif kind in {"boolean", "negative", "duplicate"}:
        assert isinstance(caught, ValueError) and "distinct positive native row receipts" in str(caught) and not reads, (
            "Invalid native exchange identities reached storage instead of failing the receipt contract"
        )
    else:
        assert isinstance(caught, ValueError) and "stored user/assistant rows" in str(caught) and len(reads) == 1, (
            "Missing or wrong-role exchange identities were accepted for derived memory"
        )


@pytest.mark.asyncio
async def test_memory_graph_routing_retains_original_native_schema_control(native_pipeline_owner, monkeypatch):
    context = native_pipeline_owner
    owner, n = context.owner, context.n
    fault = observe_original_pg_release(monkeypatch, n.pg_memory.router, "route")
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(n.did))
    caught = None
    with bind_execution_runtime(owner):
        try:
            await n.pg_memory.link_and_route_message(message_id=context.user_id,
                content="I need to ship the feature.", role="user")
        except (DatabaseError, ExecutionAuthorityError, BaseExceptionGroup, asyncio.CancelledError) as error:
            caught = error
    assert len(fault.failed_connections) == 1 and fault.failed_connections[0].is_closed()
    assert fault.sources and is_execution_control_error(fault.sources[0]) and fault.error in tuple(_explicit_control_chain(fault.sources[0]))
    assert caught is not None and is_execution_control_error(caught) and fault.error in tuple(_explicit_control_chain(caught)), (
        "Memory graph routing discarded its original native schema control"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["ordinary-query", "native-reset"])
async def test_memory_receipt_lookup_preserves_controls_but_keeps_query_errors_advisory(
    native_pipeline_owner, monkeypatch, kind,
):
    context = native_pipeline_owner
    owner, n = context.owner, context.n
    ordinary = kind == "ordinary-query"
    if ordinary:
        await n.pg_storage.db.execute(
            "ALTER TABLE conversation_history RENAME COLUMN content TO owned_unreadable_content"
        )
    fault = observe_original_pg_release(monkeypatch, n.pg_storage.conversation,
        "get_messages_by_ids", fail_reset=not ordinary)
    owner._execution_custody = ExecutionCustody(ProcessRuntimeExecutionFence(n.did))
    caught = None
    try:
        with bind_execution_runtime(owner):
            try:
                await owner._post_response_pipeline(user_message_id=context.user_id,
                    assistant_message_id=context.assistant_id)
            except (DatabaseError, ExecutionAuthorityError, BaseExceptionGroup, asyncio.CancelledError) as error:
                caught = error
            finally:
                tasks = tuple(owner._background_tasks)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    for task in tasks:
                        owner._finish_background_task(task)
    finally:
        if ordinary:
            await n.pg_storage.db.execute(
                "ALTER TABLE conversation_history RENAME COLUMN owned_unreadable_content TO content"
            )
    assert fault.sources, "Receipt lookup did not reach the original native failure producer"
    if ordinary:
        assert not is_execution_control_error(fault.sources[0])
        assert any(isinstance(source, asyncpg.UndefinedColumnError)
            for source in _explicit_control_chain(fault.sources[0]))
        assert caught is None and not owner._background_tasks, (
            "An ordinary native receipt lookup failure escaped advisory enrichment"
        )
    else:
        assert len(fault.failed_connections) == 1 and fault.failed_connections[0].is_closed()
        assert is_execution_control_error(fault.sources[0]) and fault.error in tuple(_explicit_control_chain(fault.sources[0]))
        assert caught is not None and is_execution_control_error(caught) and fault.error in tuple(_explicit_control_chain(caught)), (
            "Receipt lookup discarded its original native lease control"
        )
