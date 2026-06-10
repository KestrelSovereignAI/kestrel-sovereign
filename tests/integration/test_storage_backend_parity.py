"""SQLite/PostgreSQL semantic parity contracts for storage seams."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from kestrel_sovereign.endpoints.database import (
    _get_table_columns,
    _list_table_names,
    list_database_tables,
    query_database_table,
)
from kestrel_sovereign.a2a.stores.unified import TaskStore
from kestrel_sovereign.a2a.types import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sovereign.features.webhooks.feature import WebhookFeature
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


def _project_rows(rows):
    return [(row[1], row[2]) for row in rows]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_conversation_session_queries_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    other_agent_id = f"did:test:{uuid4()}"
    start = datetime(2026, 4, 16, 12, 0, 0)

    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                agent_id,
                "system",
                "[New conversation started]",
                '{"new_session": true, "type": "session_marker"}',
                start,
            ),
            (agent_id, "user", "hello", '{"topic": "parity"}', start + timedelta(minutes=1)),
            (agent_id, "assistant", "hi there", "{}", start + timedelta(minutes=2)),
            (other_agent_id, "user", "not yours", "{}", start + timedelta(minutes=3)),
        ],
    )

    inserted = await storage.db.fetchall(
        """
        SELECT id, role, content, metadata, created_at
        FROM conversation_history
        WHERE agent_id = ?
        ORDER BY created_at ASC
        """,
        (agent_id,),
    )
    session_id = str(inserted[0][0])

    listed = await privacy_storage.query_conversations(agent_id, limit=10)
    assert _project_rows(listed) == [
        ("assistant", "hi there"),
        ("user", "hello"),
        ("system", "[New conversation started]"),
    ]

    start_row = await privacy_storage.query_conversation_start(session_id, agent_id)
    assert start_row is not None

    session_rows = await privacy_storage.query_conversation_messages(
        agent_id,
        start_row[0],
        limit=10,
    )
    assert _project_rows(session_rows) == [
        ("system", "[New conversation started]"),
        ("user", "hello"),
        ("assistant", "hi there"),
    ]

    other_rows = await privacy_storage.query_conversation_messages(
        other_agent_id,
        start_row[0],
        limit=10,
    )
    assert _project_rows(other_rows) == [("user", "not yours")]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a2a_task_store_filters_and_payloads_are_backend_neutral(db_backend):
    store = TaskStore(db_backend)
    await store.initialize()

    session_a = f"session-{uuid4()}"
    session_b = f"session-{uuid4()}"
    user_a = f"user-{uuid4()}"
    user_b = f"user-{uuid4()}"
    task_a = f"task-{uuid4()}"
    task_b = f"task-{uuid4()}"
    other_task = f"task-{uuid4()}"

    await store.save(
        Task(
            id=task_a,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[Message(role="user", parts=[TextPart(text="build the thing")])],
            metadata={"task_type": "audit", "user_id": user_a, "marker": "first"},
        )
    )
    await store.save(
        Task(
            id=task_b,
            sessionId=session_a,
            status=TaskStatus(state=TaskState.COMPLETED),
            metadata={"task_type": "audit", "user_id": user_a, "marker": "second"},
        )
    )
    await store.save(
        Task(
            id=other_task,
            sessionId=session_b,
            status=TaskStatus(state=TaskState.SUBMITTED),
            metadata={"task_type": "audit", "user_id": user_b, "marker": "other"},
        )
    )

    await store.update_status(
        task_a,
        TaskStatus(
            state=TaskState.WORKING,
            message=Message(role="agent", parts=[TextPart(text="underway")]),
        ),
    )
    await store.add_artifact(
        task_a,
        Artifact(
            name="result.txt",
            parts=[TextPart(text="semantic parity")],
        ),
    )

    retrieved = await store.get(task_a)
    assert retrieved is not None
    assert retrieved.sessionId == session_a
    assert retrieved.status.state == TaskState.WORKING
    assert retrieved.status.message is not None
    assert retrieved.status.message.parts[0].text == "underway"
    assert retrieved.history is not None
    assert retrieved.history[0].parts[0].text == "build the thing"
    assert retrieved.artifacts is not None
    assert retrieved.artifacts[0].parts[0].text == "semantic parity"
    assert retrieved.metadata == {"task_type": "audit", "user_id": user_a, "marker": "first"}

    session_tasks = await store.list_tasks(session_id=session_a, user_id=user_a, limit=10)
    assert {task.id for task in session_tasks} == {task_a, task_b}
    assert {task.metadata["marker"] for task in session_tasks} == {"first", "second"}

    working_tasks = await store.list_tasks(user_id=user_a, status=TaskState.WORKING, limit=10)
    assert [task.id for task in working_tasks] == [task_a]

    assert await store.delete(task_a) is True
    assert await store.get(task_a) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_database_introspection_helpers_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()

    table_names = await _list_table_names(storage.db)
    assert "conversation_history" in table_names
    assert "graph_nodes" in table_names

    columns = await _get_table_columns(storage.db, "conversation_history")
    by_name = {column["name"]: column for column in columns}

    assert by_name["id"]["pk"] is True
    assert by_name["agent_id"]["nullable"] is False
    assert by_name["role"]["nullable"] is False
    assert by_name["content"]["type"]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_webhook_registration_and_audit_history_are_backend_neutral(db_backend):
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()

    agent_id = f"did:test:{uuid4()}"
    agent = SimpleNamespace(
        did=agent_id,
        agent_id=agent_id,
        storage=SimpleNamespace(db=storage.db),
        _raw_storage=None,
        features=[],
    )
    feature = WebhookFeature(agent)
    await feature.initialize()

    # Webhook tools migrated to ToolResult (#1061 wave 26).
    from kestrel_sdk.tools.result import ToolResultStatus
    webhook_name = f"audit-{uuid4().hex}"
    registered = await feature.webhooks_register(
        name=webhook_name,
        auth_type="none",
        event_type="sync",
        rate_limit=0,
    )
    assert registered.status is ToolResultStatus.OK

    listed = await feature.webhooks_list()
    assert listed.data["count"] == 1
    assert listed.data["webhooks"][0]["name"] == webhook_name

    await feature.log_webhook_event(
        webhook_name=webhook_name,
        source_ip="127.0.0.1",
        authenticated=True,
        status_code=200,
        payload_hash="abc123",
    )

    history = await feature.webhooks_history(limit=5)
    assert history.data["count"] == 1
    assert history.data["events"][0]["webhook_name"] == webhook_name
    assert history.data["events"][0]["authenticated"] is True
    assert history.data["events"][0]["payload_hash"] == "abc123"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_scopes_rows_to_requesting_agent(db_backend):
    """#1651: the /api/db/tables explorer must only return the requesting
    agent's rows for agent-scoped tables, never another agent's data in a
    shared multi-agent database — and the scope must survive a free-text
    search."""
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    other_agent_id = f"did:test:{uuid4()}"
    start = datetime(2026, 4, 16, 12, 0, 0)

    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (agent_id, "user", "mine-1", "{}", start),
            (agent_id, "assistant", "mine-2", "{}", start + timedelta(minutes=1)),
            (other_agent_id, "user", "NOT-YOURS", "{}", start + timedelta(minutes=2)),
        ],
    )

    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "conversation_history", limit=50, offset=0, search=None
    )
    contents = {r["content"] for r in result["rows"]}
    assert contents == {"mine-1", "mine-2"}
    assert result["total_rows"] == 2
    assert all(r["agent_id"] == agent_id for r in result["rows"])

    # The agent scope must AND with search — the other agent's "NOT-YOURS"
    # matches the term but must stay invisible.
    searched = await query_database_table(
        request, "conversation_history", limit=50, offset=0, search="YOURS"
    )
    assert searched["rows"] == []

    # list_database_tables row counts are scoped too.
    listing = await list_database_tables(request)
    conv = next(t for t in listing["tables"] if t["name"] == "conversation_history")
    assert conv["row_count"] == 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_hides_agent_rows_in_ephemeral_mode(db_backend):
    """#1651: for agent-scoped tables, EPHEMERAL/ISOLATED modes must not
    surface persisted rows through the raw explorer."""
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    await storage.db.execute_many(
        """
        INSERT INTO conversation_history (agent_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(agent_id, "user", "secret", "{}", datetime(2026, 4, 16, 12, 0, 0))],
    )

    privacy_storage.set_privacy_mode(PrivacyMode.EPHEMERAL)
    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "conversation_history", limit=50, offset=0, search=None
    )
    assert result["rows"] == []
    assert result["total_rows"] == 0
    assert "privacy mode" in result.get("note", "").lower()

    # The listing must not reveal the row exists via its count either.
    listing = await list_database_tables(request)
    conv = next(t for t in listing["tables"] if t["name"] == "conversation_history")
    assert conv["row_count"] == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_scopes_graph_nodes_by_properties_agent_id(db_backend):
    """#1651: graph_nodes stores agent ownership in the JSON `properties`
    (no agent_id column), and the app scopes by it — the explorer must too,
    or another agent's graph leaks via /api/db/tables/graph_nodes."""
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    other_agent_id = f"did:test:{uuid4()}"

    await storage.db.execute_many(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
        [
            (f"n-{uuid4()}", "concept", "mine", json.dumps({"agent_id": agent_id})),
            (f"n-{uuid4()}", "concept", "also-mine", json.dumps({"agent_id": agent_id})),
            (f"n-{uuid4()}", "concept", "theirs", json.dumps({"agent_id": other_agent_id})),
        ],
    )

    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "graph_nodes", limit=50, offset=0, search=None
    )
    labels = {r["label"] for r in result["rows"]}
    assert labels == {"mine", "also-mine"}
    assert result["total_rows"] == 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_db_explorer_scopes_graph_edges_via_node_membership(db_backend):
    """#1651: graph_edges have no direct owner — an edge belongs to the agent
    if it touches one of the agent's nodes. Exercises the two-subquery scope
    and the [agent_id, agent_id] + search params ordering."""
    storage = AsyncStorage.from_backend(db_backend)
    await storage.initialize()
    privacy_storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)

    agent_id = f"did:test:{uuid4()}"
    other = f"did:test:{uuid4()}"
    a1, a2 = f"a1-{uuid4()}", f"a2-{uuid4()}"
    b1, b2 = f"b1-{uuid4()}", f"b2-{uuid4()}"

    await storage.db.execute_many(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?, ?, ?, ?)",
        [
            (a1, "concept", "a1", json.dumps({"agent_id": agent_id})),
            (a2, "concept", "a2", json.dumps({"agent_id": agent_id})),
            (b1, "concept", "b1", json.dumps({"agent_id": other})),
            (b2, "concept", "b2", json.dumps({"agent_id": other})),
        ],
    )
    await storage.db.execute_many(
        "INSERT INTO graph_edges (source_id, target_id, label, properties) VALUES (?, ?, ?, ?)",
        [
            (a1, a2, "mine_internal", "{}"),    # both endpoints A -> agent's
            (b1, b2, "theirs_internal", "{}"),  # both endpoints B -> not agent's
            (a1, b1, "mine_crosslink", "{}"),   # source is A's -> agent's
        ],
    )

    agent = SimpleNamespace(agent_id=agent_id, storage=privacy_storage)
    request = SimpleNamespace(state=SimpleNamespace(agent=agent))

    result = await query_database_table(
        request, "graph_edges", limit=50, offset=0, search=None
    )
    assert {r["label"] for r in result["rows"]} == {"mine_internal", "mine_crosslink"}
    assert result["total_rows"] == 2

    # Scope must AND with search: "theirs_internal" matches the term but is
    # excluded by node membership — proving scope params precede search params.
    searched = await query_database_table(
        request, "graph_edges", limit=50, offset=0, search="internal"
    )
    assert {r["label"] for r in searched["rows"]} == {"mine_internal"}
