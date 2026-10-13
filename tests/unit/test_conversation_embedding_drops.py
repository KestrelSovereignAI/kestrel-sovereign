"""A conversation write that drops its embedding is reported, not silent (#2913).

Production wrote every turn of a PostgreSQL agent without its vector: the
``conversation_history.embedding_vec`` column was ``vector(768)`` and the
profile produced 1536 dimensions. Each turn persisted and logged three INFO
lines. The store now counts each dropped embedding, warns (rate-limited), and
surfaces the count at ``/health/detailed``; at startup it compares the profile
with the column's declared width once and reports a disagreement as an ERROR
and a health misconfiguration, without refusing to boot.

A drop is counted only once its turn has committed: a turn whose transaction
rolls back never persisted, and the count and WARNING cannot be taken back. The
store defers the count with ``AsyncDatabase.call_after_commit``, so it lands
when the transaction that owns the INSERT commits, whoever owns it: the
caller, ``AsyncStorage``'s semantic-recall fence, or no one (an INSERT that
commits on its own).

Every case writes through the real store onto a real database. The SQLite cases
run everywhere. The PostgreSQL cases reproduce the production failure itself
(pgvector refusing the width) and run when ``TEST_POSTGRES_URL`` is set, each in
a schema of its own.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.health.checks import (
    check_conversation_embeddings,
    run_standard_checks,
)
from kestrel_sovereign.features.storage_access import resolve_conversation_store
from kestrel_sovereign.identity.runtime_identity import load_agent_identity
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.knowledge import (
    Assertion,
    DirectLineage,
    EpistemicState,
    IRI,
    Literal,
    OntologyRef,
    SourceOccurrence,
    XSD_STRING,
)
from kestrel_sovereign.llm.embedding_service import derive_embedding_profile
from kestrel_sovereign.security.assertion_tenant_resolver import (
    _resolve_authenticated_agent_assertion_capability,
)
from kestrel_sovereign.storage import async_conversation_store as store_module
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.db import TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from tests.utils.postgres_schema import (
    disposable_postgres_schema,
    pgvector_schema,
    quoted_search_path,
    with_search_path,
)

STORE_LOGGER = store_module.logger.name
AGENT_ID = "did:test:embedding-drops"


class _FixedWidthEmbeddings:
    """An embedding service whose vectors all have one width."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.profile = derive_embedding_profile(
            provider="test", model=f"fixed-{dim}", dim=dim
        )

    async def aembed(self, text: str):
        return [0.01 * (index + 1) for index in range(self.dim)]

    def describe(self):
        return self.profile

    def current_profile_id(self) -> str:
        return self.profile.profile_id


def _store(db, service) -> AsyncConversationStore:
    store = AsyncConversationStore(db, agent_id=AGENT_ID)
    store._lazy_embedding_service = lambda: service  # type: ignore[assignment]
    return store


def _agent_holding(store) -> SimpleNamespace:
    """An agent whose raw storage owns *store*, as a booted agent's does."""
    return SimpleNamespace(_raw_storage=SimpleNamespace(conversation=store))


def _configure_column_width(monkeypatch, dim: int) -> None:
    """The width the process believes the column has (``KESTREL_EMBEDDING_DIM``)."""
    monkeypatch.setattr(
        "kestrel_sovereign.storage.sqla.conversation_message."
        "CONVERSATION_MESSAGE_EMBEDDING_DIM",
        dim,
    )


def _store_records(caplog, level: int):
    return [
        record for record in caplog.records
        if record.name == STORE_LOGGER and record.levelno >= level
    ]


async def _stored_turns(db, store):
    """Every persisted turn: decrypted content and whether it holds a vector."""
    rows = await db.fetchall(
        "SELECT content, metadata, embedding_vec IS NOT NULL, embedding_profile_id "
        "FROM conversation_history WHERE agent_id = ? ORDER BY id",
        (store.agent_id,),
    )
    return [
        (
            store.decrypt_stored_content(content, json.loads(metadata or "{}")),
            bool(has_vector),
            profile_id,
        )
        for content, metadata, has_vector, profile_id in rows
    ]


@pytest.fixture
async def sqlite_db(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "embedding_drops.db"))
    try:
        yield db
    finally:
        await db.close()


async def _reject_vectors_on_insert(db, message: str) -> None:
    """Make SQLite refuse every vector-bearing INSERT, as pgvector does on width.

    The trigger lives on this test's private database file only.
    """
    await db.execute(
        "CREATE TRIGGER reject_embedding_width BEFORE INSERT ON conversation_history "
        f"WHEN NEW.embedding_vec IS NOT NULL BEGIN SELECT RAISE(ABORT, '{message}'); END",
        (),
    )


# ------------------------------------------------------------- write path


async def test_mismatched_width_write_persists_the_turn_warns_and_counts(
    sqlite_db, monkeypatch, caplog
):
    """The provider's width disagrees with the configured column width."""
    _configure_column_width(monkeypatch, 4)
    service = _FixedWidthEmbeddings(6)
    store = _store(sqlite_db, service)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        await store.add_conversation(role="assistant", content="the heron nests")

    assert await _stored_turns(sqlite_db, store) == [
        ("the heron nests", False, None)
    ]
    warnings = _store_records(caplog, logging.WARNING)
    assert [record.levelno for record in warnings] == [logging.WARNING]
    message = warnings[0].getMessage()
    assert AGENT_ID in message
    assert service.profile.profile_id in message
    assert "produced 6 dimensions" in message
    assert "takes 4 dimensions" in message
    assert store.dropped_embedding_count == 1

    health = await check_conversation_embeddings(_agent_holding(store))
    assert health["status"] == "warn"
    assert health["details"]["dropped"] == 1
    assert health["details"]["misconfigured"] is False
    assert health["details"]["last_profile_id"] == service.profile.profile_id
    assert health["details"]["last_embedding_dim"] == 6
    assert health["details"]["last_column_dim"] == 4
    json.dumps(health)


async def test_a_vector_the_database_refuses_is_dropped_through_every_fallback(
    sqlite_db, monkeypatch, caplog
):
    """Production's path: every vector-bearing INSERT shape fails, the turn lands."""
    _configure_column_width(monkeypatch, 6)
    await _reject_vectors_on_insert(sqlite_db, "expected 4 dimensions, not 6")
    service = _FixedWidthEmbeddings(6)
    store = _store(sqlite_db, service)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        await store.add_conversation(role="user", content="where is the heron?")

    assert await _stored_turns(sqlite_db, store) == [
        ("where is the heron?", False, None)
    ]
    # The retries are diagnostic detail; the outcome is the one WARNING.
    assert not [
        r for r in _store_records(caplog, logging.INFO) if r.levelno == logging.INFO
    ]
    warnings = _store_records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "expected 4 dimensions, not 6" in warnings[0].getMessage()
    assert store.dropped_embedding_count == 1
    last = store.last_dropped_embedding
    assert last.profile_id == service.profile.profile_id
    assert last.embedding_dim == 6
    assert "expected 4 dimensions, not 6" in last.error

    health = await check_conversation_embeddings(_agent_holding(store))
    assert health["status"] == "warn"
    assert health["details"]["dropped"] == 1
    assert "expected 4 dimensions, not 6" in health["details"]["last_error"]


async def test_repeated_drops_warn_once_per_window_and_count_every_one(
    sqlite_db, monkeypatch, caplog
):
    _configure_column_width(monkeypatch, 6)
    await _reject_vectors_on_insert(sqlite_db, "expected 4 dimensions, not 6")
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))
    window = store_module.DROPPED_EMBEDDING_WARNING_INTERVAL_S
    now = [1000.0]
    # The store's own clock reference, not ``time.monotonic``: the event loop
    # reads that one.
    monkeypatch.setattr(store_module, "monotonic", lambda: now[0])

    async def drop_at(seconds: float) -> None:
        now[0] = 1000.0 + seconds
        await store.add_conversation(role="user", content=f"turn at {seconds}")

    with caplog.at_level(logging.WARNING, logger=STORE_LOGGER):
        await drop_at(0)
        await drop_at(1)
        await drop_at(window - 1)
        await drop_at(window)          # the window has elapsed: warn again
        await drop_at(window + 1)
        await drop_at(2 * window - 1)  # within the re-armed window
        await drop_at(2 * window)

    messages = [r.getMessage() for r in _store_records(caplog, logging.WARNING)]
    assert len(messages) == 3
    # Each line carries the running count, so suppressed drops are not lost.
    assert "1 embedding(s) dropped since start" in messages[0]
    assert "4 embedding(s) dropped since start" in messages[1]
    assert "7 embedding(s) dropped since start" in messages[2]
    assert store.dropped_embedding_count == 7
    assert len(await _stored_turns(sqlite_db, store)) == 7


async def test_a_matching_width_write_stores_the_vector_and_counts_nothing(
    sqlite_db, monkeypatch, caplog
):
    _configure_column_width(monkeypatch, 6)
    service = _FixedWidthEmbeddings(6)
    store = _store(sqlite_db, service)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        await store.add_conversation(role="assistant", content="the heron nests")

    assert await _stored_turns(sqlite_db, store) == [
        ("the heron nests", True, service.profile.profile_id)
    ]
    assert store.dropped_embedding_count == 0
    assert store.last_dropped_embedding is None
    assert not _store_records(caplog, logging.WARNING)
    health = await check_conversation_embeddings(_agent_holding(store))
    assert health["status"] == "pass"


async def test_a_write_that_fails_outright_is_not_counted_as_a_drop(
    sqlite_db, monkeypatch
):
    """Only a persisted turn can have dropped its vector; a failed write raises."""
    _configure_column_width(monkeypatch, 4)
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))
    await sqlite_db.execute(
        "CREATE TRIGGER reject_every_turn BEFORE INSERT ON conversation_history "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END",
        (),
    )

    with pytest.raises(Exception, match="disk full"):
        await store.add_conversation(role="assistant", content="lost")

    assert store.dropped_embedding_count == 0


async def test_a_row_excluded_from_retrieval_is_not_counted_as_a_drop(
    sqlite_db, monkeypatch
):
    """A stale semantic-recall derivative never carries a vector, by design."""
    _configure_column_width(monkeypatch, 4)
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))
    prepared = await store._prepare_conversation_write(  # noqa: SLF001 - fence preparation contract
        "assistant", "stale semantic result", None, None, None, None, None,
    )
    assert prepared.embedding_drop is not None

    await store._exclude_prepared_conversation_from_retrieval(prepared)  # noqa: SLF001 - fence preparation contract
    await store._persist_prepared_conversation(prepared)  # noqa: SLF001 - fence preparation contract

    assert await _stored_turns(sqlite_db, store) == [
        ("stale semantic result", False, None)
    ]
    assert store.dropped_embedding_count == 0


# ------------------------------------------------------------- transactions


class _RollBack(Exception):
    """Raised inside a transaction so that it rolls back."""


async def _drop_every_vector(db, monkeypatch, mechanism: str) -> None:
    """Make every 6-dimension embedding drop, by *mechanism*.

    ``configured_width``: the process believes the column is 4 wide, so the
    store drops the vector before any INSERT. ``refused_by_database``: every
    vector-bearing INSERT shape fails, as pgvector's width check does in
    production, so the store falls back to a row without one.
    """
    if mechanism == "configured_width":
        _configure_column_width(monkeypatch, 4)
    else:
        _configure_column_width(monkeypatch, 6)
        await _reject_vectors_on_insert(db, "expected 4 dimensions, not 6")


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_a_turn_its_transaction_rolls_back_counts_no_drop_and_warns_nothing(
    sqlite_db, monkeypatch, caplog, mechanism
):
    """The INSERT joins the caller's transaction; the turn never persisted."""
    await _drop_every_vector(sqlite_db, monkeypatch, mechanism)
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        with pytest.raises(TransactionError) as caught:
            async with sqlite_db.transaction():
                await store.add_conversation(role="user", content="rolled back")
                raise _RollBack()

    assert isinstance(caught.value.__cause__, _RollBack)
    assert await _stored_turns(sqlite_db, store) == []
    assert store.dropped_embedding_count == 0
    assert store.last_dropped_embedding is None
    assert not _store_records(caplog, logging.WARNING)
    health = await check_conversation_embeddings(_agent_holding(store))
    assert health["status"] == "pass"
    assert health["message"] == "No conversation embeddings dropped"


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_a_turn_its_transaction_commits_counts_one_drop_after_the_commit(
    sqlite_db, monkeypatch, caplog, mechanism
):
    """The INSERT joins the caller's transaction; its commit counts the drop."""
    await _drop_every_vector(sqlite_db, monkeypatch, mechanism)
    service = _FixedWidthEmbeddings(6)
    store = _store(sqlite_db, service)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        async with sqlite_db.transaction():
            await store.add_conversation(role="user", content="committed")
            # Written, not yet committed: nothing counted, nothing logged.
            assert store.dropped_embedding_count == 0
            assert not _store_records(caplog, logging.WARNING)

    assert await _stored_turns(sqlite_db, store) == [("committed", False, None)]
    assert store.dropped_embedding_count == 1
    assert store.last_dropped_embedding.profile_id == service.profile.profile_id
    assert len(_store_records(caplog, logging.WARNING)) == 1
    health = await check_conversation_embeddings(_agent_holding(store))
    assert health["status"] == "warn"
    assert health["details"]["dropped"] == 1


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_a_turn_in_a_rolled_back_savepoint_counts_no_drop(
    sqlite_db, monkeypatch, caplog, mechanism
):
    """Only the turn that survives the savepoint rollback counts, after commit."""
    await _drop_every_vector(sqlite_db, monkeypatch, mechanism)
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        async with sqlite_db.transaction():
            with pytest.raises(_RollBack):
                async with sqlite_db.transaction(savepoint=True):
                    await store.add_conversation(role="user", content="undone")
                    raise _RollBack()
            await store.add_conversation(role="user", content="kept")
            assert store.dropped_embedding_count == 0

    assert await _stored_turns(sqlite_db, store) == [("kept", False, None)]
    assert store.dropped_embedding_count == 1
    assert len(_store_records(caplog, logging.WARNING)) == 1


_ONTOLOGY = OntologyRef(
    "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
    "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
    "semantic-kb-v1",
)


@asynccontextmanager
async def _storage_with_one_assertion(tmp_path, open_storage):
    """Storage bound to an assertion tenant, holding one current assertion.

    A conversation write that names that assertion as a semantic-recall
    dependency is persisted inside ``AsyncStorage``'s lifecycle fence: a
    transaction ``AsyncStorage.add_conversation`` owns and commits itself.
    *open_storage* builds the ``AsyncStorage`` from the tenant's agent id and
    capability. Yields the storage and the write's metadata.
    """
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Embedding drops",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    storage = open_storage(
        credentials.agent_did,
        _resolve_authenticated_agent_assertion_capability(
            credentials.agent_did, identity,
        ),
    )
    await storage.initialize()
    try:
        source = SourceOccurrence(
            source_occurrence_id="source:heron", source_kind="fixture",
            locator="fixture:heron", received_at="2026-10-13T00:00:00Z",
        )
        fact = Assertion(
            tenant_id=storage.agent_id, owning_agent_id=storage.agent_id,
            subject=IRI("https://example.test/heron"),
            predicate=IRI("https://example.test/nests"),
            object=Literal("by the river", XSD_STRING),
            revision_id="heron-revision", confidence="1",
            confidence_method="fixture", confidence_basis="fixture",
            epistemic_state=EpistemicState.ASSERTED,
            asserted_at="2026-10-13T00:00:00Z", ontology_version=_ONTOLOGY,
            lineage=DirectLineage((source.source_occurrence_id,)),
            privacy_classification="normal", release_policy_reference="policy:test",
        )
        result = await storage.put_assertion(fact, source_occurrences=(source,))
        assert result.accepted
        metadata = {
            "semantic_recall_dependencies": [
                {"assertion_id": fact.assertion_id, "revision_id": fact.revision_id}
            ]
        }
        yield storage, metadata
    finally:
        await storage.close()


@pytest.fixture
async def fenced_storage(tmp_path):
    """SQLite storage holding one current assertion; see ``_storage_with_one_assertion``."""

    def open_sqlite(agent_id, capability):
        return AsyncStorage(
            ":memory:", agent_id=agent_id, _assertion_tenant_capability=capability,
        )

    async with _storage_with_one_assertion(tmp_path, open_sqlite) as fenced:
        yield fenced


def _observe_persist(store, monkeypatch, *, then_raise: bool = False):
    """Record the drop count and transaction state the moment the INSERT returns."""
    seen = []
    persist = store._persist_prepared_conversation  # noqa: SLF001 - fence persistence seam

    async def observed(prepared):
        await persist(prepared)
        seen.append((store.dropped_embedding_count, store.db.owns_open_transaction))
        if then_raise:
            raise _RollBack()

    monkeypatch.setattr(store, "_persist_prepared_conversation", observed)
    return seen


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_a_fenced_turn_counts_its_drop_only_after_the_fence_commits(
    fenced_storage, monkeypatch, caplog, mechanism
):
    """The fence owns the transaction, so the drop counts when the fence commits."""
    storage, metadata = fenced_storage
    store = storage.conversation
    await _drop_every_vector(storage.db, monkeypatch, mechanism)
    service = _FixedWidthEmbeddings(6)
    store._lazy_embedding_service = lambda: service
    seen = _observe_persist(store, monkeypatch)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        await storage.add_conversation("assistant", "the heron nests", metadata=metadata)

    # Inside the fence the row was written but not committed: nothing counted.
    assert seen == [(0, True)]
    [(content, has_vector, profile_id)] = await _stored_turns(storage.db, store)
    assert (content, has_vector, profile_id) == ("the heron nests", False, None)
    stored_metadata = json.loads(await storage.db.fetchval(
        "SELECT metadata FROM conversation_history WHERE agent_id = ?",
        (store.agent_id,),
    ))
    # The fence admitted the derivative; an excluded one drops nothing.
    assert "excluded_from_context" not in stored_metadata
    assert store.dropped_embedding_count == 1
    assert store.last_dropped_embedding.profile_id == service.profile.profile_id
    assert len(_store_records(caplog, logging.WARNING)) == 1


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_a_fenced_turn_the_fence_rolls_back_counts_no_drop(
    fenced_storage, monkeypatch, caplog, mechanism
):
    """The reviewer's case: the fence's transaction fails after the INSERT."""
    storage, metadata = fenced_storage
    store = storage.conversation
    await _drop_every_vector(storage.db, monkeypatch, mechanism)
    store._lazy_embedding_service = lambda: _FixedWidthEmbeddings(6)
    seen = _observe_persist(store, monkeypatch, then_raise=True)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        with pytest.raises(TransactionError) as caught:
            await storage.add_conversation("assistant", "rolled back", metadata=metadata)

    assert isinstance(caught.value.__cause__, _RollBack)
    assert seen == [(0, True)]
    assert await _stored_turns(storage.db, store) == []
    assert store.dropped_embedding_count == 0
    assert store.last_dropped_embedding is None
    assert not _store_records(caplog, logging.WARNING)


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_a_fenced_turn_joining_a_committed_transaction_counts_one_drop(
    fenced_storage, monkeypatch, caplog, mechanism
):
    """The fence joins the caller's transaction: one drop, at the caller's commit."""
    storage, metadata = fenced_storage
    store = storage.conversation
    await _drop_every_vector(storage.db, monkeypatch, mechanism)
    store._lazy_embedding_service = lambda: _FixedWidthEmbeddings(6)
    seen = _observe_persist(store, monkeypatch)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        async with storage.transaction():
            await storage.add_conversation(
                "assistant", "the heron nests", metadata=metadata
            )
            # The fence has exited; the caller's transaction is still open.
            assert store.dropped_embedding_count == 0

    assert seen == [(0, True)]
    assert await _stored_turns(storage.db, store) == [
        ("the heron nests", False, None)
    ]
    assert store.dropped_embedding_count == 1
    assert len(_store_records(caplog, logging.WARNING)) == 1


@pytest.fixture
async def plain_storage(tmp_path):
    """``AsyncStorage`` with no assertion tenant: writes take the no-fence path."""
    storage = AsyncStorage(str(tmp_path / "plain.db"), agent_id=AGENT_ID)
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.close()


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_an_ordinary_storage_write_counts_its_drop(
    plain_storage, monkeypatch, caplog, mechanism
):
    """``AsyncStorage.add_conversation`` with no semantic-recall lineage."""
    store = plain_storage.conversation
    await _drop_every_vector(plain_storage.db, monkeypatch, mechanism)
    store._lazy_embedding_service = lambda: _FixedWidthEmbeddings(6)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        await plain_storage.add_conversation("user", "where is the heron?")

    assert await _stored_turns(plain_storage.db, store) == [
        ("where is the heron?", False, None)
    ]
    assert store.dropped_embedding_count == 1
    assert len(_store_records(caplog, logging.WARNING)) == 1


@pytest.mark.parametrize("mechanism", ["configured_width", "refused_by_database"])
async def test_an_ordinary_storage_write_in_a_transaction_counts_at_its_outcome(
    plain_storage, monkeypatch, caplog, mechanism
):
    """Committed: one drop after the commit. Rolled back: none."""
    store = plain_storage.conversation
    await _drop_every_vector(plain_storage.db, monkeypatch, mechanism)
    store._lazy_embedding_service = lambda: _FixedWidthEmbeddings(6)

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        with pytest.raises(TransactionError):
            async with plain_storage.transaction():
                await plain_storage.add_conversation("user", "rolled back")
                raise _RollBack()
        assert store.dropped_embedding_count == 0

        async with plain_storage.transaction():
            await plain_storage.add_conversation("user", "committed")
            assert store.dropped_embedding_count == 0

    assert await _stored_turns(plain_storage.db, store) == [
        ("committed", False, None)
    ]
    assert store.dropped_embedding_count == 1
    assert len(_store_records(caplog, logging.WARNING)) == 1


async def test_a_backend_that_cannot_report_its_commits_warns_without_counting(
    monkeypatch, caplog
):
    """No commit to wait for and no proof of one: the drop is still a WARNING."""
    _configure_column_width(monkeypatch, 4)

    class _CommitBlindDatabase:
        backend_type = "sqlite"

        def call_after_commit(self, callback):
            return False

    store = _store(_CommitBlindDatabase(), _FixedWidthEmbeddings(6))
    drop = store._column_width_drop([0.0] * 6, "profile-x")  # noqa: SLF001 - drop accounting seam

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        store._count_embedding_drop_on_commit(drop)  # noqa: SLF001 - drop accounting seam

    [warning] = _store_records(caplog, logging.WARNING)
    assert "not counted" in warning.getMessage()
    assert AGENT_ID in warning.getMessage()
    assert store.dropped_embedding_count == 0


# ------------------------------------------------------------- startup check


async def test_startup_check_is_silent_when_sqlite_declares_no_width(
    sqlite_db, caplog
):
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        assert await store.check_embedding_column_dimensions() is None

    assert not _store_records(caplog, logging.INFO)
    assert store.embedding_column_mismatch is None
    health = await check_conversation_embeddings(_agent_holding(store))
    assert health["status"] == "pass"


async def test_startup_check_is_silent_when_the_width_cannot_be_read(caplog):
    """A catalog read that fails is a skipped check, never a failed boot."""

    class _UnreadableDatabase:
        backend_type = "unknown"

    store = _store(_UnreadableDatabase(), _FixedWidthEmbeddings(6))

    with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
        assert await store.check_embedding_column_dimensions() is None

    assert not _store_records(caplog, logging.INFO)
    assert store.embedding_column_mismatch is None


# ------------------------------------------------------------- health surface


async def test_health_reads_the_store_beneath_privacy_wrapped_storage(
    tmp_path, monkeypatch, recwarn
):
    """A booted agent's public storage is privacy-wrapped; its deprecated
    ``conversation`` passthrough warns on every access and must not be used."""
    _configure_column_width(monkeypatch, 4)
    from kestrel_sovereign.privacy import PrivacyMode

    raw = AsyncStorage(str(tmp_path / "wrapped.db"), agent_id=AGENT_ID)
    await raw.initialize()
    try:
        store = raw.conversation
        store._lazy_embedding_service = lambda: _FixedWidthEmbeddings(6)
        await store.add_conversation(role="assistant", content="hello")
        agent = SimpleNamespace(storage=PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL))

        assert resolve_conversation_store(agent) is store
        health = await check_conversation_embeddings(agent)

        assert health["details"]["dropped"] == 1
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)
                    and "conversation" in str(w.message)]
    finally:
        await raw.close()


async def test_standard_checks_include_conversation_embeddings(
    sqlite_db, monkeypatch
):
    """Both /health/detailed paths run ``run_standard_checks``."""
    _configure_column_width(monkeypatch, 4)
    store = _store(sqlite_db, _FixedWidthEmbeddings(6))
    await store.add_conversation(role="assistant", content="hello")

    checks = await run_standard_checks(_agent_holding(store), sqlite_db)

    [embeddings] = [c for c in checks if c["name"] == "conversation_embeddings"]
    assert embeddings["status"] == "warn"
    assert embeddings["details"]["dropped"] == 1


async def test_health_reports_no_loss_for_a_stand_in_store():
    """A test double's auto-created attributes are not a reported loss."""
    from unittest.mock import MagicMock

    health = await check_conversation_embeddings(_agent_holding(MagicMock()))
    assert health["status"] == "pass"
    health = await check_conversation_embeddings(SimpleNamespace())
    assert health["status"] == "pass"


# ------------------------------------------------------------- PostgreSQL


async def _run_no_schema_ddl(db):
    """Schema initializer for a connection that must not boot its schema."""


@dataclass(frozen=True)
class _PostgresSchema:
    db: AsyncDatabase
    vector_schema: str

    async def size_embedding_column(self, dim: int) -> None:
        """Declare ``embedding_vec`` as ``vector(dim)``, whatever the boot made.

        The boot's ``embedding_vec`` and ``embedding_profile_id`` migrations
        probe ``information_schema`` without a schema filter, so on a shared
        database either can skip this schema because another one has the
        column. Both are therefore created here.
        """
        await self.db.execute(
            "ALTER TABLE conversation_history DROP COLUMN IF EXISTS embedding_vec",
            (),
        )
        await self.db.execute(
            "ALTER TABLE conversation_history ADD COLUMN embedding_vec "
            f'"{self.vector_schema}".vector({dim})',
            (),
        )
        await self.db.execute(
            "ALTER TABLE conversation_history "
            "ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT",
            (),
        )


@asynccontextmanager
async def _own_postgres_schema(url):
    """Boot the core schema into a fresh schema on *url*'s database."""
    admin = await AsyncDatabase.postgres(url, schema_initializer=_run_no_schema_ddl)
    try:
        vector_schema = await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "embedding_drops") as schema:
            boot = await AsyncDatabase.postgres(with_search_path(url, schema))
            await boot.close()
            db = await AsyncDatabase.postgres(
                with_search_path(url, quoted_search_path(schema, vector_schema)),
                schema_initializer=_run_no_schema_ddl,
            )
            try:
                yield _PostgresSchema(db, vector_schema)
            finally:
                await db.close()
    finally:
        await admin.close()


# Parametrized with "postgres" so the #3381 guard fails these cases, rather
# than letting them skip, in a job that provides PostgreSQL.
@pytest.fixture(params=["postgres"])
def postgres_url(request):
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    return url


async def test_postgres_width_mismatch_is_an_error_at_startup_and_a_drop_per_write(
    postgres_url, monkeypatch, caplog
):
    """The production shape: ``vector(4)`` column, a 6-dimension profile, and a
    process that believes the column is 6 wide (the provider-derived width).

    Once the startup check has read the declared width, a write is checked
    against it rather than against the width the process believes in.
    """
    _configure_column_width(monkeypatch, 6)
    async with _own_postgres_schema(postgres_url) as case:
        await case.size_embedding_column(4)
        service = _FixedWidthEmbeddings(6)
        store = _store(case.db, service)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            mismatch = await store.check_embedding_column_dimensions()

        assert (mismatch.profile_id, mismatch.profile_dim, mismatch.column_dim) == (
            service.profile.profile_id, 6, 4,
        )
        assert store.embedding_column_mismatch is mismatch
        errors = _store_records(caplog, logging.ERROR)
        assert len(errors) == 1
        assert "vector(4)" in errors[0].getMessage()
        assert service.profile.profile_id in errors[0].getMessage()
        health = await check_conversation_embeddings(_agent_holding(store))
        assert health["status"] == "warn"
        assert health["details"]["misconfigured"] is True
        assert health["details"]["misconfiguration"] == {
            "profile_id": service.profile.profile_id,
            "profile_dim": 6,
            "column_dim": 4,
        }
        assert health["details"]["dropped"] == 0

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            await store.add_conversation(role="assistant", content="the heron nests")

        assert await _stored_turns(case.db, store) == [
            ("the heron nests", False, None)
        ]
        warnings = _store_records(caplog, logging.WARNING)
        assert [record.levelno for record in warnings] == [logging.WARNING]
        message = warnings[0].getMessage()
        assert "takes 4 dimensions" in message
        # Dropped against the width the startup check read, before an INSERT
        # pgvector would refuse.
        assert "declared vector(4)" in message
        assert not [
            r for r in _store_records(caplog, logging.INFO)
            if r.levelno == logging.INFO
        ]
        assert store.dropped_embedding_count == 1

        health = await check_conversation_embeddings(_agent_holding(store))
        assert health["details"]["dropped"] == 1
        assert health["details"]["misconfigured"] is True
        assert health["details"]["last_column_dim"] == 4
        json.dumps(health)


async def test_postgres_matching_width_is_silent_and_stores_the_vector(
    postgres_url, monkeypatch, caplog
):
    _configure_column_width(monkeypatch, 6)
    async with _own_postgres_schema(postgres_url) as case:
        await case.size_embedding_column(6)
        service = _FixedWidthEmbeddings(6)
        store = _store(case.db, service)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            assert await store.check_embedding_column_dimensions() is None
            await store.add_conversation(role="assistant", content="the heron nests")

        assert not _store_records(caplog, logging.WARNING)
        assert store.embedding_column_mismatch is None
        assert store.dropped_embedding_count == 0
        assert await _stored_turns(case.db, store) == [
            ("the heron nests", True, service.profile.profile_id)
        ]
        health = await check_conversation_embeddings(_agent_holding(store))
        assert health["status"] == "pass"


async def test_postgres_startup_check_is_silent_without_an_embedding_profile(
    postgres_url, caplog
):
    async with _own_postgres_schema(postgres_url) as case:
        await case.size_embedding_column(4)
        store = _store(case.db, None)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            assert await store.check_embedding_column_dimensions() is None

        assert not _store_records(caplog, logging.WARNING)
        assert store.embedding_column_mismatch is None




# Every PostgreSQL mechanism by which a 6-dimension embedding misses a
# ``vector(4)`` column. See ``_drop_every_postgres_vector``.
_POSTGRES_MECHANISMS = ["configured_width", "declared_width", "refused_by_database"]


async def _drop_every_postgres_vector(case, store, monkeypatch, caplog, mechanism):
    """Make every 6-dimension embedding drop against a ``vector(4)`` column.

    ``configured_width``: the process believes the column is 4 wide.
    ``declared_width``: production's shape. The process believes the column
    is 6 wide, as the provider does, and the startup check has read its
    declared width, so the store drops the vector before any INSERT.
    ``refused_by_database``: the same without the startup check, so pgvector
    refuses every vector-bearing INSERT and the store falls back to a row
    without one. Inside a transaction each refusal must not abort it.
    """
    await case.size_embedding_column(4)
    _configure_column_width(monkeypatch, 4 if mechanism == "configured_width" else 6)
    if mechanism == "declared_width":
        assert await store.check_embedding_column_dimensions() is not None
        caplog.clear()  # its ERROR is the startup check's, tested above


def _assert_dropped_by(store, mechanism) -> None:
    """The drop was decided where *mechanism* says, not by a later fallback."""
    error = store.last_dropped_embedding.error
    if mechanism == "configured_width":
        assert "configured column width is 4" in error
    elif mechanism == "declared_width":
        assert "declared vector(4)" in error
    else:
        assert "expected 4 dimensions, not 6" in error


@pytest.mark.parametrize("mechanism", _POSTGRES_MECHANISMS)
async def test_postgres_turn_its_transaction_rolls_back_counts_no_drop(
    postgres_url, monkeypatch, caplog, mechanism
):
    """The caller's transaction rolls back, so its drop is never counted."""
    async with _own_postgres_schema(postgres_url) as case:
        store = _store(case.db, _FixedWidthEmbeddings(6))
        await _drop_every_postgres_vector(case, store, monkeypatch, caplog, mechanism)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            with pytest.raises(TransactionError) as caught:
                async with case.db.transaction():
                    await store.add_conversation(role="user", content="rolled back")
                    raise _RollBack()

        assert isinstance(caught.value.__cause__, _RollBack)
        assert await _stored_turns(case.db, store) == []
        assert store.dropped_embedding_count == 0
        assert store.last_dropped_embedding is None
        assert not _store_records(caplog, logging.WARNING)

        # The same write outside a transaction commits on its own and counts.
        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            await store.add_conversation(role="user", content="committed")

        assert await _stored_turns(case.db, store) == [("committed", False, None)]
        assert store.dropped_embedding_count == 1
        assert len(_store_records(caplog, logging.WARNING)) == 1
        _assert_dropped_by(store, mechanism)


@pytest.mark.parametrize("mechanism", _POSTGRES_MECHANISMS)
async def test_postgres_turn_its_transaction_commits_counts_one_drop_after_the_commit(
    postgres_url, monkeypatch, caplog, mechanism
):
    """The reviewer's case: a mismatched column inside a caller's transaction.

    A vector pgvector refuses aborts the transaction it runs in. Before the
    store checked the declared width and confined each attempt to a
    savepoint, the fallback INSERT then failed too, so the turn and the
    caller's own writes were lost and no drop was reported.
    """
    async with _own_postgres_schema(postgres_url) as case:
        await case.db.execute(
            "CREATE TABLE caller_writes (note TEXT NOT NULL)", (),
        )
        store = _store(case.db, _FixedWidthEmbeddings(6))
        await _drop_every_postgres_vector(case, store, monkeypatch, caplog, mechanism)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            async with case.db.transaction():
                await case.db.execute(
                    "INSERT INTO caller_writes (note) VALUES (?)", ("before",),
                )
                await store.add_conversation(role="user", content="committed")
                await case.db.execute(
                    "INSERT INTO caller_writes (note) VALUES (?)", ("after",),
                )
                assert store.dropped_embedding_count == 0
                assert not _store_records(caplog, logging.WARNING)

        assert await _stored_turns(case.db, store) == [("committed", False, None)]
        notes = await case.db.fetchall(
            "SELECT note FROM caller_writes ORDER BY note", (),
        )
        assert [note for (note,) in notes] == ["after", "before"]
        assert store.dropped_embedding_count == 1
        assert len(_store_records(caplog, logging.WARNING)) == 1
        _assert_dropped_by(store, mechanism)


@asynccontextmanager
async def _fenced_postgres_storage(url, tmp_path):
    """``_storage_with_one_assertion`` on a fresh schema of *url*'s database."""
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    admin = await AsyncDatabase.postgres(url, schema_initializer=_run_no_schema_ddl)
    try:
        vector_schema = await pgvector_schema(admin)
        async with disposable_postgres_schema(admin, "embedding_drops_fence") as schema:
            dsn = with_search_path(url, quoted_search_path(schema, vector_schema))

            def open_postgres(agent_id, capability):
                return AsyncStorage(
                    backend=PostgresBackend(dsn),
                    agent_id=agent_id,
                    _assertion_tenant_capability=capability,
                )

            async with _storage_with_one_assertion(tmp_path, open_postgres) as (
                storage, metadata,
            ):
                yield _PostgresSchema(storage.db, vector_schema), storage, metadata
    finally:
        await admin.close()


@pytest.mark.parametrize("mechanism", _POSTGRES_MECHANISMS)
async def test_postgres_fenced_turn_counts_one_drop_after_the_fence_commits(
    postgres_url, tmp_path, monkeypatch, caplog, mechanism
):
    """``AsyncStorage``'s semantic-recall fence owns the transaction here."""
    async with _fenced_postgres_storage(postgres_url, tmp_path) as (
        case, storage, metadata,
    ):
        store = storage.conversation
        service = _FixedWidthEmbeddings(6)
        store._lazy_embedding_service = lambda: service
        await _drop_every_postgres_vector(case, store, monkeypatch, caplog, mechanism)
        seen = _observe_persist(store, monkeypatch)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            await storage.add_conversation(
                "assistant", "the heron nests", metadata=metadata
            )

        # Inside the fence the row was written but not committed: nothing counted.
        assert seen == [(0, True)]
        assert await _stored_turns(storage.db, store) == [
            ("the heron nests", False, None)
        ]
        stored_metadata = json.loads(await storage.db.fetchval(
            "SELECT metadata FROM conversation_history WHERE agent_id = ?",
            (store.agent_id,),
        ))
        # The fence admitted the derivative; an excluded one drops nothing.
        assert "excluded_from_context" not in stored_metadata
        assert store.dropped_embedding_count == 1
        assert store.last_dropped_embedding.profile_id == service.profile.profile_id
        assert len(_store_records(caplog, logging.WARNING)) == 1
        _assert_dropped_by(store, mechanism)


@pytest.mark.parametrize(
    ("embedded", "profile_column"),
    [(True, True), (True, False), (False, True)],
    ids=["embedded", "embedded-without-profile-column", "unembedded"],
)
async def test_postgres_schema_without_lexical_columns_falls_back_inside_a_transaction(
    postgres_url, monkeypatch, caplog, embedded, profile_column
):
    """The schema-shape fallbacks share the vector retries' savepoints.

    Each first INSERT names columns an unmigrated schema lacks. Inside a
    caller's transaction that refusal must not abort it, or the fallback
    INSERT, and the turn, fails with it. Without ``embedding_profile_id``
    too, the vector-and-profile fallback is refused as well, and the vector
    lands without its profile stamp.
    """
    _configure_column_width(monkeypatch, 6)
    async with _own_postgres_schema(postgres_url) as case:
        await case.size_embedding_column(6)
        dropped_columns = ["lexical_index_id", "lexical_index_version"]
        if not profile_column:
            dropped_columns.append("embedding_profile_id")
        await case.db.execute(
            "ALTER TABLE conversation_history "
            + ", ".join(f"DROP COLUMN {column}" for column in dropped_columns),
            (),
        )
        service = _FixedWidthEmbeddings(6) if embedded else None
        store = _store(case.db, service)

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            async with case.db.transaction():
                await store.add_conversation(role="user", content="unmigrated")

        rows = await case.db.fetchall(
            "SELECT embedding_vec IS NOT NULL FROM conversation_history "
            "WHERE agent_id = ?",
            (store.agent_id,),
        )
        assert [bool(has_vector) for (has_vector,) in rows] == [embedded]
        if profile_column:
            expected_profile = service.profile.profile_id if embedded else None
            assert await _stored_turns(case.db, store) == [
                ("unmigrated", embedded, expected_profile)
            ]
        assert store.dropped_embedding_count == 0
        assert not _store_records(caplog, logging.WARNING)


async def test_postgres_best_effort_writes_after_a_dropped_vector_keep_the_transaction(
    postgres_url, monkeypatch, caplog
):
    """After the vector is refused, two best-effort writes follow inside the
    caller's transaction: discarding the turn's now-unreferenced lexical tokens,
    and registering its embedding profile. Either one failing used to abort the
    transaction, and COMMIT then rolled the turn back without an error while
    its drop was still counted.

    The cleanup is confined to a savepoint; the registry write is skipped,
    because the row it would describe carries no profile id.
    """
    from kestrel_sovereign.storage.sqla import embedding_profile

    # The registry write is cached per process; start with it unwritten.
    monkeypatch.setattr(embedding_profile, "_PROFILE_UPSERT_CACHE", set())
    async with _own_postgres_schema(postgres_url) as case:
        store = _store(case.db, _FixedWidthEmbeddings(6))
        await _drop_every_postgres_vector(
            case, store, monkeypatch, caplog, "refused_by_database"
        )
        # This schema is the test's own, so its triggers touch nothing shared.
        await case.db.execute(
            "CREATE FUNCTION refuse_best_effort_write() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN "
            "RAISE EXCEPTION 'best-effort write refused'; END $$",
            (),
        )
        for table, event in (
            ("conversation_lexical_tokens", "DELETE"),
            ("embedding_profiles", "INSERT"),
        ):
            await case.db.execute(
                f"CREATE TRIGGER refuse_{table} BEFORE {event} ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION refuse_best_effort_write()",
                (),
            )

        with caplog.at_level(logging.DEBUG, logger=STORE_LOGGER):
            async with case.db.transaction():
                await store.add_conversation(
                    role="user", content="the heron nests by the river"
                )

        assert await _stored_turns(case.db, store) == [
            ("the heron nests by the river", False, None)
        ]
        cleanup_failures = [
            r for r in _store_records(caplog, logging.DEBUG)
            if "Could not clean orphan lexical tokens" in r.getMessage()
        ]
        assert len(cleanup_failures) == 1
        assert "best-effort write refused" in cleanup_failures[0].getMessage()
        assert store.dropped_embedding_count == 1
        assert len(_store_records(caplog, logging.WARNING)) == 1
