"""A conversation write that drops its embedding is reported, not silent (#2913).

Production wrote every turn of a PostgreSQL agent without its vector: the
``conversation_history.embedding_vec`` column was ``vector(768)`` and the
profile produced 1536 dimensions. Each turn persisted and logged three INFO
lines. The store now counts each dropped embedding, warns (rate-limited), and
surfaces the count at ``/health/detailed``; at startup it compares the profile
with the column's declared width once and reports a disagreement as an ERROR
and a health misconfiguration, without refusing to boot.

A drop is counted only once its turn has committed: a turn whose transaction
rolls back never persisted, and the count and WARNING cannot be taken back.

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


_ONTOLOGY = OntologyRef(
    "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
    "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
    "semantic-kb-v1",
)


@pytest.fixture
async def fenced_storage(tmp_path):
    """Storage bound to an assertion tenant, holding one current assertion.

    A conversation write that names that assertion as a semantic-recall
    dependency is persisted inside ``AsyncStorage``'s lifecycle fence: a
    transaction ``AsyncStorage.add_conversation`` owns and commits itself.
    Yields the storage and the write's metadata.
    """
    identity_dir = tmp_path / "identity"
    credentials = await create_kestrel_identity_async(
        str(identity_dir), identity_method="did:pkh", agent_name="Embedding drops",
    )
    key_id = f"kestrel_{credentials.agent_did.rsplit(':', 1)[-1]}"
    identity = load_agent_identity(key_id, identity_dir)
    storage = AsyncStorage(
        ":memory:",
        agent_id=credentials.agent_did,
        _assertion_tenant_capability=_resolve_authenticated_agent_assertion_capability(
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
    """``AsyncStorage`` owns the transaction, so it publishes after the commit."""
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
    process that believes the column is 6 wide (the provider-derived width)."""
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
        assert "expected 4 dimensions, not 6" in message
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


async def test_postgres_turn_its_transaction_rolls_back_counts_no_drop(
    postgres_url, monkeypatch, caplog
):
    """PostgreSQL reports the caller's open transaction, so nothing is published.

    Only the configured-width drop is reachable here: inside a PostgreSQL
    transaction a refused INSERT aborts the transaction, so the fallback
    chain raises instead of landing a row without its vector.
    """
    _configure_column_width(monkeypatch, 4)
    async with _own_postgres_schema(postgres_url) as case:
        await case.size_embedding_column(4)
        store = _store(case.db, _FixedWidthEmbeddings(6))

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
