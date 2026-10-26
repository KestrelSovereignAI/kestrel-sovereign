"""A durable, audited SOUL revision path, and honest reporting of file-only edits (#3779).

The canonical SOUL is the current encrypted ``agent.soul.markdown`` version;
``SOUL.md`` is only its seed and cache, and every boot serves canonical over
it. These tests pin that a Sovereign revision survives a restart, that a
file-only edit loaded by ``bootstrap_reload`` is reported as not durable (and
does revert, loudly), and that the volatile-privacy refusal (#2672) holds.
A rename renames the canonical SOUL and never publishes over a revision that
landed while it ran (#3813).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from kestrel_sdk.storage.database.interface import QueryError
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.bootstrap import service as service_module
from kestrel_sovereign.bootstrap.service import (
    BootstrapService,
    SoulRevisionOutcome,
    _next_boot_after_failed_revision,
)
from kestrel_sovereign.bootstrap.soul_durability import (
    SoulFileState,
    SoulFileStatus,
    inspect_soul_file,
    soul_sha256,
)
from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.features.bootstrap import feature as feature_module
from kestrel_sovereign.features.bootstrap.feature import (
    BootstrapFeature,
    _update_soul_name,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage, SOUL_MARKDOWN_RESOURCE_TYPE
from kestrel_sovereign.storage import async_graph_store as graph_store_module
from kestrel_sovereign.storage.agent_resource_store import (
    ExpectedCurrentVersion,
    ResourceVersionConflictError,
)
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

AGENT_ID = "did:test:soul-revision"
ORIGINAL = "# SOUL.md - You Are Claw\n\n## Role\nAccept work orders from did:pkh only.\n"
REVISED = (
    "# SOUL.md - You Are Claw\n\n## Role\n"
    "Accept work orders from did:pkh and did:web.\n"
)
VOLATILE_MODES = [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED]


@pytest.fixture(autouse=True)
def data_key(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "test-soul-revision-key")


@pytest.fixture
def agent_dir(tmp_path):
    path = tmp_path / "agent_data" / "claw"
    path.mkdir(parents=True)
    return path


async def _open(tmp_path, mode=PrivacyMode.NORMAL):
    raw = AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID)
    await raw.initialize()
    return raw, PrivacyEnforcingStorage(raw, mode)


def _service(raw, storage, agent_dir) -> BootstrapService:
    return BootstrapService(
        db=raw.db,
        agent_id=AGENT_ID,
        agent_name="Claw",
        llm_service=None,
        agent_data_path=str(agent_dir),
        storage=storage,
    )


async def _boot(tmp_path, agent_dir) -> tuple[AsyncStorage, ContextBuilder]:
    """What a restart does: open the store and run the boot-time SOUL load."""
    raw, storage = await _open(tmp_path)
    builder = ContextBuilder(storage, agent_data_path=str(agent_dir))
    runtime = SimpleNamespace(agent_id=AGENT_ID, storage=storage, context_builder=builder)
    await KestrelAgent._load_or_promote_soul_resource(runtime, str(agent_dir))
    return raw, builder


async def _seed_canonical(tmp_path, agent_dir) -> None:
    """An agent past inception: canonical v1 and a matching SOUL.md."""
    raw, storage = await _open(tmp_path)
    try:
        assert await _service(raw, storage, agent_dir).save_soul_md(ORIGINAL) is True
    finally:
        await raw.close()


async def _version_count(raw) -> int:
    rows = await raw.db.fetchall(
        "SELECT version FROM agent_identity_resources WHERE agent_id = ? AND resource_type = ?",
        (AGENT_ID, SOUL_MARKDOWN_RESOURCE_TYPE),
    )
    return len(rows)


# ---------------------------------------------------------------------------
# The revision path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revision_survives_a_restart_and_records_who_and_why(tmp_path, agent_dir):
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")

    raw, storage = await _open(tmp_path)
    try:
        outcome = await _service(raw, storage, agent_dir).revise_soul(
            reason="Sovereign-approved: accept Emma's signing DID",
            revised_by=CallerContext.sovereign(),
        )
    finally:
        await raw.close()

    assert outcome.revised is True, outcome.message
    assert (outcome.previous_version, outcome.version) == (1, 2)
    assert outcome.content_sha256 == soul_sha256(REVISED)

    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        assert builder._soul_content == REVISED
        current = await raw.get_current_agent_resource()
        assert current.version == 2
        provenance = current.provenance
        assert provenance["reason"] == "Sovereign-approved: accept Emma's signing DID"
        assert provenance["revised_by"] == {
            "role": "sovereign", "auth_method": "api_key", "identity": "api_key",
        }
        assert provenance["created_by"] == "sovereign:api_key"
        assert provenance["source"] == f"soul_revision:{agent_dir / 'SOUL.md'}"
        assert provenance["previous_version"] == 1
        assert provenance["previous_content_hash"] == soul_sha256(ORIGINAL)
        status = await inspect_soul_file(raw, agent_dir)
        assert status.state is SoulFileState.IN_SYNC
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_revision_of_an_unchanged_file_writes_no_version(tmp_path, agent_dir):
    await _seed_canonical(tmp_path, agent_dir)
    raw, storage = await _open(tmp_path)
    try:
        outcome = await _service(raw, storage, agent_dir).revise_soul(
            reason="no-op", revised_by=CallerContext.sovereign()
        )
        assert outcome.revised is False
        assert outcome.unchanged is True
        assert outcome.version == 1
        assert await _version_count(raw) == 1
    finally:
        await raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller",
    [None, CallerContext.authenticated("someone@example.com"), CallerContext.anonymous()],
)
async def test_revision_refuses_a_non_sovereign_caller(tmp_path, agent_dir, caller):
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    raw, storage = await _open(tmp_path)
    try:
        with pytest.raises(PermissionError):
            await _service(raw, storage, agent_dir).revise_soul(
                reason="self-edit", revised_by=caller
            )
        assert await _version_count(raw) == 1
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_revision_requires_a_reason(tmp_path, agent_dir):
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    raw, storage = await _open(tmp_path)
    try:
        with pytest.raises(ValueError):
            await _service(raw, storage, agent_dir).revise_soul(
                reason="   ", revised_by=CallerContext.sovereign()
            )
        assert await _version_count(raw) == 1
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_revision_reports_a_failed_promotion_instead_of_succeeding(
    tmp_path, agent_dir, monkeypatch
):
    """``save_soul_md`` treats promotion as best-effort; a revision must not."""
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    raw, storage = await _open(tmp_path)
    try:
        service = _service(raw, storage, agent_dir)
        monkeypatch.setattr(
            raw, "promote_soul_seed", AsyncMock(side_effect=RuntimeError("disk full"))
        )
        outcome = await service.revise_soul(
            reason="fix role", revised_by=CallerContext.sovereign()
        )
        assert outcome.revised is False
        assert "next restart serves canonical v1" in outcome.message
        assert (await raw.get_current_agent_resource()).content == ORIGINAL

        # The inception writers keep their best-effort contract.
        assert await service.save_soul_md(REVISED) is True
    finally:
        await raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", VOLATILE_MODES)
async def test_revision_is_refused_in_a_volatile_privacy_mode(tmp_path, agent_dir, mode):
    """#2672: a volatile mode persists no SOUL, revision included."""
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    raw, storage = await _open(tmp_path, mode)
    try:
        node_before = (await raw.get_node(f"{AGENT_ID}#soul")).properties
        outcome = await _service(raw, storage, agent_dir).revise_soul(
            reason="fix role", revised_by=CallerContext.sovereign()
        )
        assert outcome.revised is False
        assert outcome.skipped_privacy is True
        assert await _version_count(raw) == 1
        assert (await raw.get_current_agent_resource()).content == ORIGINAL
        assert (await raw.get_node(f"{AGENT_ID}#soul")).properties == node_before
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == REVISED
    finally:
        await raw.close()


# ---------------------------------------------------------------------------
# Canonical publication is atomic, on both engines
# ---------------------------------------------------------------------------


@pytest.fixture(params=["sqlite", "postgres"])
async def engine(request, tmp_path):
    """Storage for one DID of this test's own; each ``make()`` is a fresh open."""
    handle = SimpleNamespace(
        did=f"did:test:soul-revision-{uuid4()}", name=request.param
    )
    # ``make`` reads ``handle.did`` when called, so a test may pick its DID.
    if request.param == "postgres":
        url = os.environ.get("TEST_POSTGRES_URL")
        if not url:
            pytest.skip("TEST_POSTGRES_URL is not set")

        def make():
            return AsyncStorage(backend="postgres", dsn=url, agent_id=handle.did)
    else:
        path = str(tmp_path / "kestrel.db")

        def make():
            return AsyncStorage(path, agent_id=handle.did)

    handle.make = make
    yield handle
    # A PostgreSQL database outlives the run: remove what this test wrote.
    raw = make()
    await raw.initialize()
    try:
        await raw.delete_node(f"{handle.did}#soul")
        await raw.delete_node(handle.did)
        await raw.db.execute(
            "DELETE FROM agent_identity_resources WHERE agent_id = ?", (handle.did,)
        )
        await raw.db.execute(
            "DELETE FROM agent_metadata WHERE agent_id = ?", (handle.did,)
        )
    finally:
        await raw.close()


async def _engine_open(engine) -> tuple[AsyncStorage, PrivacyEnforcingStorage]:
    raw = engine.make()
    await raw.initialize()
    return raw, PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)


def _engine_service(engine, raw, storage, agent_dir) -> BootstrapService:
    return BootstrapService(
        db=raw.db,
        agent_id=engine.did,
        agent_name="Claw",
        llm_service=None,
        agent_data_path=str(agent_dir),
        storage=storage,
    )


async def _soul_rows(raw, did) -> list[tuple[int, bool]]:
    rows = await raw.db.fetchall(
        "SELECT version, is_current FROM agent_identity_resources "
        "WHERE agent_id = ? AND resource_type = ? ORDER BY version",
        (did, SOUL_MARKDOWN_RESOURCE_TYPE),
    )
    return [(int(version), bool(current)) for version, current in rows]


async def _seed_engine(engine, agent_dir):
    """Canonical v1 and a matching SOUL.md, then a Sovereign's edit on disk."""
    raw, storage = await _engine_open(engine)
    try:
        assert await _engine_service(engine, raw, storage, agent_dir).save_soul_md(
            ORIGINAL
        ) is True
        v1 = await raw.get_current_agent_resource()
        node = await raw.get_node(f"{engine.did}#soul")
    finally:
        await raw.close()
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    return v1, node.properties


async def _assert_restart_keeps_v1(engine, agent_dir, v1, node_before) -> None:
    """The next boot serves v1, as v1, and does not adopt the file as a seed."""
    raw, storage = await _engine_open(engine)
    try:
        builder = ContextBuilder(storage, agent_data_path=str(agent_dir))
        runtime = SimpleNamespace(
            agent_id=engine.did, storage=storage, context_builder=builder
        )
        await KestrelAgent._load_or_promote_soul_resource(runtime, str(agent_dir))
        assert builder._soul_content == ORIGINAL
        assert await _soul_rows(raw, engine.did) == [(1, True)]
        current = await raw.get_current_agent_resource()
        assert current.provenance == v1.provenance
        assert "reason" not in current.provenance
        assert (await raw.get_node(f"{engine.did}#soul")).properties == node_before
        status = await inspect_soul_file(raw, agent_dir)
        assert status.state is SoulFileState.FILE_DIFFERS
        assert status.canonical_version == 1
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_revision_that_fails_after_deselecting_v1_leaves_v1_current(
    engine, agent_dir, monkeypatch
):
    """The replacement insert fails after the previous selection was cleared.

    Separately committed, that left v1 stored with ``is_current = 0`` and no
    version current, and the next boot promoted the edited file as v2 with
    none of the revision's reason or attribution. One transaction rolls the
    deselection back with the insert.
    """
    v1, node_before = await _seed_engine(engine, agent_dir)

    raw, storage = await _engine_open(engine)
    try:
        statements: list[str] = []
        real_execute = raw.db.execute

        async def failing_insert(sql, params=()):
            text = " ".join(sql.split())
            statements.append(text)
            if text.startswith("INSERT INTO agent_identity_resources"):
                raise QueryError("injected: replacement insert rejected")
            return await real_execute(sql, params)

        monkeypatch.setattr(raw.db, "execute", failing_insert)
        outcome = await _engine_service(engine, raw, storage, agent_dir).revise_soul(
            reason="accept Emma's signing DID", revised_by=CallerContext.sovereign()
        )
        monkeypatch.setattr(raw.db, "execute", real_execute)

        # The failure landed between clearing the selection and the insert.
        writes = [s for s in statements if "agent_identity_resources" in s]
        assert writes[0].startswith("UPDATE agent_identity_resources SET is_current = 0")
        assert writes[1].startswith("INSERT INTO agent_identity_resources")
        assert outcome.revised is False
        assert outcome.previous_version == 1
        assert "the next restart serves canonical v1" in outcome.message
        assert await _soul_rows(raw, engine.did) == [(1, True)]
    finally:
        await raw.close()

    await _assert_restart_keeps_v1(engine, agent_dir, v1, node_before)


@pytest.mark.asyncio
async def test_the_resource_store_alone_allocates_and_selects_atomically(
    engine, monkeypatch
):
    """A direct store caller gets the same unit, without the facade's transaction."""
    raw, _ = await _engine_open(engine)
    try:
        store = raw.agent_resources
        await store.promote_soul_seed(ORIGINAL, created_by=engine.did)
        real_execute = raw.db.execute

        async def failing_insert(sql, params=()):
            if " ".join(sql.split()).startswith("INSERT INTO agent_identity_resources"):
                raise QueryError("injected: replacement insert rejected")
            return await real_execute(sql, params)

        monkeypatch.setattr(raw.db, "execute", failing_insert)
        with pytest.raises(Exception, match="replacement insert rejected"):
            await store.promote_soul_seed(REVISED, created_by=engine.did)
        monkeypatch.setattr(raw.db, "execute", real_execute)

        assert await _soul_rows(raw, engine.did) == [(1, True)]
        assert (await store.get_current()).content == ORIGINAL
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_the_generic_resource_door_publishes_a_soul_atomically_too(
    engine, monkeypatch
):
    """``create_agent_resource_version`` for a SOUL is the same one unit."""
    raw, _ = await _engine_open(engine)
    try:
        await raw.promote_soul_seed(ORIGINAL, created_by=engine.did)
        real_add_node = raw.add_node

        async def failing_soul_node(node, **kwargs):
            if node.node_id == f"{engine.did}#soul":
                raise QueryError("injected: graph reference rejected")
            return await real_add_node(node, **kwargs)

        monkeypatch.setattr(raw, "add_node", failing_soul_node)
        with pytest.raises(Exception, match="graph reference rejected"):
            await raw.create_agent_resource_version(
                SOUL_MARKDOWN_RESOURCE_TYPE,
                REVISED,
                created_by=engine.did,
                source="operator",
            )
        monkeypatch.setattr(raw, "add_node", real_add_node)

        assert await _soul_rows(raw, engine.did) == [(1, True)]
        assert (await raw.get_node(f"{engine.did}#soul")).properties["current_version"] == 1
    finally:
        await raw.close()


def _reservation_key(node_id: str) -> int:
    """The PostgreSQL advisory key a graph write reserves for ``node_id``."""
    shard = int.from_bytes(
        hashlib.sha256(f"kestrel:graph-node:{node_id}".encode()).digest()[:8], "big"
    ) % graph_store_module._GRAPH_NODE_RESERVATION_SHARDS
    return graph_store_module._GRAPH_NODE_RESERVATION_KEY_BY_SHARD[shard]


def _did_whose_soul_shard_sorts(order: str) -> str:
    for _ in range(10000):
        did = f"did:test:soul-revision-{uuid4()}"
        agent_key, soul_key = _reservation_key(did), _reservation_key(f"{did}#soul")
        if (agent_key < soul_key) == (order == "agent_first") and agent_key != soul_key:
            return did
    raise AssertionError("no DID found with the requested shard order")


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["agent_first", "soul_first"])
async def test_the_soul_graph_link_is_published_whatever_the_shard_order(
    engine, agent_dir, order
):
    """The SOUL node and its agent edge are written in one transaction.

    PostgreSQL refuses a nested graph write whose reservation shard sorts
    before one the transaction already holds. Writing the SOUL node first, an
    agent DID whose shard sorts lower failed the edge, and the reference writer
    swallowed that as "agent node missing": the edge silently went unwritten.
    """
    engine.did = _did_whose_soul_shard_sorts(order)
    raw, storage = await _engine_open(engine)
    try:
        await raw.add_node(
            GraphNode(node_id=engine.did, node_type="agent", label="Claw", properties={})
        )
        assert await _engine_service(engine, raw, storage, agent_dir).save_soul_md(
            ORIGINAL
        ) is True
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        outcome = await _engine_service(engine, raw, storage, agent_dir).revise_soul(
            reason="accept Emma's signing DID", revised_by=CallerContext.sovereign()
        )
        assert outcome.revised is True, outcome.message

        edges = [
            (edge.target_id, edge.label) for edge in await raw.get_edges_from(engine.did)
        ]
        assert edges == [(f"{engine.did}#soul", "has_private_identity_resource")]
        node = await raw.get_node(f"{engine.did}#soul")
        assert node.properties["current_version"] == 2
    finally:
        await raw.close()


@pytest.mark.parametrize(
    "status, expected",
    [
        (SoulFileStatus(SoulFileState.FILE_DIFFERS, "p", canonical_version=3),
         "the next restart serves canonical v3."),
        (SoulFileStatus(SoulFileState.FILE_MISSING, "p", canonical_version=3),
         "the next restart serves canonical v3."),
        (SoulFileStatus(SoulFileState.NO_CANONICAL, "p"),
         "no canonical SOUL is current, so the next restart promotes SOUL.md as a "
         "seed, without this revision's reason or attribution."),
        (SoulFileStatus(SoulFileState.IN_SYNC, "p", canonical_version=4),
         "the canonical SOUL v4 matches it although the write reported a failure; "
         "see the agent log."),
        (SoulFileStatus(SoulFileState.UNVERIFIED, "p", error="db gone"),
         "what the next restart serves is unknown: db gone"),
    ],
)
def test_a_failed_revision_reports_what_the_next_boot_serves(status, expected):
    """Read back, not assumed: each canonical state gets its own report."""
    assert _next_boot_after_failed_revision(status) == expected


THIRD = REVISED + "\n## Reports\nSource every report.\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["storage", "resource_store"])
async def test_concurrent_promotions_on_postgresql_allocate_distinct_versions(
    engine, monkeypatch, door
):
    """Two writers racing for the next version both land, one after the other.

    The first holds its publication open past its writes; the second must wait
    for it, then allocate v3. Read concurrently instead, both see v1 as the
    latest and the second fails on the unique version. The storage facade also
    serializes on the SOUL's graph write set; the resource store, used alone,
    has only its own version lock.
    """
    if engine.name != "postgres":
        pytest.skip("SQLite serializes writers through its writer slot; no race to lose")
    first, _ = await _engine_open(engine)
    second, _ = await _engine_open(engine)
    observer, _ = await _engine_open(engine)
    written, release = asyncio.Event(), asyncio.Event()
    tasks: list[asyncio.Task] = []

    def writer(handle):
        return handle if door == "storage" else handle.agent_resources

    try:
        await writer(first).promote_soul_seed(ORIGINAL, created_by=engine.did)
        # The last step inside each door's transaction, after every write.
        held_target, held_name = (
            (first, "add_node") if door == "storage"
            else (first.agent_resources, "get_version")
        )
        real_step = getattr(held_target, held_name)

        async def held_publication(*args, **kwargs):
            result = await real_step(*args, **kwargs)
            if not written.is_set():
                written.set()
                await release.wait()
            return result

        monkeypatch.setattr(held_target, held_name, held_publication)
        tasks.append(asyncio.create_task(
            writer(first).promote_soul_seed(REVISED, created_by=engine.did)
        ))
        await asyncio.wait_for(written.wait(), timeout=10)
        tasks.append(asyncio.create_task(
            writer(second).promote_soul_seed(THIRD, created_by=engine.did)
        ))

        async def racer_waits_on_a_lock():
            while True:
                row = await observer.db.fetchone(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock' AND pid <> pg_backend_pid()"
                )
                if row and row[0]:
                    return
                await asyncio.sleep(0.02)

        await asyncio.wait_for(racer_waits_on_a_lock(), timeout=10)
        release.set()
        revised, third = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

        assert (revised.version, third.version) == (2, 3)
        assert await _soul_rows(observer, engine.did) == [(1, False), (2, False), (3, True)]
        assert (await observer.get_current_agent_resource()).content == THIRD
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for handle in (first, second, observer):
            await handle.close()


@pytest.mark.asyncio
async def test_concurrent_revisions_record_the_version_each_one_replaced(
    engine, agent_dir, monkeypatch
):
    """Both revisions inspect v1; the one stored second replaced v2, not v1.

    The store reads the version it replaces under its version lock. Taken
    from the revision's own earlier inspection, v3 claimed to replace v1.
    """
    await _seed_engine(engine, agent_dir)
    first_raw, first_storage = await _engine_open(engine)
    second_raw, second_storage = await _engine_open(engine)
    real_inspect = service_module.inspect_soul_file
    interleaved: list = []

    async def inspect_then_let_another_revision_land(storage, agent_data_path):
        status = await real_inspect(storage, agent_data_path)
        if storage is first_storage and not interleaved:
            interleaved.append(status)
            (agent_dir / "SOUL.md").write_text(THIRD, encoding="utf-8")
            interleaved.append(
                await _engine_service(
                    engine, second_raw, second_storage, agent_dir
                ).revise_soul(
                    reason="second operator", revised_by=CallerContext.sovereign()
                )
            )
        return status

    monkeypatch.setattr(
        service_module, "inspect_soul_file", inspect_then_let_another_revision_land
    )
    try:
        outcome = await _engine_service(
            engine, first_raw, first_storage, agent_dir
        ).revise_soul(reason="first operator", revised_by=CallerContext.sovereign())
    finally:
        await first_raw.close()
        await second_raw.close()

    inspected, other = interleaved
    assert (inspected.state, inspected.canonical_version) == (SoulFileState.FILE_DIFFERS, 1)
    assert other.revised is True, other.message
    assert (other.previous_version, other.version) == (1, 2)
    assert outcome.revised is True, outcome.message
    assert (outcome.previous_version, outcome.version) == (2, 3)
    assert "(was v2)" in outcome.message

    raw, _ = await _engine_open(engine)
    try:
        v2 = await raw.agent_resources.get_version(version=2)
        v3 = await raw.get_current_agent_resource()
        assert (v3.version, v3.content) == (3, REVISED)
        assert v3.provenance["reason"] == "first operator"
        assert v3.provenance["previous_version"] == 2
        assert v3.provenance["previous_content_hash"] == v2.content_hash
        assert v2.content_hash == soul_sha256(THIRD)
        assert v2.provenance["reason"] == "second operator"
        assert v2.provenance["previous_version"] == 1
        assert v2.provenance["previous_content_hash"] == soul_sha256(ORIGINAL)
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_the_store_records_the_current_version_a_new_one_replaced(engine):
    """Recorded for every writer, and it names the current version, not the latest."""
    raw, _ = await _engine_open(engine)
    try:
        store = raw.agent_resources
        first = await store.promote_soul_seed(ORIGINAL, created_by=engine.did)
        assert "previous_version" not in first.provenance
        assert "previous_content_hash" not in first.provenance

        draft = await store.create_version(
            SOUL_MARKDOWN_RESOURCE_TYPE,
            REVISED,
            created_by=engine.did,
            source="draft",
            make_current=False,
        )
        assert draft.version == 2
        assert "previous_version" not in draft.provenance

        third = await store.promote_soul_seed(THIRD, created_by=engine.did)
        assert third.version == 3
        assert third.provenance["previous_version"] == 1
        assert third.provenance["previous_content_hash"] == soul_sha256(ORIGINAL)
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_failed_graph_reference_rolls_the_new_version_back(
    engine, agent_dir, monkeypatch
):
    """The version and the graph reference to it are one publication."""
    v1, node_before = await _seed_engine(engine, agent_dir)

    raw, storage = await _engine_open(engine)
    try:
        real_add_node = raw.add_node
        refused: list[str] = []

        async def failing_soul_node(node, **kwargs):
            if node.node_id == f"{engine.did}#soul":
                refused.append(node.node_id)
                raise QueryError("injected: graph reference rejected")
            return await real_add_node(node, **kwargs)

        monkeypatch.setattr(raw, "add_node", failing_soul_node)
        outcome = await _engine_service(engine, raw, storage, agent_dir).revise_soul(
            reason="accept Emma's signing DID", revised_by=CallerContext.sovereign()
        )
        monkeypatch.setattr(raw, "add_node", real_add_node)

        assert refused == [f"{engine.did}#soul"]
        assert outcome.revised is False
        assert "the next restart serves canonical v1" in outcome.message
        assert await _soul_rows(raw, engine.did) == [(1, True)]
    finally:
        await raw.close()

    await _assert_restart_keeps_v1(engine, agent_dir, v1, node_before)


@pytest.mark.asyncio
async def test_a_failed_soul_edge_write_keeps_the_published_version(
    engine, agent_dir, monkeypatch
):
    """The agent edge is best-effort inside the unit; its failure costs only itself.

    The edge writer opens its own transaction scope, which nests as a
    savepoint on PostgreSQL. A statement failing inside it is undone alone;
    joined without one, it would abort the whole unit and the swallowed error
    would let the commit discard a version the revision reports as stored.
    """
    await _seed_engine(engine, agent_dir)
    raw, storage = await _engine_open(engine)
    try:
        attempted: list[tuple] = []

        async def failing_edge(*args, **kwargs):
            async with raw.db.transaction():
                attempted.append(args)
                await raw.db.execute("SELECT * FROM no_such_table_3779")

        monkeypatch.setattr(raw, "add_edge", failing_edge)
        outcome = await _engine_service(engine, raw, storage, agent_dir).revise_soul(
            reason="accept Emma's signing DID", revised_by=CallerContext.sovereign()
        )
        monkeypatch.undo()

        assert [args[:2] for args in attempted] == [(engine.did, f"{engine.did}#soul")]
        assert outcome.revised is True, outcome.message
        assert outcome.version == 2
    finally:
        await raw.close()

    raw, _ = await _engine_open(engine)
    try:
        assert await _soul_rows(raw, engine.did) == [(1, False), (2, True)]
        node = await raw.get_node(f"{engine.did}#soul")
        assert node.properties["current_version"] == 2
        assert node.properties["provenance"]["reason"] == "accept Emma's signing DID"
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_revision_on_either_engine_survives_a_restart(engine, agent_dir):
    """The atomic unit still commits: v2 is current, with its provenance."""
    await _seed_engine(engine, agent_dir)
    raw, storage = await _engine_open(engine)
    try:
        outcome = await _engine_service(engine, raw, storage, agent_dir).revise_soul(
            reason="accept Emma's signing DID", revised_by=CallerContext.sovereign()
        )
    finally:
        await raw.close()
    assert outcome.revised is True, outcome.message

    raw, storage = await _engine_open(engine)
    try:
        builder = ContextBuilder(storage, agent_data_path=str(agent_dir))
        runtime = SimpleNamespace(
            agent_id=engine.did, storage=storage, context_builder=builder
        )
        await KestrelAgent._load_or_promote_soul_resource(runtime, str(agent_dir))
        assert builder._soul_content == REVISED
        assert await _soul_rows(raw, engine.did) == [(1, False), (2, True)]
        current = await raw.get_current_agent_resource()
        assert current.provenance["reason"] == "accept Emma's signing DID"
        node = await raw.get_node(f"{engine.did}#soul")
        assert node.properties["current_version"] == 2
    finally:
        await raw.close()


# ---------------------------------------------------------------------------
# File-only edits: reported, not silently reverted
# ---------------------------------------------------------------------------


def _reload_feature(storage, builder, agent_dir) -> BootstrapFeature:
    agent = MagicMock()
    agent.storage = storage
    agent.context_builder = builder
    agent.bootstrap_service = SimpleNamespace(agent_data_path=str(agent_dir))
    return BootstrapFeature(agent)


@pytest.mark.asyncio
async def test_file_only_edit_plus_reload_is_reported_as_not_durable(
    tmp_path, agent_dir, caplog
):
    await _seed_canonical(tmp_path, agent_dir)
    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        result = await _reload_feature(builder.storage, builder, agent_dir).bootstrap_reload()

        assert builder._soul_content == REVISED  # live for this session
        assert result.status is ToolResultStatus.PARTIAL
        assert "live for this session only" in result.error
        assert "!revise-soul" in result.error
        assert result.data["soul_file"]["state"] == "file_differs"
        assert result.data["soul_file"]["file_survives_restart"] is False
        assert result.data["soul_file"]["canonical_version"] == 1
    finally:
        await raw.close()

    # The next boot serves canonical, and says so.
    with caplog.at_level(logging.WARNING):
        raw, builder = await _boot(tmp_path, agent_dir)
    try:
        assert builder._soul_content == ORIGINAL
        drift = [r for r in caplog.records if "SOUL file drift" in r.getMessage()]
        assert len(drift) == 1
        assert drift[0].levelno == logging.WARNING
        assert "differs from the canonical SOUL v1" in drift[0].getMessage()
        assert "!revise-soul" in drift[0].getMessage()
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_reload_of_the_canonical_file_is_ok_and_boot_is_quiet(
    tmp_path, agent_dir, caplog
):
    await _seed_canonical(tmp_path, agent_dir)
    with caplog.at_level(logging.WARNING):
        raw, builder = await _boot(tmp_path, agent_dir)
    try:
        assert not [r for r in caplog.records if "SOUL file drift" in r.getMessage()]
        result = await _reload_feature(builder.storage, builder, agent_dir).bootstrap_reload()
        assert result.status is ToolResultStatus.OK
        assert result.data["soul_file"]["state"] == "in_sync"
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_reload_context_command_reports_a_non_durable_soul(tmp_path, agent_dir):
    await _seed_canonical(tmp_path, agent_dir)
    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        handler = CommandHandler(
            SimpleNamespace(storage=builder.storage, context_builder=builder),
            task_manager=MagicMock(),
        )
        message = await handler.handle("!reload-context")
        assert "SOUL.md" in message
        assert "live for this session only" in message
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_bootstrap_status_warns_of_a_differing_file(tmp_path, agent_dir):
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    raw, storage = await _open(tmp_path)
    try:
        status = await _service(raw, storage, agent_dir).get_bootstrap_status()
        assert "file_differs" in status
        assert "next restart serves v1" in status
    finally:
        await raw.close()


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_soul_file_states(tmp_path, agent_dir):
    raw, storage = await _open(tmp_path)
    try:
        assert (await inspect_soul_file(storage, agent_dir)).state is SoulFileState.NO_CANONICAL
        await raw.promote_soul_seed(ORIGINAL, created_by=AGENT_ID)

        missing = await inspect_soul_file(storage, agent_dir)
        assert missing.state is SoulFileState.FILE_MISSING
        assert missing.file_survives_restart is False

        (agent_dir / "SOUL.md").write_text("   \n", encoding="utf-8")
        assert (await inspect_soul_file(storage, agent_dir)).state is SoulFileState.FILE_MISSING

        (agent_dir / "SOUL.md").write_text(ORIGINAL, encoding="utf-8")
        in_sync = await inspect_soul_file(storage, agent_dir)
        assert in_sync.state is SoulFileState.IN_SYNC
        assert in_sync.file_sha256 == in_sync.canonical_sha256 == soul_sha256(ORIGINAL)
        assert in_sync.reload_caveat() is None and in_sync.boot_warning() is None

        # A trailing newline is a difference: the boot serves canonical's bytes.
        (agent_dir / "SOUL.md").write_text(ORIGINAL + "\n", encoding="utf-8")
        assert (await inspect_soul_file(storage, agent_dir)).state is SoulFileState.FILE_DIFFERS
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_inspect_soul_file_reports_an_unreadable_side_as_unverified(agent_dir):
    failing = SimpleNamespace(
        get_current_agent_resource=AsyncMock(side_effect=RuntimeError("db gone"))
    )
    status = await inspect_soul_file(failing, agent_dir)
    assert status.state is SoulFileState.UNVERIFIED
    assert "db gone" in status.error
    assert status.file_survives_restart is False
    assert "db gone" in status.reload_caveat()

    not_text = SimpleNamespace(
        get_current_agent_resource=AsyncMock(
            return_value=SimpleNamespace(content=b"bytes", version=1)
        )
    )
    status = await inspect_soul_file(not_text, agent_dir)
    assert status.state is SoulFileState.UNVERIFIED
    assert "not text" in status.error

    (agent_dir / "SOUL.md").mkdir()  # a directory where the file should be
    status = await inspect_soul_file(SimpleNamespace(), agent_dir)
    assert status.state is SoulFileState.UNVERIFIED
    assert "SOUL.md read failed" in status.error


@pytest.mark.asyncio
async def test_extra_provenance_cannot_replace_store_written_fields(tmp_path):
    raw, _ = await _open(tmp_path)
    try:
        for field in (
            "created_by",
            "source",
            "created_at",
            "signature",
            "previous_version",
            "previous_content_hash",
        ):
            with pytest.raises(ValueError, match=field):
                await raw.promote_soul_seed(
                    ORIGINAL, created_by=AGENT_ID, provenance={field: "forged"}
                )
        assert await _version_count(raw) == 0
    finally:
        await raw.close()


# ---------------------------------------------------------------------------
# The !revise-soul command
# ---------------------------------------------------------------------------


def _command_agent(outcome: SoulRevisionOutcome):
    service = SimpleNamespace(revise_soul=AsyncMock(return_value=outcome))
    builder = SimpleNamespace(load_canonical_soul_resource=AsyncMock(return_value=True))
    agent = SimpleNamespace(bootstrap_service=service, context_builder=builder)
    return agent, service, builder


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller",
    [None, CallerContext.authenticated("someone@example.com"), CallerContext.a2a_transport()],
)
async def test_revise_soul_command_requires_sovereign_authority(caller):
    agent, service, _ = _command_agent(SoulRevisionOutcome(revised=True, message="done"))
    handler = CommandHandler(agent, task_manager=MagicMock())
    message = await handler.handle("!revise-soul fix role", caller=caller)
    assert "Unauthorized" in message
    service.revise_soul.assert_not_awaited()


@pytest.mark.asyncio
async def test_revise_soul_command_records_the_gated_caller_and_reloads():
    agent, service, builder = _command_agent(
        SoulRevisionOutcome(revised=True, message="SOUL.md is now canonical SOUL v2")
    )
    caller = CallerContext.sovereign()
    handler = CommandHandler(agent, task_manager=MagicMock())
    message = await handler.handle("!revise-soul  accept Emma's signing DID ", caller=caller)
    service.revise_soul.assert_awaited_once_with(
        reason="accept Emma's signing DID", revised_by=caller
    )
    builder.load_canonical_soul_resource.assert_awaited_once()
    assert message.startswith("✅")


@pytest.mark.asyncio
async def test_revise_soul_command_without_a_reason_shows_usage():
    agent, service, _ = _command_agent(SoulRevisionOutcome(revised=True, message="done"))
    handler = CommandHandler(agent, task_manager=MagicMock())
    message = await handler.handle("!revise-soul", caller=CallerContext.sovereign())
    assert message.startswith("Usage: !revise-soul <reason>")
    service.revise_soul.assert_not_awaited()


@pytest.mark.asyncio
async def test_revise_soul_command_reports_a_refusal_as_a_failure():
    agent, _, builder = _command_agent(
        SoulRevisionOutcome(revised=False, skipped_privacy=True, message="Not revised")
    )
    handler = CommandHandler(agent, task_manager=MagicMock())
    message = await handler.handle("!revise-soul fix", caller=CallerContext.sovereign())
    assert message == "❌ Not revised"
    builder.load_canonical_soul_resource.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_loads", [True, False])
async def test_revise_soul_command_reloads_the_session_when_unchanged(session_loads):
    agent, _, builder = _command_agent(
        SoulRevisionOutcome(
            revised=False, unchanged=True, version=1,
            message="SOUL.md already is the canonical SOUL (v1); nothing to revise.",
        )
    )
    builder.load_canonical_soul_resource.return_value = session_loads
    handler = CommandHandler(agent, task_manager=MagicMock())
    message = await handler.handle("!revise-soul fix", caller=CallerContext.sovereign())
    builder.load_canonical_soul_resource.assert_awaited_once()
    if session_loads:
        assert message.startswith("✅ SOUL.md already is the canonical SOUL (v1)")
    else:
        assert message.startswith("⚠")
        assert "did not load the canonical SOUL" in message


def _real_command_handler(raw, builder, agent_dir) -> CommandHandler:
    agent = SimpleNamespace(
        bootstrap_service=_service(raw, builder.storage, agent_dir),
        context_builder=builder,
        storage=builder.storage,
    )
    return CommandHandler(agent, task_manager=MagicMock())


@pytest.mark.asyncio
async def test_an_unchanged_revision_still_reloads_a_divergent_session(
    tmp_path, agent_dir
):
    """A SOUL.md reloaded and then restored leaves the session out of step.

    The file is canonical again, so nothing is stored, but the running
    session still served the reloaded text until the command reloaded it.
    """
    await _seed_canonical(tmp_path, agent_dir)
    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        await _reload_feature(builder.storage, builder, agent_dir).bootstrap_reload()
        assert builder._soul_content == REVISED
        (agent_dir / "SOUL.md").write_text(ORIGINAL, encoding="utf-8")

        message = await _real_command_handler(raw, builder, agent_dir).handle(
            "!revise-soul restore the canonical role", caller=CallerContext.sovereign()
        )

        assert message.startswith("✅"), message
        assert "already is the canonical SOUL (v1)" in message
        assert builder._soul_content == ORIGINAL
        assert await _version_count(raw) == 1
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_revise_soul_command_end_to_end_survives_a_restart(tmp_path, agent_dir):
    """Edit, reload, revise through the real command, restart: the edit stays."""
    await _seed_canonical(tmp_path, agent_dir)
    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        await _reload_feature(builder.storage, builder, agent_dir).bootstrap_reload()
        message = await _real_command_handler(raw, builder, agent_dir).handle(
            "!revise-soul accept Emma's signing DID", caller=CallerContext.sovereign()
        )
        assert message.startswith("✅ SOUL.md is now canonical SOUL v2 (was v1)"), message
        assert builder._soul_content == REVISED
    finally:
        await raw.close()

    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        assert builder._soul_content == REVISED
        current = await raw.get_current_agent_resource()
        assert current.version == 2
        assert current.provenance["reason"] == "accept Emma's signing DID"
        assert current.provenance["revised_by"]["role"] == "sovereign"
        assert current.provenance["previous_version"] == 1
    finally:
        await raw.close()


def test_soul_revision_is_not_an_agent_tool():
    """The agent cannot revise its own SOUL: no tool it can call does it."""
    feature = BootstrapFeature(MagicMock())
    for agent_tool in feature.get_tools():
        assert "revise" not in agent_tool.name
        assert agent_tool.schema.command_prefix != "!revise-soul"


# ---------------------------------------------------------------------------
# A rename does not publish over a concurrent revision (#3813)
# ---------------------------------------------------------------------------


def _renaming_agent(agent_id, storage, agent_dir) -> SimpleNamespace:
    """The agent shape ``_update_soul_name`` uses, on one storage connection."""
    return SimpleNamespace(
        agent_id=agent_id,
        storage=storage,
        context_builder=ContextBuilder(storage, agent_data_path=str(agent_dir)),
        bootstrap_service=SimpleNamespace(agent_data_path=str(agent_dir)),
    )


def _renamed(content: str) -> str:
    return content.replace("You Are Claw", "You Are Talon")


@pytest.mark.asyncio
async def test_a_rename_does_not_publish_over_a_revision_that_lands_while_it_runs(
    engine, agent_dir, monkeypatch
):
    """The lost update of #3813, on two connections to one database.

    The rename reads v1. The Sovereign then edits SOUL.md and revises it on the
    other connection, publishing the approved role change as v2. Published as
    read, the rename would make "v1 renamed" v3 and drop the role change from
    the canonical SOUL. The store refuses a write computed from v1 once v2 is
    current, and the rename reads v2 again and renames that.
    """
    raw, storage = await _engine_open(engine)
    other_raw, other_storage = await _engine_open(engine)
    try:
        assert await _engine_service(engine, raw, storage, agent_dir).save_soul_md(
            ORIGINAL
        ) is True
        real_publish = storage.promote_soul_seed
        revisions: list[SoulRevisionOutcome] = []

        async def a_revision_lands_before_the_rename_publishes(*args, **kwargs):
            if not revisions:
                (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
                revisions.append(
                    await _engine_service(
                        engine, other_raw, other_storage, agent_dir
                    ).revise_soul(
                        reason="Sovereign-approved role change",
                        revised_by=CallerContext.sovereign(),
                    )
                )
            return await real_publish(*args, **kwargs)

        monkeypatch.setattr(
            storage, "promote_soul_seed", a_revision_lands_before_the_rename_publishes
        )
        agent = _renaming_agent(engine.did, storage, agent_dir)

        assert await _update_soul_name(agent, "Claw", "Talon") is True

        (revision,) = revisions
        assert revision.revised is True, revision.message
        assert (revision.previous_version, revision.version) == (1, 2)
        assert await _soul_rows(raw, engine.did) == [(1, False), (2, False), (3, True)]
        v2 = await raw.agent_resources.get_version(version=2)
        assert (v2.content, v2.provenance["reason"]) == (
            REVISED, "Sovereign-approved role change"
        )
        v3 = await raw.get_current_agent_resource()
        assert v3.content == _renamed(REVISED)
        assert v3.provenance["source"].startswith("rename:")
        assert v3.provenance["previous_version"] == 2
        assert v3.provenance["previous_content_hash"] == soul_sha256(REVISED)
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(REVISED)
        assert agent.context_builder._soul_content == _renamed(REVISED)
    finally:
        await raw.close()
        await other_raw.close()

    raw, storage = await _engine_open(engine)
    try:
        builder = ContextBuilder(storage, agent_data_path=str(agent_dir))
        runtime = SimpleNamespace(
            agent_id=engine.did, storage=storage, context_builder=builder
        )
        await KestrelAgent._load_or_promote_soul_resource(runtime, str(agent_dir))
        assert builder._soul_content == _renamed(REVISED)
        assert (await inspect_soul_file(raw, agent_dir)).state is SoulFileState.IN_SYNC
    finally:
        await raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("landing", ["before_canonical_read", "after_canonical_read"])
async def test_a_rename_renames_a_revision_that_lands_around_its_canonical_read(
    engine, agent_dir, monkeypatch, landing
):
    """The canonical SOUL is read before SOUL.md, so either landing ends in step.

    Landing after the canonical read, the revision falls between the rename's
    two reads: its write from v1 is refused, and the next attempt finds the
    file and v2 in step. Landing just before it, the rename reads v2 and the
    revised file. Read file first, that landing gave the rename the older
    file text and v2: it renamed v2 but left SOUL.md differing from it.
    """
    raw, storage = await _engine_open(engine)
    other_raw, other_storage = await _engine_open(engine)
    try:
        assert await _engine_service(engine, raw, storage, agent_dir).save_soul_md(
            ORIGINAL
        ) is True
        real_read = storage.get_current_agent_resource
        revisions: list[SoulRevisionOutcome] = []

        async def a_revision_lands_once():
            if not revisions:
                (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
                revisions.append(
                    await _engine_service(
                        engine, other_raw, other_storage, agent_dir
                    ).revise_soul(
                        reason="Sovereign-approved role change",
                        revised_by=CallerContext.sovereign(),
                    )
                )

        async def a_revision_lands_around_the_read(*args, **kwargs):
            if landing == "before_canonical_read":
                await a_revision_lands_once()
            current = await real_read(*args, **kwargs)
            if landing == "after_canonical_read":
                await a_revision_lands_once()
            return current

        monkeypatch.setattr(
            storage, "get_current_agent_resource", a_revision_lands_around_the_read
        )
        agent = _renaming_agent(engine.did, storage, agent_dir)

        assert await _update_soul_name(agent, "Claw", "Talon") is True

        (revision,) = revisions
        assert revision.version == 2, revision.message
        assert await _soul_rows(raw, engine.did) == [(1, False), (2, False), (3, True)]
        current = await raw.get_current_agent_resource()
        assert current.content == _renamed(REVISED)
        assert current.provenance["previous_version"] == 2
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(REVISED)
        assert (await inspect_soul_file(raw, agent_dir)).state is SoulFileState.IN_SYNC
    finally:
        await raw.close()
        await other_raw.close()


@pytest.mark.asyncio
async def test_a_rename_that_keeps_losing_to_concurrent_writes_changes_nothing(
    engine, agent_dir, monkeypatch
):
    """Bounded: after the last refused attempt the conflict is raised.

    Another connection makes a new version current before each attempt
    publishes. The rename writes no version and leaves SOUL.md and the
    session as they were: the canonical SOUL is written before the file.
    """
    raw, storage = await _engine_open(engine)
    other_raw, _ = await _engine_open(engine)
    try:
        assert await _engine_service(engine, raw, storage, agent_dir).save_soul_md(
            ORIGINAL
        ) is True
        real_publish = storage.promote_soul_seed
        refused: list[dict] = []

        async def another_write_lands_first(*args, **kwargs):
            refused.append(kwargs)
            await other_raw.promote_soul_seed(
                THIRD + f"\nWrite {len(refused)}.\n", created_by=engine.did
            )
            return await real_publish(*args, **kwargs)

        monkeypatch.setattr(storage, "promote_soul_seed", another_write_lands_first)
        agent = _renaming_agent(engine.did, storage, agent_dir)
        agent.context_builder = MagicMock()

        with pytest.raises(ResourceVersionConflictError) as conflict:
            await _update_soul_name(agent, "Claw", "Talon")

        attempts = feature_module._SOUL_RENAME_ATTEMPTS
        assert len(refused) == attempts
        assert [call["expected_current"] for call in refused] == [
            ExpectedCurrentVersion(version) for version in range(1, attempts + 1)
        ]
        assert (conflict.value.expected, conflict.value.actual) == (attempts, attempts + 1)
        assert "nothing was written" in str(conflict.value)
        rows = await _soul_rows(raw, engine.did)
        assert rows == [(v, v == attempts + 1) for v in range(1, attempts + 2)]
        current = await raw.get_current_agent_resource()
        assert current.provenance["source"] != f"rename:{agent_dir / 'SOUL.md'}"
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == ORIGINAL
        agent.context_builder.load_canonical_soul_resource.assert_not_called()
    finally:
        await raw.close()
        await other_raw.close()


async def _metadata_name(raw, did):
    row = await raw.db.fetchone(
        "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = 'name'", (did,)
    )
    return row[0] if row else None


async def _agent_node_name(raw, did) -> tuple[str, str]:
    node = await raw.get_node(did)
    return node.label, node.properties["name"]


@pytest.mark.asyncio
async def test_a_rename_that_loses_every_soul_attempt_changes_no_name_and_its_retry_renames(
    engine, agent_dir, monkeypatch
):
    """Through ``!rename``: a refused SOUL rename leaves the name unchanged.

    The SOUL rename finds the old name in the SOUL's text, and a retry takes
    the old name from the live one. The name used to be written to the
    metadata, the graph and the live agent first, and the SOUL last. Losing
    every SOUL attempt then left Talon live, so running ``!rename Talon``
    again renamed Talon to Talon and reported success over a canonical SOUL,
    SOUL.md and session still saying "You Are Claw". The SOUL is renamed
    first now: losing it fails the rename with the name unchanged everywhere,
    and the retry renames Claw.
    """
    raw, storage = await _engine_open(engine)
    other_raw, _ = await _engine_open(engine)
    try:
        await raw.add_node(
            GraphNode(
                node_id=engine.did,
                node_type="agent",
                label="Claw",
                properties={"name": "Claw"},
            )
        )
        assert await _engine_service(engine, raw, storage, agent_dir).save_soul_md(
            ORIGINAL
        ) is True
        real_publish = storage.promote_soul_seed
        concurrent: list[str] = []

        async def a_revision_lands_first(*args, **kwargs):
            # As a revision writes: SOUL.md, then the canonical version.
            text = THIRD + f"\nWrite {len(concurrent) + 1}.\n"
            concurrent.append(text)
            (agent_dir / "SOUL.md").write_text(text, encoding="utf-8")
            await other_raw.promote_soul_seed(text, created_by=engine.did)
            return await real_publish(*args, **kwargs)

        monkeypatch.setattr(storage, "promote_soul_seed", a_revision_lands_first)
        agent = _renaming_agent(engine.did, storage, agent_dir)
        agent._agent_name = "Claw"
        agent.bootstrap_service.agent_name = "Claw"
        agent._raw_storage = raw
        feature = BootstrapFeature(agent)

        refused = await feature.rename_agent("Talon")

        attempts = feature_module._SOUL_RENAME_ATTEMPTS
        assert len(concurrent) == attempts
        assert refused.status is ToolResultStatus.ERROR
        assert refused.error.startswith("Failed to rename: SOUL.md update failed: ")
        assert "nothing was written" in refused.error
        assert not refused.data["rename_outcome"]["db_row_written"]
        assert not refused.data["rename_outcome"]["graph_updated"]
        assert not refused.data["rename_outcome"]["memory_updated"]
        assert not refused.data["rename_outcome"]["soul_md_updated"]
        assert (agent._agent_name, agent.bootstrap_service.agent_name) == ("Claw", "Claw")
        assert await _agent_node_name(raw, engine.did) == ("Claw", "Claw")
        assert await _metadata_name(raw, engine.did) is None
        last = concurrent[-1]
        assert (await raw.get_current_agent_resource()).content == last
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == last

        # The concurrent SOUL writes stop, and the same rename is run again.
        monkeypatch.setattr(storage, "promote_soul_seed", real_publish)
        renamed = await feature.rename_agent("Talon")

        assert renamed.status is ToolResultStatus.OK, renamed.error
        assert "SOUL.md updated." in renamed.confirmation
        assert (agent._agent_name, agent.bootstrap_service.agent_name) == (
            "Talon", "Talon"
        )
        assert await _agent_node_name(raw, engine.did) == ("Talon", "Talon")
        assert await _metadata_name(raw, engine.did) == "Talon"
        current = await raw.get_current_agent_resource()
        assert current.content == _renamed(last)
        assert current.provenance["previous_version"] == attempts + 1
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(last)
        assert agent.context_builder._soul_content == _renamed(last)
    finally:
        await raw.close()
        await other_raw.close()

    raw, storage = await _engine_open(engine)
    try:
        builder = ContextBuilder(storage, agent_data_path=str(agent_dir))
        runtime = SimpleNamespace(
            agent_id=engine.did, storage=storage, context_builder=builder
        )
        await KestrelAgent._load_or_promote_soul_resource(runtime, str(agent_dir))
        assert builder._soul_content == _renamed(last)
        assert (await inspect_soul_file(raw, agent_dir)).state is SoulFileState.IN_SYNC
        assert await _agent_node_name(raw, engine.did) == ("Talon", "Talon")
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_that_fails_after_the_soul_leaves_a_retry_that_completes_it(
    tmp_path, agent_dir, monkeypatch
):
    """A store failing after the SOUL was renamed: the retry renames the rest.

    The SOUL already names Talon and the live name is still Claw, so the
    retry finds nothing left to rename in the SOUL and writes the name.
    """
    await _seed_canonical(tmp_path, agent_dir)
    raw, storage = await _open(tmp_path)
    try:
        await raw.add_node(
            GraphNode(
                node_id=AGENT_ID,
                node_type="agent",
                label="Claw",
                properties={"name": "Claw"},
            )
        )
        agent = _renaming_agent(AGENT_ID, storage, agent_dir)
        agent._agent_name = "Claw"
        agent.bootstrap_service.agent_name = "Claw"
        agent._raw_storage = raw
        real_execute = raw.db.execute
        metadata_writes: list[tuple] = []

        async def the_metadata_store_fails_once(query, params=None, *args, **kwargs):
            if "agent_metadata" in query and not metadata_writes:
                metadata_writes.append(params)
                raise QueryError("metadata store unavailable")
            return await real_execute(query, params, *args, **kwargs)

        monkeypatch.setattr(raw.db, "execute", the_metadata_store_fails_once)
        feature = BootstrapFeature(agent)

        partial = await feature.rename_agent("Talon")

        assert partial.status is ToolResultStatus.PARTIAL
        assert partial.data["rename_outcome"]["soul_md_updated"] is True
        assert not partial.data["rename_outcome"]["db_row_written"]
        assert "SOUL.md has the new name" in partial.error
        assert "metadata DB still has the old name" in partial.error
        assert agent._agent_name == "Claw"
        assert (await raw.get_current_agent_resource()).content == _renamed(ORIGINAL)

        renamed = await feature.rename_agent("Talon")

        assert renamed.status is ToolResultStatus.OK, renamed.error
        assert renamed.data["rename_outcome"]["soul_md_updated"] is False
        assert agent._agent_name == "Talon"
        assert await _agent_node_name(raw, AGENT_ID) == ("Talon", "Talon")
        assert await _metadata_name(raw, AGENT_ID) == "Talon"
        assert await _version_count(raw) == 2
        assert (await raw.get_current_agent_resource()).content == _renamed(ORIGINAL)
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(ORIGINAL)
    finally:
        await raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["storage", "resource_store"])
async def test_the_store_refuses_a_version_derived_from_one_no_longer_current(
    engine, door
):
    """Checked under the version lock; a refusal writes no version and no graph link."""
    raw, _ = await _engine_open(engine)
    try:
        writer = raw if door == "storage" else raw.agent_resources
        with pytest.raises(ResourceVersionConflictError) as refused:
            await writer.promote_soul_seed(
                ORIGINAL, expected_current=ExpectedCurrentVersion(1)
            )
        assert (refused.value.expected, refused.value.actual) == (1, None)
        assert await _soul_rows(raw, engine.did) == []

        first = await writer.promote_soul_seed(
            ORIGINAL, expected_current=ExpectedCurrentVersion(None)
        )
        assert first.version == 1
        node_before = await raw.get_node(f"{engine.did}#soul")
        # Only the storage door publishes the graph link with the version.
        assert (node_before is not None) == (door == "storage")

        for expected in (None, 2):
            with pytest.raises(ResourceVersionConflictError) as refused:
                await writer.promote_soul_seed(
                    REVISED, expected_current=ExpectedCurrentVersion(expected)
                )
            assert (refused.value.expected, refused.value.actual) == (expected, 1)
        assert await _soul_rows(raw, engine.did) == [(1, True)]
        node_after = await raw.get_node(f"{engine.did}#soul")
        assert getattr(node_after, "properties", None) == getattr(
            node_before, "properties", None
        )

        # A version that is not current neither satisfies nor defeats the check.
        await raw.agent_resources.create_version(
            SOUL_MARKDOWN_RESOURCE_TYPE,
            THIRD,
            created_by=engine.did,
            source="draft",
            make_current=False,
        )
        with pytest.raises(ResourceVersionConflictError):
            await writer.promote_soul_seed(
                REVISED, expected_current=ExpectedCurrentVersion(2)
            )
        second = await writer.promote_soul_seed(
            REVISED, expected_current=ExpectedCurrentVersion(1)
        )
        assert (second.version, second.provenance["previous_version"]) == (3, 1)
        assert await _soul_rows(raw, engine.did) == [(1, False), (2, False), (3, True)]
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_does_not_make_an_unrevised_file_edit_canonical(
    tmp_path, agent_dir
):
    """A rename renames the canonical SOUL, not whatever SOUL.md holds.

    Before #3813 the rename published the file, so an edit no Sovereign
    revised became canonical, attributed to the rename. The edit stays in
    the file, renamed too, and is still reported as differing.
    """
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        agent = _renaming_agent(AGENT_ID, builder.storage, agent_dir)
        agent.context_builder = builder

        assert await _update_soul_name(agent, "Claw", "Talon") is True

        current = await raw.get_current_agent_resource()
        assert (current.version, current.content) == (2, _renamed(ORIGINAL))
        assert current.provenance["previous_version"] == 1
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(REVISED)
        assert builder._soul_content == _renamed(ORIGINAL)
        status = await inspect_soul_file(raw, agent_dir)
        assert (status.state, status.canonical_version) == (SoulFileState.FILE_DIFFERS, 2)
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_without_soul_md_renames_the_canonical_soul(tmp_path, agent_dir):
    """The canonical SOUL is renamed; the missing cache is not recreated."""
    await _seed_canonical(tmp_path, agent_dir)
    (agent_dir / "SOUL.md").unlink()
    raw, builder = await _boot(tmp_path, agent_dir)
    try:
        agent = _renaming_agent(AGENT_ID, builder.storage, agent_dir)
        agent.context_builder = builder

        assert await _update_soul_name(agent, "Nobody", "Talon") is False
        assert await _version_count(raw) == 1

        assert await _update_soul_name(agent, "Claw", "Talon") is True
        current = await raw.get_current_agent_resource()
        assert (current.version, current.content) == (2, _renamed(ORIGINAL))
        assert not (agent_dir / "SOUL.md").exists()
        assert builder._soul_content == _renamed(ORIGINAL)
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_without_a_canonical_soul_publishes_the_renamed_file(
    tmp_path, agent_dir
):
    """As boot would: the renamed SOUL.md becomes the first canonical version."""
    (agent_dir / "SOUL.md").write_text(ORIGINAL, encoding="utf-8")
    raw, storage = await _open(tmp_path)
    try:
        agent = _renaming_agent(AGENT_ID, storage, agent_dir)

        assert await _update_soul_name(agent, "Claw", "Talon") is True

        current = await raw.get_current_agent_resource()
        assert (current.version, current.content) == (1, _renamed(ORIGINAL))
        assert current.provenance["source"].startswith("rename:")
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(ORIGINAL)
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_does_not_overwrite_a_file_edit_written_while_it_ran(
    tmp_path, agent_dir, monkeypatch
):
    """SOUL.md is replaced only if it still holds the text the rename read."""
    await _seed_canonical(tmp_path, agent_dir)
    raw, storage = await _open(tmp_path)
    try:
        real_publish = storage.promote_soul_seed

        async def publish_then_the_file_is_edited(*args, **kwargs):
            published = await real_publish(*args, **kwargs)
            (agent_dir / "SOUL.md").write_text(THIRD, encoding="utf-8")
            return published

        monkeypatch.setattr(storage, "promote_soul_seed", publish_then_the_file_is_edited)
        agent = _renaming_agent(AGENT_ID, storage, agent_dir)

        assert await _update_soul_name(agent, "Claw", "Talon") is True

        assert (await raw.get_current_agent_resource()).content == _renamed(ORIGINAL)
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == THIRD
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_of_only_an_unrevised_file_edit_stores_no_version(
    tmp_path, agent_dir
):
    """The canonical SOUL names nobody; only the file's pending edit is renamed."""
    unnamed = "# SOUL.md\n\n## Role\nAccept work orders from did:pkh only.\n"
    raw, storage = await _open(tmp_path)
    try:
        assert await _service(raw, storage, agent_dir).save_soul_md(unnamed) is True
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        agent = _renaming_agent(AGENT_ID, storage, agent_dir)

        assert await _update_soul_name(agent, "Claw", "Talon") is True

        assert await _version_count(raw) == 1
        assert (await raw.get_current_agent_resource()).content == unnamed
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == _renamed(REVISED)
        assert await _update_soul_name(agent, "Claw", "Talon") is False
    finally:
        await raw.close()


@pytest.mark.asyncio
async def test_a_rename_that_could_write_nothing_reports_no_soul_update(agent_dir):
    """A storage that can read but not publish the canonical SOUL, and no file."""
    storage = SimpleNamespace(
        get_current_agent_resource=AsyncMock(
            return_value=SimpleNamespace(content=ORIGINAL, version=1)
        )
    )
    builder = MagicMock()
    agent = SimpleNamespace(
        agent_id=AGENT_ID,
        storage=storage,
        context_builder=builder,
        bootstrap_service=SimpleNamespace(agent_data_path=str(agent_dir)),
    )

    assert await _update_soul_name(agent, "Claw", "Talon") is False
    assert not (agent_dir / "SOUL.md").exists()
    builder.load_canonical_soul_resource.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical_names_the_agent", [True, False])
async def test_a_failed_soul_md_write_after_the_canonical_rename_still_reloads(
    tmp_path, agent_dir, monkeypatch, caplog, canonical_names_the_agent
):
    """The canonical SOUL is renamed first; SOUL.md is only its cache.

    A cache write that fails after the canonical SOUL was renamed is logged
    and the session still loads the renamed SOUL. With nothing renamed but
    the file, the failure is the rename's own and is raised.
    """
    canonical = ORIGINAL if canonical_names_the_agent else (
        "# SOUL.md\n\n## Role\nAccept work orders from did:pkh only.\n"
    )
    raw, storage = await _open(tmp_path)
    try:
        assert await _service(raw, storage, agent_dir).save_soul_md(canonical) is True
        (agent_dir / "SOUL.md").write_text(REVISED, encoding="utf-8")
        real_write_text = type(agent_dir).write_text

        def full_disk(path, *args, **kwargs):
            if path.name == "SOUL.md":
                raise OSError(28, "No space left on device")
            return real_write_text(path, *args, **kwargs)

        monkeypatch.setattr(type(agent_dir), "write_text", full_disk)
        agent = _renaming_agent(AGENT_ID, storage, agent_dir)

        if canonical_names_the_agent:
            with caplog.at_level(logging.WARNING):
                assert await _update_soul_name(agent, "Claw", "Talon") is True
            assert "could not be updated after the canonical SOUL" in caplog.text
            assert (await raw.get_current_agent_resource()).content == _renamed(ORIGINAL)
            assert agent.context_builder._soul_content == _renamed(ORIGINAL)
        else:
            with pytest.raises(OSError):
                await _update_soul_name(agent, "Claw", "Talon")
            assert await _version_count(raw) == 1
        assert (agent_dir / "SOUL.md").read_text(encoding="utf-8") == REVISED
    finally:
        await raw.close()
