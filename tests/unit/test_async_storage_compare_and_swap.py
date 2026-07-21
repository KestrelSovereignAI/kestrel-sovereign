"""compare_and_swap_node — the atomic conditional-update primitive (#2661).

``add_node`` is a whole-row clobber, so every caller that needs "update X only
if nobody changed it since I read it" was stuck hand-rolling a TOCTOU-racy retry
loop. ``compare_and_swap_node`` closes that window: the predicate check and the
write are one serialized unit at the storage layer.

Coverage:
  1. swap succeeds when the predicate (last-read snapshot) still holds
  2. predicate_failed leaves the existing row — including a concurrent writer's
     post-read update — completely untouched
  3. not_found when the row is genuinely absent
  4. compare-and-create semantics for ``expected is None``
  5. explicit concurrent-writer race: N parallel swaps on one node, exactly one
     succeeds and the rest report predicate_failed (the load-bearing guarantee)
  6. the guarantee survives the AsyncStorage facade AND the privacy wrapper
     passthrough (proving the wrapper does NOT decompose into get_node+add_node)
  7. backwards-compat: add_node still clobbers exactly as before

The dual-backend tests run against SQLite always and real PostgreSQL when
TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL is set (skipped
otherwise) — see the ``db_backend`` fixture in tests/conftest.py. The unit CI
job supplies none of those, so the Postgres path is skipped here; the atomicity
guarantees are re-exercised against the live pgvector service in the
integration job — see
``tests/integration/test_async_storage_compare_and_swap_postgres.py`` (#2661
review P2).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    NodeSwapResult,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from kestrel_sovereign.privacy import PrivacyMode


def _nid(prefix: str = "cas") -> str:
    """Unique node id so shared-backend (Postgres) tests never collide."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _node(node_id: str, properties: dict, *, label: str = "L", node_type: str = "cas_node") -> GraphNode:
    return GraphNode(node_id=node_id, node_type=node_type, label=label, properties=properties)


@pytest_asyncio.fixture
async def graph_store(db_backend):
    """AsyncGraphStore over the parametrized backend (SQLite + optional PG)."""
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return AsyncGraphStore(db)


# =====================================================================
# Result type
# =====================================================================


class TestNodeSwapResult:
    def test_is_str_enum(self):
        assert NodeSwapResult.SWAPPED == "swapped"
        assert NodeSwapResult.PREDICATE_FAILED == "predicate_failed"
        assert NodeSwapResult.NOT_FOUND == "not_found"

    def test_value_passes_through_string_boundary(self):
        # A caller may JSON-encode / stringify the result — the value must be
        # the plain string, not "NodeSwapResult.SWAPPED".
        assert NodeSwapResult.SWAPPED.value == "swapped"
        assert str(NodeSwapResult.SWAPPED) in ("swapped", "NodeSwapResult.SWAPPED")


# =====================================================================
# Happy path + failure classification (dual backend)
# =====================================================================


class TestCompareAndSwap:

    async def test_swap_succeeds_when_predicate_holds(self, graph_store):
        nid = _nid()
        await graph_store.add_node(_node(nid, {"status": "pending", "n": 1}))

        snapshot = (await graph_store.get_node(nid)).properties
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "passed", "n": 2})
        )

        assert result == NodeSwapResult.SWAPPED
        after = await graph_store.get_node(nid)
        assert after.properties == {"status": "passed", "n": 2}

    async def test_predicate_failed_leaves_post_read_update_untouched(self, graph_store):
        """The core safety property: a swap that loses the race must not clobber
        the winner's write."""
        nid = _nid()
        await graph_store.add_node(_node(nid, {"status": "pending"}))

        # Caller reads a snapshot...
        snapshot = (await graph_store.get_node(nid)).properties

        # ...then a concurrent writer lands a real decision (e.g. runtime audit
        # FAILED) via a plain add_node BEFORE the caller's swap fires.
        await graph_store.add_node(_node(nid, {"status": "failed", "audited": True}))

        # The stale-snapshot swap must refuse and leave the FAILED row intact.
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "passed"})
        )

        assert result == NodeSwapResult.PREDICATE_FAILED
        after = await graph_store.get_node(nid)
        assert after.properties == {"status": "failed", "audited": True}

    async def test_not_found_when_node_absent(self, graph_store):
        nid = _nid("absent")
        result = await graph_store.compare_and_swap_node(
            nid, {"status": "pending"}, _node(nid, {"status": "passed"})
        )
        assert result == NodeSwapResult.NOT_FOUND
        assert await graph_store.get_node(nid) is None

    async def test_compare_and_create_succeeds_when_absent(self, graph_store):
        nid = _nid("create")
        result = await graph_store.compare_and_swap_node(
            nid, None, _node(nid, {"status": "fresh"})
        )
        assert result == NodeSwapResult.SWAPPED
        created = await graph_store.get_node(nid)
        assert created is not None
        assert created.properties == {"status": "fresh"}

    async def test_compare_and_create_fails_when_present(self, graph_store):
        nid = _nid("create")
        await graph_store.add_node(_node(nid, {"status": "existing"}))

        result = await graph_store.compare_and_swap_node(
            nid, None, _node(nid, {"status": "clobbered"})
        )
        assert result == NodeSwapResult.PREDICATE_FAILED
        # The existing row is untouched — create-if-absent never overwrites.
        after = await graph_store.get_node(nid)
        assert after.properties == {"status": "existing"}

    async def test_swap_updates_properties_only(self, graph_store):
        """CAS is properties-only: a swap rewrites ``properties`` but leaves the
        existing node's ``node_type`` / ``label`` untouched (they are set at
        creation). ``new_node``'s type/label are ignored on the swap path."""
        nid = _nid()
        await graph_store.add_node(
            _node(nid, {"k": "v"}, node_type="orig_type", label="Orig")
        )
        snapshot = (await graph_store.get_node(nid)).properties
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"k": "v2"}, node_type="ignored_type", label="Ignored")
        )
        assert result == NodeSwapResult.SWAPPED
        after = await graph_store.get_node(nid)
        assert after.node_type == "orig_type"   # unchanged — set at creation
        assert after.label == "Orig"            # unchanged — set at creation
        assert after.properties == {"k": "v2"}  # swapped

    async def test_swap_does_not_clobber_concurrent_type_or_label_change(self, graph_store):
        """P1 regression (#2661 review): a properties swap must not silently
        revert a concurrent writer's ``node_type`` / ``label`` change. Because
        CAS is properties-only, a writer that changed only ``label`` (leaving
        ``properties`` intact) keeps its label AND our properties swap still
        lands — the two coexist rather than one clobbering the other."""
        nid = _nid()
        await graph_store.add_node(
            _node(nid, {"status": "pending"}, node_type="t", label="Before")
        )
        snapshot = (await graph_store.get_node(nid)).properties

        # Concurrent writer relabels the node but leaves properties unchanged.
        await graph_store.add_node(
            _node(nid, {"status": "pending"}, node_type="t", label="After")
        )

        # Our properties snapshot still holds, so the swap succeeds...
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "done"}, node_type="t", label="Ours")
        )
        assert result == NodeSwapResult.SWAPPED
        after = await graph_store.get_node(nid)
        # ...but the concurrent writer's label survives (never clobbered, and
        # our own new_node.label="Ours" is ignored on the swap path)...
        assert after.label == "After"
        # ...and our properties change landed.
        assert after.properties == {"status": "done"}

    async def test_snapshot_round_trips_through_get_node(self, graph_store):
        """A snapshot obtained via get_node must be an accepted predicate even
        with nested structures / unicode / numbers (JSON-value equality)."""
        nid = _nid()
        rich = {
            "genesis_audit": {"status": "pending", "risk": 1},
            "tags": ["a", "b"],
            "score": 0.5,
            "note": "café ☕",
            "count": 100,
        }
        await graph_store.add_node(_node(nid, rich))
        snapshot = (await graph_store.get_node(nid)).properties
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {**rich, "score": 0.9})
        )
        assert result == NodeSwapResult.SWAPPED


# =====================================================================
# Non-canonical stored rows — the predicate compares JSON *value*, not
# raw bytes (regression for the byte-exact comparison, #2661 review P2)
# =====================================================================


class TestNonCanonicalRows:
    """``get_node`` decodes ``properties`` (``NULL``/``''`` → ``{}``,
    minified-or-spaced JSON → the same dict), so the CAS predicate must accept a
    snapshot taken from such a row. A byte-exact ``properties = ?`` comparison
    re-serialized the snapshot and mismatched the stored text — reporting
    ``predicate_failed`` for an unchanged row. These rows are inserted with raw
    SQL because ``add_node`` always writes canonical ``json.dumps`` output and so
    could never reproduce the bug."""

    async def _insert_raw(self, graph_store, node_id, properties_sql_value):
        """Insert a graph_nodes row with a caller-controlled raw ``properties``
        text (or NULL), bypassing add_node's canonical serialization."""
        await graph_store.db.execute_commit(
            "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
            "VALUES (?, ?, ?, ?)",
            (node_id, "cas_node", "L", properties_sql_value),
        )

    async def test_swap_accepts_snapshot_of_null_properties_row(self, graph_store):
        nid = _nid("null")
        # A row whose properties column is SQL NULL (legacy / non-add_node writer).
        await self._insert_raw(graph_store, nid, None)

        # get_node decodes NULL → {}, which is what a caller would read + pass back.
        snapshot = (await graph_store.get_node(nid)).properties
        assert snapshot == {}

        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "set"})
        )
        assert result == NodeSwapResult.SWAPPED
        assert (await graph_store.get_node(nid)).properties == {"status": "set"}

    async def test_null_properties_row_still_fails_on_real_mismatch(self, graph_store):
        """The NULL→{} coalesce must not turn every predicate into a pass: a
        snapshot that does NOT match the (empty) stored value still fails."""
        nid = _nid("null")
        await self._insert_raw(graph_store, nid, None)

        result = await graph_store.compare_and_swap_node(
            nid, {"status": "stale"}, _node(nid, {"status": "set"})
        )
        assert result == NodeSwapResult.PREDICATE_FAILED
        # Row untouched (still decodes to {}).
        assert (await graph_store.get_node(nid)).properties == {}

    async def test_swap_accepts_snapshot_of_minified_row(self, graph_store):
        nid = _nid("minified")
        # Row persisted by some writer that used compact separators — the stored
        # text differs from default json.dumps(...) spacing, same JSON value.
        import json as _json

        minified = _json.dumps({"status": "pending", "n": 1}, separators=(",", ":"))
        assert " " not in minified  # sanity: genuinely minified
        await self._insert_raw(graph_store, nid, minified)

        snapshot = (await graph_store.get_node(nid)).properties
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "passed", "n": 2})
        )
        assert result == NodeSwapResult.SWAPPED
        assert (await graph_store.get_node(nid)).properties == {"status": "passed", "n": 2}


# =====================================================================
# Frinz #626 scenario: PASSED only if not already FAILED
# =====================================================================


class TestGenesisAuditScenario:
    """The concrete race that motivated the primitive (Frinz PR #626): persist
    genesis_audit=PASSED on a companion node only if a concurrent runtime audit
    hasn't already written FAILED since we read it."""

    async def test_passed_lands_when_no_concurrent_failed(self, graph_store):
        nid = _nid("agent")
        await graph_store.add_node(
            _node(nid, {"genesis_audit": {"status": "pending"}})
        )
        snapshot = (await graph_store.get_node(nid)).properties
        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"genesis_audit": {"status": "passed"}})
        )
        assert result == NodeSwapResult.SWAPPED
        after = await graph_store.get_node(nid)
        assert after.properties["genesis_audit"]["status"] == "passed"

    async def test_passed_refused_when_failed_slipped_in(self, graph_store):
        nid = _nid("agent")
        await graph_store.add_node(
            _node(nid, {"genesis_audit": {"status": "pending"}})
        )
        snapshot = (await graph_store.get_node(nid)).properties

        # Concurrent runtime audit writes FAILED after our read.
        await graph_store.add_node(
            _node(nid, {"genesis_audit": {"status": "genesis_audit_failed"}})
        )

        result = await graph_store.compare_and_swap_node(
            nid, snapshot, _node(nid, {"genesis_audit": {"status": "passed"}})
        )
        # The real safety decision (FAILED) is preserved — not reversed.
        assert result == NodeSwapResult.PREDICATE_FAILED
        after = await graph_store.get_node(nid)
        assert after.properties["genesis_audit"]["status"] == "genesis_audit_failed"


# =====================================================================
# Concurrent-writer race — the load-bearing guarantee (dual backend)
# =====================================================================


class TestConcurrentWriters:

    @pytest.mark.parametrize("n_writers", [8, 16])
    async def test_exactly_one_swap_wins(self, graph_store, n_writers):
        """Fire N compare_and_swap_node calls in parallel on one node, all with
        the SAME last-read snapshot. Exactly one must win; the rest must report
        predicate_failed. This is the property a hand-rolled retry loop can never
        guarantee."""
        nid = _nid("race")
        await graph_store.add_node(_node(nid, {"status": "pending", "v": 0}))
        snapshot = (await graph_store.get_node(nid)).properties

        async def swap(i: int) -> NodeSwapResult:
            return await graph_store.compare_and_swap_node(
                nid, snapshot, _node(nid, {"status": "won", "winner": i})
            )

        results = await asyncio.gather(*(swap(i) for i in range(n_writers)))

        swapped = [r for r in results if r == NodeSwapResult.SWAPPED]
        failed = [r for r in results if r == NodeSwapResult.PREDICATE_FAILED]
        assert len(swapped) == 1, f"expected exactly one winner, got {results}"
        assert len(failed) == n_writers - 1
        # The persisted row is one of the racers' writes, fully applied.
        winner_idx = results.index(NodeSwapResult.SWAPPED)
        after = await graph_store.get_node(nid)
        assert after.properties == {"status": "won", "winner": winner_idx}

    async def test_concurrent_create_exactly_one_wins(self, graph_store):
        """Compare-and-create under concurrency: N parallel creators of the same
        absent node, exactly one inserts."""
        nid = _nid("create-race")

        async def create(i: int) -> NodeSwapResult:
            return await graph_store.compare_and_swap_node(
                nid, None, _node(nid, {"creator": i})
            )

        results = await asyncio.gather(*(create(i) for i in range(8)))
        assert sum(1 for r in results if r == NodeSwapResult.SWAPPED) == 1
        assert sum(1 for r in results if r == NodeSwapResult.PREDICATE_FAILED) == 7


# =====================================================================
# Backwards-compat: add_node behaviour is unchanged
# =====================================================================


class TestAddNodeUnchanged:

    async def test_add_node_still_clobbers(self, graph_store):
        """add_node remains a whole-row upsert — no accidental CAS semantics."""
        nid = _nid()
        await graph_store.add_node(_node(nid, {"status": "a"}))
        await graph_store.add_node(_node(nid, {"status": "b"}))
        after = await graph_store.get_node(nid)
        assert after.properties == {"status": "b"}


# =====================================================================
# Facade + privacy wrapper (SQLite; proves the delegation chain is atomic)
# =====================================================================


class TestFacadeAndPrivacyWrapper:

    async def test_facade_delegates(self, tmp_path):
        storage = await AsyncStorage.create_sqlite(str(tmp_path / "facade.db"))
        try:
            nid = _nid()
            await storage.add_node(_node(nid, {"status": "pending"}))
            snapshot = (await storage.get_node(nid)).properties
            result = await storage.compare_and_swap_node(
                nid, snapshot, _node(nid, {"status": "passed"})
            )
            assert result == NodeSwapResult.SWAPPED
            assert (await storage.get_node(nid)).properties == {"status": "passed"}
        finally:
            await storage.close()

    async def test_privacy_wrapper_passthrough_preserves_atomicity(self, tmp_path):
        """If the wrapper decomposed CAS into get_node + add_node, every racer
        would read the same snapshot, pass the predicate, and clobber — so more
        than one would "win". A true passthrough yields exactly one winner."""
        storage = await AsyncStorage.create_sqlite(str(tmp_path / "priv.db"))
        wrapped = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        try:
            nid = _nid("priv-race")
            await wrapped.add_node(_node(nid, {"status": "pending"}))
            snapshot = (await wrapped.get_node(nid)).properties

            async def swap(i: int):
                return await wrapped.compare_and_swap_node(
                    nid, snapshot, _node(nid, {"status": "won", "winner": i})
                )

            results = await asyncio.gather(*(swap(i) for i in range(8)))
            assert sum(1 for r in results if r == NodeSwapResult.SWAPPED) == 1
            assert sum(1 for r in results if r == NodeSwapResult.PREDICATE_FAILED) == 7
        finally:
            await storage.close()

    async def test_privacy_wrapper_works_in_ephemeral_mode(self, tmp_path):
        """Graph CAS is structural metadata, not PII — allowed in EPHEMERAL just
        like add_node (which the wrapper explicitly permits)."""
        storage = await AsyncStorage.create_sqlite(str(tmp_path / "eph.db"))
        wrapped = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        try:
            nid = _nid("eph")
            result = await wrapped.compare_and_swap_node(
                nid, None, _node(nid, {"status": "fresh"})
            )
            assert result == NodeSwapResult.SWAPPED
        finally:
            await storage.close()


# =====================================================================
# Bound-store tenant ownership — CAS honours the same isolation boundary
# as add_node / get_node when the store is bound to an agent (#2649)
# =====================================================================


@pytest_asyncio.fixture
async def bound_pair(db_backend):
    """Two AsyncGraphStores bound to different agents over one shared DB.

    The unbound ``graph_store`` fixture above exercises the ownerless CAS
    behaviour; these two bound stores exercise cross-tenant isolation.
    """
    db = AsyncDatabase(db_backend)
    await db._init_schema()
    db._initialized = True
    return (
        AsyncGraphStore(db, agent_id="agent-a"),
        AsyncGraphStore(db, agent_id="agent-b"),
    )


class TestBoundOwnershipCAS:
    """When bound, ``compare_and_swap_node`` must (a) record ownership on
    compare-and-create so the creator can read its own node back, and (b) scope
    the swap/failure-classification through the ownership predicate so one tenant
    can never overwrite — or even observe — another tenant's node. Regression for
    the #2649 review: the primitive merged in from #2661 bypassed both."""

    async def test_bound_compare_and_create_is_visible_and_owned(self, bound_pair):
        store_a, store_b = bound_pair
        nid = _nid("bound-create")

        result = await store_b.compare_and_swap_node(
            nid, None, _node(nid, {"status": "fresh"})
        )
        assert result == NodeSwapResult.SWAPPED

        # The creator can read its own new node back...
        created = await store_b.get_node(nid)
        assert created is not None
        assert created.properties == {"status": "fresh"}

        # ...backed by exactly one ownership witness (the creator)...
        owners = await store_b.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?", (nid,)
        )
        assert {row[0] for row in owners} == {"agent-b"}

        # ...and the node is invisible to a different tenant.
        assert await store_a.get_node(nid) is None

    async def test_bound_swap_cannot_overwrite_foreign_node(self, bound_pair):
        store_a, store_b = bound_pair
        nid = _nid("bound-foreign")

        # Agent A owns the node.
        await store_a.add_node(_node(nid, {"status": "A-owned"}))
        snapshot = (await store_a.get_node(nid)).properties

        # Agent B knows the id AND the exact properties, yet still cannot swap
        # it: the node is outside B's ownership scope, so it reads as NOT_FOUND
        # (invisible), never PREDICATE_FAILED (leaks existence) or SWAPPED (the
        # pre-fix hijack).
        result = await store_b.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "B-hijacked"})
        )
        assert result == NodeSwapResult.NOT_FOUND

        # A's node is completely untouched, and B still cannot see it.
        after = await store_a.get_node(nid)
        assert after.properties == {"status": "A-owned"}
        assert await store_b.get_node(nid) is None

    async def test_bound_owner_can_swap_its_own_node(self, bound_pair):
        """The isolation predicate must not lock the legitimate owner out."""
        store_a, _store_b = bound_pair
        nid = _nid("bound-own")
        await store_a.add_node(_node(nid, {"status": "pending"}))
        snapshot = (await store_a.get_node(nid)).properties

        result = await store_a.compare_and_swap_node(
            nid, snapshot, _node(nid, {"status": "done"})
        )
        assert result == NodeSwapResult.SWAPPED
        assert (await store_a.get_node(nid)).properties == {"status": "done"}

    async def test_bound_create_rejects_foreign_declared_owner(self, bound_pair):
        """A bound store refuses a new_node that declares a different agent_id —
        the same guard add_node applies, so ownership can't be spoofed via CAS."""
        _store_a, store_b = bound_pair
        nid = _nid("bound-spoof")
        with pytest.raises(ValueError):
            await store_b.compare_and_swap_node(
                nid, None, _node(nid, {"agent_id": "agent-a"})
            )
        # The guard runs before any write, so nothing was created.
        assert await store_b.get_node(nid) is None

    async def test_bound_create_conflict_on_foreign_node_is_not_found(
        self, bound_pair
    ):
        """Compare-and-create against an id already taken by ANOTHER tenant
        cannot insert (the node_id primary key blocks it), but it must NOT report
        PREDICATE_FAILED: that would let B tell a foreign-owned id apart from an
        absent one, an existence leak across the tenant boundary. The scoped
        re-read makes the foreign row invisible, so it reports NOT_FOUND exactly
        like get_node — and the foreign row stays owned solely by its original
        tenant (no shadow row, no B witness)."""
        store_a, store_b = bound_pair
        nid = _nid("bound-create-conflict")
        await store_a.add_node(_node(nid, {"status": "A-owned"}))

        result = await store_b.compare_and_swap_node(
            nid, None, _node(nid, {"status": "B-shadow"})
        )
        # NOT_FOUND (invisible), never PREDICATE_FAILED (leaks existence) or
        # SWAPPED (would shadow-create) — the same verdict get_node gives B.
        assert result == NodeSwapResult.NOT_FOUND
        assert await store_b.get_node(nid) is None

        # No B ownership witness was recorded, and A's node is unchanged.
        owners = await store_a.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?", (nid,)
        )
        assert {row[0] for row in owners} == {"agent-a"}
        assert (await store_a.get_node(nid)).properties == {"status": "A-owned"}

    async def test_bound_create_conflict_on_own_node_is_predicate_failed(
        self, bound_pair
    ):
        """Compare-and-create against an id this SAME tenant already owns is a
        genuine, visible conflict — the node is inside B's scope, so B is
        entitled to learn it exists. That path stays PREDICATE_FAILED (distinct
        from the foreign NOT_FOUND above), and the existing row is untouched."""
        _store_a, store_b = bound_pair
        nid = _nid("bound-create-own-conflict")

        first = await store_b.compare_and_swap_node(
            nid, None, _node(nid, {"status": "B-first"})
        )
        assert first == NodeSwapResult.SWAPPED

        again = await store_b.compare_and_swap_node(
            nid, None, _node(nid, {"status": "B-second"})
        )
        assert again == NodeSwapResult.PREDICATE_FAILED
        # The original create is left untouched (no clobber via the create path).
        assert (await store_b.get_node(nid)).properties == {"status": "B-first"}
        owners = await store_b.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?", (nid,)
        )
        assert {row[0] for row in owners} == {"agent-b"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
