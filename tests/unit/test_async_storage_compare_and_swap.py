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
    NodeDeleteResult,
    NodeSwapResult,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyViolationError,
    acquire_control_plane_capability,
)
from kestrel_sovereign.privacy import PrivacyMode


def _control_plane_capability():
    """Obtain the real control-plane capability via a genuine trusted-module call.

    The token is closure-private (no importable singleton — #2672 review P2); the
    only legitimate way to get it is a call whose caller frame IS a trusted
    module's namespace, so bind a throwaway probe into that module's ``__dict__``.
    """
    import importlib

    module = importlib.import_module("kestrel_sovereign.bootstrap.service")
    namespace = module.__dict__
    exec("def __cp_probe():\n    return acquire_control_plane_capability()", namespace)
    try:
        return namespace["__cp_probe"]()
    finally:
        namespace.pop("__cp_probe", None)


_CONTROL_PLANE_CAPABILITY = _control_plane_capability()

# A realistic 64-hex SHA-256 digest — the per-field validators (#2672 review P1)
# require content-free structural fields to be hash / timestamp / enum shaped.
_HEX = "0123456789abcdef" * 4
_HEX2 = "fedcba9876543210" * 4


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

    def test_delete_result_is_str_enum(self):
        assert NodeDeleteResult.DELETED == "deleted"
        assert NodeDeleteResult.PREDICATE_FAILED == "predicate_failed"
        assert NodeDeleteResult.NOT_FOUND == "not_found"


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

    async def test_expected_identity_refuses_post_read_relabel(self, graph_store):
        """A caller that owns one graph shape can atomically refuse a row that
        was relabeled after its read, even when the properties still match."""
        nid = _nid("identity-label")
        await graph_store.add_node(
            _node(nid, {"status": "pending"}, node_type="owned", label="Before")
        )
        snapshot = (await graph_store.get_node(nid)).properties

        # Another whole-row writer changes identity but preserves the exact
        # properties snapshot, reproducing the gap in a properties-only CAS.
        await graph_store.add_node(
            _node(nid, dict(snapshot), node_type="owned", label="After")
        )

        result = await graph_store.compare_and_swap_node(
            nid,
            snapshot,
            _node(nid, {"status": "done"}, node_type="owned", label="Before"),
            expected_node_type="owned",
            expected_label="Before",
        )

        assert result == NodeSwapResult.PREDICATE_FAILED
        after = await graph_store.get_node(nid)
        assert after.label == "After"
        assert after.properties == {"status": "pending"}

    async def test_expected_identity_refuses_post_read_retype(self, graph_store):
        nid = _nid("identity-type")
        await graph_store.add_node(
            _node(nid, {"status": "pending"}, node_type="owned", label="Stable")
        )
        snapshot = (await graph_store.get_node(nid)).properties
        await graph_store.add_node(
            _node(nid, dict(snapshot), node_type="foreign", label="Stable")
        )

        result = await graph_store.compare_and_swap_node(
            nid,
            snapshot,
            _node(nid, {"status": "done"}, node_type="owned", label="Stable"),
            expected_node_type="owned",
            expected_label="Stable",
        )

        assert result == NodeSwapResult.PREDICATE_FAILED
        after = await graph_store.get_node(nid)
        assert after.node_type == "foreign"
        assert after.properties == {"status": "pending"}

    async def test_expected_identity_allows_compare_and_create(self, graph_store):
        nid = _nid("identity-create")

        result = await graph_store.compare_and_swap_node(
            nid,
            None,
            _node(nid, {"status": "fresh"}, node_type="owned", label="Stable"),
            expected_node_type="owned",
            expected_label="Stable",
        )

        assert result == NodeSwapResult.SWAPPED
        created = await graph_store.get_node(nid)
        assert created is not None
        assert created.node_type == "owned"
        assert created.label == "Stable"

    async def test_expected_identity_requires_type_and_label_together(
        self, graph_store
    ):
        nid = _nid("identity-partial")
        with pytest.raises(ValueError, match="expected_node_type.*expected_label"):
            await graph_store.compare_and_swap_node(
                nid,
                None,
                _node(nid, {"status": "fresh"}),
                expected_node_type="cas_node",
            )
        assert await graph_store.get_node(nid) is None

    async def test_expected_identity_rejects_a_different_new_node_shape(
        self, graph_store
    ):
        nid = _nid("identity-new-shape")
        await graph_store.add_node(
            _node(nid, {"status": "pending"}, node_type="owned", label="Stable")
        )
        snapshot = (await graph_store.get_node(nid)).properties

        with pytest.raises(ValueError, match="new_node identity must match"):
            await graph_store.compare_and_swap_node(
                nid,
                snapshot,
                _node(
                    nid,
                    {"status": "done"},
                    node_type="owned",
                    label="Different",
                ),
                expected_node_type="owned",
                expected_label="Stable",
            )

        after = await graph_store.get_node(nid)
        assert after.label == "Stable"
        assert after.properties == {"status": "pending"}

    async def test_empty_allowed_type_set_denies_existing_swap(self, graph_store):
        """An explicit empty allowlist must deny every effective node type."""
        nid = _nid("empty-types")
        await graph_store.add_node(_node(nid, {"status": "pending"}))
        snapshot = (await graph_store.get_node(nid)).properties

        result = await graph_store.compare_and_swap_node(
            nid,
            snapshot,
            _node(nid, {"status": "done"}),
            allowed_node_types=frozenset(),
        )

        assert result == NodeSwapResult.TYPE_NOT_ALLOWED
        assert (await graph_store.get_node(nid)).properties == {"status": "pending"}

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
# Atomic compare-and-delete by graph identity (dual backend)
# =====================================================================


class TestCompareAndDelete:

    async def test_matching_identity_is_deleted(self, graph_store):
        nid = _nid("delete-match")
        await graph_store.add_node(
            _node(nid, {"status": "stale"}, node_type="owned", label="Stable")
        )

        result = await graph_store.compare_and_delete_node(
            nid, expected_node_type="owned", expected_label="Stable"
        )

        assert result == "deleted"
        assert await graph_store.get_node(nid) is None

    @pytest.mark.parametrize(
        "replacement_type,replacement_label",
        (("foreign", "Stable"), ("owned", "After")),
    )
    async def test_post_read_identity_change_is_not_deleted(
        self, graph_store, replacement_type, replacement_label
    ):
        nid = _nid("delete-race")
        await graph_store.add_node(
            _node(nid, {"status": "stale"}, node_type="owned", label="Before")
        )
        observed = await graph_store.get_node(nid)
        assert observed.node_type == "owned"
        assert observed.label == "Before"

        # Reproduce a replacement after the caller's read but before its delete.
        await graph_store.add_node(
            _node(
                nid,
                {"status": "replacement", "sentinel": "must survive"},
                node_type=replacement_type,
                label=replacement_label,
            )
        )

        result = await graph_store.compare_and_delete_node(
            nid, expected_node_type="owned", expected_label="Before"
        )

        assert result == "predicate_failed"
        after = await graph_store.get_node(nid)
        assert after is not None
        assert after.node_type == replacement_type
        assert after.label == replacement_label
        assert after.properties == {
            "status": "replacement",
            "sentinel": "must survive",
        }

    async def test_absent_node_is_not_found(self, graph_store):
        result = await graph_store.compare_and_delete_node(
            _nid("delete-absent"),
            expected_node_type="owned",
            expected_label="Stable",
        )
        assert result == "not_found"

    async def test_public_immediate_transaction_locks_before_outer_read(
        self, tmp_path
    ):
        """A public atomic read/write scope takes SQLite's slot up front.

        A nested graph mutation cannot upgrade a deferred snapshot after a
        competing connection commits.  The storage and privacy facades must
        therefore expose the backend's immediate mode so callers composing a
        read with conditional deletion can serialize before that first read.
        """

        db_path = str(tmp_path / "nested-delete.db")
        first_storage = await AsyncStorage.create_sqlite(db_path)
        first = PrivacyEnforcingStorage(first_storage, PrivacyMode.NORMAL)
        second = await AsyncStorage.create_sqlite(db_path)
        nid = _nid("nested-delete")
        await first.add_node(
            _node(nid, {"status": "stale"}, node_type="owned", label="Before")
        )

        replacement = None
        try:
            async with first.transaction(immediate=True):
                observed = await first.get_node(nid)
                assert observed is not None
                assert observed.label == "Before"
                replacement = asyncio.create_task(
                    second.add_node(
                        _node(
                            nid,
                            {"status": "replacement", "sentinel": "must survive"},
                            node_type="owned",
                            label="After",
                        )
                    )
                )
                await asyncio.sleep(0.1)
                assert not replacement.done(), (
                    "replacement passed the public immediate transaction"
                )
                result = await first.compare_and_delete_node(
                    nid,
                    expected_node_type="owned",
                    expected_label="Before",
                )
            assert result == NodeDeleteResult.DELETED

            await asyncio.wait_for(replacement, timeout=5)
            after = await second.get_node(nid)
            assert after is not None
            assert after.label == "After"
            assert after.properties["sentinel"] == "must survive"
        finally:
            if replacement is not None and not replacement.done():
                replacement.cancel()
                await asyncio.gather(replacement, return_exceptions=True)
            await first_storage.close()
            await second.close()

    async def test_ordinary_unbound_delete_cleans_dangling_graph_records(
        self, graph_store
    ):
        """Physical maintenance deletion still repairs an absent graph row."""

        nid = _nid("dangling-unbound")
        source = _nid("dangling-source")
        await graph_store.db.execute_commit(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            (nid, "agent-a"),
        )
        await graph_store.db.execute_commit(
            "INSERT INTO graph_edges (source_id, target_id, label, properties) "
            "VALUES (?, ?, ?, ?)",
            (source, nid, "dangling", "{}"),
        )
        await graph_store.db.execute_commit(
            "INSERT INTO graph_edge_owners "
            "(source_id, target_id, label, agent_id) VALUES (?, ?, ?, ?)",
            (source, nid, "dangling", "agent-a"),
        )

        await graph_store.delete_node(nid)

        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_node_owners WHERE node_id = ?", (nid,)
        ) is None
        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_edges WHERE target_id = ?", (nid,)
        ) is None
        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_edge_owners WHERE target_id = ?", (nid,)
        ) is None

    async def test_ordinary_bound_delete_releases_dangling_owned_records(
        self, graph_store
    ):
        """A tenant can repair its witnesses even after the node disappeared."""

        agent_id = f"agent:{uuid.uuid4().hex}"
        bound = AsyncGraphStore(graph_store.db, agent_id=agent_id)
        nid = _nid("dangling-bound")
        source = _nid("dangling-source")
        await graph_store.db.execute_commit(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            (nid, agent_id),
        )
        await graph_store.db.execute_commit(
            "INSERT INTO graph_edges (source_id, target_id, label, properties) "
            "VALUES (?, ?, ?, ?)",
            (source, nid, "dangling", "{}"),
        )
        await graph_store.db.execute_commit(
            "INSERT INTO graph_edge_owners "
            "(source_id, target_id, label, agent_id) VALUES (?, ?, ?, ?)",
            (source, nid, "dangling", agent_id),
        )

        await bound.delete_node(nid)

        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_node_owners WHERE node_id = ? AND agent_id = ?",
            (nid, agent_id),
        ) is None
        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_edges WHERE target_id = ?", (nid,)
        ) is None
        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_edge_owners WHERE target_id = ?", (nid,)
        ) is None


# =====================================================================
# Edge admission and endpoint deletion serialization
# =====================================================================


class TestEdgeAdmission:
    @pytest.mark.parametrize("missing", ["source", "target"])
    async def test_unbound_edge_rejects_a_missing_endpoint(
        self, graph_store, missing
    ):
        """Maintenance callers cannot introduce a newly dangling edge."""

        source_id = _nid("edge-source")
        target_id = _nid("edge-target")
        existing_id = target_id if missing == "source" else source_id
        await graph_store.add_node(_node(existing_id, {"status": "present"}))

        with pytest.raises(Exception, match="endpoints"):
            await graph_store.add_edge(source_id, target_id, "references")

        assert await graph_store.db.fetchone(
            "SELECT 1 FROM graph_edges "
            "WHERE source_id = ? AND target_id = ? AND label = ?",
            (source_id, target_id, "references"),
        ) is None


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

    async def test_privacy_wrapper_forwards_expected_identity(self, tmp_path):
        storage = await AsyncStorage.create_sqlite(str(tmp_path / "identity.db"))
        wrapped = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        try:
            nid = _nid("wrapped-identity")
            await wrapped.add_node(
                _node(
                    nid,
                    {"status": "pending"},
                    node_type="owned",
                    label="Before",
                )
            )
            snapshot = (await wrapped.get_node(nid)).properties
            await storage.add_node(
                _node(
                    nid,
                    dict(snapshot),
                    node_type="owned",
                    label="After",
                )
            )

            result = await wrapped.compare_and_swap_node(
                nid,
                snapshot,
                _node(
                    nid,
                    {"status": "done"},
                    node_type="owned",
                    label="Before",
                ),
                expected_node_type="owned",
                expected_label="Before",
            )

            assert result == NodeSwapResult.PREDICATE_FAILED
            assert (await storage.get_node(nid)).properties == {"status": "pending"}
        finally:
            await storage.close()

    async def test_facade_and_graph_privacy_proxy_compare_and_delete(self, tmp_path):
        storage = await AsyncStorage.create_sqlite(str(tmp_path / "delete.db"))
        wrapped = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
        try:
            facade_id = _nid("facade-delete")
            await storage.add_node(
                _node(facade_id, {}, node_type="owned", label="Facade")
            )
            assert await storage.compare_and_delete_node(
                facade_id,
                expected_node_type="owned",
                expected_label="Facade",
            ) == "deleted"

            wrapper_id = _nid("wrapper-delete")
            await wrapped.add_node(
                _node(wrapper_id, {}, node_type="owned", label="Wrapper")
            )
            assert await wrapped.compare_and_delete_node(
                wrapper_id,
                expected_node_type="owned",
                expected_label="Wrapper",
            ) == "deleted"

            proxy_id = _nid("proxy-delete")
            await wrapped.add_node(
                _node(proxy_id, {}, node_type="owned", label="Proxy")
            )
            assert await wrapped.graph.compare_and_delete_node(
                proxy_id,
                expected_node_type="owned",
                expected_label="Proxy",
            ) == "deleted"
        finally:
            await storage.close()

    async def test_privacy_wrapper_governs_graph_cas_in_ephemeral_mode(self, tmp_path):
        """CAS is privacy-governed in EPHEMERAL (#2672), without decomposing the
        atomic primitive.

        A durable graph write is not "structural, not PII". Two tiers: a
        user-derived / unknown node CAS is default-denied and writes no row; a
        content-free structural type (here ``document``) is admitted on the
        ordinary path with strict per-field validation and lands atomically; and
        the ``agent`` control-plane type is admitted through the unforgeable
        control-plane capability, but only as a SWAP of an existing node whose
        identity label is carried along unchanged (a compare-and-create with a
        fresh free-text label is refused by the label carry-along boundary —
        #2672 review P1). In every case the governance inspects ``new_node`` up
        front and then delegates the single atomic CAS — it never becomes
        get_node + add_node.
        """
        storage = await AsyncStorage.create_sqlite(str(tmp_path / "eph.db"))
        wrapped = PrivacyEnforcingStorage(storage, PrivacyMode.EPHEMERAL)
        try:
            # Unknown/user-derived node type → fail closed, no durable row.
            blocked = _nid("eph-blocked")
            with pytest.raises(PrivacyViolationError):
                await wrapped.compare_and_swap_node(
                    blocked, None, _node(blocked, {"status": "fresh"})
                )
            assert await storage.get_node(blocked) is None

            # Content-free structural type → admitted on the ordinary path. The
            # document ``hash`` must be a real digest AND equal the node id.
            allowed = _HEX
            result = await wrapped.compare_and_swap_node(
                allowed,
                None,
                _node(
                    allowed,
                    {"hash": allowed, "type": "Constitution",
                     "created_at": "2026-01-01T00:00:00+00:00"},
                    label="KESTREL_CONSTITUTION",
                    node_type="document",
                ),
            )
            assert result == NodeSwapResult.SWAPPED
            assert (await storage.get_node(allowed)) is not None

            # Control-plane ``agent`` type → the realistic CAS is a properties SWAP
            # of the inception-written node, carrying the identity label along. It
            # is denied without the capability, admitted only when the caller
            # presents the unforgeable control-plane capability.
            cp = _nid("agent")
            base_props = {
                "constitution_hash": _HEX,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            # Seed the inception-written agent node directly (raw store).
            await storage.add_node(_node(
                cp, dict(base_props), label="Kestrel", node_type="agent",
            ))
            swapped_props = {**base_props, "bootstrap_state": "complete"}
            cp_node = _node(cp, swapped_props, label="Kestrel", node_type="agent")

            with pytest.raises(PrivacyViolationError):
                await wrapped.compare_and_swap_node(cp, base_props, cp_node)
            assert (await storage.get_node(cp)).properties.get("bootstrap_state") != "complete"

            result = await wrapped.compare_and_swap_node(
                cp, base_props, cp_node, capability=_CONTROL_PLANE_CAPABILITY
            )
            assert result == NodeSwapResult.SWAPPED
            assert (await storage.get_node(cp)).properties["bootstrap_state"] == "complete"
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

    async def test_bound_compare_delete_cannot_observe_foreign_node(
        self, bound_pair
    ):
        store_a, store_b = bound_pair
        nid = _nid("bound-delete-foreign")
        await store_a.add_node(
            _node(nid, {"status": "A-owned"}, node_type="owned", label="Stable")
        )

        result = await store_b.compare_and_delete_node(
            nid,
            expected_node_type="owned",
            expected_label="Stable",
        )

        assert result == NodeDeleteResult.NOT_FOUND
        assert await store_b.get_node(nid) is None
        assert (await store_a.get_node(nid)).properties == {"status": "A-owned"}

    async def test_bound_compare_delete_releases_only_callers_shared_witness(
        self, bound_pair
    ):
        store_a, store_b = bound_pair
        nid = _nid("bound-delete-shared")
        unbound = AsyncGraphStore(store_a.db)
        await unbound.add_node(
            _node(nid, {"status": "shared"}, node_type="owned", label="Stable")
        )
        await store_a.db.execute_commit(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            (nid, "agent-a"),
        )
        await store_a.db.execute_commit(
            "INSERT INTO graph_node_owners (node_id, agent_id) VALUES (?, ?)",
            (nid, "agent-b"),
        )

        result = await store_a.compare_and_delete_node(
            nid,
            expected_node_type="owned",
            expected_label="Stable",
        )

        assert result == NodeDeleteResult.DELETED
        assert await store_a.get_node(nid) is None
        remaining = await store_b.get_node(nid)
        assert remaining is not None
        assert remaining.properties == {"status": "shared"}
        owners = await store_a.db.fetchall(
            "SELECT agent_id FROM graph_node_owners WHERE node_id = ?", (nid,)
        )
        assert {row[0] for row in owners} == {"agent-b"}

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
