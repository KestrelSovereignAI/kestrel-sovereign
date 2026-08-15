"""Co-ownership of a fleet-shared graph node — the two questions (#2893).

A content-addressed node (the governing constitution document, or a
Sovereign-signed reanchor artifact) may carry more than one owner, because
every tenant that possesses the bytes computes the same node id. ``add_node``
decides that with two separate checks, and each of them was found missing in
review:

1. **Is this property set shareable at all?** It used to be asked only when a
   row already existed, so acceptance depended on insertion order: a property
   set outside the shared shape was stored happily for whichever agent got
   there first, and every sibling presenting the same content afterwards rolled
   back with an ownership error. That is exactly the split fleet #2893 exists to
   remove — one agent governed by v2, the next stuck on v1.

2. **Do the stored row and the incoming node agree?** Two property sets can each
   be shareable in isolation while describing *different* content. Without an
   agreement check the second agent is simply added as an owner, the stored row
   is deliberately retained, and the reanchor reports success — leaving an agent
   owning a row that says something other than what it just anchored.

The agreement check has to be per shape, which is the trap the second fix has to
avoid: every field of a signed artifact is covered by the signature and must
match, but a constitution anchor's ``created_at`` is when *that* tenant first
stored the document and legitimately differs across a fleet. A wholesale dict
comparison would pass these tests and break the sharing that already works —
``test_a_constitution_anchor_shares_despite_a_different_created_at`` is the one
that fails if the agreement set is drawn too wide.

These run against SQLite always and real PostgreSQL when TEST_POSTGRES_URL is
set — the shared-fleet case is a PostgreSQL one, but the rule is the storage
layer's and must not differ by backend.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from kestrel_sovereign.constitution.amendment_artifact import ARTIFACT_TYPE
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_file_store import AsyncFileStore
from kestrel_sovereign.storage.async_graph_store import (
    AsyncGraphStore,
    GraphNode,
    NodeSwapResult,
)

pytestmark = pytest.mark.asyncio

AGENT_A = "did:web:example.com:first"
AGENT_B = "did:web:example.com:second"
AGENT_C = "did:web:example.com:third"
SIGNER = "did:web:example.com:kestrel-sovereign"

CONSTITUTION_HASH = "b" * 64

ARTIFACT_TYPE_LABEL = "constitution_amendment_artifact"
ARTIFACT_LABEL = "Signed Constitution Reanchor Artifact"

#: A refusal raised *inside* a transaction reaches the caller as the backend's
#: ``TransactionError``, not as the ``ValueError`` the store raised — both
#: backends wrap it. The message survives, so that is what these tests pin.
#:
#: ``add_node`` checks the shared shape *before* opening its transaction, so
#: that one refusal arrives unwrapped and those tests can pin ``ValueError``
#: directly. ``compare_and_swap_node`` cannot: deciding a swap needs the stored
#: row, and reading it outside the transaction would race the very writer the
#: primitive exists to serialise against. So the same rule surfaces as two
#: exception types depending on the door — worth knowing when catching it.
REFUSAL = Exception


@pytest_asyncio.fixture
async def db(db_backend):
    database = AsyncDatabase(db_backend)
    await database._init_schema()
    database._initialized = True
    return database


@pytest.fixture
def artifact_bytes():
    """Content unique to this test.

    The PostgreSQL backend is a live shared database that is not torn down
    between tests, so fixed content would give every test the same node id and
    let one test's rows decide another's outcome — silently, since a leftover
    owner row turns a refusal into a pass.
    """
    return f'{{"artifact": "{uuid.uuid4().hex}"}}'.encode()


@pytest.fixture
def constitution_bytes():
    return f"# KESTREL CONSTITUTION\n\n{uuid.uuid4().hex}\n".encode()


def _graph(db: AsyncDatabase, agent_did: str) -> AsyncGraphStore:
    """A store bound to one tenant, as every runtime callsite binds it."""
    store = AsyncGraphStore(db)
    store.bind_agent(agent_did)
    return store


async def _take_possession(
    db: AsyncDatabase, agent_did: str, content: bytes, name: str
) -> str:
    """Store the blob under this agent, returning its content hash.

    Co-ownership follows *possession of the content*, not knowledge of a hash:
    ``add_node`` requires a ``file_owners`` row for this agent before it will
    admit them onto a shared row. Storing the real bytes is how a runtime agent
    earns that row, so the tests earn it the same way rather than inserting the
    witness by hand.
    """
    files = AsyncFileStore(db, agent_did)
    return await files.store_file(content, name)


def _artifact_node(node_id: str, **overrides) -> GraphNode:
    properties = {
        "hash": node_id,
        "type": "SignedConstitutionAmendment",
        "artifact_type": ARTIFACT_TYPE,
        "constitution_hash": CONSTITUTION_HASH,
        "signer": SIGNER,
        "created_at": "2026-08-11T00:00:00+00:00",
    }
    properties.update(overrides)
    return GraphNode(
        node_id=node_id,
        node_type=ARTIFACT_TYPE_LABEL,
        label=ARTIFACT_LABEL,
        properties=properties,
    )


def _constitution_node(node_id: str, created_at: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type="document",
        label="KESTREL_CONSTITUTION",
        properties={
            "hash": node_id,
            "type": "Constitution",
            "created_at": created_at,
        },
    )


async def _owners(db: AsyncDatabase, node_id: str) -> list[str]:
    rows = await db.fetchall(
        "SELECT agent_id FROM graph_node_owners WHERE node_id = ?", (node_id,)
    )
    return sorted(row[0] for row in rows)


# =====================================================================
# 1. Shareability is decided before the first row, not after it
# =====================================================================


class TestAcceptanceDoesNotDependOnOrder:
    async def test_an_unshareable_artifact_is_refused_on_an_empty_database(
        self, db, artifact_bytes
    ):
        """``created_at`` is a *signed* field the verifier does not constrain.

        An artifact carrying an unbounded one verifies fine, so it reaches the
        store. Nothing existed at this node id, so the shape checks used to be
        skipped entirely and the row was written.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)

        with pytest.raises(ValueError, match="fleet-shared"):
            await graph.add_node(_artifact_node(node_id, created_at="x" * 65))

        assert await graph.get_node(node_id) is None
        assert await _owners(db, node_id) == []

    async def test_neither_agent_can_be_the_lucky_first(
        self, db, artifact_bytes
    ):
        """The property that makes the fleet coherent.

        Whichever agent presents the unshareable artifact first, the answer is
        the same. Before this check the first caller committed and every one
        after it rolled back — one agent governed by the new constitution and
        the rest refused, from a single signed authorization.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_B, artifact_bytes, "a.json")

        for agent in (AGENT_A, AGENT_B):
            with pytest.raises(ValueError, match="fleet-shared"):
                await _graph(db, agent).add_node(
                    _artifact_node(node_id, signer="not-a-did")
                )

        assert await _owners(db, node_id) == []

    async def test_a_per_agent_field_is_refused_on_a_first_write_too(
        self, db, artifact_bytes
    ):
        """An operator filesystem path cannot reach a fleet-shared row by being
        first to it either."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")

        with pytest.raises(ValueError, match="fleet-shared"):
            await _graph(db, AGENT_A).add_node(
                _artifact_node(node_id, source_path="/home/operator/a.json")
            )

    async def test_the_canonical_artifact_is_still_written(
        self, db, artifact_bytes
    ):
        """The refusal must be narrow. A check that rejected the shape this
        codebase actually writes would brick every reanchor, so pin the
        accepting case as hard as the refusing ones."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)

        await graph.add_node(_artifact_node(node_id))

        stored = await graph.get_node(node_id)
        assert stored is not None
        assert stored.properties["signer"] == SIGNER
        assert await _owners(db, node_id) == [AGENT_A]

    async def test_an_ordinary_node_is_untouched_by_any_of_this(self, db):
        """Only the two content-addressed shapes are governed by the table; a
        node type that is not fleet-shared keeps writing whatever it likes."""
        graph = _graph(db, AGENT_A)
        node_id = f"episode:{uuid.uuid4().hex}"
        node = GraphNode(
            node_id=node_id,
            node_type="episode",
            label="A Tuesday",
            properties={"source_path": "/home/operator/notes.md", "n": 1},
        )

        await graph.add_node(node)

        stored = await graph.get_node(node_id)
        assert stored.properties["source_path"] == "/home/operator/notes.md"


# =====================================================================
# 2. Two co-owners must agree on what the row says
# =====================================================================


class TestCoOwnersMustAgree:
    async def test_a_second_artifact_disagreeing_on_a_signed_field_is_refused(
        self, db, artifact_bytes
    ):
        """Both rows are shareable in isolation — that is the whole difficulty.

        Neither carries a per-agent field, so the shareability predicate has
        nothing to object to. They simply describe different authorizations. The
        stored row is deliberately retained on a co-ownership admission, so
        without an agreement check the second agent would end up owning a row
        naming a signer it never verified.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_B, artifact_bytes, "a.json")
        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))

        with pytest.raises(REFUSAL, match="owned by another agent"):
            await _graph(db, AGENT_B).add_node(
                _artifact_node(node_id, signer="did:web:someone-else.example")
            )

        assert await _owners(db, node_id) == [AGENT_A]
        stored = await _graph(db, AGENT_A).get_node(node_id)
        assert stored.properties["signer"] == SIGNER

    async def test_a_disagreeing_created_at_is_refused_on_an_artifact(
        self, db, artifact_bytes
    ):
        """``created_at`` is inside ``canonical_amendment_bytes``, so two agents
        holding the same artifact compute the same value. A different one means
        a different record, even though the field is shaped correctly."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_B, artifact_bytes, "a.json")
        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))

        with pytest.raises(REFUSAL, match="owned by another agent"):
            await _graph(db, AGENT_B).add_node(
                _artifact_node(node_id, created_at="2019-01-01T00:00:00+00:00")
            )

        assert await _owners(db, node_id) == [AGENT_A]

    async def test_an_identical_artifact_still_gains_the_second_owner(
        self, db, artifact_bytes
    ):
        """The feature itself: one signed artifact, one row, two owners."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_B, artifact_bytes, "a.json")

        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))
        await _graph(db, AGENT_B).add_node(_artifact_node(node_id))

        assert await _owners(db, node_id) == sorted([AGENT_A, AGENT_B])
        rows = await db.fetchall(
            "SELECT node_id FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        assert len(rows) == 1

    async def test_a_constitution_anchor_shares_despite_a_different_created_at(
        self, db, constitution_bytes
    ):
        """The guard against over-fixing, and the reason agreement is per shape.

        ``created_at`` on a constitution anchor is when *this* tenant first
        stored the document — not a property of the document. Two agents
        adopting the same constitution on different days both anchor it, and
        the first tenant's row is kept rather than rewritten. Comparing whole
        property dicts would refuse the second agent here and break the
        multi-tenant anchoring that already ships (#2890).
        """
        node_id = await _take_possession(
            db, AGENT_A, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        await _take_possession(
            db, AGENT_B, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )

        await _graph(db, AGENT_A).add_node(
            _constitution_node(node_id, "2026-01-01T00:00:00+00:00")
        )
        await _graph(db, AGENT_B).add_node(
            _constitution_node(node_id, "2026-08-11T00:00:00+00:00")
        )

        assert await _owners(db, node_id) == sorted([AGENT_A, AGENT_B])
        stored = await _graph(db, AGENT_B).get_node(node_id)
        assert stored.properties["created_at"] == "2026-01-01T00:00:00+00:00"

    async def test_a_constitution_anchor_disagreeing_on_its_hash_is_refused(
        self, db, constitution_bytes
    ):
        """Narrowing the anchor's agreement set opens no hole.

        Note what actually refuses this: the *shareability* predicate, not the
        agreement check. An anchor's agreement set is ``{hash, type}``, and its
        predicate already pins ``hash == node_id`` and ``type ==
        "Constitution"`` — so two rows that are each shareable under the same
        node id necessarily agree, and the anchor's agreement check can never
        fire on its own. That is the point: everything left over is either
        pinned by the predicate or per-tenant by design, so dropping
        ``created_at`` from the set gives up nothing. This test exists to keep
        that true — if the anchor's predicate is ever loosened, one of these two
        checks still has to refuse a row describing a different document, and
        this is where that gets noticed.
        """
        node_id = await _take_possession(
            db, AGENT_A, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        await _take_possession(
            db, AGENT_B, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        await _graph(db, AGENT_A).add_node(
            _constitution_node(node_id, "2026-01-01T00:00:00+00:00")
        )
        # A stored row that no longer describes this document, reachable only
        # by a writer that computed its properties from something else.
        await db.execute(
            "UPDATE graph_nodes SET properties = ? WHERE node_id = ?",
            (
                '{"hash": "' + "c" * 64 + '", "type": "Constitution"}',
                node_id,
            ),
        )

        with pytest.raises(REFUSAL, match="owned by another agent"):
            await _graph(db, AGENT_B).add_node(
                _constitution_node(node_id, "2026-08-11T00:00:00+00:00")
            )

        assert await _owners(db, node_id) == [AGENT_A]

    async def test_possession_of_the_content_is_still_required(
        self, db, artifact_bytes
    ):
        """Agreement does not replace possession. An agent that never stored
        the bytes cannot join a shared row by presenting properties it guessed
        from the node id."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))

        with pytest.raises(REFUSAL, match="owned by another agent"):
            await _graph(db, AGENT_B).add_node(_artifact_node(node_id))

        assert await _owners(db, node_id) == [AGENT_A]


# =====================================================================
# 3. The other door: compare_and_swap_node
# =====================================================================


class TestTheSwapDoorObeysTheSameRules:
    """``add_node`` is not the only writer of graph rows.

    A rule enforced at one door is not a rule. ``compare_and_swap_node`` is a
    public primitive reachable through ``AsyncStorage`` and the privacy
    wrapper, and it writes ``properties`` onto whatever row already exists —
    so an agent that legitimately co-owns the fleet's artifact row could have
    swapped an operator path onto it, or rewritten the signer every co-owner
    is governed by, and been told ``SWAPPED``.
    """

    async def test_a_create_cannot_seed_an_unshareable_shared_row(
        self, db, artifact_bytes
    ):
        """Compare-and-create writes the caller's shape verbatim, so it is the
        same order-dependence hole as ``add_node`` had, through another door."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")

        with pytest.raises(REFUSAL, match="fleet-shared"):
            await _graph(db, AGENT_A).compare_and_swap_node(
                node_id, None, _artifact_node(node_id, created_at="x" * 65)
            )

        assert await _owners(db, node_id) == []

    async def test_a_swap_cannot_put_a_per_agent_field_on_a_shared_row(
        self, db, artifact_bytes
    ):
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))
        snapshot = (await graph.get_node(node_id)).properties

        with pytest.raises(REFUSAL, match="fleet-shared"):
            await graph.compare_and_swap_node(
                node_id,
                snapshot,
                _artifact_node(node_id, source_path="/home/operator/a.json"),
            )

        stored = await graph.get_node(node_id)
        assert "source_path" not in stored.properties

    async def test_a_swap_cannot_dodge_the_rules_by_declaring_another_type(
        self, db, artifact_bytes
    ):
        """The shape must be read off the *stored* row.

        A swap ignores ``new_node.node_type`` and ``label`` entirely — it
        writes onto whatever row is at ``node_id``. Gating on the shape the
        caller declares would therefore be no gate at all: claim to be writing
        an episode and walk straight onto the fleet's artifact row.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))
        snapshot = (await graph.get_node(node_id)).properties

        disguised = GraphNode(
            node_id=node_id,
            node_type="episode",
            label="A Tuesday",
            properties={"source_path": "/home/operator/a.json"},
        )
        with pytest.raises(REFUSAL, match="fleet-shared"):
            await graph.compare_and_swap_node(node_id, snapshot, disguised)

        stored = await graph.get_node(node_id)
        assert "source_path" not in stored.properties

    async def test_a_co_owner_cannot_swap_a_row_the_fleet_shares(
        self, db, artifact_bytes
    ):
        """Even a well-formed swap is refused once the row has two owners.

        ``add_node`` hands a second tenant an ownership witness and leaves the
        canonical bytes alone — it will not rewrite a row more than one tenant
        owns. A swap must not be the door that does what the front door
        refuses, or one co-owner silently redefines what the others are
        governed by.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_B, artifact_bytes, "a.json")
        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))
        graph_b = _graph(db, AGENT_B)
        await graph_b.add_node(_artifact_node(node_id))
        assert await _owners(db, node_id) == sorted([AGENT_A, AGENT_B])

        snapshot = (await graph_b.get_node(node_id)).properties
        with pytest.raises(REFUSAL, match="owned by another agent"):
            await graph_b.compare_and_swap_node(
                node_id,
                snapshot,
                _artifact_node(node_id, signer="did:web:someone-else.example"),
            )

        stored = await graph_b.get_node(node_id)
        assert stored.properties["signer"] == SIGNER

    async def test_a_lone_owner_can_still_swap_its_own_shared_row(
        self, db, artifact_bytes
    ):
        """The refusal is about co-owners, not about the shape being shared.

        A single owner rewriting its own row affects nobody else, and
        ``add_node`` allows exactly that — so refusing it here would be a
        stricter rule at one door than the other, which is the shape of bug
        this whole change is about.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))
        snapshot = (await graph.get_node(node_id)).properties

        result = await graph.compare_and_swap_node(
            node_id,
            snapshot,
            _artifact_node(node_id, created_at="2027-01-01T00:00:00+00:00"),
        )

        assert result == NodeSwapResult.SWAPPED
        stored = await graph.get_node(node_id)
        assert stored.properties["created_at"] == "2027-01-01T00:00:00+00:00"

    async def test_an_ordinary_node_swaps_freely(self, db):
        """The new read must not change behaviour for the 99% of node types
        that are not fleet-shared."""
        graph = _graph(db, AGENT_A)
        node_id = f"episode:{uuid.uuid4().hex}"
        node = GraphNode(node_id, "episode", "A Tuesday", {"n": 1})
        await graph.add_node(node)

        result = await graph.compare_and_swap_node(
            node_id, {"n": 1}, GraphNode(node_id, "episode", "A Tuesday", {"n": 2})
        )

        assert result == NodeSwapResult.SWAPPED
        assert (await graph.get_node(node_id)).properties == {"n": 2}

    async def test_a_foreign_shared_row_still_reports_not_found(
        self, db, artifact_bytes
    ):
        """The refusal must not become an existence oracle.

        ``compare_and_swap_node`` is careful never to let a bound tenant tell
        "absent" apart from "owned by someone else" — both are NOT_FOUND, the
        same answer ``get_node`` gives. A shared-shape check that read the
        stored row *unscoped* would **raise** for a foreign row while an absent
        one returns NOT_FOUND, handing back that distinction in the shape of an
        exception. B never stored these bytes, so it is not a co-owner and must
        learn nothing.

        The probe has to be one the check would actually object to. An earlier
        version of this test swapped in *well-formed* artifact properties onto
        a single-owner row: nothing to refuse, so it fell through to the scoped
        UPDATE and reported NOT_FOUND either way. Removing the scope left it
        green — a surviving mutant, and the test's fault rather than the code's.
        Both probes below trip a different refusal.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_C, artifact_bytes, "a.json")
        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))
        # A and C co-own it. B is the outsider throughout.
        await _graph(db, AGENT_C).add_node(_artifact_node(node_id))

        absent_id = "f" * 64
        graph_b = _graph(db, AGENT_B)

        # Probe 1: properties the shape check refuses outright.
        unshareable = await graph_b.compare_and_swap_node(
            node_id,
            {"anything": True},
            _artifact_node(node_id, source_path="/home/operator/a.json"),
        )
        unshareable_absent = await graph_b.compare_and_swap_node(
            absent_id,
            {"anything": True},
            _artifact_node(absent_id, source_path="/home/operator/a.json"),
        )

        # Probe 2: well-formed properties onto a row that already has two
        # owners, which the co-owner rule refuses.
        multi_owner = await graph_b.compare_and_swap_node(
            node_id, {"anything": True}, _artifact_node(node_id)
        )
        multi_owner_absent = await graph_b.compare_and_swap_node(
            absent_id, {"anything": True}, _artifact_node(absent_id)
        )

        assert unshareable == NodeSwapResult.NOT_FOUND
        assert multi_owner == NodeSwapResult.NOT_FOUND
        assert unshareable == unshareable_absent, (
            "an unshareable swap distinguishes a foreign row from an absent one"
        )
        assert multi_owner == multi_owner_absent, (
            "a co-owned foreign row is distinguishable from an absent one"
        )
