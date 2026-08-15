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
        self, db, constitution_bytes
    ):
        """The co-owner rule, isolated from the identity rule.

        Deliberately on the *anchor* rather than the artifact. Every artifact
        field is an identity field, so any swap worth refusing there is already
        refused for disagreeing on identity, and this test would prove nothing
        about co-ownership — it would pass with the co-owner check deleted.
        ``created_at`` is the one field a fleet-shared row legitimately differs
        on, which makes it the only way to ask this question on its own.

        ``add_node`` hands a second tenant an ownership witness and leaves the
        first tenant's stamp alone. A swap must not be the door that does what
        the front door refuses.
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
        graph_b = _graph(db, AGENT_B)
        await graph_b.add_node(
            _constitution_node(node_id, "2026-08-11T00:00:00+00:00")
        )
        assert await _owners(db, node_id) == sorted([AGENT_A, AGENT_B])

        snapshot = (await graph_b.get_node(node_id)).properties
        with pytest.raises(REFUSAL, match="owned by another agent"):
            await graph_b.compare_and_swap_node(
                node_id,
                snapshot,
                _constitution_node(node_id, "2027-01-01T00:00:00+00:00"),
            )

        stored = await graph_b.get_node(node_id)
        assert stored.properties["created_at"] == "2026-01-01T00:00:00+00:00"

    async def test_a_lone_owner_cannot_swap_the_identity_either(
        self, db, artifact_bytes
    ):
        """Being first is not a licence to redefine the row.

        An earlier version of this test asserted the opposite — that a sole
        owner may rewrite its own shared row, on the reasoning that nobody else
        is on it yet so nobody is harmed. That reasoning is wrong, and the test
        was pinning the hole open. A node id here IS the hash of the bytes, so
        identity cannot legitimately change while the id stays: a different
        artifact is a different node. Swap in another well-formed ``signer``
        and every sibling that later anchors the genuine file disagrees with
        the stored row and rolls back — this issue's fleet split, reachable by
        one agent acting alone, and unrepairable afterwards.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))
        snapshot = (await graph.get_node(node_id)).properties

        with pytest.raises(REFUSAL, match="content-derived identity"):
            await graph.compare_and_swap_node(
                node_id,
                snapshot,
                _artifact_node(node_id, signer="did:web:someone-else.example"),
            )

        stored = await graph.get_node(node_id)
        assert stored.properties["signer"] == SIGNER

    async def test_a_lone_owner_cannot_rewrite_the_identity_through_add_node(
        self, db, artifact_bytes
    ):
        """The same rule at the front door, or the swap check is theatre."""
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))

        with pytest.raises(REFUSAL, match="content-derived identity"):
            await graph.add_node(
                _artifact_node(node_id, signer="did:web:someone-else.example")
            )

        stored = await graph.get_node(node_id)
        assert stored.properties["signer"] == SIGNER

    async def test_the_owner_may_still_restate_the_same_artifact(
        self, db, artifact_bytes
    ):
        """Immutable identity is not a frozen row.

        Re-anchoring the same artifact — an idempotent repeat, which the
        reanchor writer does — presents identical properties, agrees with
        itself, and must still land. A rule that refused this would brick the
        ordinary path.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))

        await graph.add_node(_artifact_node(node_id))
        result = await graph.compare_and_swap_node(
            node_id,
            (await graph.get_node(node_id)).properties,
            _artifact_node(node_id),
        )

        assert result == NodeSwapResult.SWAPPED
        assert await _owners(db, node_id) == [AGENT_A]

    async def test_an_anchors_own_timestamp_is_still_the_owners_to_change(
        self, db, constitution_bytes
    ):
        """The guard against over-fixing, on the swap door.

        ``created_at`` is outside the anchor's identity set — it is when *this*
        tenant stored the document, not a property of the document — so a lone
        owner may still update it. Folding it into identity would make the
        rule wrong in the other direction.
        """
        node_id = await _take_possession(
            db, AGENT_A, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        graph = _graph(db, AGENT_A)
        await graph.add_node(
            _constitution_node(node_id, "2026-01-01T00:00:00+00:00")
        )
        snapshot = (await graph.get_node(node_id)).properties

        result = await graph.compare_and_swap_node(
            node_id,
            snapshot,
            _constitution_node(node_id, "2027-01-01T00:00:00+00:00"),
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


class TestTheShapeIsReadOffTheStoredRow:
    """The relabelling escape, found by review after the swap door was fixed.

    ``add_node`` is a whole-row upsert: it writes ``node_type`` and ``label``
    as well as ``properties``. Every guard above keys off the shared-shape
    table, so deriving that key from the *incoming* node alone meant a sole
    owner could skip all of them by declaring a different type — the row's id
    is what identifies it, not the label the caller puts on it.

    The same mistake, in the same change, at the other door. The swap path
    already read the shape off the stored row; ``add_node`` did not.
    """

    async def test_a_sole_owner_cannot_relabel_a_shared_row(
        self, db, artifact_bytes
    ):
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(_artifact_node(node_id))

        disguised = GraphNode(
            node_id=node_id,
            node_type="episode",
            label="A Tuesday",
            properties={"source_path": "/home/operator/a.json"},
        )
        with pytest.raises(REFUSAL, match="node_type or label"):
            await graph.add_node(disguised)

        stored = await graph.get_node(node_id)
        assert stored.node_type == ARTIFACT_TYPE_LABEL
        assert stored.properties["signer"] == SIGNER

    async def test_the_next_agent_can_still_anchor_the_genuine_artifact(
        self, db, artifact_bytes
    ):
        """The consequence the refusal exists to prevent, stated as a test.

        Without it, A's relabelling lands and B — holding the same signed
        artifact — can never join the row.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _take_possession(db, AGENT_B, artifact_bytes, "a.json")
        graph_a = _graph(db, AGENT_A)
        await graph_a.add_node(_artifact_node(node_id))

        with pytest.raises(REFUSAL, match="node_type or label"):
            await graph_a.add_node(
                GraphNode(node_id, "episode", "A Tuesday", {"n": 1})
            )
        await _graph(db, AGENT_B).add_node(_artifact_node(node_id))

        assert await _owners(db, node_id) == sorted([AGENT_A, AGENT_B])

    async def test_a_foreign_caller_learns_nothing_new(
        self, db, artifact_bytes
    ):
        """The refusal must not become a better error for an outsider.

        B does not own the row and must keep getting the ownership answer —
        anything more specific tells it the shape of a row it cannot read.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        await _graph(db, AGENT_A).add_node(_artifact_node(node_id))

        with pytest.raises(REFUSAL, match="owned by another agent"):
            await _graph(db, AGENT_B).add_node(
                GraphNode(node_id, "episode", "A Tuesday", {"n": 1})
            )

    async def test_an_ordinary_node_can_still_be_retyped(self, db):
        """add_node has always been a whole-row upsert for everything else, and
        this refusal must not quietly become a global rule."""
        graph = _graph(db, AGENT_A)
        node_id = f"episode:{uuid.uuid4().hex}"
        await graph.add_node(GraphNode(node_id, "episode", "A Tuesday", {"n": 1}))

        await graph.add_node(GraphNode(node_id, "concept", "Tuesdays", {"n": 2}))

        stored = await graph.get_node(node_id)
        assert stored.node_type == "concept"
        assert stored.label == "Tuesdays"


class TestTheSwapDecidesAtomically:
    """A check is not a lock, and a policy denial is not an exception.

    Both found by review after the swap guards went in — the guards were
    right, the way they reached the write was not.
    """

    @staticmethod
    def _retype_mid_flight(monkeypatch, db, node_id, node_type, label):
        """Land a retype between the stored-shape read and the UPDATE.

        The hazard is a genuine race, and a race is not something a test can
        pin by running two coroutines and hoping. So the retype is injected at
        the exact point it would have to land to matter: right after the read
        that decides the shape, before the write that trusts it. Matching on
        the statement is blunt, but it names the window precisely, which is the
        whole point of the test.
        """
        original = db.fetchone
        fired = {"done": False}

        async def hooked(sql, params=None):
            row = await original(sql, params)
            if (
                not fired["done"]
                and "SELECT node_type, label, properties FROM graph_nodes" in sql
            ):
                fired["done"] = True
                await db.execute(
                    "UPDATE graph_nodes SET node_type = ?, label = ? "
                    "WHERE node_id = ?",
                    (node_type, label, node_id),
                )
            return row

        monkeypatch.setattr(db, "fetchone", hooked)
        return fired

    async def test_a_row_that_becomes_shared_mid_flight_is_not_written(
        self, db, artifact_bytes, monkeypatch
    ):
        """The window the guards had: read, then write.

        The stored-shape read is not a lock. On PostgreSQL a concurrent writer
        can retype an ordinary row into a fleet-shared one in between — and the
        swap, having validated against the shape it *saw* rather than the shape
        it *writes to*, lands unshareable properties on a row the fleet now
        shares. The decision has to travel into the UPDATE predicate, not just
        precede it.
        """
        node_id = await _take_possession(db, AGENT_A, artifact_bytes, "a.json")
        graph = _graph(db, AGENT_A)
        await graph.add_node(GraphNode(node_id, "episode", "A Tuesday", {"n": 1}))
        snapshot = (await graph.get_node(node_id)).properties
        fired = self._retype_mid_flight(
            monkeypatch, db, node_id, ARTIFACT_TYPE_LABEL, ARTIFACT_LABEL
        )

        result = await graph.compare_and_swap_node(
            node_id,
            snapshot,
            GraphNode(
                node_id,
                "episode",
                "A Tuesday",
                {"source_path": "/home/operator/a.json"},
            ),
        )

        assert fired["done"], "the race was never injected; the test proves nothing"
        assert result != NodeSwapResult.SWAPPED
        row = await db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        assert "source_path" not in (row[0] or ""), (
            "an unshareable payload landed on a row that became fleet-shared"
        )

    async def test_a_shared_row_that_stops_being_shared_mid_flight_is_not_written(
        self, db, constitution_bytes, monkeypatch
    ):
        """The hazard runs both ways: shared when checked, something else when
        written. The guards passed against a row that is no longer the row.

        On the *anchor* rather than the artifact, and not for convenience: every
        artifact field is an identity field, so the only swap that survives the
        pre-write checks there is one that changes nothing, and a test that
        writes nothing cannot show where the write landed. ``created_at`` is
        outside the anchor's identity set, so this is a real change that clears
        every check and must still be stopped by the predicate.
        """
        node_id = await _take_possession(
            db, AGENT_A, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        graph = _graph(db, AGENT_A)
        await graph.add_node(
            _constitution_node(node_id, "2026-01-01T00:00:00+00:00")
        )
        snapshot = (await graph.get_node(node_id)).properties
        fired = self._retype_mid_flight(
            monkeypatch, db, node_id, "episode", "A Tuesday"
        )

        result = await graph.compare_and_swap_node(
            node_id,
            snapshot,
            _constitution_node(node_id, "2027-01-01T00:00:00+00:00"),
        )

        assert fired["done"], "the race was never injected; the test proves nothing"
        assert result != NodeSwapResult.SWAPPED
        row = await db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        assert "2026-01-01" in (row[0] or ""), (
            "the swap landed on a row that had stopped being the shape it checked"
        )

    async def test_an_owner_joining_mid_flight_does_not_lose_its_stamp(
        self, db, constitution_bytes, monkeypatch
    ):
        """The owner count is a read, not a lock.

        A second tenant can commit its ownership witness between the count and
        the UPDATE, and the swap would then rewrite a row that had become
        co-owned in the interval — the exact write the count exists to refuse,
        landing anyway. Injected after the *owner* read specifically: injecting
        earlier would just make the count see two owners and raise, which is
        the case already covered and proves nothing about the window.
        """
        node_id = await _take_possession(
            db, AGENT_A, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        await _take_possession(
            db, AGENT_B, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        graph = _graph(db, AGENT_A)
        await graph.add_node(
            _constitution_node(node_id, "2026-01-01T00:00:00+00:00")
        )
        snapshot = (await graph.get_node(node_id)).properties

        original = db.fetchall
        fired = {"done": False}

        async def hooked(sql, params=None):
            rows = await original(sql, params)
            if (
                not fired["done"]
                and "FROM graph_node_owners WHERE node_id" in sql
            ):
                fired["done"] = True
                await db.execute(
                    "INSERT INTO graph_node_owners (node_id, agent_id) "
                    "VALUES (?, ?)",
                    (node_id, AGENT_B),
                )
            return rows

        monkeypatch.setattr(db, "fetchall", hooked)

        result = await graph.compare_and_swap_node(
            node_id,
            snapshot,
            _constitution_node(node_id, "2027-01-01T00:00:00+00:00"),
        )

        assert fired["done"], "the race was never injected; the test proves nothing"
        assert result != NodeSwapResult.SWAPPED
        row = await db.fetchone(
            "SELECT properties FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        assert "2026-01-01" in (row[0] or ""), (
            "a co-owner's stamp was overwritten by a swap that raced its arrival"
        )

    async def test_an_ordinary_concurrent_retype_still_coexists(self, db):
        """The documented behaviour this must not quietly break.

        ``compare_and_swap_node`` promises that a concurrent ``node_type`` /
        ``label`` change is neither detected nor clobbered — it writes
        ``properties`` only. A blanket "type must not have changed" predicate
        would have been the easy fix and would have broken that promise for
        every node type in the system.
        """
        graph = _graph(db, AGENT_A)
        node_id = f"episode:{uuid.uuid4().hex}"
        await graph.add_node(GraphNode(node_id, "episode", "A Tuesday", {"n": 1}))
        snapshot = (await graph.get_node(node_id)).properties

        await db.execute(
            "UPDATE graph_nodes SET node_type = ?, label = ? WHERE node_id = ?",
            ("concept", "Tuesdays", node_id),
        )

        result = await graph.compare_and_swap_node(
            node_id, snapshot, GraphNode(node_id, "episode", "A Tuesday", {"n": 2})
        )

        assert result == NodeSwapResult.SWAPPED
        stored = await graph.get_node(node_id)
        assert stored.properties == {"n": 2}
        assert stored.node_type == "concept"

    async def test_a_disallowed_stored_type_is_a_result_not_an_exception(
        self, db, constitution_bytes
    ):
        """The privacy wrapper's contract.

        It passes ``allowed_node_types`` and converts TYPE_NOT_ALLOWED into a
        PrivacyViolationError. Validating the new properties against the stored
        row's shared shape first raised instead — a wrapped TransactionError
        for what is a documented policy denial, skipping that conversion
        entirely.
        """
        node_id = await _take_possession(
            db, AGENT_A, constitution_bytes, "KESTREL_CONSTITUTION.md"
        )
        graph = _graph(db, AGENT_A)
        await graph.add_node(
            _constitution_node(node_id, "2026-01-01T00:00:00+00:00")
        )
        snapshot = (await graph.get_node(node_id)).properties

        result = await graph.compare_and_swap_node(
            node_id,
            snapshot,
            GraphNode(node_id, "audit_anchor", "Audit", {"checked": True}),
            allowed_node_types=frozenset({"audit_anchor"}),
        )

        assert result == NodeSwapResult.TYPE_NOT_ALLOWED
