"""Agent-routed observability reads return only the routed agent's rows (#3215).

The defect these cover is invisible on a SQLite-per-agent deployment,
because the file boundary does the scoping the query never asked for. So
these seed ONE store with TWO agents' rows — the shape a shared
PostgreSQL table has on a multi-agent host — and read through the real
endpoints.

`GET /api/observability/summary` queried with no agent predicate at all,
and `GET /api/observability/metrics/{name}` took `agent_name` as a query
parameter: omitting it summarised every agent, supplying it addressed
someone else's.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
)
from kestrel_sovereign.endpoints.observability import router


@pytest.fixture
def identities():
    """Two agent DIDs unique to this test.

    DIDs, not display names: `a2a_observability.agent_name` holds the
    DID despite its column name — every live recorder writes
    `agent_name=self.did`. The first version of this fixture seeded the
    display name and the endpoint read the display name, so the two
    agreed with each other and disagreed with production; a fix that
    scoped by `agent.agent_name` passed 4/4 while returning zero rows on
    a real deployment. The stub below therefore carries a `did` and a
    DIFFERENT `agent_name`, so reading the wrong one cannot pass.

    A shared PostgreSQL database is not torn down between tests, so fixed
    names accumulate rows from every earlier run and every earlier
    backend param — the first Postgres run of these tests read 9 events
    where it seeded 1. Exact-count assertions need identities nobody else
    has used; unique names give that without truncating a table this
    fixture does not own.
    """
    unique = uuid4().hex[:12]
    return (
        f"did:pkh:eip155:1:0xMINE{unique}",
        f"did:pkh:eip155:1:0xTHEIRS{unique}",
    )


@pytest.fixture
async def shared_store(db_backend, identities):
    """One store holding two agents' events.

    Built over the dual-backend fixture, so this runs on SQLite and — when
    `TEST_POSTGRES_URL` is set — on PostgreSQL, which is where the defect
    actually bites: one table serves the whole host there, while
    SQLite-per-agent hides it behind the file boundary.
    """
    mine, theirs = identities
    store = ObservabilityStore(db_backend)
    await store.initialize()

    await store.log_metric(agent_name=mine, metric_name=f"turns-{mine}", metric_value=1.0)
    await store.log_metric(agent_name=theirs, metric_name=f"turns-{mine}", metric_value=99.0)
    await store.log_metric(
        agent_name=theirs, metric_name=f"theirs-only-{mine}", metric_value=7.0
    )
    return store


def _client(store, agent_did: str) -> httpx.AsyncClient:
    """An in-loop ASGI client.

    Not `TestClient`: it drives the app on its own event loop in another
    thread, and an asyncpg pool created in the test's loop cannot be used
    from there — the Postgres runs failed with "connection was closed in
    the middle of operation" before the assertions were ever reached.
    `ASGITransport` keeps the request in this loop, so the same test can
    exercise both backends.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.agent = type(
        "Agent",
        (),
        {
            "observability_store": store,
            "did": agent_did,
            # Deliberately different from the DID. Rows are written under
            # the DID, so an endpoint that scopes by this reads nothing.
            "agent_name": f"display-name-for-{agent_did[-8:]}",
        },
    )()
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_summary_counts_only_the_routed_agents_events(
    shared_store, identities
):
    MINE, THEIRS = identities
    async with _client(shared_store, MINE) as client:
        mine = (await client.get("/api/observability/summary?minutes=60")).json()
    async with _client(shared_store, THEIRS) as client:
        theirs = (await client.get("/api/observability/summary?minutes=60")).json()

    assert mine["total_events"] == 1, mine
    assert theirs["total_events"] == 2, theirs
    # A real control, unlike the sum that used to sit here: that was
    # implied by the two equalities above and could not fail
    # independently. This one asserts the reads are non-empty for the
    # identity PRODUCTION writes, which is the failure an
    # empty-but-well-formed 200 produces (#3215 review F1).
    assert mine["events_by_type"], mine
    assert mine["metrics_by_name"], mine


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_metric_summary_sees_only_the_routed_agents_rows(
    shared_store, identities
):
    MINE, THEIRS = identities
    async with _client(shared_store, MINE) as client:
        mine = (await client.get(f"/api/observability/metrics/turns-{MINE}")).json()
    async with _client(shared_store, THEIRS) as client:
        theirs = (await client.get(f"/api/observability/metrics/turns-{MINE}")).json()

    assert mine["count"] == 1, mine
    assert theirs["count"] == 1, theirs
    assert list(mine.get("by_agent", {})) == [MINE], mine
    assert list(theirs.get("by_agent", {})) == [THEIRS], theirs


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_caller_supplied_agent_name_cannot_widen_the_scope(
    shared_store, identities
):
    """Routing to an agent is not authority over another one.

    The parameter is gone rather than validated: a validated one would
    have to answer "no such metric" differently from "not your agent",
    which leaks the same fact one step removed. FastAPI ignores unknown
    query parameters, so an old caller passing it now simply gets its
    own rows.
    """
    MINE, THEIRS = identities
    async with _client(shared_store, MINE) as client:
        # `/summary` too. It had no smuggle coverage, so reinstating a
        # caller-supplied override there — the ticket's own defect class,
        # on the other route — survived the whole suite.
        smuggled_summary = (
            await client.get(
                f"/api/observability/summary?minutes=60&agent_name={THEIRS}"
            )
        ).json()
        assert smuggled_summary["total_events"] == 1, smuggled_summary

        smuggled = (
            await client.get(
                f"/api/observability/metrics/turns-{MINE}?agent_name={THEIRS}"
            )
        ).json()
        other_metric = (
            await client.get(
                f"/api/observability/metrics/theirs-only-{MINE}?agent_name={THEIRS}"
            )
        ).json()

    assert smuggled["count"] == 1, smuggled
    assert list(smuggled.get("by_agent", {})) == [MINE], smuggled
    # A metric only the other agent ever emitted must read as absent,
    # not as forbidden — the endpoint should not confirm it exists.
    assert other_metric["count"] == 0, other_metric


def test_the_metrics_route_no_longer_accepts_an_agent_name_parameter():
    """Read off the signature, so the parameter cannot quietly return."""
    import inspect

    from kestrel_sovereign.endpoints.observability import get_metric_summary

    assert "agent_name" not in inspect.signature(get_metric_summary).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity",
    [
        pytest.param({}, id="attribute-absent"),
        pytest.param({"did": None}, id="did-is-none"),
        pytest.param({"did": ""}, id="did-is-empty-string"),
    ],
)
async def test_an_agent_without_a_resolvable_identity_is_refused(
    shared_store, identity
):
    """No identity means no read — never an unscoped one.

    The empty string is the dangerous one, and the reason the guard
    tests truthiness rather than `is None`. `query_events` and
    `get_metric_summary` both gate on `if agent_name:`, so `""` does not
    narrow the query — it silences the predicate entirely and returns
    every agent's rows in a well-formed 200. That is this ticket's own
    defect, reached through the fix for it.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.agent = type(
        "Agent",
        (),
        {
            "observability_store": shared_store,
            "agent_name": "display",
            **identity,
        },
    )()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        summary = await client.get("/api/observability/summary?minutes=60")
        metric = await client.get("/api/observability/metrics/turns")

    assert summary.status_code == 503, summary.text
    assert metric.status_code == 503, metric.text
