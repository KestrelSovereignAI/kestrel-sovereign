"""Real bridge SQL/tools preserve original native denial before fallback.

Host/request fields are declared component inputs. Actual scoped PostgreSQL
and fixture-owned SQLite queries execute before the actual retired PG lease
denial is delivered at named callbacks. No Task, SQL result or ACK is faked.
This is not full Agent READY, authentication or native lost-COMMIT proof.
"""

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.features.bridge import router as bridge_router
from kestrel_sovereign.features.bridge.feature import BridgeFeature
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg

DOORS = (
    "initialize",
    "status-sessions",
    "status-log",
    "connections",
    "history",
    "session-lookup",
    "session-create",
    "session-touch",
    "log-invocation",
    "capabilities",
    "session-json",
    "health-resolver",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", DOORS)
@pytest.mark.parametrize("kind", ["native-control", "ordinary", "healthy"])
async def test_actual_bridge_consumers_preserve_original_native_denial(
    native_memory_owners, monkeypatch, door, kind
):
    n = native_memory_owners
    # Exercise the portable hour-count consumer on actual SQLite as well as
    # the PostgreSQL status case. Neither backend supplies a fabricated count.
    db = (
        n.sqlite_saved.db
        if door in {"status-log", "health-resolver"}
        else n.pg_storage.db
    )
    agent = SimpleNamespace(
        did=n.did, agent_id=n.did, storage=SimpleNamespace(db=db), features={}
    )
    feature = BridgeFeature(agent)
    agent.features["BridgeFeature"] = feature
    sources, reached, subsequent_sql = [], [], []

    def fault():
        reached.append(door)
        if kind == "native-control":
            try:
                n.lease.require_live()
            except BaseException as original:
                sources.append(original)
                raise
        if kind == "ordinary":
            raise OSError("ordinary local bridge observation unavailable")

    def selected(sql):
        return {
            "initialize": "CREATE TABLE IF NOT EXISTS bridge_sessions" in sql,
            "status-sessions": "COUNT(*) FROM bridge_sessions" in sql,
            "status-log": "COUNT(*) FROM bridge_log" in sql,
            "connections": "FROM bridge_sessions" in sql and "ORDER BY" in sql,
            "history": "FROM bridge_log" in sql and "ORDER BY" in sql,
            "session-lookup": "WHERE gateway_session_id" in sql,
            "session-create": "INSERT INTO bridge_sessions" in sql,
            "session-touch": "UPDATE bridge_sessions" in sql,
            "log-invocation": "INSERT INTO bridge_log" in sql,
        }.get(door, False)

    def observe_sql(name):
        actual = getattr(db, name)

        async def observed(sql, *args, **kwargs):
            if sources:
                subsequent_sql.append(sql)
            result = await actual(sql, *args, **kwargs)
            if selected(sql):
                fault()
            return result

        monkeypatch.setattr(db, name, observed)

    try:
        if door != "initialize":
            await feature.initialize()
            assert feature._db is db
            session = await feature.get_or_create_session("existing-gateway")
            await feature.log_invocation(session.id, "inbound", "actual prior entry")
            if kind == "healthy" and door.startswith("status-"):
                await db.execute(
                    "INSERT INTO bridge_log (id, agent_id, direction, created_at) VALUES (?,?,?,?)",
                    (
                        "old-bridge-entry",
                        n.did,
                        "inbound",
                        (
                            datetime.now(timezone.utc) - timedelta(minutes=61)
                        ).isoformat(),
                    ),
                )
            if door == "session-lookup":
                feature._sessions.clear()
        for name in ("execute", "fetchone", "fetchall"):
            observe_sql(name)
        if door == "capabilities":
            actual_tools = feature.get_tools

            def tools():
                result = actual_tools()
                fault()
                return result

            monkeypatch.setattr(feature, "get_tools", tools)

        async def receive():
            return {
                "type": "http.request",
                "body": b'{"session_id":"new-gateway"}',
                "more_body": False,
            }

        request = Request(
            {"type": "http", "method": "POST", "path": "/component"}, receive
        )
        if door == "session-json":
            actual_json = request.json

            async def body():
                result = await actual_json()
                fault()
                return result

            monkeypatch.setattr(request, "json", body)
            monkeypatch.setattr(bridge_router, "get_agent", lambda request: agent)
        elif door == "health-resolver":

            def resolve(request):
                # Declared request-context input. The compatibility HTTP
                # carrier preserves the SAME actual native lease denial.
                try:
                    fault()
                except BaseException as original:
                    raise HTTPException(503, "bridge agent unavailable") from original
                return agent

            monkeypatch.setattr(bridge_router, "get_agent", resolve)
        caught = result = None
        try:
            if door == "initialize":
                result = await feature.initialize()
            elif door.startswith("status-"):
                result = await feature.bridge_status()
            elif door == "connections":
                result = await feature.bridge_connections()
            elif door == "history":
                result = await feature.bridge_history()
            elif door in {"session-lookup", "session-touch"}:
                result = await feature.get_or_create_session("existing-gateway")
            elif door == "session-create":
                result = await feature.get_or_create_session("new-gateway")
            elif door == "log-invocation":
                result = await feature.log_invocation(
                    session.id, "outbound", "actual new entry"
                )
            elif door == "capabilities":
                result = feature.get_capabilities()
            elif door == "health-resolver":
                endpoint = next(
                    route.endpoint
                    for route in bridge_router.get_router().routes
                    if route.path.endswith("/health")
                )
                result = await inspect.unwrap(endpoint)(request)
            else:
                endpoint = next(
                    route.endpoint
                    for route in bridge_router.get_router().routes
                    if route.path.endswith("/session")
                )
                result = await inspect.unwrap(endpoint)(request)
        except BaseException as error:
            caught = error
        assert reached, (
            f"declared actual bridge boundary not reached: {door} {caught!r} {result!r}"
        )
        if kind == "native-control":
            assert (
                sources
                and caught is not None
                and sources[0] in tuple(_explicit_control_chain(caught))
            ), f"bridge normalized original native denial: {door} {caught!r} {result!r}"
            assert is_execution_control_error(caught)
            assert not subsequent_sql, (
                "native denial admitted another bridge query/effect"
            )
        else:
            assert caught is None, (
                f"ordinary/healthy compatibility changed: {door} {caught!r}"
            )
            assert not is_execution_control_error(caught)
            if kind == "healthy" and door.startswith("status-"):
                assert result.status is ToolResultStatus.OK
                assert result.data["total_sessions_db"] == 1
                assert result.data["recent_invocations_1h"] == 1
        assert not n.requests
    finally:
        await feature.shutdown()
