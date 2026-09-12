"""Bounded asyncpg worker for ``kestrel doctor``.

Doctor launches this module with the same executable, environment, and working
directory as the agent.  The process boundary lets the parent enforce a finite
deadline without changing any asyncpg connection setting.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys

ERROR_KIND_CONNECTION = "connection"
ERROR_KIND_QUERY = "query"
ERROR_KIND_DIAGNOSTIC = "diagnostic"


class ProbeError(RuntimeError):
    """A worker failure safe to return over the JSON protocol."""


class ProbeConnectionError(ProbeError):
    """The spawned runtime's asyncpg connection failed."""


class ProbeQueryError(ProbeError):
    """The connection opened, but the governance query failed."""


def _emit_phase(*, connected: bool) -> None:
    """Flush one bounded, non-secret breadcrumb outside the JSON channel."""
    phase = "connected; querying" if connected else "connecting"
    sys.stderr.write(f"PostgreSQL diagnostic phase: {phase}\n")
    sys.stderr.flush()


async def _fetch_rows(
    dsn: str,
    sql: str,
    params: tuple | list,
    *,
    connect=None,
    expected_cluster_identity: str | None = None,
) -> list:
    if connect is None:
        try:
            import asyncpg
        except ImportError as exc:
            raise ProbeError(
                "spawned agent Python environment could not import asyncpg"
            ) from exc
        connect = asyncpg.connect

    _emit_phase(connected=False)
    try:
        connection = await connect(dsn)
    except Exception as exc:
        raise ProbeConnectionError(str(exc)) from exc

    _emit_phase(connected=True)
    try:
        if expected_cluster_identity is not None:
            actual_cluster_identity = await connection.fetchval(
                "SELECT system_identifier::text "
                "FROM pg_catalog.pg_control_system()"
            )
            if (
                not isinstance(actual_cluster_identity, str)
                or not actual_cluster_identity.strip()
                or actual_cluster_identity != expected_cluster_identity
            ):
                raise ProbeQueryError(
                    "diagnostic query connection is not on the expected "
                    "PostgreSQL cluster"
                )
        records = await connection.fetch(sql, *params)
    except ProbeQueryError:
        with contextlib.suppress(Exception):
            await connection.close()
        raise
    except Exception as exc:
        with contextlib.suppress(Exception):
            await connection.close()
        raise ProbeQueryError(str(exc)) from exc

    rows = [list(record) for record in records]
    try:
        await connection.close()
    except Exception as exc:
        raise ProbeError(
            "asyncpg could not close the diagnostic connection cleanly"
        ) from exc
    return rows


def fetch_rows_in_process(
    dsn: str,
    sql: str,
    params: tuple | list,
    *,
    connect=None,
    expected_cluster_identity: str | None = None,
) -> list:
    """Connect and query through asyncpg in the current process."""
    return asyncio.run(
        _fetch_rows(
            dsn,
            sql,
            params,
            connect=connect,
            expected_cluster_identity=expected_cluster_identity,
        )
    )


def main() -> None:
    """Serve one JSON request on stdin and one JSON response on stdout."""
    try:
        request = json.loads(sys.stdin.read())
        rows = fetch_rows_in_process(
            request["dsn"],
            request["sql"],
            request.get("params", []),
            expected_cluster_identity=request.get("expected_cluster_identity"),
        )
        output = json.dumps({"ok": True, "rows": rows})
    except ProbeConnectionError as exc:
        output = json.dumps(
            {"ok": False, "kind": ERROR_KIND_CONNECTION, "error": str(exc)}
        )
    except ProbeQueryError as exc:
        output = json.dumps(
            {"ok": False, "kind": ERROR_KIND_QUERY, "error": str(exc)}
        )
    except Exception as exc:  # noqa: BLE001 - protocol carries redacted text later
        output = json.dumps(
            {"ok": False, "kind": ERROR_KIND_DIAGNOSTIC, "error": str(exc)}
        )
    sys.stdout.write(output)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    main()
