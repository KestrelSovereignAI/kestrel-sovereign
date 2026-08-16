"""Minimal environment-isolated PostgreSQL worker for ``kestrel doctor``.

This module intentionally imports no Kestrel runtime code.  Doctor launches it
in a short-lived process whose ``PG*`` namespace has been removed, preventing
libpq-only environment settings from changing the asyncpg connection that the
parent already translated into an explicit DSN.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"


class ProbeError(RuntimeError):
    """A driver failure safe to return over the worker protocol."""


class ProbeConnectionError(ProbeError):
    """The equivalent libpq connection itself failed."""


class ProbeQueryError(ProbeError):
    """The connection opened, but the governance query failed."""


def _query_postgres_driver(driver, dsn: str, sql: str, params) -> list:
    """Run one query through a psycopg2-compatible driver."""
    try:
        connection = driver.connect(dsn)
    except Exception as exc:
        raise ProbeConnectionError(str(exc)) from exc
    try:
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as exc:
            raise ProbeQueryError(str(exc)) from exc
    finally:
        connection.close()


def fetch_rows_in_process(
    dsn: str,
    sql: str,
    params: tuple | list,
    *,
    absent_passfile_sentinel: str,
    driver=None,
) -> list:
    """Connect and query inside the already-isolated worker process."""
    if driver is None:
        import psycopg2 as driver

    effective_dsn = dsn
    with tempfile.TemporaryDirectory(prefix="kestrel-doctor-") as temp_dir:
        # Windows libpq searches %APPDATA%/postgresql for default TLS files,
        # while asyncpg searches %USERPROFILE%/.postgresql.  The parent has
        # frozen any asyncpg-visible files into the DSN, so hide libpq's
        # unrelated defaults for the duration of this one child-local query.
        old_appdata = os.environ.get("APPDATA")
        had_appdata = "APPDATA" in os.environ
        if _IS_WINDOWS:
            os.environ["APPDATA"] = str(Path(temp_dir) / "absent-appdata")
        if driver.extensions.parse_dsn(dsn).get("passfile") == absent_passfile_sentinel:
            # The worker directory has private mkdtemp custody.  Its ``absent``
            # child is never created, so no process can race a passfile into
            # place between translation and libpq's open.
            passfile = Path(temp_dir) / "absent" / "pgpass"
            effective_dsn = driver.extensions.make_dsn(dsn, passfile=str(passfile))

        try:
            try:
                return _query_postgres_driver(driver, effective_dsn, sql, params)
            except ProbeError as exc:
                message = str(exc).replace(temp_dir, "<probe-temp>")
                raise type(exc)(message) from exc
            except Exception as exc:
                message = str(exc).replace(temp_dir, "<probe-temp>")
                raise ProbeError(message) from exc
        finally:
            if _IS_WINDOWS:
                if had_appdata:
                    os.environ["APPDATA"] = old_appdata or ""
                else:
                    os.environ.pop("APPDATA", None)


def main() -> None:
    """Serve one JSON request on stdin and one JSON response on stdout."""
    try:
        request = json.loads(sys.stdin.read())
        rows = fetch_rows_in_process(
            request["dsn"],
            request["sql"],
            request.get("params", []),
            absent_passfile_sentinel=request["absent_passfile_sentinel"],
        )
        # Deliberately no ``default=str``: every current governance query
        # returns JSON-native scalars.  A future query that does not must fail
        # closed rather than silently changing bytes/datetime/Decimal values.
        output = json.dumps({"ok": True, "rows": rows})
    except ProbeConnectionError as exc:
        output = json.dumps({"ok": False, "kind": "connection", "error": str(exc)})
    except ProbeQueryError as exc:
        output = json.dumps({"ok": False, "kind": "query", "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - protocol carries safe text only
        output = json.dumps({"ok": False, "kind": "diagnostic", "error": str(exc)})
    sys.stdout.write(output)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    main()
