"""#2892 — ``kestrel doctor`` against a PostgreSQL runtime.

Cannot be reproduced on SQLite, where ``kestrel_prime.db`` *is* the database
the runtime reads. With ``KESTREL_DB_BACKEND=postgres`` they are two databases:
the anchor holds the birth record (#2871, kept byte-for-byte) and the agent's
live governance is in PostgreSQL. Doctor read the anchor unconditionally, so it
reported birth-time state as current — and after #2890 made
``kestrel constitution reanchor`` write PostgreSQL and deliberately leave the
anchor alone, that means doctor flags drift *permanently* while the repair it
prescribes correctly answers "nothing to do".

The unit suite fakes psycopg2. This runs the real driver against a real
database, because "the query I wrote is the query PostgreSQL accepts" is
exactly what a fake cannot tell me.

Run against any throwaway PostgreSQL:

    TEST_POSTGRES_URL=postgresql://u:p@127.0.0.1:5432/db pytest \
        tests/integration/test_doctor_postgres.py

Skipped when that is not set, so CI stays green.

The ``sslmode=prefer`` plus direct-negotiation cases use asyncpg's parser as
their runtime precondition. Asyncpg 0.30 attempts direct TLS before plaintext,
and that transport attempt is rejected before its fallback on stock servers
that lack direct TLS (PostgreSQL 16) or require ALPN for it (PostgreSQL 17).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.doctor import (
    _LIBPQ_COMPILED_DSN_DEFAULTS,
    diagnose,
)
from kestrel_sovereign.multi_agent.config import (
    MULTI_AGENT_CONFIG_FILENAME,
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
)
from kestrel_sovereign.setup.env_file import write_env
from kestrel_sovereign.setup.toml_file import write_toml

pytestmark = [pytest.mark.integration]

POSTGRES_URL = (
    os.environ.get("TEST_POSTGRES_URL")
    or os.environ.get("KESTREL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)

if not POSTGRES_URL:  # pragma: no cover - environment gate
    pytest.skip(
        "TEST_POSTGRES_URL / KESTREL_DATABASE_URL / DATABASE_URL required",
        allow_module_level=True,
    )

CONSTITUTION = b"# Kestrel Constitution\n\nv1, for the doctor.\n"
AGENT_DID = "did:web:doctor.example:kestrel"
NEIGHBOUR_DID = "did:web:doctor.example:neighbour"


@pytest.fixture
def runtime_db():
    """A throwaway database, and a hand to seed it with.

    Doctor's whole job here is to read what the *runtime* holds, so the fixture
    writes rows the way the runtime would rather than going through storage.

    "The way the runtime would" includes the ownership ledgers. A bound
    ``AsyncGraphStore`` matches on ``graph_node_owners`` / ``graph_edge_owners``
    rather than on ``node_id``, so a fixture that seeds only the rows describes
    a database in which the agent can see nothing at all — and any assertion
    about what doctor reports would then be about the wrong host entirely.
    """
    import psycopg2

    name = f"kestrel_doctor_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(POSTGRES_URL)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE "{name}"')
    finally:
        admin.close()

    dsn = POSTGRES_URL.rsplit("/", 1)[0] + "/" + name
    connection = psycopg2.connect(dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        # The real columns (async_database._SCHEMA). A hand-rolled subset
        # made the bound store raise on a plain get_node, which reads as
        # "database unreachable" to any caller that fails closed.
        cursor.execute(
            "CREATE TABLE graph_nodes ("
            " node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL,"
            " label TEXT NOT NULL, properties TEXT)"
        )
        cursor.execute(
            "CREATE TABLE graph_edges ("
            " source_id TEXT NOT NULL, target_id TEXT NOT NULL,"
            " label TEXT NOT NULL, properties TEXT,"
            " PRIMARY KEY (source_id, target_id, label))"
        )
        cursor.execute(
            "CREATE TABLE graph_node_owners (node_id TEXT, agent_id TEXT)"
        )
        cursor.execute(
            "CREATE TABLE graph_edge_owners ("
            " source_id TEXT, target_id TEXT, label TEXT, agent_id TEXT)"
        )
        # Every real database carries this (CORE_SCHEMA), and one the runtime
        # has opened has #2649 recorded — which is what makes a *missing*
        # ownership witness permanent rather than merely pending.
        cursor.execute(
            "CREATE TABLE schema_backfills ("
            " name TEXT PRIMARY KEY,"
            " completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cursor.execute(
            "INSERT INTO schema_backfills (name) VALUES ('ownership_2649')"
        )

    def seed(
        did: str,
        properties: dict,
        governed_by: str | None,
        *,
        witness_node: bool = True,
        witness_edge: bool = True,
    ):
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO graph_nodes "
                "(node_id, node_type, label, properties) "
                "VALUES (%s, 'agent', %s, %s)",
                (did, properties.get("name", "agent"), json.dumps(properties)),
            )
            if witness_node:
                cursor.execute(
                    "INSERT INTO graph_node_owners (node_id, agent_id) "
                    "VALUES (%s, %s)",
                    (did, did),
                )
            if governed_by is not None:
                cursor.execute(
                    "INSERT INTO graph_edges (source_id, target_id, label) "
                    "VALUES (%s, %s, 'governed_by')",
                    (did, governed_by),
                )
                if witness_edge:
                    cursor.execute(
                        "INSERT INTO graph_edge_owners "
                        "(source_id, target_id, label, agent_id) "
                        "VALUES (%s, %s, 'governed_by', %s)",
                        (did, governed_by, did),
                    )

    seed.dsn = dsn
    try:
        yield seed
    finally:
        connection.close()
        admin = psycopg2.connect(POSTGRES_URL)
        admin.autocommit = True
        try:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    (name,),
                )
                cursor.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            admin.close()


def _seed_governance_schema(dsn: str, schema: str, constitution_hash: str) -> None:
    """Create one complete, owned governance state outside ``public``."""
    import psycopg2
    from psycopg2 import sql

    connection = psycopg2.connect(dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )
            for table in (
                "graph_nodes",
                "graph_edges",
                "graph_node_owners",
                "graph_edge_owners",
                "schema_backfills",
            ):
                cursor.execute(
                    sql.SQL("CREATE TABLE {}.{} (LIKE {})").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(table),
                    )
                )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {}.graph_nodes "
                    "(node_id, node_type, label, properties) "
                    "VALUES (%s, 'agent', 'Test', %s)"
                ).format(sql.Identifier(schema)),
                (
                    AGENT_DID,
                    json.dumps(
                        {"name": "Test", "constitution_hash": constitution_hash}
                    ),
                ),
            )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {}.graph_node_owners (node_id, agent_id) "
                    "VALUES (%s, %s)"
                ).format(sql.Identifier(schema)),
                (AGENT_DID, AGENT_DID),
            )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {}.graph_edges "
                    "(source_id, target_id, label) "
                    "VALUES (%s, %s, 'governed_by')"
                ).format(sql.Identifier(schema)),
                (AGENT_DID, constitution_hash),
            )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {}.graph_edge_owners "
                    "(source_id, target_id, label, agent_id) "
                    "VALUES (%s, %s, 'governed_by', %s)"
                ).format(sql.Identifier(schema)),
                (AGENT_DID, constitution_hash, AGENT_DID),
            )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {}.schema_backfills (name) "
                    "VALUES ('ownership_2649')"
                ).format(sql.Identifier(schema))
            )
    finally:
        connection.close()


def _seed_project(tmp_path: Path, anchored_hash: str) -> Path:
    """A ready project whose *anchor* carries ``anchored_hash``."""
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
            "OPENAI_API_KEY": "sk-x",
        },
    )
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {
                "route_priority": ["openai:api"],
                "vendors": {
                    "openai": {
                        "is_cloud": True,
                        "routes": {
                            "api": {
                                "adapter": "OpenAIAdapter",
                                "api_key_env": "OPENAI_API_KEY",
                            }
                        },
                    }
                },
            }
        },
    )
    MultiAgentConfig(
        host=HostConfig(),
        agents={
            "Test": LocalAgentConfig(
                data_dir=Path("agent_data/test"), port=8801, autostart=True
            )
        },
    ).save(tmp_path / MULTI_AGENT_CONFIG_FILENAME)

    db_dir = tmp_path / "agent_data" / "test"
    db_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_dir / "kestrel_prime.db")
    try:
        connection.execute(
            "CREATE TABLE graph_nodes ("
            " node_id TEXT PRIMARY KEY, node_type TEXT,"
            " label TEXT, properties TEXT)"
        )
        # The real columns and the real ownership ledgers. A hand-rolled
        # subset has bitten three times on this branch: a bound read selects
        # ``properties`` and joins ``graph_edge_owners``, so a fixture missing
        # either makes the store raise — which every caller that fails closed
        # reads as "database unreachable", not as "your fixture is wrong".
        connection.execute(
            "CREATE TABLE graph_edges ("
            " source_id TEXT NOT NULL, target_id TEXT NOT NULL,"
            " label TEXT NOT NULL, properties TEXT,"
            " PRIMARY KEY (source_id, target_id, label))"
        )
        connection.execute(
            "CREATE TABLE graph_node_owners (node_id TEXT, agent_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE graph_edge_owners ("
            " source_id TEXT, target_id TEXT, label TEXT, agent_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE schema_backfills ("
            " name TEXT PRIMARY KEY,"
            " completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
            "INSERT INTO schema_backfills (name) VALUES ('ownership_2649')"
        )
        connection.execute(
            "INSERT INTO graph_nodes VALUES (?, 'agent', ?, ?)",
            (
                AGENT_DID,
                "Test",
                json.dumps({"name": "Test", "constitution_hash": anchored_hash}),
            ),
        )
        connection.execute(
            "INSERT INTO graph_node_owners VALUES (?, ?)", (AGENT_DID, AGENT_DID)
        )
        connection.execute(
            "INSERT INTO graph_edges VALUES (?, ?, 'governed_by', NULL)",
            (AGENT_DID, anchored_hash),
        )
        connection.execute(
            "INSERT INTO graph_edge_owners VALUES (?, ?, 'governed_by', ?)",
            (AGENT_DID, anchored_hash, AGENT_DID),
        )
        connection.commit()
    finally:
        connection.close()
    return tmp_path


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    path = tmp_path / "KESTREL_CONSTITUTION.md"
    path.write_bytes(CONSTITUTION)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(path))
    return hashlib.sha256(CONSTITUTION).hexdigest()


def test_doctor_reads_the_database_the_agent_is_governed_by(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """The anchor matches the canonical file; PostgreSQL is one reanchor behind.

    Reading the anchor gives a clean bill of health to an agent that is
    actually governed by something else.
    """
    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID, {"name": "Test", "constitution_hash": stale}, governed_by=stale
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert any(
        "constitution drift" in m and stale[:12] in m for m in report.fail
    ), f"ok={report.ok} warn={report.warn} fail={report.fail}"


def test_doctor_agrees_with_the_runtime_after_a_reanchor(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """The case #2892 is really about: a legitimate reanchor moved PostgreSQL
    and deliberately left the anchor on the birth record (#2890). Doctor must
    now say the agent is current — otherwise it flags drift forever and
    prescribes a repair that answers "nothing to do"."""
    birth = hashlib.sha256(b"the constitution at inception").hexdigest()
    _seed_project(tmp_path, anchored_hash=birth)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert not any("constitution drift" in m for m in report.fail), report.fail
    assert any(
        "constitution anchored to current file" in m for m in report.ok
    ), report.ok


def test_doctor_does_not_answer_about_a_neighbouring_agent(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """One PostgreSQL holds every local agent. Seed the neighbour so that a
    ``LIMIT 1`` read would have something wrong to find — with the tenant scope
    removed the neighbour's stale hash is what comes back."""
    stale = hashlib.sha256(b"the neighbour's older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=canonical)
    # Neighbour first, so incidental row order favours it.
    runtime_db(
        NEIGHBOUR_DID,
        {"name": "Neighbour", "constitution_hash": stale},
        governed_by=stale,
    )
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert not any(stale[:12] in m for m in report.fail + report.warn), (
        "doctor answered about a neighbouring agent"
    )
    assert any("constitution anchored to current file" in m for m in report.ok)


def test_the_project_env_alone_is_enough_to_reach_postgres(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """The configuration an operator actually has, against a real database.

    Every other test here *exports* the two settings, which is not how a
    Kestrel host is set up: ``.env.example`` documents them in the project
    ``.env`` and ``kestrel setup`` writes them there. Neither ``cmd_doctor``
    nor ``setup --check`` loads that file, so a doctor reading ``os.environ``
    alone silently fell back to the anchor on every real PostgreSQL host —
    passing this file's entire suite while fixing nothing.

    So this one deletes the exports and puts the settings only where they
    really live.
    """
    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID, {"name": "Test", "constitution_hash": stale}, governed_by=stale
    )
    from psycopg2.extensions import parse_dsn

    parsed = parse_dsn(runtime_db.dsn)
    pg_env_names = {
        "host": "PGHOST",
        "port": "PGPORT",
        "user": "PGUSER",
        "password": "PGPASSWORD",
        "sslmode": "PGSSLMODE",
        "sslrootcert": "PGSSLROOTCERT",
    }
    connection_env = {
        env_name: parsed[dsn_name]
        for dsn_name, env_name in pg_env_names.items()
        if parsed.get(dsn_name) is not None
    }
    for name in (
        "KESTREL_DB_BACKEND",
        "KESTREL_DATABASE_URL",
        *connection_env,
    ):
        monkeypatch.delenv(name, raising=False)
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DB_BACKEND": "postgres",
            # Only the database remains in the URI. Host, account, password,
            # and TLS data live exclusively in the project .env's PG* keys.
            "KESTREL_DATABASE_URL": (
                "postgresql:///" + quote(parsed["dbname"], safe="")
            ),
            **connection_env,
        },
    )

    report = diagnose(tmp_path)

    assert any(
        "constitution drift" in m and stale[:12] in m for m in report.fail
    ), f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert "KESTREL_DB_BACKEND" not in os.environ, (
        "a diagnostic must not export the project's environment"
    )
    assert all(name not in os.environ for name in connection_env)


def test_relative_pgpassfile_is_resolved_from_the_agent_working_directory(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Doctor can be invoked elsewhere, but the spawned agent runs here."""
    from psycopg2.extensions import parse_dsn

    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    parsed = parse_dsn(runtime_db.dsn)
    if not parsed.get("password"):
        pytest.skip("live PostgreSQL URL has no password to exercise pgpass")
    if any(character in parsed["password"] for character in (":", "\\", "\n")):
        pytest.skip("asyncpg pgpass fixture requires an unescaped password")

    passfile = tmp_path / "secrets" / "runtime.pgpass"
    passfile.parent.mkdir()
    passfile.write_text(f'*:*:*:*:{parsed["password"]}\n')
    passfile.chmod(0o600)

    for name in (
        "KESTREL_DB_BACKEND",
        "KESTREL_DATABASE_URL",
        "PGHOST",
        "PGPORT",
        "PGUSER",
        "PGPASSWORD",
        "PGPASSFILE",
        "PGSSLMODE",
    ):
        monkeypatch.delenv(name, raising=False)
    project_env = {
        "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
        "OPENAI_API_KEY": "sk-x",
        "KESTREL_DB_BACKEND": "postgres",
        "KESTREL_DATABASE_URL": (
            "postgresql:///" + quote(parsed["dbname"], safe="")
        ),
        "PGHOST": parsed["host"],
        "PGPORT": parsed["port"],
        "PGUSER": parsed["user"],
        "PGPASSFILE": "secrets/runtime.pgpass",
    }
    if parsed.get("sslmode"):
        project_env["PGSSLMODE"] = parsed["sslmode"]
    write_env(tmp_path / ".env", project_env)
    invocation_dir = tmp_path / "nested" / "invocation"
    invocation_dir.mkdir(parents=True)
    monkeypatch.chdir(invocation_dir)

    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"


async def test_multi_host_pgpass_password_is_frozen_before_fallback(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Doctor and asyncpg reuse the first host's credential on host two.

    Raw libpq would read the deliberately-wrong second row after the refused
    first connection. The translated diagnostic must instead fold the single
    password asyncpg selected before it began trying hosts.
    """
    import asyncpg
    from psycopg2.extensions import parse_dsn

    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    parsed = parse_dsn(runtime_db.dsn)
    required = ("host", "port", "user", "password", "dbname")
    if any(not parsed.get(field) for field in required):
        pytest.skip("live PostgreSQL URL needs TCP host, user, and password")
    if any(
        character in parsed[field]
        for field in ("host", "user", "password", "dbname")
        for character in (":", "\\", "\n")
    ):
        pytest.skip("adversarial pgpass fixture requires unescaped fields")

    bad_port = "1" if parsed["port"] != "1" else "2"
    passfile_host = (
        "localhost" if parsed["host"].startswith("/") else parsed["host"]
    )
    passfile = tmp_path / "fallback.pgpass"
    passfile.write_text(
        f'{passfile_host}:{bad_port}:{parsed["dbname"]}:'
        f'{parsed["user"]}:{parsed["password"]}\n'
        f'{passfile_host}:{parsed["port"]}:{parsed["dbname"]}:'
        f'{parsed["user"]}:deliberately-wrong-password\n'
    )
    passfile.chmod(0o600)
    query = {
        "host": f'{parsed["host"]},{parsed["host"]}',
        "port": f'{bad_port},{parsed["port"]}',
        "user": parsed["user"],
        "passfile": str(passfile),
    }
    if parsed.get("sslmode"):
        query["sslmode"] = parsed["sslmode"]
    def build_dsn(values):
        return (
            "postgresql:///"
            + quote(parsed["dbname"], safe="")
            + "?"
            + "&".join(
                f"{name}={quote(value, safe='')}"
                for name, value in values.items()
            )
        )

    runtime_dsn = build_dsn(query)
    wrong_password_query = {
        **query,
        "host": parsed["host"],
        "port": parsed["port"],
        "password": "deliberately-wrong-password",
    }
    wrong_password_query.pop("passfile")

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_dsn)
    with pytest.MonkeyPatch.context() as connection_env:
        for name in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGPASSFILE"):
            connection_env.delenv(name, raising=False)
        try:
            wrong_connection = await asyncpg.connect(
                build_dsn(wrong_password_query)
            )
        except asyncpg.InvalidPasswordError:
            pass
        else:
            await wrong_connection.close()
            pytest.skip("live PostgreSQL server does not require a password")
        connection = await asyncpg.connect(runtime_dsn)
        await connection.close()
        report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"


@pytest.mark.skip(
    reason=(
        "production SQLAlchemy DSN parity for asyncpg startup settings is "
        "tracked by #2984"
    )
)
def test_asyncpg_search_path_is_applied_to_doctors_postgres_session(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Exercise startup-setting translation after production parity lands.

    The governance tables exist only in the tenant schema, so reading public
    would produce a different verdict rather than merely exercising URI
    parsing.
    """
    import psycopg2
    from psycopg2 import sql

    stale = hashlib.sha256(b"tenant schema was not read").hexdigest()
    _seed_project(tmp_path, anchored_hash=stale)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    schema = f"doctor_{uuid.uuid4().hex[:12]}"
    connection = psycopg2.connect(runtime_db.dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )
            for table in (
                "graph_nodes",
                "graph_edges",
                "graph_node_owners",
                "graph_edge_owners",
                "schema_backfills",
            ):
                cursor.execute(
                    sql.SQL("ALTER TABLE {} SET SCHEMA {}").format(
                        sql.Identifier(table), sql.Identifier(schema)
                    )
                )
    finally:
        connection.close()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv(
        "KESTREL_DATABASE_URL",
        runtime_db.dsn + "?search_path=" + quote(schema, safe=""),
    )

    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert any(
        "constitution anchored to current file" in message
        for message in report.ok
    ), report.ok


async def test_direct_startup_setting_outranks_options_for_doctor_and_asyncpg(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Both drivers must read tenant even when URI order suggests public."""
    import asyncpg

    stale = hashlib.sha256(b"public schema must lose precedence").hexdigest()
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": stale},
        governed_by=stale,
    )
    schema = f"tenant_{uuid.uuid4().hex[:12]}"
    _seed_governance_schema(runtime_db.dsn, schema, canonical)
    runtime_dsn = (
        runtime_db.dsn
        + "?search_path="
        + quote(schema, safe="")
        + "&options=-c%20search_path%3Dpublic"
    )

    connection = await asyncpg.connect(runtime_dsn)
    try:
        assert await connection.fetchval("SELECT current_schema()") == schema
    finally:
        await connection.close()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_dsn)
    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert any(
        "constitution anchored to current file" in message
        for message in report.ok
    ), report.ok


def test_libpq_only_pgoptions_cannot_change_doctors_schema(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Doctor ignores the same ambient PGOPTIONS that asyncpg ignores.

    The leaked schema deliberately carries stale governance. If the translated
    DSN stops stating ``options=``, libpq inherits PGOPTIONS, reads that stale
    tenant, and this test reports drift instead of passing vacuously.
    """
    stale = hashlib.sha256(b"ambient PGOPTIONS changed the schema").hexdigest()
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    schema = f"leaked_{uuid.uuid4().hex[:12]}"
    _seed_governance_schema(runtime_db.dsn, schema, stale)

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={schema}")

    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert any(
        "constitution anchored to current file" in message
        for message in report.ok
    ), report.ok


def test_libpq_service_recipe_cannot_redirect_doctor(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """A libpq recipe must not redirect doctor away from asyncpg's database."""
    stale = hashlib.sha256(b"libpq service selected the anchor").hexdigest()
    _seed_project(tmp_path, anchored_hash=stale)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)
    service_file = tmp_path / "pg_service.conf"
    service_file.write_text(
        "[libpq_only_recipe]\n"
        "host=127.0.0.1\n"
        "port=1\n"
        "dbname=not_the_runtime_database\n"
        "user=private_service_identity\n"
        "password=private_service_password\n"
        "sslmode=verify-full\n"
        "sslrootcert=/private/service/root.crt\n"
        "require_auth=gss\n"
    )
    # This context must close before ``runtime_db`` tears down: its finalizer
    # also connects through libpq and must not inherit the deliberately
    # divergent recipe used only for this diagnostic.
    with pytest.MonkeyPatch.context() as service_env:
        service_env.setenv("PGSERVICE", "libpq_only_recipe")
        service_env.setenv("PGSERVICEFILE", str(service_file))
        report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert any(
        "constitution anchored to current file" in message
        for message in report.ok
    ), report.ok
    assert "private_service_identity" not in " ".join(
        report.ok + report.warn + report.fail
    )


def test_missing_ambient_libpq_service_cannot_block_doctor(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Libpq must not resolve a service name that asyncpg never reads."""
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)
    service_name = "private_missing_service_identity"

    with pytest.MonkeyPatch.context() as service_env:
        service_env.setenv("PGSERVICE", service_name)
        service_env.setenv("PGSERVICEFILE", str(tmp_path / "missing.conf"))
        report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert service_name not in " ".join(report.ok + report.warn + report.fail)


async def test_bare_slash_database_cannot_fall_through_to_pgdatabase(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Doctor and asyncpg must avoid the unrelated PGDATABASE target."""
    import asyncpg
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    _seed_project(tmp_path, anchored_hash=canonical)
    unrelated_hash = "f" * 64
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": unrelated_hash},
        governed_by=unrelated_hash,
    )
    bare_database_dsn = runtime_db.dsn.rsplit("/", 1)[0] + "/"
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", bare_database_dsn)
    monkeypatch.setenv(
        "PGDATABASE", urlsplit(runtime_db.dsn).path.removeprefix("/")
    )

    translated = parse_dsn(
        _doctor_postgres_dsn(
            bare_database_dsn,
            dict(os.environ),
            tmp_path,
        )
    )
    assert translated["dbname"] == ""

    try:
        connection = await asyncpg.connect(bare_database_dsn)
    except Exception:  # noqa: BLE001 - parity includes server rejection
        asyncpg_connected = False
    else:
        asyncpg_connected = True
        try:
            selected_database = await connection.fetchval(
                "SELECT current_database()"
            )
        finally:
            await connection.close()
        assert selected_database != urlsplit(runtime_db.dsn).path.removeprefix(
            "/"
        )

    report = diagnose(tmp_path)

    if not asyncpg_connected:
        assert not report.ready, f"ok={report.ok} warn={report.warn}"
        assert any("cannot read PostgreSQL" in message for message in report.fail)
    else:
        # A standard PostgreSQL installation has a database named after its
        # bootstrap role (usually ``postgres``), so an empty dbname can be
        # reachable. In that case doctor may validly report a pending anchor
        # replication, but it must never read the unrelated PGDATABASE target.
        findings = " ".join(report.ok + report.warn + report.fail)
        assert unrelated_hash[:12] not in findings
        assert any(
            "PostgreSQL holds no record for this agent yet" in message
            and "kestrel_prime.db" in message
            for message in report.warn
        )


def test_all_libpq_only_environment_is_absent_from_the_probe_process(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Independent libpq-only poison values cannot steer the child probe."""
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    libpq_only_poison = {
        "PGDATESTYLE": "not-a-datestyle",
        "PGTZ": "Definitely/Not_A_Timezone",
        "PGGEQO": "not-a-boolean",
        "PGKEEPALIVES": "not-a-boolean",
        "PGKEEPALIVESIDLE": "not-an-integer",
        "PGKEEPALIVESINTERVAL": "not-an-integer",
        "PGKEEPALIVESCOUNT": "not-an-integer",
        "PGTCP_USER_TIMEOUT": "not-an-integer",
    }
    # Keep poison out of the runtime database fixture's teardown connection.
    with pytest.MonkeyPatch.context() as libpq_only_env:
        for env_name, value in libpq_only_poison.items():
            libpq_only_env.setenv(env_name, value)
        report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"


def test_compiled_libpq_defaults_remain_connectable_without_environment(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Unconditional asyncpg-equivalent defaults work with real libpq."""
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)
    for env_name in ("PGGSSENCMODE", "PGCHANNELBINDING"):
        monkeypatch.delenv(env_name, raising=False)

    report = diagnose(tmp_path)

    assert _LIBPQ_COMPILED_DSN_DEFAULTS == (
        ("gssencmode", "disable"),
        ("channel_binding", "disable"),
    )
    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"


@pytest.mark.parametrize("query", ["connect_timeout=30", "keepalives=1"])
def test_libpq_connection_names_rejected_by_runtime_are_not_reported_healthy(
    tmp_path, monkeypatch, canonical, runtime_db, query
):
    """asyncpg sends both names as GUCs, which PostgreSQL rejects."""
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn + "?" + query)

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    assert any("governance NOT verified" in message for message in report.fail)


@pytest.mark.parametrize(
    "query",
    [
        "sslmode=prefer",
        "options=-c%20statement_timeout%3D5000",
    ],
)
def test_asyncpg_recognized_and_startup_options_remain_connectable(
    tmp_path, monkeypatch, canonical, runtime_db, query
):
    """The two non-GUC query shapes retain their distinct semantics."""
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn + "?" + query)

    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"


@pytest.mark.parametrize(
    ("query", "tls_env", "doctor_ready"),
    [
        ("?sslmode=disable&sslnegotiation=direct", {}, True),
        ("?sslmode=disable", {"PGSSLNEGOTIATION": "direct"}, True),
        ("?sslnegotiation=direct", {"PGSSLMODE": "disable"}, True),
        (
            "",
            {"PGSSLMODE": "disable", "PGSSLNEGOTIATION": "direct"},
            True,
        ),
        ("?sslmode=allow&sslnegotiation=direct", {}, False),
        (
            "",
            {"PGSSLMODE": "allow", "PGSSLNEGOTIATION": "direct"},
            False,
        ),
        ("?sslmode=prefer&sslnegotiation=direct", {}, False),
        (
            "",
            {"PGSSLMODE": "prefer", "PGSSLNEGOTIATION": "direct"},
            False,
        ),
    ],
    ids=(
        "disable-query",
        "disable-query-environment-direct",
        "disable-query-direct-environment",
        "disable-environment",
        "allow-query",
        "allow-environment",
        "prefer-query",
        "prefer-environment",
    ),
)
async def test_direct_negotiation_matches_runtime_or_fails_closed(
    tmp_path,
    monkeypatch,
    canonical,
    runtime_db,
    query,
    tls_env,
    doctor_ready,
):
    """Unrepresentable weak direct TLS fails closed; disable stays exact."""
    import asyncpg
    from asyncpg.connect_utils import (
        SSLMode,
        SSLNegotiation,
        _parse_connect_dsn_and_args,
    )

    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    runtime_dsn = runtime_db.dsn + query
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_dsn)

    # PGSSL* affects psycopg2 even when its fixture passes an explicit DSN.
    # Drop these overrides before runtime_db's finalizer reconnects to drop
    # the throwaway database.
    with pytest.MonkeyPatch.context() as tls:
        for name in ("PGSSLMODE", "PGSSLNEGOTIATION"):
            tls.delenv(name, raising=False)
        for name, value in tls_env.items():
            tls.setenv(name, value)

        _, runtime_params = _parse_connect_dsn_and_args(
            dsn=runtime_dsn,
            host=None,
            port=None,
            user=None,
            password=None,
            passfile=None,
            database=None,
            ssl=None,
            direct_tls=None,
            server_settings=None,
            target_session_attrs=None,
            krbsrvname=None,
            gsslib=None,
        )
        if runtime_params.sslmode == SSLMode.prefer:
            # This is the runtime shape under test, but its first transport
            # attempt cannot be used as a portable live-server precondition;
            # see the module docstring. Doctor must still fail translation
            # closed rather than probe a different connection policy.
            assert runtime_params.ssl_negotiation == SSLNegotiation.direct
            assert runtime_params.ssl is not None
        else:
            connection = await asyncpg.connect(runtime_dsn)
            await connection.close()
        report = diagnose(tmp_path)

    assert report.ready is doctor_ready, (
        f"ok={report.ok} warn={report.warn} fail={report.fail}"
    )
    if not doctor_ready:
        assert any("cannot read PostgreSQL" in message for message in report.fail)


async def test_explicit_missing_tls_file_fails_like_the_asyncpg_runtime(
    tmp_path,
    monkeypatch,
    canonical,
    runtime_db,
):
    """Libpq must not forgive a client certificate asyncpg requires."""
    import asyncpg

    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    missing_certificate = tmp_path / "private" / "missing-client.crt"
    runtime_dsn = (
        runtime_db.dsn
        + "?sslmode=prefer&sslcert="
        + quote(str(missing_certificate), safe="")
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_dsn)

    with pytest.raises(FileNotFoundError):
        await asyncpg.connect(runtime_dsn)
    report = diagnose(tmp_path)

    assert not report.ready
    assert any("cannot read PostgreSQL" in message for message in report.fail)
    assert all(str(missing_certificate) not in message for message in report.fail)


def test_a_row_the_bound_runtime_cannot_see_is_not_a_clean_bill_of_health(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Everything is in place except the ownership witnesses.

    ``AsyncGraphStore`` binds to the agent's DID on this backend, and its
    ``_node_scope`` / ``_edge_scope`` require a row in ``graph_node_owners`` /
    ``graph_edge_owners`` — not a matching ``node_id``. The boot integrity
    audit reads through that bound store, so without the witnesses it sees no
    agent node at all.

    This test's expectation has moved twice, each time because the model of
    boot got more accurate, and it is worth recording where it landed.

    First it asserted a refusal. Then — on learning that boot computes a
    birth-record shortfall and replicates from the anchor before auditing — it
    asserted pending replication. Both were wrong for the same missing fact:
    ``AsyncGraphStore.add_node`` raises ``Cannot claim or overwrite an unowned
    graph node``, so replication cannot repair *this* state. The row is there,
    the agent cannot see it, and nothing at boot will fix it.

    So it is a failure, and specifically an ownership failure — naming it drift
    would send an operator to a reanchor that cannot clear it.
    """
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
        witness_node=False,
        witness_edge=False,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    assert any(
        "is not owned by" in m and "reanchoring will not clear it" in m
        for m in report.fail
    ), f"fail={report.fail}"
    assert not any("holds no record for this agent yet" in m for m in report.warn)


async def test_a_stale_anchor_awaiting_replication_is_not_ready(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """The case the pending-replication path exists for.

    PostgreSQL holds nothing for this agent, so boot will copy the anchor and
    audit it — and this anchor is drifted. Reporting "nothing to check" would
    pass a host whose very next boot safe-modes.
    """
    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=stale)
    # Nothing seeded into PostgreSQL at all: an agent not yet replicated.
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert any(
        "holds no record for this agent yet" in m for m in report.warn
    ), f"warn={report.warn}"
    assert any(
        "constitution drift" in m and stale[:12] in m for m in report.fail
    ), f"ok={report.ok} warn={report.warn} fail={report.fail}"


async def test_a_never_booted_postgres_is_a_first_boot_not_a_failure(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """A freshly configured database has no Kestrel schema at all.

    ``AsyncDatabase.postgres()`` creates the tables and replicates the anchor
    on first boot, so this is a valid starting state. Treating the missing
    ``graph_nodes`` as "cannot read" made doctor answer **Not ready** to a
    correctly configured first boot — and after readiness started failing on
    unreadable databases, that became a hard stop rather than a warning.
    """
    _seed_project(tmp_path, anchored_hash=canonical)
    # Drop the schema the fixture created: this database has never been booted.
    import psycopg2
    connection = psycopg2.connect(runtime_db.dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            # ``schema_backfills`` too: a database the runtime has never
            # opened has *nothing*, and leaving the marker table behind made
            # this test pass against a probe that cannot survive its absence.
            for table in (
                "graph_nodes", "graph_edges",
                "graph_node_owners", "graph_edge_owners",
                "schema_backfills",
            ):
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        connection.close()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert report.ready, f"fail={report.fail}"
    assert any(
        "holds no record for this agent yet" in m for m in report.warn
    ), f"warn={report.warn}"
    assert any("constitution anchored to current file" in m for m in report.ok)


async def test_a_stale_anchor_on_a_never_booted_postgres_still_fails(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """...and the drift in those pending bytes is still reported."""
    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=stale)
    import psycopg2
    connection = psycopg2.connect(runtime_db.dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            # ``schema_backfills`` too: a database the runtime has never
            # opened has *nothing*, and leaving the marker table behind made
            # this test pass against a probe that cannot survive its absence.
            for table in (
                "graph_nodes", "graph_edges",
                "graph_node_owners", "graph_edge_owners",
                "schema_backfills",
            ):
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        connection.close()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert not report.ready
    assert any(
        "constitution drift" in m and stale[:12] in m for m in report.fail
    ), f"fail={report.fail}"


async def test_a_boot_fabricated_placeholder_is_not_a_birth_record(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """A database damaged by a pre-#2878 boot holds a stand-in, not a record.

    ``birth_record.is_fabricated_placeholder`` matches the exact shape one code
    path writes, and boot counts such a row as an identity shortfall: it is
    replaced from the local anchor *before* the audit runs. Reading it as
    populated made doctor judge a row nobody will be governed by — warning only
    that it had no hash — and exit Ready while the stale anchor about to
    replace it safe-modes the agent.
    """
    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=stale)
    runtime_db(AGENT_DID, {"initialBalance": "100.0"}, governed_by=None)
    # The label half of the predicate, as _ensure_agent_node_present writes it.
    import psycopg2
    connection = psycopg2.connect(runtime_db.dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE graph_nodes SET label = %s WHERE node_id = %s",
                (f"Agent {AGENT_DID}", AGENT_DID),
            )
    finally:
        connection.close()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    assert any(
        "holds no record for this agent yet" in m for m in report.warn
    ), f"warn={report.warn}"
    assert any(
        "constitution drift" in m and stale[:12] in m for m in report.fail
    ), f"fail={report.fail}"


async def test_a_real_agent_row_is_not_mistaken_for_a_placeholder(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """The predicate matches an exact shape, so a genuine record still counts."""
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_db.dsn)

    report = diagnose(tmp_path)

    assert not any("holds no record" in m for m in report.warn), report.warn
    assert any("constitution anchored to current file" in m for m in report.ok)


async def test_the_prescribed_repair_reaches_a_pending_anchor(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Doctor's remedy has to work in the state doctor found.

    With PostgreSQL holding nothing for this agent, doctor reports on the
    anchor — the bytes first boot will copy and audit. Pointed at PostgreSQL,
    the reanchor would answer "no constitution_hash", change nothing, and leave
    the stale anchor to safe-mode the agent: a remedy that cannot clear the
    finding that prescribed it.
    """
    from kestrel_sovereign.setup.constitution_reanchor import (
        reanchor_constitution,
    )

    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=stale)
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION)
    agent_dir = tmp_path / "agent_data" / "test"

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=runtime_db.dsn,
    )

    # It found the anchor's stale hash rather than reporting nothing to do.
    assert result.old_hash == stale, result.error
    assert result.target_backend == "sqlite", result.target_label


async def test_a_replicated_agent_still_reanchors_in_postgres(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """The retarget is for an *empty* runtime, not a licence to prefer the file."""
    from kestrel_sovereign.setup.constitution_reanchor import (
        reanchor_constitution,
    )

    stale = hashlib.sha256(b"an older constitution").hexdigest()
    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": stale},
        governed_by=stale,
    )
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION)
    agent_dir = tmp_path / "agent_data" / "test"

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=runtime_db.dsn,
    )

    # PostgreSQL's hash, not the anchor's: the runtime holds this agent.
    assert result.old_hash == stale
    assert result.target_backend == "postgres", result.target_label


async def test_a_present_but_unanchored_postgres_node_stays_in_postgres(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Present-without-a-hash is not absent, and only absence retargets.

    ``_read_agent_anchor`` returns ``old_hash=None`` for both, so gating the
    retarget on the hash sent a *present but unanchored* PostgreSQL agent to
    SQLite: the local file would be written while the runtime node it actually
    boots from stayed unanchored and safe-mode-bound. Absence is the only state
    first-boot replication repairs.
    """
    from kestrel_sovereign.setup.constitution_reanchor import (
        reanchor_constitution,
    )

    _seed_project(tmp_path, anchored_hash=canonical)
    # The node exists in PostgreSQL, with no constitution_hash of its own.
    runtime_db(AGENT_DID, {"name": "Test"}, governed_by=None)
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION)

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=tmp_path / "agent_data" / "test",
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=runtime_db.dsn,
    )

    # ``target_backend``, not ``target_label``: the label embeds tmp_path,
    # which contains this test's own name — so a substring check for
    # "postgres" passes no matter what the code does. Mutation testing found
    # that the hard way.
    assert result.target_backend == "postgres", result.target_label
    assert result.error is not None
    assert "no constitution_hash" in result.error


async def test_an_unowned_postgres_row_does_not_retarget_the_anchor(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Invisible is not absent, and only absence retargets.

    A row present in PostgreSQL without this agent's ``graph_node_owners``
    witness reads back empty from the bound store, exactly like a missing one.
    They want opposite handling: replication repairs a missing row, and cannot
    repair an unowned one — ``add_node`` refuses to overwrite a foreign-owned
    row. Retargeting here would write the local SQLite file while PostgreSQL
    stayed unreadable and the agent unbootable.
    """
    from kestrel_sovereign.setup.constitution_reanchor import (
        reanchor_constitution,
    )

    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
        witness_node=False,
        witness_edge=False,
    )
    # The fixture records #2649 as complete, which is what makes the missing
    # witness persist: opening AsyncStorage against an *unmarked* database runs
    # the backfill and quietly creates the witness, so without this the state
    # under test repairs itself before the read and the test proves nothing.
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION)

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=tmp_path / "agent_data" / "test",
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=runtime_db.dsn,
    )

    # ``target_backend``, not ``target_label``: the label embeds tmp_path,
    # which contains this test's own name — so a substring check for
    # "postgres" passes no matter what the code does. Mutation testing found
    # that the hard way.
    assert result.target_backend == "postgres", result.target_label


async def test_an_unwitnessed_edge_is_drift_the_reanchor_will_repair(
    tmp_path, monkeypatch, canonical, runtime_db
):
    """Doctor's finding and the reanchor's verdict must use the same view.

    A physical ``governed_by`` edge at the current hash with no
    ``graph_edge_owners`` witness is invisible to the bound store the integrity
    audit reads through, so proof 2 fails and the agent safe-modes. The
    reanchor reads edges unscoped — deliberately, to find pre-ledger stale
    targets — and judging "unchanged" from that set alone reported nothing to
    do for the very drift doctor had just prescribed this command to fix.
    """
    from kestrel_sovereign.setup.constitution_reanchor import (
        reanchor_constitution,
    )

    _seed_project(tmp_path, anchored_hash=canonical)
    runtime_db(
        AGENT_DID,
        {"name": "Test", "constitution_hash": canonical},
        governed_by=canonical,
        witness_edge=False,   # the node is owned; the edge is not
    )
    constitution_path = tmp_path / "KESTREL_CONSTITUTION.md"
    constitution_path.write_bytes(CONSTITUTION)

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=tmp_path / "agent_data" / "test",
        canonical_path=constitution_path,
        force=False,
        runtime_backend="postgres",
        runtime_dsn=runtime_db.dsn,
    )

    assert not result.unchanged, "reported nothing to do for a failing proof 2"
    assert result.drift_unforced, result
