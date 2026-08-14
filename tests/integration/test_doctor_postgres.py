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
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path

import pytest
import toml
from cryptography.fernet import Fernet

from kestrel_sovereign.doctor import diagnose
from kestrel_sovereign.multi_agent.config import (
    HostConfig,
    LocalAgentConfig,
    MULTI_AGENT_CONFIG_FILENAME,
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
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DB_BACKEND": "postgres",
            "KESTREL_DATABASE_URL": runtime_db.dsn,
        },
    )

    report = diagnose(tmp_path)

    assert any(
        "constitution drift" in m and stale[:12] in m for m in report.fail
    ), f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert "KESTREL_DB_BACKEND" not in os.environ, (
        "a diagnostic must not export the project's environment"
    )


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

