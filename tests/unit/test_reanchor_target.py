"""Which database — and whose agent — ``kestrel constitution reanchor`` writes (#2890).

Governance lives in the database the *runtime* opens. On a host configured
with ``KESTREL_DB_BACKEND=postgres`` plus a DSN that is not the local
``kestrel_prime.db``: the anchor holds the birth record (#2871) and nothing the
agent reads at runtime. A reanchor that resolves the anchor unconditionally
reports success and changes nothing the agent is governed by.

The same database also holds *every* local agent on a PostgreSQL host, so
"which database" is only half the question. An unbound ``AsyncGraphStore``
scopes to ``1 = 1``, which makes "the agent node" whichever row comes back
first.

These tests pin the resolution. The end-to-end write against a real PostgreSQL
runtime, including the two-agent case, is in
``tests/integration/test_constitution_reanchor_postgres.py``.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from kestrel_sovereign.setup.constitution_reanchor import (
    ReanchorTarget,
    ReanchorTargetError,
    reanchor_constitution,
    resolve_reanchor_target,
)

AGENT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000002890"
OTHER_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000009999"


def _write_anchor(path, *dids):
    """A ``kestrel_prime.db`` holding the given agent roots.

    Not an opaque stub: ``resolve_reanchor_target`` reads the DID out of this
    file through the same reader the host uses (``_get_agent_did``), so the
    fixture has to be the shape that reader accepts.
    """
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS graph_nodes "
            "(node_id TEXT PRIMARY KEY, node_type TEXT, label TEXT, properties TEXT)"
        )
        for did in dids:
            conn.execute(
                "INSERT OR REPLACE INTO graph_nodes VALUES (?, 'agent', 'Test', '{}')",
                (did,),
            )
        conn.commit()


@pytest.fixture
def agent_dir(tmp_path):
    d = tmp_path / "agent_data" / "Test"
    d.mkdir(parents=True)
    _write_anchor(d / "kestrel_prime.db", AGENT_DID)
    return d


@pytest.fixture(autouse=True)
def _no_ambient_backend(monkeypatch):
    """The host running the tests must not decide the answer."""
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)


# ---------------------------------------------------------------------------
# Resolution — copied from agent_manager._initialize_agent, not invented
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_target_is_the_local_anchor(agent_dir):
    target = await resolve_reanchor_target(agent_dir)
    assert target.backend == "sqlite"
    assert target.writes_to_anchor is True
    assert target.anchor_path == agent_dir / "kestrel_prime.db"
    assert target.agent_did == AGENT_DID


@pytest.mark.asyncio
async def test_postgres_with_a_dsn_targets_postgres_not_the_anchor(
    agent_dir, monkeypatch
):
    """The regression: on a PostgreSQL host the write must not go to the file."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv(
        "KESTREL_DATABASE_URL", "postgresql://u:p@db.internal:5432/kestrel"
    )

    target = await resolve_reanchor_target(agent_dir)

    assert target.backend == "postgres"
    assert target.writes_to_anchor is False
    assert target.dsn == "postgresql://u:p@db.internal:5432/kestrel"
    assert target.agent_did == AGENT_DID


@pytest.mark.asyncio
async def test_postgres_without_a_dsn_is_a_sqlite_host(agent_dir, monkeypatch):
    """``agent_manager._initialize_agent`` starts an agent on PostgreSQL only
    when ``KESTREL_DB_BACKEND=postgres`` **and** ``KESTREL_DATABASE_URL`` are
    both set; otherwise it hands ``KestrelAgent`` the local file. Refusing this
    host — or targeting a PostgreSQL that nothing reads — would break a
    reanchor that works today, and there is no flag to override it."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")

    target = await resolve_reanchor_target(agent_dir)

    assert target.backend == "sqlite"
    assert target.anchor_path == agent_dir / "kestrel_prime.db"


@pytest.mark.asyncio
async def test_backend_is_case_insensitive(agent_dir, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_BACKEND", "PostgreS")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")
    target = await resolve_reanchor_target(agent_dir)
    assert target.backend == "postgres"


@pytest.mark.asyncio
async def test_surrounding_whitespace_is_not_stripped(agent_dir, monkeypatch):
    """``_initialize_agent`` compares ``db_backend.lower() == "postgres"`` with
    no strip, so ``"postgres "`` starts the agent on **SQLite**. Stripping here
    would send the reanchor to PostgreSQL instead — this very issue with the
    two databases exchanged. Copying a rule means copying it exactly,
    including the parts that look like bugs."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres ")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")

    target = await resolve_reanchor_target(agent_dir)

    assert target.backend == "sqlite"
    assert target.anchor_path == agent_dir / "kestrel_prime.db"


@pytest.mark.asyncio
async def test_explicit_arguments_override_the_environment(agent_dir, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/env")

    target = await resolve_reanchor_target(
        agent_dir, backend="postgres", dsn="postgresql://h/explicit"
    )

    assert target.dsn == "postgresql://h/explicit"


@pytest.mark.asyncio
async def test_unsupported_backend_lands_where_the_runtime_lands(
    agent_dir, monkeypatch, caplog,
):
    """``_initialize_agent`` tests ``== "postgres"`` and falls through to
    SQLite for anything else, so an agent configured ``mysql`` really is
    running on the local file. Refusing here would be stricter than the thing
    being repaired; the honest move is to follow it and say so."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "mysql")

    with caplog.at_level("WARNING"):
        target = await resolve_reanchor_target(agent_dir)

    assert target.backend == "sqlite"
    assert "mysql" in caplog.text


# ---------------------------------------------------------------------------
# Identity — the anchor names the tenant, on every backend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_missing_anchor_refuses_on_postgres_too(agent_dir, monkeypatch):
    """The anchor is not merely the SQLite write target. It is the only
    artifact that says *which* agent this reanchor is for, and boot reads it on
    a PostgreSQL host for exactly that reason."""
    (agent_dir / "kestrel_prime.db").unlink()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")

    with pytest.raises(ReanchorTargetError) as exc:
        await resolve_reanchor_target(agent_dir)

    assert "Cannot identify the agent" in str(exc.value)


@pytest.mark.asyncio
async def test_two_agent_roots_in_one_anchor_refuse(agent_dir):
    """``_get_agent_did`` refuses rather than picking by row order. Inheriting
    that refusal is the point of reusing it."""
    _write_anchor(agent_dir / "kestrel_prime.db", AGENT_DID, OTHER_DID)

    with pytest.raises(ReanchorTargetError) as exc:
        await resolve_reanchor_target(agent_dir)

    assert "Cannot identify the agent" in str(exc.value)


@pytest.mark.asyncio
async def test_the_open_storage_binds_the_agent(agent_dir):
    target = await resolve_reanchor_target(agent_dir)
    storage = target.open_storage()
    assert storage.agent_id == AGENT_DID


@pytest.mark.asyncio
async def test_the_postgres_branch_binds_the_agent_too(agent_dir, monkeypatch):
    """Both branches, separately. Deleting ``agent_id`` from the PostgreSQL
    branch alone left the whole suite green — the probe tests were carried by
    the by-DID lookup, not the binding — and PostgreSQL is the only backend
    where the binding does anything, because it is the only one that holds
    more than one agent."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://u:p@h:5432/db")

    target = await resolve_reanchor_target(agent_dir)
    storage = target.open_storage()

    assert target.backend == "postgres"
    assert storage.agent_id == AGENT_DID


# ---------------------------------------------------------------------------
# Opening the target
# ---------------------------------------------------------------------------

def test_sqlite_target_is_not_redirected_by_the_ambient_environment(
    agent_dir, monkeypatch
):
    """``AsyncStorage(path)`` alone consults ``KESTREL_DB_BACKEND`` and would
    hand back a PostgreSQL backend for a SQLite path. The target passes the
    backend explicitly so an exported variable cannot change where an
    already-decided reanchor lands."""
    target = ReanchorTarget(
        anchor_path=agent_dir / "kestrel_prime.db",
        backend="sqlite",
        agent_did=AGENT_DID,
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")

    storage = target.open_storage()

    assert storage._backend.backend_type == "sqlite"
    assert storage.db_path == str(agent_dir / "kestrel_prime.db")


def test_target_describe_redacts_the_dsn_password(agent_dir):
    target = ReanchorTarget(
        anchor_path=agent_dir / "kestrel_prime.db",
        backend="postgres",
        agent_did=AGENT_DID,
        dsn="postgresql://kestrel:sup3rs3cret@db.internal:5432/kestrel",
    )

    described = target.describe()

    assert "sup3rs3cret" not in described
    assert "kestrel:" not in described
    assert "db.internal:5432/kestrel" in described


def test_target_describe_survives_an_unparseable_dsn(agent_dir):
    """``urlsplit`` parses lazily — an invalid port raises from ``.port``, not
    from the split. ``describe()`` runs on every outcome, so an unguarded read
    turns a typo into a traceback out of the CLI."""
    target = ReanchorTarget(
        anchor_path=agent_dir / "kestrel_prime.db",
        backend="postgres",
        agent_did=AGENT_DID,
        dsn="postgresql://u:p@h:notaport/db",
    )

    described = target.describe()

    assert "unparseable" in described
    assert "p@" not in described


@pytest.mark.asyncio
async def test_sqlite_target_describe_names_the_file(agent_dir):
    target = await resolve_reanchor_target(agent_dir)
    assert target.describe() == f"sqlite:{agent_dir / 'kestrel_prime.db'}"


# ---------------------------------------------------------------------------
# The helper's use of the target
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unreachable_postgres_is_an_error_not_a_traceback(
    agent_dir, tmp_path, monkeypatch
):
    """Opening the runtime database is the first step here that touches the
    network."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    canonical = tmp_path / "KESTREL_CONSTITUTION.md"
    canonical.write_bytes(b"# Constitution\n")
    import kestrel_sovereign.config as ks_config
    monkeypatch.setattr(ks_config, "CONSTITUTION_PATH", str(canonical))

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=canonical,
        force=True,
    )

    assert result.error is not None
    assert "Could not read this agent's governance" in result.error
    assert "Nothing was written" in result.error
    assert "p@" not in result.error  # the DSN password is not in the message
    assert not list(agent_dir.glob("*.backup-*"))


@pytest.mark.asyncio
async def test_missing_anchor_refuses_on_a_sqlite_host(agent_dir, tmp_path):
    (agent_dir / "kestrel_prime.db").unlink()

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=tmp_path / "missing-constitution.md",
        force=False,
    )

    assert result.error is not None
    assert "Cannot identify the agent" in result.error


@pytest.mark.asyncio
async def test_every_result_names_the_database_it_describes(agent_dir, tmp_path):
    """An outcome that does not say which database it read is the ambiguity
    this issue is about."""
    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=tmp_path / "missing-constitution.md",
        force=False,
    )

    assert result.target_backend == "sqlite"
    assert result.target_label == f"sqlite:{agent_dir / 'kestrel_prime.db'}"


@pytest.mark.asyncio
async def test_overlay_anchor_refuses_an_unidentifiable_agent(agent_dir):
    """``anchor-overlay`` writes the property that authorizes DANGEROUS
    Amendment IX grants. Landing it on the wrong tenant, or in a database the
    runtime never opens, must not be possible."""
    from kestrel_sovereign.setup.overlay_anchor import anchor_overlay

    (agent_dir / "CONSTITUTION.md").write_bytes(b"# Overlay\n")
    (agent_dir / "kestrel_prime.db").unlink()

    result = await anchor_overlay(agent_name="Test", agent_dir=agent_dir)

    assert result.error is not None
    assert "Cannot identify the agent" in result.error


@pytest.mark.asyncio
async def test_environment_is_read_at_call_time_not_import_time(
    agent_dir, monkeypatch
):
    """Boot resolves its backend from ``os.environ`` when the agent is
    constructed. Caching the answer at import would make a CLI that loads the
    agent home's .env after import resolve the wrong database."""
    assert (await resolve_reanchor_target(agent_dir)).backend == "sqlite"
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")
    assert (await resolve_reanchor_target(agent_dir)).backend == "postgres"
    assert os.environ["KESTREL_DB_BACKEND"] == "postgres"
