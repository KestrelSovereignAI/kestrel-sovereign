"""Which database ``kestrel constitution reanchor`` reads and writes (#2890).

Governance lives in the database the *runtime* opens. On a host configured
with ``KESTREL_DB_BACKEND=postgres`` that is not the local
``kestrel_prime.db`` — the anchor holds the birth record (#2871) and nothing
the agent reads at runtime. A reanchor that resolves the anchor unconditionally
reports success and changes nothing the agent is governed by.

These tests pin the resolution itself. The end-to-end write against a real
PostgreSQL runtime is in
``tests/integration/test_constitution_reanchor_postgres.py``.
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.setup.constitution_reanchor import (
    ReanchorTarget,
    ReanchorTargetError,
    reanchor_constitution,
    resolve_reanchor_target,
)


@pytest.fixture
def agent_dir(tmp_path):
    d = tmp_path / "agent_data" / "Test"
    d.mkdir(parents=True)
    (d / "kestrel_prime.db").write_bytes(b"stub")
    return d


@pytest.fixture(autouse=True)
def _no_ambient_backend(monkeypatch):
    """The host running the tests must not decide the answer."""
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_default_target_is_the_local_anchor(agent_dir):
    target = resolve_reanchor_target(agent_dir)
    assert target.backend == "sqlite"
    assert target.writes_to_anchor is True
    assert target.anchor_path == agent_dir / "kestrel_prime.db"


def test_postgres_environment_targets_postgres_not_the_anchor(agent_dir, monkeypatch):
    """The regression: on a PostgreSQL host the write must not go to the file."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://u:p@db.internal:5432/kestrel")

    target = resolve_reanchor_target(agent_dir)

    assert target.backend == "postgres"
    assert target.writes_to_anchor is False
    assert target.dsn == "postgresql://u:p@db.internal:5432/kestrel"


def test_backend_is_case_and_whitespace_insensitive(agent_dir, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_BACKEND", "  PostgreS \n")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")
    assert resolve_reanchor_target(agent_dir).backend == "postgres"


def test_explicit_arguments_override_the_environment(agent_dir, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/env")

    target = resolve_reanchor_target(
        agent_dir, backend="postgres", dsn="postgresql://h/explicit"
    )

    assert target.dsn == "postgresql://h/explicit"


def test_postgres_without_a_dsn_refuses(agent_dir, monkeypatch):
    """``main.build_storage`` requires KESTREL_DATABASE_URL for a PostgreSQL
    runtime. An agent configured that way has no database we can name, and
    falling back to the anchor is exactly the silent no-op."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")

    with pytest.raises(ReanchorTargetError) as exc:
        resolve_reanchor_target(agent_dir)

    assert "KESTREL_DATABASE_URL" in str(exc.value)
    assert "kestrel_prime.db" in str(exc.value)


def test_unsupported_backend_refuses(agent_dir, monkeypatch):
    monkeypatch.setenv("KESTREL_DB_BACKEND", "mysql")
    with pytest.raises(ReanchorTargetError) as exc:
        resolve_reanchor_target(agent_dir)
    assert "mysql" in str(exc.value)


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
        anchor_path=agent_dir / "kestrel_prime.db", backend="sqlite"
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
        dsn="postgresql://kestrel:sup3rs3cret@db.internal:5432/kestrel",
    )

    described = target.describe()

    assert "sup3rs3cret" not in described
    assert "kestrel:" not in described
    assert "db.internal:5432/kestrel" in described


def test_sqlite_target_describe_names_the_file(agent_dir):
    target = resolve_reanchor_target(agent_dir)
    assert target.describe() == f"sqlite:{agent_dir / 'kestrel_prime.db'}"


# ---------------------------------------------------------------------------
# The helper's use of the target
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reanchor_refuses_rather_than_writing_the_anchor_on_postgres(
    agent_dir, tmp_path, monkeypatch
):
    """A PostgreSQL host with no DSN must produce an error, not a successful
    reanchor of a database the runtime never opens."""
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    before = (agent_dir / "kestrel_prime.db").read_bytes()

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=tmp_path / "KESTREL_CONSTITUTION.md",
        force=True,
    )

    assert result.error is not None
    assert "KESTREL_DATABASE_URL" in result.error
    assert result.reanchored is False
    assert (agent_dir / "kestrel_prime.db").read_bytes() == before
    assert not list(agent_dir.glob("*.backup-*"))


@pytest.mark.asyncio
async def test_missing_anchor_is_not_a_gate_on_a_postgres_host(
    agent_dir, tmp_path, monkeypatch
):
    """The anchor's existence is the "this directory is an agent" proxy and
    the write target — but only when the runtime reads it. Requiring the file
    before writing PostgreSQL would gate on a fact the write does not need."""
    (agent_dir / "kestrel_prime.db").unlink()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://127.0.0.1:1/none")

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=tmp_path / "missing-constitution.md",
        force=False,
    )

    # It fails on the canonical source it was actually given, not on the
    # absence of a local file this write does not use.
    assert result.error is not None
    assert "Agent database not found" not in result.error
    assert result.target_backend == "postgres"


@pytest.mark.asyncio
async def test_missing_anchor_still_refuses_on_a_sqlite_host(agent_dir, tmp_path):
    (agent_dir / "kestrel_prime.db").unlink()

    result = await reanchor_constitution(
        agent_name="Test",
        agent_dir=agent_dir,
        canonical_path=tmp_path / "missing-constitution.md",
        force=False,
    )

    assert result.error is not None
    assert "Agent database not found" in result.error
    assert result.target_backend == "sqlite"


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
async def test_overlay_anchor_refuses_a_postgres_host_with_no_dsn(
    agent_dir, monkeypatch
):
    """``anchor-overlay`` writes the property that authorizes DANGEROUS
    Amendment IX grants. Landing it in a database the runtime never opens
    reports success and denies every grant."""
    from kestrel_sovereign.setup.overlay_anchor import anchor_overlay

    (agent_dir / "CONSTITUTION.md").write_bytes(b"# Overlay\n")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    before = (agent_dir / "kestrel_prime.db").read_bytes()

    result = await anchor_overlay(agent_name="Test", agent_dir=agent_dir)

    assert result.error is not None
    assert "KESTREL_DATABASE_URL" in result.error
    assert (agent_dir / "kestrel_prime.db").read_bytes() == before


def test_environment_is_read_at_call_time_not_import_time(agent_dir, monkeypatch):
    """Boot resolves its backend from ``os.environ`` when the agent is
    constructed. Caching the answer at import would make a CLI that loads the
    agent home's .env after import resolve the wrong database."""
    assert resolve_reanchor_target(agent_dir).backend == "sqlite"
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://h/db")
    assert resolve_reanchor_target(agent_dir).backend == "postgres"
    assert os.environ["KESTREL_DB_BACKEND"] == "postgres"
