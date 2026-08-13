"""Unit tests for the ``kestrel constitution reanchor`` CLI surface.

These tests mock out the async reanchor helper — they assert on the
*CLI behaviour* (argument parsing, exit codes, messaging, refusal
gates). The real governance rewrite and authorization record are exercised in
``tests/integration/test_constitution_reanchor_e2e.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import toml

from kestrel_sovereign.cli import build_parser, cmd_constitution
from kestrel_sovereign.setup.constitution_reanchor import ReanchorResult


@pytest.fixture
def restore_environ():
    """Restore ``os.environ`` wholesale.

    ``load_project_env`` mutates it with ``os.environ.setdefault``, which
    ``monkeypatch`` does not track — a leaked ``KESTREL_DB_BACKEND`` would
    redirect every later test's storage.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _no_ambient_backend(monkeypatch):
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)


AGENT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000002890"


@pytest.fixture
def reanchor_env(tmp_path):
    """Project tree with one multi_agent agent + a readable kestrel_prime.db.

    The anchor holds a real agent root because the reanchor path reads this
    agent's DID out of it — a directory that cannot name its tenant is refused
    on every backend, so an opaque stub would make every test a refusal test.
    """
    import sqlite3

    agent_dir = tmp_path / "agent_data" / "Test"
    agent_dir.mkdir(parents=True)
    with sqlite3.connect(str(agent_dir / "kestrel_prime.db")) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS graph_nodes "
            "(node_id TEXT PRIMARY KEY, node_type TEXT, label TEXT, properties TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO graph_nodes VALUES (?, 'agent', 'Test', '{}')",
            (AGENT_DID,),
        )
        conn.commit()

    multi_agent = {
        "host": {"port": 8888, "bind": "0.0.0.0"},
        "agents": {
            "Test": {
                "data_dir": "agent_data/Test",
                "port": 8801,
                "autostart": True,
            }
        },
    }
    (tmp_path / "multi_agent.toml").write_text(toml.dumps(multi_agent))
    return tmp_path


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_reanchor_requires_agent_name():
    """argparse must reject a call with no --agent-name."""
    with pytest.raises(SystemExit):
        _parse(["constitution", "reanchor"])


def test_reanchor_parses_force_flag():
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    assert args.force is True
    assert args.agent_name == "Test"


def test_reanchor_force_default_false():
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])
    assert args.force is False


def test_reanchor_accepts_constitution_path_override():
    args = _parse([
        "constitution", "reanchor", "--agent-name", "Test",
        "--constitution-path", "/tmp/custom.md",
    ])
    assert args.constitution_path == "/tmp/custom.md"


def test_reanchor_parses_external_authority_paths():
    args = _parse([
        "constitution", "reanchor", "--agent-name", "Test", "--force",
        "--signed-artifact", "/secure/reanchor.signed.json",
        "--trust-root", "/secure/sovereign-root.did.json",
    ])
    assert args.signed_artifact == "/secure/reanchor.signed.json"
    assert args.trust_root == "/secure/sovereign-root.did.json"


# ---------------------------------------------------------------------------
# Refusal gates (must NOT touch the helper)
# ---------------------------------------------------------------------------

def test_reanchor_rejects_unknown_agent(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "NotInMultiAgent"])
    with patch(
        "kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env,
    ), patch(
        "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution"
    ) as mock_reanchor:
        rc = cmd_constitution(args)
    assert rc == 2
    assert "not in multi_agent" in capsys.readouterr().err.lower()
    mock_reanchor.assert_not_called()


def test_reanchor_refuses_when_agent_appears_running(reanchor_env, capsys):
    """SQLite WAL locking would corrupt the DB if the agent is running.
    The CLI must refuse before invoking the helper."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    with patch(
        "kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env,
    ), patch(
        "kestrel_sovereign.cli._agent_appears_running", return_value=True,
    ), patch(
        "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution"
    ) as mock_reanchor:
        rc = cmd_constitution(args)
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "appears to be running" in err
    assert "kestrel stop test" in err
    mock_reanchor.assert_not_called()


# ---------------------------------------------------------------------------
# Result dispatch
# ---------------------------------------------------------------------------

def _stubbed_helper(result: ReanchorResult):
    """Build an async stub that returns the given result."""
    async def _stub(**_kwargs):
        return result

    return _stub


def test_reanchor_targets_the_database_doctor_reads(reanchor_env, monkeypatch):
    """The repair and the finding must mean the same database (#2892).

    ``kestrel doctor`` resolves the target through ``spawned_agent_env`` — the
    launcher's precedence, where the project ``.env`` outranks an export,
    because that is what a spawned agent actually gets. This command loads its
    environment through ``load_project_env``, whose ``setdefault`` is the
    opposite. Left to the ambient environment, a stale exported DSN would have
    doctor report drift in database A while the repair it prescribes rewrote
    database B: the finding survives, and a database nobody reads is modified.
    """
    from kestrel_sovereign.doctor import runtime_env

    (reanchor_env / ".env").write_text(
        "KESTREL_DB_BACKEND=postgres\n"
        "KESTREL_DATABASE_URL=postgresql://from-file/db\n"
    )
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://stale-export/db")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")

    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    seen: dict = {}

    async def _capture(**kwargs):
        seen.update(kwargs)
        return ReanchorResult(
            agent_name="Test",
            db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
            canonical_path=Path("/fake/canonical.md"),
            old_hash="a" * 64,
            new_hash="a" * 64,
            backup_path=None,
            unchanged=True,
        )

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_capture,
         ):
        assert cmd_constitution(args) == 0

    doctors_view = runtime_env(reanchor_env)
    assert seen["runtime_dsn"] == doctors_view["KESTREL_DATABASE_URL"]
    assert seen["runtime_backend"] == doctors_view["KESTREL_DB_BACKEND"]
    assert seen["runtime_dsn"] == "postgresql://from-file/db"


def test_reanchor_unchanged_returns_zero(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="a" * 64,
        backup_path=None,
        unchanged=True,
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "already anchored" in out


def test_reanchor_same_hash_edge_repair_reports_removed(
    reanchor_env, capsys,
):
    """#2617 cleanup shape: unchanged anchor, stale edge repaired (#2616 flow)."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    backup_path = (
        reanchor_env / "agent_data" / "Test"
        / "kestrel_prime.db.backup-20260719-120000"
    )
    stale = "5" * 64
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="a" * 64,
        backup_path=backup_path,
        reanchored=True,
        governance_edge_drift=True,
        stale_edge_targets=(stale,),
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "governance edge repaired" in out.lower()
    assert stale[:12] in out  # which edge was removed
    assert str(backup_path) in out


def test_reanchor_stale_edge_drift_unforced_returns_one(reanchor_env, capsys):
    """Current anchor + dangling governed_by edges, no --force → drift report."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])
    stale = "5" * 64
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="a" * 64,
        backup_path=None,
        drift_unforced=True,
        governance_edge_drift=True,
        stale_edge_targets=(stale,),
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert "governance-edge drift detected" in out.lower()
    assert stale[:12] in out
    assert "--force" in out
    assert "backup" in out.lower()


def test_reanchor_success_with_stale_edges_reports_reanchored(
    reanchor_env, capsys,
):
    """A full reanchor over a drifted edge set still reports cleanly."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    stale = "5" * 64
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="b" * 64,
        backup_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db.backup-x",
        reanchored=True,
        governance_edge_drift=True,
        stale_edge_targets=(stale,),
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "reanchored" in out.lower()
    assert ("a" * 64)[:12] in out
    assert ("b" * 64)[:12] in out


def test_reanchor_drift_unforced_returns_one(reanchor_env, capsys):
    """Drift detected, --force absent → CLI prints diagnosis and exits 1."""
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="b" * 64,
        backup_path=None,
        drift_unforced=True,
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert "drift detected" in out.lower()
    assert "--force" in out  # Tells the user how to actually do it
    assert "backup" in out.lower()  # And that a backup will happen


def test_reanchor_success_prints_old_new_and_backup(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    backup_path = reanchor_env / "agent_data" / "Test" / "kestrel_prime.db.backup-20260504-120000"
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="aaaaaaaaaaaaaaaa" * 4,
        new_hash="bbbbbbbbbbbbbbbb" * 4,
        backup_path=backup_path,
        reanchored=True,
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "reanchored" in out.lower()
    assert "aaaaaaaaaaaa" in out  # truncated old hash
    assert "bbbbbbbbbbbb" in out  # truncated new hash
    assert str(backup_path) in out  # full backup path visible


def test_reanchor_helper_error_propagates(reanchor_env, capsys):
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash=None,
        new_hash=None,
        backup_path=None,
        error="Cannot read canonical constitution at /fake/canonical.md: [Errno 2]",
    )
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        rc = cmd_constitution(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "Cannot read canonical constitution" in err


def test_reanchor_passes_authority_paths_to_shared_helper(reanchor_env):
    args = _parse([
        "constitution", "reanchor", "--agent-name", "Test", "--force",
        "--signed-artifact", "/secure/reanchor.signed.json",
        "--trust-root", "/secure/sovereign-root.did.json",
    ])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="a" * 64,
        backup_path=None,
        unchanged=True,
    )
    captured = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return result

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_capture,
         ):
        assert cmd_constitution(args) == 0

    assert captured["amendment_artifact_path"] == Path(
        "/secure/reanchor.signed.json"
    )
    assert captured["sovereign_trust_root_path"] == Path(
        "/secure/sovereign-root.did.json"
    )


# ---------------------------------------------------------------------------
# Which database (#2890)
# ---------------------------------------------------------------------------

def test_reanchor_reads_the_agent_homes_env_not_the_operators_shell(
    reanchor_env, restore_environ, capsys,
):
    """The decisive case. The agent's ``.env`` says its runtime is PostgreSQL;
    nothing is exported. Before this was fixed the CLI never read that file,
    resolved ``agent_dir / "kestrel_prime.db"``, wrote it, and reported
    success — against a database the running agent never opens.

    The DSN here points at a closed port, so "it tried PostgreSQL" is provable
    from the failure. A run that had ignored the ``.env`` would have read the
    local anchor and reported drift instead.
    """
    (reanchor_env / ".env").write_text(
        "KESTREL_DB_BACKEND=postgres\n"
        "KESTREL_DATABASE_URL=postgresql://u:p@127.0.0.1:1/none\n"
    )
    anchor = reanchor_env / "agent_data" / "Test" / "kestrel_prime.db"
    before = anchor.read_bytes()
    args = _parse([
        "constitution", "reanchor", "--agent-name", "Test", "--force",
        "--signed-artifact", "/secure/reanchor.signed.json",
    ])

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False):
        rc = cmd_constitution(args)

    assert rc == 1
    err = capsys.readouterr().err
    assert "postgres" in err
    assert "Nothing was written" in err
    assert "u:p@" not in err
    # And it did not quietly rewrite the anchor on the way past.
    assert anchor.read_bytes() == before
    assert not list(anchor.parent.glob("*.backup-*"))


def test_reanchor_env_does_not_override_an_exported_value(
    reanchor_env, restore_environ,
):
    """``.env`` is a default, not an override — the same ``setdefault``
    semantics every other target-aware CLI path uses. An operator who exports
    a value deliberately keeps it."""
    (reanchor_env / ".env").write_text("KESTREL_DB_BACKEND=postgres\n")
    os.environ["KESTREL_DB_BACKEND"] = "sqlite"
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="a" * 64,
        backup_path=None,
        unchanged=True,
        target_backend="sqlite",
        target_label="sqlite:/fake/kestrel_prime.db",
    )

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        assert cmd_constitution(args) == 0

    assert os.environ["KESTREL_DB_BACKEND"] == "sqlite"


def test_reanchor_output_names_the_database_it_wrote(reanchor_env, capsys):
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="b" * 64,
        backup_path=None,
        reanchored=True,
        target_backend="postgres",
        target_label="postgresql://db.internal:5432/kestrel",
        backup_unavailable_reason="no file-level backup: governance lives in PostgreSQL",
    )
    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        assert cmd_constitution(args) == 0

    out = capsys.readouterr().out
    assert "postgresql://db.internal:5432/kestrel" in out
    assert "no file-level backup" in out
    # The stale claim: a backup file named after the local anchor.
    assert "kestrel_prime.db.backup-" not in out


def test_unforced_drift_on_postgres_does_not_promise_a_file_backup(
    reanchor_env, capsys,
):
    result = ReanchorResult(
        agent_name="Test",
        db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
        canonical_path=Path("/fake/canonical.md"),
        old_hash="a" * 64,
        new_hash="b" * 64,
        backup_path=None,
        drift_unforced=True,
        target_backend="postgres",
        target_label="postgresql://db.internal:5432/kestrel",
    )
    args = _parse(["constitution", "reanchor", "--agent-name", "Test"])

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_stubbed_helper(result),
         ):
        assert cmd_constitution(args) == 1

    out = capsys.readouterr().out
    assert "backed up to" not in out
    assert "Snapshot that database first" in out


def test_anchor_overlay_reads_the_agent_homes_env_too(
    reanchor_env, restore_environ, capsys,
):
    """``anchor-overlay`` shares the target rule, so it must share the env
    load. The overlay hash authorizes DANGEROUS Amendment IX grants; writing
    it to the local file on a PostgreSQL host leaves every grant denied while
    reporting success."""
    (reanchor_env / ".env").write_text(
        "KESTREL_DB_BACKEND=postgres\n"
        "KESTREL_DATABASE_URL=postgresql://u:p@127.0.0.1:1/none\n"
    )
    agent_dir = reanchor_env / "agent_data" / "Test"
    (agent_dir / "CONSTITUTION.md").write_bytes(b"# Overlay\n")
    before = (agent_dir / "kestrel_prime.db").read_bytes()
    args = _parse(["constitution", "anchor-overlay", "--agent-name", "Test"])

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False):
        rc = cmd_constitution(args)

    assert rc == 1
    err = capsys.readouterr().err
    assert "postgres" in err
    assert "Nothing was written" in err
    assert "u:p@" not in err
    assert (agent_dir / "kestrel_prime.db").read_bytes() == before


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

def test_constitution_with_no_subcommand_prints_usage(reanchor_env, capsys):
    args = _parse(["constitution"])
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env):
        rc = cmd_constitution(args)
    assert rc == 1
    assert "Usage" in capsys.readouterr().out


def test_reanchor_refuses_a_data_key_that_will_not_open_the_target(
    reanchor_env, monkeypatch, capsys,
):
    """The database and the key that opens it must come from one place.

    The target is resolved with the launcher's precedence (file wins) while
    ``load_project_env`` leaves an exported ``KESTREL_DATA_KEY`` authoritative.
    A shell still holding another home's credentials would therefore encrypt
    the new constitution and its artifact into database A under key B — and the
    agent, opening A with A's key, fails decryption at its next integrity
    audit. That is not a visible wrong answer; it is a governance record nobody
    can read again.
    """
    (reanchor_env / ".env").write_text("KESTREL_DATA_KEY=the-projects-key\n")
    monkeypatch.setenv("KESTREL_DATA_KEY", "a-stale-exported-key")

    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    called = False

    async def _must_not_run(**_kwargs):
        nonlocal called
        called = True

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_must_not_run,
         ):
        rc = cmd_constitution(args)

    assert rc == 2
    assert not called, "refused runs must not reach the writer"
    err = capsys.readouterr().err
    assert "KESTREL_DATA_KEY" in err
    assert "cannot decrypt" in err
    # It refuses rather than choosing: the conflict is the operator's to settle.
    assert "Unset" in err


def test_reanchor_proceeds_when_the_keys_agree(reanchor_env, monkeypatch):
    """An exported key identical to the file's is not a conflict."""
    (reanchor_env / ".env").write_text("KESTREL_DATA_KEY=the-projects-key\n")
    monkeypatch.setenv("KESTREL_DATA_KEY", "the-projects-key")

    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    reached = False

    async def _capture(**_kwargs):
        nonlocal reached
        reached = True
        return ReanchorResult(
            agent_name="Test",
            db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
            canonical_path=Path("/fake/canonical.md"),
            old_hash="a" * 64,
            new_hash="a" * 64,
            backup_path=None,
            unchanged=True,
        )

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_capture,
         ):
        assert cmd_constitution(args) == 0

    assert reached


def test_reanchor_refuses_a_blank_project_key_against_an_exported_one(
    reanchor_env, monkeypatch, capsys,
):
    """Empty is an answer, not an absence.

    A project ``.env`` that sets ``KESTREL_DATA_KEY=`` explicitly gives the
    spawned agent an empty key. Requiring both sides to be truthy let the
    exported key survive here, so the reanchor would encrypt blobs the agent
    cannot read — the same lesson the DSN resolution learned, in a place it had
    not been carried to.
    """
    (reanchor_env / ".env").write_text("KESTREL_DATA_KEY=\n")
    monkeypatch.setenv("KESTREL_DATA_KEY", "a-stale-exported-key")

    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False):
        rc = cmd_constitution(args)

    assert rc == 2
    assert "KESTREL_DATA_KEY" in capsys.readouterr().err


def test_a_key_only_in_the_environment_is_not_a_conflict(
    reanchor_env, monkeypatch,
):
    """The file saying nothing means the export reaches the agent too.

    ``spawned_agent_env`` starts from ``os.environ``, so with no key in the
    file the agent gets the exported one — the same key this command uses.
    Refusing there would block a legitimate setup.
    """
    (reanchor_env / ".env").write_text("OPENAI_API_KEY=sk-x\n")
    monkeypatch.setenv("KESTREL_DATA_KEY", "the-only-key")

    args = _parse(["constitution", "reanchor", "--agent-name", "Test", "--force"])
    reached = False

    async def _capture(**_kwargs):
        nonlocal reached
        reached = True
        return ReanchorResult(
            agent_name="Test",
            db_path=reanchor_env / "agent_data" / "Test" / "kestrel_prime.db",
            canonical_path=Path("/fake/canonical.md"),
            old_hash="a" * 64, new_hash="a" * 64,
            backup_path=None, unchanged=True,
        )

    with patch("kestrel_sovereign.cli._get_project_dir", return_value=reanchor_env), \
         patch("kestrel_sovereign.cli._agent_appears_running", return_value=False), \
         patch(
             "kestrel_sovereign.setup.constitution_reanchor.reanchor_constitution",
             side_effect=_capture,
         ):
        assert cmd_constitution(args) == 0

    assert reached
