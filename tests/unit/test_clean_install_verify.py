"""Unit tests for scripts/ci/clean_install_verify.py.

The CI workflow's correctness depends on these assertions returning the
right exit code for the right state. We test them in-process by
chdir'ing into a tmp_path that we've populated to look like a post-
wizard project — bypasses the heavy wizard run while exercising the
exact code paths CI uses.

Pure stdlib, no fixtures from the project — the script is intentionally
import-light and these tests follow.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "ci" / "clean_install_verify.py"
)


def _load_module():
    """Import the verifier as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "clean_install_verify_under_test", _SCRIPT_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verify = _load_module()


# ---------------------------------------------------------------------------
# Fixtures: build a fake post-wizard tree
# ---------------------------------------------------------------------------

def _make_post_wizard_project(tmp_path: Path, agent_name: str = "Kestrel") -> Path:
    """Lay out the files the verifier expects to find in a green project."""
    (tmp_path / ".env").write_text(
        "KESTREL_DATA_KEY=test-fernet-key-not-real\n"
        "KESTREL_API_KEY=test-api-key\n",
        encoding="utf-8",
    )
    (tmp_path / "kestrel.toml").write_text(
        '[llm]\nroute_priority = ["ollama:local"]\n', encoding="utf-8"
    )
    (tmp_path / "rookery.toml").write_text(
        "[host]\nport = 8888\nbind = \"0.0.0.0\"\n\n"
        f"[agents.{agent_name}]\n"
        f"data_dir = \"agent_data/{agent_name}\"\n"
        f"port = 8801\nautostart = true\n",
        encoding="utf-8",
    )

    # Build a minimal SQLite DB matching the schema the assertions probe.
    agent_dir = tmp_path / "agent_data" / agent_name
    agent_dir.mkdir(parents=True)
    db_path = agent_dir / "kestrel_prime.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT,
                properties TEXT
            );
            CREATE TABLE graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL
            );
            CREATE TABLE files (
                hash TEXT PRIMARY KEY,
                original_name TEXT NOT NULL
            );
            CREATE TABLE document_chunks (
                chunk_id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL
            );
            CREATE TABLE conversation_history (
                id INTEGER PRIMARY KEY,
                content TEXT
            );
            INSERT INTO graph_nodes(node_id, node_type, label) VALUES
              ('did:pkh:eip155:1:0xTESTfakeFAKEfake', 'agent', 'Kestrel'),
              ('constitution-hash-abc', 'document', 'KESTREL_CONSTITUTION');
            INSERT INTO graph_edges(source_id, target_id, label) VALUES
              ('did:pkh:eip155:1:0xTESTfakeFAKEfake', 'constitution-hash-abc', 'governed_by');
            INSERT INTO files(hash, original_name) VALUES
              ('constitution-hash-abc', 'KESTREL_CONSTITUTION.md');
            INSERT INTO document_chunks(chunk_id, file_hash) VALUES
              ('chunk-1', 'constitution-hash-abc'),
              ('chunk-2', 'constitution-hash-abc');
            """
        )
    return db_path


def _run(cmd_fn, **kwargs) -> int:
    """Call a subcommand with kwargs as argparse.Namespace attributes."""
    import argparse
    args = argparse.Namespace(**kwargs)
    return cmd_fn(args)


# ---------------------------------------------------------------------------
# wizard-artifacts
# ---------------------------------------------------------------------------

def test_wizard_artifacts_passes_on_post_wizard_tree(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_wizard_artifacts)
    assert rc == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "KESTREL_DATA_KEY set" in out


def test_wizard_artifacts_fails_when_env_missing(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    (tmp_path / ".env").unlink()
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_wizard_artifacts)
    assert rc == 1
    err = capsys.readouterr().err
    assert ".env" in err


def test_wizard_artifacts_fails_when_data_key_missing(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    (tmp_path / ".env").write_text("OTHER=value\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_wizard_artifacts)
    assert rc == 1
    assert "KESTREL_DATA_KEY" in capsys.readouterr().err


def test_wizard_artifacts_fails_when_route_priority_empty(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    (tmp_path / "kestrel.toml").write_text(
        "[llm]\nroute_priority = []\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_wizard_artifacts)
    assert rc == 1
    assert "route_priority" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_identity_passes_when_did_present(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_identity, agent_name="Kestrel")
    assert rc == 0
    out = capsys.readouterr().out
    assert "did:pkh:eip155:1:0xTESTfakeFAKEfake" in out


def test_identity_fails_when_db_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_identity, agent_name="Ghost")
    assert rc == 1
    assert "Agent database not created" in capsys.readouterr().err


def test_identity_fails_when_no_agent_node(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    db = tmp_path / "agent_data" / "Kestrel" / "kestrel_prime.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("DELETE FROM graph_nodes WHERE node_type='agent'")
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_identity, agent_name="Kestrel")
    assert rc == 1
    assert "No DID found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# constitution
# ---------------------------------------------------------------------------

def test_constitution_passes_with_full_anchor(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_constitution, agent_name="Kestrel")
    assert rc == 0
    out = capsys.readouterr().out
    assert "governed_by=1" in out
    assert "rag_chunks=2" in out


def test_constitution_fails_when_no_governed_by_edge(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    db = tmp_path / "agent_data" / "Kestrel" / "kestrel_prime.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("DELETE FROM graph_edges WHERE label='governed_by'")
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_constitution, agent_name="Kestrel")
    assert rc == 1
    assert "Constitution not anchored" in capsys.readouterr().err


def test_constitution_fails_when_no_constitution_file(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    db = tmp_path / "agent_data" / "Kestrel" / "kestrel_prime.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("DELETE FROM files")
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_constitution, agent_name="Kestrel")
    assert rc == 1


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def test_memory_passes_when_required_tables_present(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_memory, agent_name="Kestrel")
    assert rc == 0
    assert "all required present" in capsys.readouterr().out


def test_memory_fails_when_table_missing(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    db = tmp_path / "agent_data" / "Kestrel" / "kestrel_prime.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("DROP TABLE conversation_history")
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_memory, agent_name="Kestrel")
    assert rc == 1
    err = capsys.readouterr().err
    assert "conversation_history" in err


# ---------------------------------------------------------------------------
# did-persists
# ---------------------------------------------------------------------------

def test_did_persists_passes_when_db_intact(tmp_path, monkeypatch, capsys):
    _make_post_wizard_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_did_persists, agent_name="Kestrel")
    assert rc == 0
    assert "DID persists" in capsys.readouterr().out


def test_did_persists_fails_when_db_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = _run(verify.cmd_did_persists, agent_name="Vanished")
    assert rc == 1


# ---------------------------------------------------------------------------
# Helpers (unit-level)
# ---------------------------------------------------------------------------

def test_read_dotenv_handles_quotes_and_blanks(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# header\n"
        "BARE=naked\n"
        "DOUBLE=\"with spaces\"\n"
        "SINGLE='abc'\n"
        "\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    parsed = verify._read_dotenv(p)
    assert parsed == {
        "BARE": "naked",
        "DOUBLE": "with spaces",
        "SINGLE": "abc",
        "EMPTY": "",
    }


def test_read_toml_missing_returns_empty(tmp_path):
    assert verify._read_toml(tmp_path / "nope.toml") == {}


def test_agent_port_lookup(tmp_path, monkeypatch):
    _make_post_wizard_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert verify._agent_port("Kestrel") == 8801
    assert verify._agent_port("Other") is None


def test_main_dispatch_unknown_subcommand_exits():
    """Unknown subcommand should fail at argparse, not silently no-op."""
    with pytest.raises(SystemExit):
        verify.main(["nonexistent-sub"])
