"""Unit tests for kestrel_sovereign.doctor."""

from __future__ import annotations

from pathlib import Path

import toml
from cryptography.fernet import Fernet

from kestrel_sovereign.doctor import diagnose, format_report
from kestrel_sovereign.rookery.config import (
    HostConfig,
    LocalAgentConfig,
    ROOKERY_CONFIG_FILENAME,
    RookeryConfig,
)
from kestrel_sovereign.setup.env_file import write_env
from kestrel_sovereign.setup.toml_file import write_toml


def _seed_ready(tmp_path: Path) -> None:
    """Build a fully-ready project tree."""
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
    rookery = RookeryConfig(
        host=HostConfig(),
        agents={
            "Test": LocalAgentConfig(
                data_dir=Path("agent_data/test"), port=8801, autostart=True
            )
        },
    )
    rookery.save(tmp_path / ROOKERY_CONFIG_FILENAME)
    db_dir = tmp_path / "agent_data" / "test"
    db_dir.mkdir(parents=True)
    (db_dir / "kestrel_prime.db").write_bytes(b"")


def test_doctor_reports_ready_when_everything_set(tmp_path):
    _seed_ready(tmp_path)
    report = diagnose(tmp_path)
    assert report.ready, f"fail={report.fail}"
    assert report.fail == []


def test_doctor_blocks_on_missing_data_key(tmp_path):
    _seed_ready(tmp_path)
    # Wipe .env
    (tmp_path / ".env").unlink()
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("KESTREL_DATA_KEY" in m for m in report.fail)


def test_doctor_blocks_on_empty_route_priority(tmp_path):
    _seed_ready(tmp_path)
    write_toml(tmp_path / "kestrel.toml", {"llm": {"route_priority": []}}, deep_merge=False)
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("route_priority" in m for m in report.fail)


def test_doctor_blocks_on_missing_api_key_env(tmp_path):
    _seed_ready(tmp_path)
    # Remove OPENAI_API_KEY but keep route
    p = tmp_path / ".env"
    text = p.read_text()
    p.write_text("\n".join(
        line for line in text.splitlines() if not line.startswith("OPENAI_API_KEY=")
    ) + "\n")
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("OPENAI_API_KEY" in m for m in report.fail)


def test_doctor_blocks_when_no_agents(tmp_path):
    _seed_ready(tmp_path)
    (tmp_path / "rookery.toml").unlink()
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("agent" in m.lower() for m in report.fail)


def test_doctor_blocks_when_agent_db_missing(tmp_path):
    _seed_ready(tmp_path)
    (tmp_path / "agent_data" / "test" / "kestrel_prime.db").unlink()
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("kestrel_prime.db" in m for m in report.fail)


def test_format_report_renders_lines(tmp_path):
    _seed_ready(tmp_path)
    report = diagnose(tmp_path)
    text = format_report(report)
    assert "✅" in text
    assert "Ready" in text


def test_format_report_says_not_ready_when_blocked(tmp_path):
    """Empty project should produce a not-ready message."""
    report = diagnose(tmp_path)
    text = format_report(report)
    assert "Not ready" in text
    assert "❌" in text


# ---------------------------------------------------------------------------
# Constitution drift detection
#
# Strategy: build a real SQLite DB with the schema the inception service
# produces, plus a real KESTREL_CONSTITUTION.md, and let the doctor read
# it. We synthesize the DB directly so these tests don't need to spin
# up the heavy async inception path.
# ---------------------------------------------------------------------------


import hashlib  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
from unittest.mock import patch  # noqa: E402

from kestrel_sovereign.doctor import (  # noqa: E402
    _NoAgentNode,
    _NoHashProperty,
    _UnreadableDB,
    _read_anchored_constitution_hash,
)


def _seed_with_anchored_constitution(
    tmp_path: Path,
    *,
    constitution_text: bytes,
    stored_hash: str | None,
) -> None:
    """Build a project tree where the agent's anchored hash is exactly ``stored_hash``.

    Pass ``stored_hash=None`` to omit the constitution_hash property
    entirely (older-agent scenario).

    Always writes to ``tmp_path / "agent_data" / "test"`` to match the
    lowercase data_dir produced by ``_seed_ready``. On case-sensitive
    filesystems (Linux) ``"test"`` and ``"Test"`` are different
    directories, and the rookery's ``data_dir=Path("agent_data/test")``
    is what doctor reads — anchoring elsewhere would silently
    write into the wrong location.
    """
    _seed_ready(tmp_path)

    # Match the path _seed_ready uses (rookery says data_dir is
    # "agent_data/test" — lowercase, regardless of the agent's name).
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    properties: dict = {"name": "Test"}
    if stored_hash is not None:
        properties["constitution_hash"] = stored_hash

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT,
                properties TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO graph_nodes(node_id, node_type, label, properties) "
            "VALUES (?, 'agent', ?, ?)",
            ("did:test:Test", "Test", json.dumps(properties)),
        )
        conn.commit()


def _patch_canonical(tmp_path: Path, content: bytes) -> Path:
    """Write a fake canonical constitution and return its path.

    Returns the path so callers can monkeypatch
    ``kestrel_sovereign.config.CONSTITUTION_PATH`` to point at it.
    """
    p = tmp_path / "FAKE_CONSTITUTION.md"
    p.write_bytes(content)
    return p


def test_constitution_drift_passes_when_hashes_match(tmp_path, monkeypatch):
    text = b"# Kestrel Constitution\nv1\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=text,
        stored_hash=hashlib.sha256(text).hexdigest(),
    )
    report = diagnose(tmp_path)
    assert any(
        "constitution anchored to current file" in m for m in report.ok
    )
    # No drift fails generated.
    assert not any("constitution drift" in m for m in report.fail)


def test_constitution_drift_fails_when_file_changed(tmp_path, monkeypatch):
    """The bug we're catching: file edited, agent never reanchored."""
    original = b"# Kestrel Constitution\nv1\n"
    edited = b"# Kestrel Constitution\nv2 - amended\n"

    canonical = _patch_canonical(tmp_path, edited)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=original,
        stored_hash=hashlib.sha256(original).hexdigest(),
    )
    report = diagnose(tmp_path)
    drift_msgs = [m for m in report.fail if "constitution drift" in m]
    assert len(drift_msgs) == 1
    msg = drift_msgs[0]
    # The message must include both hashes (truncated) so the user can
    # see exactly what diverged. The reanchor CLI lands in a follow-up
    # PR, so the message must NOT name a command that doesn't exist yet
    # — instead it states that reanchor support is planned.
    assert hashlib.sha256(original).hexdigest()[:12] in msg
    assert hashlib.sha256(edited).hexdigest()[:12] in msg
    assert "reanchor support is planned" in msg.lower()
    assert "kestrel constitution reanchor" not in msg.lower(), (
        "Drift message must not promise a CLI that does not exist yet"
    )


def test_constitution_drift_warns_on_missing_hash_property(tmp_path, monkeypatch):
    """Older agent that never anchored. Surfaced as a warning, not a fail —
    blocking would prevent users from upgrading to a hash-anchored agent."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    _seed_with_anchored_constitution(
        tmp_path, constitution_text=text, stored_hash=None,
    )
    report = diagnose(tmp_path)
    assert any(
        "missing constitution_hash property" in m for m in report.warn
    )
    assert not any("constitution drift" in m for m in report.fail)


def test_constitution_drift_warns_on_unreadable_db(tmp_path, monkeypatch):
    """Sqlcipher-encrypted DB looks like 'file is not a database' to stock
    sqlite3 — must skip, not crash, not fail the whole doctor run."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    _seed_ready(tmp_path)
    # Replace the empty-bytes DB with garbage that will fail to parse.
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"
    db_path.write_bytes(b"this is not a sqlite database, deliberately")
    report = diagnose(tmp_path)
    assert any(
        "constitution drift check skipped" in m and "DB unreadable" in m
        for m in report.warn
    )
    # And no fails introduced.
    assert not any("constitution drift" in m for m in report.fail)


def test_constitution_drift_warns_when_canonical_file_missing(
    tmp_path, monkeypatch
):
    """Cannot drift-check if we can't read the canonical file."""
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH",
        str(tmp_path / "does_not_exist.md"),
    )
    _seed_ready(tmp_path)
    report = diagnose(tmp_path)
    assert any(
        "Constitution drift check skipped — cannot read canonical" in m
        for m in report.warn
    )


def test_constitution_drift_silent_when_no_agents(tmp_path, monkeypatch):
    """No registered agents → no per-agent drift checks. The empty
    rookery is already failing the check_rookery step elsewhere; we
    should not pile on additional drift noise."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    # No rookery written → no agents.
    report = diagnose(tmp_path)
    # Match only the exact phrases the drift check itself emits, NOT
    # substrings — pytest's tmp_path names include the test function
    # name (which contains "constitution" and "drift"), so a substring
    # match against arbitrary report messages would catch path strings.
    drift_phrases = (
        "constitution drift",
        "constitution anchored to current file",
        "constitution drift check skipped",
        "Constitution drift check skipped",
    )

    def _is_drift_msg(m: str) -> bool:
        return any(p in m for p in drift_phrases)

    drift_msgs = [
        m for m in (*report.warn, *report.fail, *report.ok)
        if _is_drift_msg(m)
    ]
    assert drift_msgs == []


def test_constitution_drift_skips_when_db_missing(tmp_path, monkeypatch):
    """Missing DB is already a fail in check_rookery; drift check should
    not pile on a duplicate message."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    _seed_ready(tmp_path)
    (tmp_path / "agent_data" / "test" / "kestrel_prime.db").unlink()
    report = diagnose(tmp_path)
    drift_phrases = (
        "constitution drift",
        "constitution anchored to current file",
        "constitution drift check skipped",
        "Constitution drift check skipped",
    )
    drift_msgs = [
        m for m in (*report.warn, *report.fail, *report.ok)
        if any(p in m for p in drift_phrases)
    ]
    # Only the no-db fail remains; nothing added by drift check itself.
    assert drift_msgs == []


# ---------------------------------------------------------------------------
# Helper-level tests for _read_anchored_constitution_hash
# ---------------------------------------------------------------------------


def test_read_hash_returns_string_on_happy_path(tmp_path):
    db = tmp_path / "k.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(
            """CREATE TABLE graph_nodes (
                node_id TEXT, node_type TEXT, label TEXT, properties TEXT
            );"""
        )
        conn.execute(
            "INSERT INTO graph_nodes VALUES (?, 'agent', ?, ?)",
            ("did:x", "x", json.dumps({"constitution_hash": "abc123"})),
        )
        conn.commit()
    assert _read_anchored_constitution_hash(db) == "abc123"


def test_read_hash_handles_no_graph_nodes_table(tmp_path):
    db = tmp_path / "k.db"
    sqlite3.connect(str(db)).close()  # Empty DB.
    assert isinstance(_read_anchored_constitution_hash(db), _UnreadableDB)


def test_read_hash_handles_no_agent_node(tmp_path):
    db = tmp_path / "k.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(
            """CREATE TABLE graph_nodes (
                node_id TEXT, node_type TEXT, label TEXT, properties TEXT
            );"""
        )
        conn.commit()
    assert isinstance(_read_anchored_constitution_hash(db), _NoAgentNode)


def test_read_hash_handles_corrupt_properties_json(tmp_path):
    db = tmp_path / "k.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript(
            """CREATE TABLE graph_nodes (
                node_id TEXT, node_type TEXT, label TEXT, properties TEXT
            );"""
        )
        conn.execute(
            "INSERT INTO graph_nodes VALUES (?, 'agent', ?, ?)",
            ("did:x", "x", "{not valid json"),
        )
        conn.commit()
    assert isinstance(_read_anchored_constitution_hash(db), _NoHashProperty)
