"""Unit tests for kestrel_sovereign.doctor."""

from __future__ import annotations

from pathlib import Path

import toml
from cryptography.fernet import Fernet

from kestrel_sovereign.doctor import (
    diagnose,
    format_report,
)
from kestrel_sovereign.multi_agent.config import (
    MULTI_AGENT_CONFIG_FILENAME,
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
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
    multi_agent = MultiAgentConfig(
        host=HostConfig(),
        agents={
            "Test": LocalAgentConfig(
                data_dir=Path("agent_data/test"), port=8801, autostart=True
            )
        },
    )
    multi_agent.save(tmp_path / MULTI_AGENT_CONFIG_FILENAME)
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
    write_toml(
        tmp_path / "kestrel.toml",
        {"llm": {"route_priority": []}},
        deep_merge=False,
    )
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("route_priority" in m for m in report.fail)


def test_doctor_blocks_on_missing_api_key_env(tmp_path):
    _seed_ready(tmp_path)
    # Remove OPENAI_API_KEY but keep route
    p = tmp_path / ".env"
    text = p.read_text()
    p.write_text(
        "\n".join(
            line for line in text.splitlines() if not line.startswith("OPENAI_API_KEY=")
        )
        + "\n"
    )
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("OPENAI_API_KEY" in m for m in report.fail)


def test_doctor_accepts_openrouter_management_key_only_undeclared_env(tmp_path):
    """Management-key-only OpenRouter passes doctor even when the route
    TOML omits ``management_api_key_env`` (#2245).

    setup --check accepts this shape via the vendor default fallback;
    doctor must agree or a valid setup fails ``kestrel doctor``.
    """
    _seed_ready(tmp_path)
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
            "OPENROUTER_MANAGEMENT_API_KEY": "sk-mgmt-x",
        },
    )
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {
                "route_priority": ["openrouter:api"],
                "vendors": {
                    "openrouter": {
                        "is_cloud": True,
                        "routes": {
                            "api": {
                                "adapter": "OpenRouterAdapter",
                                # NOTE: no management_api_key_env declared —
                                # doctor must fall back to the vendor alt key.
                                "api_key_env": "OPENROUTER_API_KEY",
                            }
                        },
                    }
                },
            }
        },
        deep_merge=False,
    )
    report = diagnose(tmp_path)
    assert report.ready, f"fail={report.fail}"
    assert not any("OPENROUTER_API_KEY not set" in m for m in report.fail)


def test_doctor_blocks_when_no_agents(tmp_path):
    _seed_ready(tmp_path)
    (tmp_path / "multi_agent.toml").unlink()
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


def test_doctor_reports_legacy_identity_export_permissions_without_reading(tmp_path):
    _seed_ready(tmp_path)
    export_root = tmp_path / "agent_data"
    legacy = export_root / "identity_legacy.json"
    secret = "doctor-must-not-echo-or-parse-this-secret"
    legacy.write_text(secret, encoding="utf-8")
    legacy.chmod(0o644)

    report = diagnose(tmp_path)

    warning = next(message for message in report.warn if "identity-export" in message)
    assert "metadata-only" in warning
    assert "harden-exports" in warning
    assert secret not in warning
    assert legacy.read_text(encoding="utf-8") == secret


def test_doctor_accepts_private_identity_export_metadata(tmp_path):
    _seed_ready(tmp_path)
    export_root = tmp_path / "agent_data"
    export_root.chmod(0o700)
    private = export_root / "identity_private.json"
    private.write_text("private", encoding="utf-8")
    private.chmod(0o600)

    report = diagnose(tmp_path)

    assert not any("identity-export" in message for message in report.warn)


# ---------------------------------------------------------------------------
# Constitution drift detection
#
# Strategy: build a real SQLite DB with the schema the inception service
# produces, plus a real KESTREL_CONSTITUTION.md, and let the doctor read
# it. We synthesize the DB directly so these tests don't need to spin
# up the heavy async inception path.
# ---------------------------------------------------------------------------


import hashlib
import json
import os
import sqlite3

import pytest

from kestrel_sovereign.doctor import (
    _anchored_constitution_hash,
    _anchored_emancipation_contract,
    _GovernanceSource,
    _NoAgentNode,
    _NoHashProperty,
    _UnreadableDB,
)

_TEST_DID = "did:test:Test"


def _anchor(
    db_path, agent_did: str = _TEST_DID, *, ownership_settled: bool = True
) -> _GovernanceSource:
    """A source that reads the local anchor — the SQLite host's shape.

    The readers take a :class:`_GovernanceSource` rather than a path, because
    on a PostgreSQL host the anchor holds the *birth record* and the live
    governance is elsewhere (#2892). These helper-level tests are about the
    SQLite reading itself, so they say which database they mean rather than
    relying on a default.

    ``agent_did`` is required on both backends: the runtime's store is bound to
    it everywhere, so every governance read carries an ownership predicate.

    ``ownership_settled`` defaults to True — a database the runtime has already
    booted against, which is what these reader tests are about. It is what
    selects the scoped SQL; before #2649 is recorded complete the legacy reads
    are the faithful ones, because boot is about to assign the witnesses a
    scoped read would hide.
    """
    return _GovernanceSource(
        anchor_path=db_path,
        agent_did=agent_did,
        ownership_settled=ownership_settled,
    )


# Sentinel default: "make the governed_by edge target the stored hash",
# distinguishable from an explicit None (= no edge at all).
_EDGE_MATCHES_ANCHOR = object()


def _seed_with_anchored_constitution(
    tmp_path: Path,
    *,
    constitution_text: bytes,
    stored_hash: str | None,
    governed_by_target: object = _EDGE_MATCHES_ANCHOR,
    overlay_anchor: object = None,
    create_edges_table: bool = True,
    witness_node: bool = True,
    witness_edge: bool = True,
) -> None:
    """Build a project tree where the agent's anchored hash is exactly ``stored_hash``.

    Pass ``stored_hash=None`` to omit the constitution_hash property
    entirely (older-agent scenario).

    By default the DB carries the realistic inception governance wiring:
    a ``graph_edges`` table with an ``agent --governed_by--> stored_hash``
    edge. ``governed_by_target`` overrides the edge target (a stale-anchor
    DB) or, when ``None``, omits the edge. ``create_edges_table=False``
    synthesizes a legacy DB with no ``graph_edges`` table at all.
    ``overlay_anchor`` sets ``constitution_overlay_hash`` on the agent node.

    Always writes to ``tmp_path / "agent_data" / "test"`` to match the
    lowercase data_dir produced by ``_seed_ready``. On case-sensitive
    filesystems (Linux) ``"test"`` and ``"Test"`` are different
    directories, and the multi_agent's ``data_dir=Path("agent_data/test")``
    is what doctor reads — anchoring elsewhere would silently
    write into the wrong location.
    """
    _seed_ready(tmp_path)

    # Match the path _seed_ready uses (multi_agent says data_dir is
    # "agent_data/test" — lowercase, regardless of the agent's name).
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    properties: dict = {"name": "Test"}
    if stored_hash is not None:
        properties["constitution_hash"] = stored_hash
    if overlay_anchor is not None:
        properties["constitution_overlay_hash"] = overlay_anchor

    if governed_by_target is _EDGE_MATCHES_ANCHOR:
        governed_by_target = stored_hash

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT,
                properties TEXT
            );
            CREATE TABLE graph_node_owners (
                node_id TEXT NOT NULL,
                agent_id TEXT NOT NULL
            );
            CREATE TABLE graph_edge_owners (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                label TEXT NOT NULL,
                agent_id TEXT NOT NULL
            );
            CREATE TABLE schema_backfills (
                name TEXT PRIMARY KEY,
                completed_at TIMESTAMP
            );
            """
        )
        if create_edges_table:
            conn.executescript(
                """
                CREATE TABLE graph_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties TEXT
                );
                """
            )
        conn.execute(
            "INSERT INTO graph_nodes(node_id, node_type, label, properties) "
            "VALUES (?, 'agent', ?, ?)",
            ("did:test:Test", "Test", json.dumps(properties)),
        )
        # The ownership witness a real write always lays down beside the row
        # (``AsyncGraphStore.add_node`` -> ``record_graph_node_owner``). It is
        # what the bound runtime store actually matches on, so a seed without
        # it is not a healthy agent — it is the invisible-row case.
        conn.execute("INSERT INTO schema_backfills VALUES ('ownership_2649', NULL)")
        if witness_node:
            conn.execute(
                "INSERT INTO graph_node_owners(node_id, agent_id) VALUES (?, ?)",
                ("did:test:Test", "did:test:Test"),
            )
        if create_edges_table and governed_by_target is not None:
            conn.execute(
                "INSERT INTO graph_edges(source_id, target_id, label, properties) "
                "VALUES (?, ?, 'governed_by', NULL)",
                ("did:test:Test", governed_by_target),
            )
            if witness_edge:
                conn.execute(
                    "INSERT INTO graph_edge_owners"
                    "(source_id, target_id, label, agent_id) "
                    "VALUES (?, ?, 'governed_by', ?)",
                    ("did:test:Test", governed_by_target, "did:test:Test"),
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
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=text,
        stored_hash=hashlib.sha256(text).hexdigest(),
    )
    report = diagnose(tmp_path)
    assert any("constitution anchored to current file" in m for m in report.ok)
    # No drift fails generated.
    assert not any("constitution drift" in m for m in report.fail)


def test_constitution_drift_fails_when_file_changed(tmp_path, monkeypatch):
    """The bug we're catching: file edited, agent never reanchored."""
    original = b"# Kestrel Constitution\nv1\n"
    edited = b"# Kestrel Constitution\nv2 - amended\n"

    canonical = _patch_canonical(tmp_path, edited)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=original,
        stored_hash=hashlib.sha256(original).hexdigest(),
    )
    report = diagnose(tmp_path)
    drift_msgs = [m for m in report.fail if "constitution drift" in m]
    assert len(drift_msgs) == 1
    msg = drift_msgs[0]
    # The message must include both hashes (truncated) and a remediation
    # hint that points at the actual CLI command. The "Test" agent name
    # must appear in the suggested command, parameterised so users can
    # copy-paste.
    assert hashlib.sha256(original).hexdigest()[:12] in msg
    assert hashlib.sha256(edited).hexdigest()[:12] in msg
    assert "kestrel constitution reanchor --agent-name Test --force" in msg
    assert "DB is backed up first" in msg


def test_constitution_drift_warns_on_missing_hash_property(tmp_path, monkeypatch):
    """Older agent that never anchored. Surfaced as a warning, not a fail —
    blocking would prevent users from upgrading to a hash-anchored agent."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=text,
        stored_hash=None,
    )
    report = diagnose(tmp_path)
    assert any("missing constitution_hash property" in m for m in report.warn)
    assert not any("constitution drift" in m for m in report.fail)


def test_constitution_drift_warns_on_unreadable_db(tmp_path, monkeypatch):
    """Sqlcipher-encrypted DB looks like 'file is not a database' to stock
    sqlite3 — must skip, not crash, not fail the whole doctor run."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
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


def test_constitution_drift_warns_when_canonical_file_missing(tmp_path, monkeypatch):
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
    multi_agent is already failing the check_multi_agent step elsewhere; we
    should not pile on additional drift noise."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    # No multi_agent written → no agents.
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
        m for m in (*report.warn, *report.fail, *report.ok) if _is_drift_msg(m)
    ]
    assert drift_msgs == []


def test_constitution_drift_skips_when_db_missing(tmp_path, monkeypatch):
    """Missing DB is already a fail in check_multi_agent; drift check should
    not pile on a duplicate message."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
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
        m
        for m in (*report.warn, *report.fail, *report.ok)
        if any(p in m for p in drift_phrases)
    ]
    # Only the no-db fail remains; nothing added by drift check itself.
    assert drift_msgs == []


# ---------------------------------------------------------------------------
# Helper-level tests for _read_anchored_constitution_hash
# ---------------------------------------------------------------------------


def test_reads_the_hash_off_the_properties_it_was_given():
    assert _anchored_constitution_hash({"constitution_hash": "abc123"}) == "abc123"


@pytest.mark.parametrize(
    "properties",
    [
        None,  # corrupt/absent properties column
        {},  # older agent, never anchored
        {"constitution_hash": ""},  # present but empty
        {"constitution_hash": 12345},  # present but not a string
        "not a dict",  # properties parsed to a scalar
    ],
)
def test_no_usable_hash_is_its_own_verdict(properties):
    """``_NoHashProperty`` is distinct from "unreadable" on purpose: doctor
    tells an older agent to re-incept, and tells an unreadable database
    nothing at all."""
    assert isinstance(_anchored_constitution_hash(properties), _NoHashProperty)


def test_the_emancipation_contract_comes_from_the_same_row():
    """No second query, so no second failure to mistake for "no contract".

    Reading it separately meant one transient PostgreSQL error made doctor
    render the dormant constitution for an emancipated agent and report drift
    that was not there — a governance failure manufactured from a hiccup.
    """
    properties = {
        "constitution_hash": "abc123",
        "emancipation_contract": '{"enabled": true}',
    }
    assert _anchored_constitution_hash(properties) == "abc123"
    assert _anchored_emancipation_contract(properties) == '{"enabled": true}'


@pytest.mark.parametrize("properties", [None, {}, "not a dict"])
def test_an_agent_without_a_contract_reads_as_dormant(properties):
    """``None`` is the right answer *for an absent contract* — it resolves to
    the canonical bytes, which is what a non-emancipated agent should hash
    against. It is only wrong as the answer to a failed read."""
    assert _anchored_emancipation_contract(properties) is None


# ---------------------------------------------------------------------------
# Anchor consistency (#2616): governed_by edge + overlay anchoring
#
# The fail-closed integrity audit safe-modes agents at boot on a
# mis-targeted/missing governed_by edge (proof 2) or an unanchored/
# drifted/removed overlay. Doctor must surface these BEFORE an upgrade
# into that enforcement — the 2026-07-18 incident shape.
# ---------------------------------------------------------------------------


from kestrel_sovereign.doctor import (
    _read_agent_node,
    _read_governed_by_targets,
)


def _seed_matching_anchor(tmp_path, monkeypatch, **kwargs) -> str:
    """Seed a project whose base anchor matches the canonical file.

    Keeps the base drift check green so anchor-consistency asserts are
    isolated. Returns the stored hash.
    """
    text = b"# Kestrel Constitution\nv1\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    stored = hashlib.sha256(text).hexdigest()
    _seed_with_anchored_constitution(
        tmp_path, constitution_text=text, stored_hash=stored, **kwargs
    )
    return stored


def test_governed_by_edge_match_passes(tmp_path, monkeypatch):
    stored = _seed_matching_anchor(tmp_path, monkeypatch)
    report = diagnose(tmp_path)
    assert any(
        "governed_by edge targets the anchored constitution" in m and stored[:12] in m
        for m in report.ok
    )
    assert not any("anchor drift" in m for m in report.fail)


def test_governed_by_edge_mistargeted_fails(tmp_path, monkeypatch):
    """The 2026-07-18 incident: property + blob reanchored, edge still on
    the ancient anchor → proof 2 fails closed at boot."""
    ancient = hashlib.sha256(b"ancient constitution").hexdigest()
    stored = _seed_matching_anchor(tmp_path, monkeypatch, governed_by_target=ancient)
    report = diagnose(tmp_path)
    drift = [m for m in report.fail if "anchor drift" in m]
    assert len(drift) == 1
    msg = drift[0]
    assert ancient[:12] in msg
    assert stored[:12] in msg
    assert "safe-mode" in msg
    assert "kestrel constitution reanchor --agent-name Test --force" in msg
    # The base hash check must still pass — this drift is edge-only.
    assert any("constitution anchored to current file" in m for m in report.ok)


def test_governed_by_edge_missing_fails(tmp_path, monkeypatch):
    stored = _seed_matching_anchor(tmp_path, monkeypatch, governed_by_target=None)
    report = diagnose(tmp_path)
    drift = [m for m in report.fail if "anchor drift" in m]
    assert len(drift) == 1
    assert "no governed_by edge" in drift[0]
    assert stored[:12] in drift[0]
    assert "kestrel constitution reanchor --agent-name Test --force" in drift[0]


def test_governed_by_check_fails_without_edges_table(tmp_path, monkeypatch):
    """A DB whose graph_edges cannot be read (missing table/corruption)
    while the agent node CAN be read is not a skip — the runtime fails
    closed either way (an edge-read error is an integrity failure, and a
    missing table is auto-created empty, leaving proof 2 with no edge).
    Doctor must not report Ready and upgrade the operator into safe mode."""
    _seed_matching_anchor(tmp_path, monkeypatch, create_edges_table=False)
    report = diagnose(tmp_path)
    unverifiable = [
        m for m in report.fail if "cannot verify the governed_by governance edge" in m
    ]
    assert len(unverifiable) == 1
    assert "safe-mode" in unverifiable[0]
    assert "kestrel constitution reanchor --agent-name Test --force" in unverifiable[0]
    assert not report.ready


def test_governed_by_stale_extra_edge_warns_but_passes(tmp_path, monkeypatch):
    """Correct edge present + a stale extra target: proof 2 passes (it only
    requires one edge at the anchor), so boot succeeds — ok + warn, not fail."""
    stored = _seed_matching_anchor(tmp_path, monkeypatch)
    ancient = hashlib.sha256(b"ancient constitution").hexdigest()
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO graph_edges(source_id, target_id, label, properties) "
            "VALUES (?, ?, 'governed_by', NULL)",
            ("did:test:Test", ancient),
        )
        # With its witness: this is a dangling edge a *real* reanchor left
        # behind, so the agent genuinely owns it and genuinely sees it. An
        # unwitnessed edge is a different case, covered separately — and it
        # would make this test pass for the wrong reason, by being invisible.
        conn.execute(
            "INSERT INTO graph_edge_owners"
            "(source_id, target_id, label, agent_id) "
            "VALUES (?, ?, 'governed_by', ?)",
            ("did:test:Test", ancient, "did:test:Test"),
        )
        conn.commit()
    report = diagnose(tmp_path)
    assert any(
        "governed_by edge targets the anchored constitution" in m and stored[:12] in m
        for m in report.ok
    )
    stale = [m for m in report.warn if "stale extra governed_by edge" in m]
    assert len(stale) == 1
    assert ancient[:12] in stale[0]
    assert "kestrel constitution reanchor --agent-name Test --force" in stale[0]
    assert not any("anchor drift" in m for m in report.fail)


def test_governed_by_check_silent_without_base_anchor(tmp_path, monkeypatch):
    """No constitution_hash property → nothing for the edge to agree with.
    The base check already warns; the edge check must stay silent."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=text,
        stored_hash=None,
    )
    report = diagnose(tmp_path)
    assert not any(
        "governed_by" in m or "governance-edge" in m
        for m in (*report.ok, *report.warn, *report.fail)
    )


def test_overlay_unanchored_fails(tmp_path, monkeypatch):
    """Present-but-unanchored overlay (the incident's Emma shape) fails
    with the anchor-overlay remediation."""
    _seed_matching_anchor(tmp_path, monkeypatch)
    overlay = tmp_path / "agent_data" / "test" / "CONSTITUTION.md"
    overlay.write_bytes(b"# Overlay\ngrant: nothing\n")
    report = diagnose(tmp_path)
    unanchored = [m for m in report.fail if "NOT anchored" in m]
    assert len(unanchored) == 1
    assert "safe-mode" in unanchored[0]
    assert "kestrel constitution anchor-overlay --agent-name Test" in unanchored[0]


def test_overlay_anchored_matching_passes(tmp_path, monkeypatch):
    overlay_text = b"# Overlay\ngrant: nothing\n"
    overlay_hash = hashlib.sha256(overlay_text).hexdigest()
    _seed_matching_anchor(tmp_path, monkeypatch, overlay_anchor=overlay_hash)
    overlay = tmp_path / "agent_data" / "test" / "CONSTITUTION.md"
    overlay.write_bytes(overlay_text)
    report = diagnose(tmp_path)
    assert any(
        "constitution overlay anchored" in m and overlay_hash[:12] in m
        for m in report.ok
    )
    assert report.ready, f"fail={report.fail}"


def test_overlay_modified_fails(tmp_path, monkeypatch):
    anchored_text = b"# Overlay\nv1\n"
    anchored_hash = hashlib.sha256(anchored_text).hexdigest()
    _seed_matching_anchor(tmp_path, monkeypatch, overlay_anchor=anchored_hash)
    overlay = tmp_path / "agent_data" / "test" / "CONSTITUTION.md"
    modified = b"# Overlay\nv2 - edited\n"
    overlay.write_bytes(modified)
    report = diagnose(tmp_path)
    drift = [m for m in report.fail if "constitution overlay drift" in m]
    assert len(drift) == 1
    assert anchored_hash[:12] in drift[0]
    assert hashlib.sha256(modified).hexdigest()[:12] in drift[0]
    assert "kestrel constitution anchor-overlay --agent-name Test" in drift[0]


def test_overlay_anchor_missing_file_fails(tmp_path, monkeypatch):
    anchored_hash = hashlib.sha256(b"# Overlay\n").hexdigest()
    _seed_matching_anchor(tmp_path, monkeypatch, overlay_anchor=anchored_hash)
    # No overlay file written.
    report = diagnose(tmp_path)
    missing = [
        m
        for m in report.fail
        if "anchored constitution overlay is missing from disk" in m
    ]
    assert len(missing) == 1
    assert anchored_hash[:12] in missing[0]


def test_overlay_malformed_anchor_with_overlay_present_fails(tmp_path, monkeypatch):
    """A truthy non-string anchor counts as anchored at runtime (truthiness,
    not isinstance) and can never equal the overlay sha → drift fail, not
    'present but NOT anchored', and definitely not silence."""
    _seed_matching_anchor(
        tmp_path, monkeypatch, overlay_anchor={"hash": "not-a-string"}
    )
    overlay = tmp_path / "agent_data" / "test" / "CONSTITUTION.md"
    overlay.write_bytes(b"# Overlay\n")
    report = diagnose(tmp_path)
    drift = [m for m in report.fail if "constitution overlay drift" in m]
    assert len(drift) == 1
    assert "malformed non-string value" in drift[0]
    assert "kestrel constitution anchor-overlay --agent-name Test" in drift[0]
    assert not report.ready


def test_overlay_malformed_anchor_without_overlay_fails(tmp_path, monkeypatch):
    """Overlay file absent + truthy non-string anchor: the runtime treats
    the anchor as present → 'anchored overlay missing' tampering failure.
    Doctor must not silently coerce the malformed anchor to absent."""
    _seed_matching_anchor(
        tmp_path, monkeypatch, overlay_anchor={"hash": "not-a-string"}
    )
    report = diagnose(tmp_path)
    missing = [
        m
        for m in report.fail
        if "anchored constitution overlay is missing from disk" in m
    ]
    assert len(missing) == 1
    assert "malformed non-string value" in missing[0]
    assert not report.ready


def test_overlay_unreadable_with_anchor_fails(tmp_path, monkeypatch):
    """An overlay that exists but cannot be read is treated as ABSENT by the
    runtime; with an anchor set that is the tampering failure → safe mode.
    Doctor must fail, not warn. (A directory named CONSTITUTION.md exists()
    but raises OSError on read_bytes — cross-platform unreadability.)"""
    anchored_hash = hashlib.sha256(b"# Overlay\n").hexdigest()
    _seed_matching_anchor(tmp_path, monkeypatch, overlay_anchor=anchored_hash)
    (tmp_path / "agent_data" / "test" / "CONSTITUTION.md").mkdir()
    report = diagnose(tmp_path)
    unreadable = [m for m in report.fail if "cannot be read" in m]
    assert len(unreadable) == 1
    assert "safe-mode" in unreadable[0]
    assert anchored_hash[:12] in unreadable[0]
    assert not report.ready


def test_overlay_unreadable_without_anchor_warns(tmp_path, monkeypatch):
    """Unreadable overlay + no anchor: the runtime treats it as 'no overlay'
    and boots fine — doctor warns that verification was incomplete but must
    not block readiness (matrix-exact with verify_constitution_overlay)."""
    _seed_matching_anchor(tmp_path, monkeypatch)
    (tmp_path / "agent_data" / "test" / "CONSTITUTION.md").mkdir()
    report = diagnose(tmp_path)
    incomplete = [m for m in report.warn if "overlay anchor check incomplete" in m]
    assert len(incomplete) == 1
    assert not any("overlay" in m.lower() for m in report.fail)
    assert report.ready, f"fail={report.fail}"


def test_overlay_silent_when_absent_and_unanchored(tmp_path, monkeypatch):
    """Normal agent: no overlay file, no anchor → zero overlay messages."""
    _seed_matching_anchor(tmp_path, monkeypatch)
    report = diagnose(tmp_path)
    assert not any("overlay" in m for m in (*report.ok, *report.warn, *report.fail))


def test_overlay_checked_for_legacy_agent_without_base_anchor(tmp_path, monkeypatch):
    """#1722 legacy shape: anchored overlay but NO base constitution_hash.
    The overlay check must still run (the runtime audits it first and
    unconditionally)."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    overlay_text = b"# Overlay\n"
    overlay_hash = hashlib.sha256(overlay_text).hexdigest()
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=text,
        stored_hash=None,
        overlay_anchor=overlay_hash,
    )
    (tmp_path / "agent_data" / "test" / "CONSTITUTION.md").write_bytes(overlay_text)
    report = diagnose(tmp_path)
    assert any("constitution overlay anchored" in m for m in report.ok)


# Helper-level tests for the new readers.

_GRAPH_SCHEMA = """
CREATE TABLE graph_nodes (
    node_id TEXT, node_type TEXT, label TEXT, properties TEXT
);
CREATE TABLE graph_edges (
    source_id TEXT, target_id TEXT, label TEXT, properties TEXT
);
CREATE TABLE graph_node_owners (node_id TEXT, agent_id TEXT);
CREATE TABLE graph_edge_owners (
    source_id TEXT, target_id TEXT, label TEXT, agent_id TEXT
);
CREATE TABLE schema_backfills (
    name TEXT PRIMARY KEY, completed_at TIMESTAMP
);
"""


def _graph_db(
    path,
    *,
    nodes=(),
    edges=(),
    node_owners=(),
    edge_owners=(),
    ownership_settled=True,
):
    """A Kestrel graph with the ownership ledgers a real write maintains.

    Seeding rows without their witnesses is what an earlier version of these
    tests did, and it made every reader look correct while the bound runtime
    saw nothing — so the ledgers are part of the schema here, and a test that
    wants an unwitnessed row has to say so.
    """
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_GRAPH_SCHEMA)
        conn.executemany("INSERT INTO graph_nodes VALUES (?, ?, ?, ?)", nodes)
        conn.executemany("INSERT INTO graph_edges VALUES (?, ?, ?, NULL)", edges)
        conn.executemany("INSERT INTO graph_node_owners VALUES (?, ?)", node_owners)
        conn.executemany(
            "INSERT INTO graph_edge_owners VALUES (?, ?, ?, ?)", edge_owners
        )
        if ownership_settled:
            # A database the runtime has opened has #2649 recorded. Without the
            # marker a missing witness is *pending*, not permanent — boot
            # backfills it — so tests about permanence must say which they mean.
            conn.execute("INSERT INTO schema_backfills VALUES ('ownership_2649', NULL)")
        conn.commit()
    return path


def test_read_agent_node_returns_id_and_properties(tmp_path):
    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc123"}))],
        node_owners=[("did:x", "did:x")],
    )
    node_id, _label, properties = _read_agent_node(_anchor(db, "did:x"))
    assert node_id == "did:x"
    assert properties == {"constitution_hash": "abc123"}


def test_read_agent_node_none_properties_on_corrupt_json(tmp_path):
    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", "{not valid json")],
        node_owners=[("did:x", "did:x")],
    )
    node_id, _label, properties = _read_agent_node(_anchor(db, "did:x"))
    assert node_id == "did:x"
    assert properties is None


def test_read_agent_node_sentinels(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    assert isinstance(_read_agent_node(_anchor(empty, "did:x")), _UnreadableDB)

    no_agent = _graph_db(tmp_path / "noagent.db")
    assert isinstance(_read_agent_node(_anchor(no_agent, "did:x")), _NoAgentNode)


def test_an_agent_node_without_its_ownership_witness_is_invisible(tmp_path):
    """The runtime's bound store matches on the ledger, not on ``node_id``.

    ``AsyncGraphStore._node_scope`` requires a ``graph_node_owners`` row for
    the bound DID, and the boot integrity audit reads through it
    (``storage.get_node``). A raw row without that witness is a row the agent
    cannot see, so doctor must not see it either — otherwise it certifies an
    agent healthy and the agent safe-modes at its next boot.
    """
    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc"}))],
        node_owners=[],  # the row exists; nobody witnesses owning it
    )
    assert isinstance(_read_agent_node(_anchor(db, "did:x")), _NoAgentNode)


def test_an_agent_node_owned_by_someone_else_is_invisible(tmp_path):
    """One PostgreSQL holds every local agent; a neighbour's witness is not
    this agent's capability."""
    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc"}))],
        node_owners=[("did:x", "did:neighbour")],
    )
    assert isinstance(_read_agent_node(_anchor(db, "did:x")), _NoAgentNode)


def test_read_governed_by_targets_filters_by_source_and_label(tmp_path):
    db = _graph_db(
        tmp_path / "k.db",
        edges=[
            ("did:x", "hash-a", "governed_by"),
            ("did:x", "hash-b", "knows"),
            ("did:other", "hash-c", "governed_by"),
        ],
        edge_owners=[
            ("did:x", "hash-a", "governed_by", "did:x"),
            ("did:x", "hash-b", "knows", "did:x"),
            ("did:other", "hash-c", "governed_by", "did:other"),
        ],
    )
    assert _read_governed_by_targets(_anchor(db, "did:x"), "did:x") == ("hash-a",)


def test_a_governed_by_edge_without_its_witness_is_invisible(tmp_path):
    """The proof-2 counterpart: ``_edge_scope`` joins ``graph_edge_owners``,
    and the audit reads ``storage.get_edges_from``. An unwitnessed edge cannot
    satisfy proof 2, so reporting it as satisfying proof 2 is the failure
    mode — doctor says Ready, boot safe-modes."""
    db = _graph_db(
        tmp_path / "k.db",
        edges=[("did:x", "hash-a", "governed_by")],
        edge_owners=[],
    )
    assert _read_governed_by_targets(_anchor(db, "did:x"), "did:x") == ()


def test_read_governed_by_targets_unreadable_without_table(tmp_path):
    db = tmp_path / "k.db"
    sqlite3.connect(str(db)).close()
    assert isinstance(
        _read_governed_by_targets(_anchor(db, "did:x"), "did:x"), _UnreadableDB
    )


# ---------------------------------------------------------------------------
# A database from before the ownership migration (#2649) has no ledgers at all.
# Doctor is *meant* to run before boot (#2616), which is exactly when it meets
# one — and boot is what creates and backfills them.
# ---------------------------------------------------------------------------


def _legacy_graph_db(path, *, nodes=(), edges=()):
    """A pre-#2649 database: graph tables, no ownership ledgers."""
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT, node_type TEXT, label TEXT, properties TEXT
            );
            CREATE TABLE graph_edges (
                source_id TEXT, target_id TEXT, label TEXT, properties TEXT
            );
            """
        )
        conn.executemany("INSERT INTO graph_nodes VALUES (?, ?, ?, ?)", nodes)
        conn.executemany("INSERT INTO graph_edges VALUES (?, ?, ?, NULL)", edges)
        conn.commit()
    return path


def test_a_pre_migration_database_is_still_read(tmp_path, monkeypatch):
    """Absent ledger ≠ unwitnessed row.

    Scoping unconditionally made the query raise ``no such table``, which
    became a warning, which skipped the hash and edge checks — and warnings
    leave ``ready`` true, so doctor certified governance it had not looked at.
    Every row in a per-agent file belongs to that agent; the backfill at boot
    is what will shortly say so.
    """
    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _legacy_graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc"}))],
        edges=[("did:x", "abc", "governed_by")],
    )
    source = _resolve_governance_source(db, {}, tmp_path)

    assert source.ownership_ledger is False
    node_id, _label, properties = _read_agent_node(source)
    assert node_id == "did:x"
    assert properties == {"constitution_hash": "abc"}
    assert _read_governed_by_targets(source, "did:x") == ("abc",)


def test_a_migrated_database_is_detected_as_such(tmp_path):
    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", "{}")],
        node_owners=[("did:x", "did:x")],
    )
    assert _resolve_governance_source(db, {}, tmp_path).ownership_ledger is True


# ---------------------------------------------------------------------------
# Driver errors reach report.warn, which operators read on a terminal and CI
# archives. libpq echoes the DSN it could not parse.
# ---------------------------------------------------------------------------


def test_an_unreachable_postgres_is_not_ready(tmp_path, monkeypatch):
    """ "I did not check" must not print as "Ready".

    ``ready`` is ``not report.fail``, so warning here made ``kestrel doctor``
    exit 0 and print Ready having inspected no governance at all — on a host
    whose database is down, misconfigured, or refusing this account, which is
    the state an operator most needs told about. The runtime cannot reach that
    database either, so readiness really is false.
    """
    _seed_matching_anchor(tmp_path, monkeypatch)

    class _Unreachable:
        extensions = _real_psycopg2_extensions()

        def connect(self, dsn, **kwargs):
            raise OSError("connection timed out")

    _postgres_host(monkeypatch, _Unreachable())
    # The fake intentionally exposes no ``psycopg2.Error``. Ambient libpq-only
    # variables must not make the capability probe raise before ``connect``
    # reports the database's actual outage.
    monkeypatch.setenv("PGAPPNAME", "ambient-libpq-only-name")

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    assert any("governance NOT verified" in m for m in report.fail), report.fail
    assert any("connection timed out" in m for m in report.fail), report.fail
    assert any("equivalent libpq connection failed" in m for m in report.fail)
    assert not any(
        "Runtime database reachability was not established" in m for m in report.fail
    )
    assert not any(
        "cannot represent runtime connection option" in m for m in report.fail
    ), report.fail
    # Not phrased as drift: the remedy is access, not a reanchor.
    assert not any("reanchor" in m for m in report.fail)


def test_an_unreadable_sqlite_anchor_still_only_warns(tmp_path, monkeypatch):
    """``KESTREL_DB_KEY`` at inception makes a whole-DB sqlcipher file.

    Stock ``sqlite3`` cannot open it and the agent reads it perfectly well — a
    supported configuration in which doctor alone is blind. Failing would mark
    every sqlcipher host permanently not-ready for a problem that is not there.
    """
    _seed_ready(tmp_path)
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"SQLite format 3\x00" + b"\xde\xad\xbe\xef" * 64)

    report = diagnose(tmp_path)

    assert any("drift check skipped" in m for m in report.warn), report.warn
    assert not any("governance NOT verified" in m for m in report.fail)


def test_one_probe_serves_a_whole_postgres_fleet(tmp_path, monkeypatch):
    """Every local agent shares the database; the schema question is about it.

    Probing per agent meant a black-holed endpoint cost the connection timeout
    once per agent — a ten-agent fleet waiting fifty seconds under a
    five-second bound.
    """
    _seed_matching_anchor(tmp_path, monkeypatch)
    # A second agent pointed at the same anchor directory: one shared DSN.
    multi_agent_path = tmp_path / MULTI_AGENT_CONFIG_FILENAME
    raw = toml.load(multi_agent_path)
    raw["agents"]["Second"] = dict(raw["agents"]["Test"], port=8802)
    multi_agent_path.write_text(toml.dumps(raw))

    attempts = []

    class _Unreachable:
        extensions = _real_psycopg2_extensions()

        def connect(self, dsn, **kwargs):
            attempts.append(dsn)
            raise OSError("connection timed out")

    _postgres_host(monkeypatch, _Unreachable())

    diagnose(tmp_path)

    assert len(attempts) == 1, f"probed {len(attempts)}× for one shared DSN"


def test_an_anchor_with_two_agent_roots_is_refused(tmp_path):
    """The refusal boot already makes.

    ``identity.local_anchor.read_anchor_agent_did`` rejects an anchor holding
    more than one agent root rather than choosing by row order, and boot goes
    through it. Choosing here would let doctor scope its checks to an arbitrary
    tenant, find that one healthy, and report Ready for an agent the runtime
    will not start.
    """
    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _graph_db(
        tmp_path / "k.db",
        nodes=[
            ("did:one", "agent", "One", "{}"),
            ("did:two", "agent", "Two", "{}"),
        ],
        node_owners=[("did:one", "did:one"), ("did:two", "did:two")],
    )

    result = _resolve_governance_source(db, {}, tmp_path)

    assert isinstance(result, _UnreadableDB)
    assert "more than one agent root" in result.reason


def test_an_unreachable_database_is_probed_once(tmp_path, monkeypatch):
    """The ledger probe must not spend a connection timeout and throw it away.

    It returned ``True`` on failure, so the node read opened the same DSN and
    waited again — putting back, one function earlier, the doubled timeout that
    sharing a per-agent reading had just removed.
    """
    from kestrel_sovereign.doctor import _resolve_governance_source

    _seed_matching_anchor(tmp_path, monkeypatch)
    attempts = []

    class _Unreachable:
        extensions = _real_psycopg2_extensions()

        def connect(self, dsn, **kwargs):
            attempts.append(dsn)
            raise OSError("connection timed out")

    _postgres_host(monkeypatch, _Unreachable())
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"

    result = _resolve_governance_source(
        db_path,
        {
            "KESTREL_DB_BACKEND": "postgres",
            "KESTREL_DATABASE_URL": "postgresql://durable.example/kestrel",
        },
        tmp_path,
    )

    assert isinstance(result, _UnreadableDB)
    assert len(attempts) == 1, f"connected {len(attempts)}× : {attempts}"


def test_postgres_repairs_do_not_promise_a_backup(tmp_path, monkeypatch):
    """PostgreSQL reanchor takes no backup — there is no file to copy.

    Repeating the SQLite promise would send an operator to mutate live
    governance believing a rollback copy exists.
    """
    stored = _seed_matching_anchor(tmp_path, monkeypatch)
    drifted = hashlib.sha256(b"what postgres holds").hexdigest()
    properties = json.dumps({"name": "Test", "constitution_hash": drifted})
    fake = _FakePostgres(
        {
            # Three columns, as the real probe asks: graph schema, ledger,
            # then whether #2649 is recorded complete.
            "SELECT to_regclass": [(True, True, True)],
            "SELECT node_id, label, properties FROM graph_nodes": [
                ("did:test:Test", "Test", properties)
            ],
            "FROM graph_edges": [(drifted,)],
        }
    )
    _postgres_host(monkeypatch, fake)

    report = diagnose(tmp_path)

    drift = [m for m in report.fail if "constitution drift" in m]
    assert drift, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert "DB is backed up first" not in drift[0]
    assert "snapshot that database first" in drift[0]
    # PostgreSQL's hash is the anchored one; the canonical file's is what it
    # failed to match. Both belong in the message.
    assert drifted[:12] in drift[0]
    assert stored[:12] in drift[0]


def test_sqlite_repairs_still_promise_the_backup(tmp_path, monkeypatch):
    """It is true there: the reanchor copies the anchor aside before writing."""
    text = b"# Kestrel Constitution\nv1\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr("kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical))
    _seed_with_anchored_constitution(
        tmp_path, constitution_text=text, stored_hash="f" * 64
    )

    report = diagnose(tmp_path)

    drift = [m for m in report.fail if "constitution drift" in m]
    assert drift, f"fail={report.fail}"
    assert "DB is backed up first" in drift[0]


def test_a_driver_error_never_carries_the_dsn(tmp_path):
    """An unmatched ``[`` in the URI is enough to make libpq quote the whole
    connection string back, password included."""
    from kestrel_sovereign.doctor import _GovernanceSource, _read_agent_node

    dsn = "postgresql://kestrel:hunter2@[bad/kestrel"
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=dsn
    )

    result = _read_agent_node(source)

    assert isinstance(result, _UnreadableDB)
    assert "hunter2" not in result.reason
    assert dsn not in result.reason


@pytest.mark.parametrize(
    "message, leaked",
    [
        ('could not translate host name "db.internal" to address', "db.internal"),
        (
            'FATAL:  password authentication failed for user "kestrel_prod"',
            "kestrel_prod",
        ),
        ('FATAL:  database "governance_prod" does not exist', "governance_prod"),
    ],
)
def test_routine_failures_do_not_leak_the_database_estate(
    tmp_path,
    message,
    leaked,
):
    """The common case, and the one a whole-DSN replace never catches.

    libpq quotes the connection string back only for a *malformed* URI. Every
    ordinary DNS, authentication, or missing-database failure names the fields
    individually instead — so a redaction that handled only the verbatim echo
    left hostnames and account names in every routine outage message doctor
    prints to a terminal and CI archives.
    """
    from kestrel_sovereign.doctor import _GovernanceSource, _safe

    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db",
        agent_did="did:x",
        dsn=("postgresql://kestrel_prod:hunter2@db.internal:5432/governance_prod"),
    )

    redacted = _safe(message, source)

    assert leaked not in redacted
    # Still a usable message: the failure itself survives redaction.
    assert "FATAL" in redacted or "could not translate" in redacted


def test_redaction_leaves_short_values_alone(tmp_path):
    """A one- or two-character host or user is an ordinary English fragment.

    Replacing it would corrupt the very message redaction exists to keep
    readable.
    """
    from kestrel_sovereign.doctor import _GovernanceSource, _safe

    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db",
        agent_did="did:x",
        dsn="postgresql://a:p@db/x",
    )

    assert _safe("could not connect: a database is required", source) == (
        "could not connect: a database is required"
    )


def test_the_password_is_redacted_even_if_the_dsn_is_not_quoted_whole(tmp_path):
    """The whole-string replace only catches a byte-identical echo, and libpq
    is free to truncate or normalise what it quotes. The password is redacted
    on its own for that reason."""
    from kestrel_sovereign.doctor import _GovernanceSource, _safe

    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db",
        agent_did="did:x",
        dsn="postgresql://kestrel:hunter2@db.internal:5432/kestrel",
    )

    truncated = 'invalid dsn: ... in URI: "postgresql://kestrel:hunter2@db.int'
    assert "hunter2" not in _safe(truncated, source)


def test_doctor_accepts_openrouter_management_key_only(tmp_path):
    """#2245: a management-key-only OpenRouter route (api_key_env unset in .env,
    but management_api_key_env present) must NOT be flagged — doctor must agree
    with `kestrel setup --check`, which now accepts this shape."""
    _seed_ready(tmp_path)
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {
                "route_priority": ["openrouter:api"],
                "vendors": {
                    "openrouter": {
                        "routes": {
                            "api": {
                                "adapter": "OpenRouterAdapter",
                                "api_key_env": "OPENROUTER_API_KEY",
                                "management_api_key_env": "OPENROUTER_MANAGEMENT_API_KEY",
                                "model": "auto",
                            }
                        }
                    }
                },
            }
        },
        deep_merge=False,
    )
    # Only the management key is present in .env (no OPENROUTER_API_KEY).
    p = tmp_path / ".env"
    p.write_text(p.read_text() + "\nOPENROUTER_MANAGEMENT_API_KEY=sk-or-mgmt-xyz\n")
    report = diagnose(tmp_path)
    assert report.ready, f"fail={report.fail}"
    assert not any("OPENROUTER_API_KEY" in m for m in report.fail)


def test_the_backend_rule_is_the_runtimes_rule():
    """Copied from ``agent_manager._initialize_agent``, and copied *exactly*.

    `KESTREL_DB_BACKEND=postgres` with no DSN is a host whose agents really do
    run on SQLite, so treating it as PostgreSQL would send doctor to a database
    the agent never opens — the same defect with the two exchanged.
    """
    from kestrel_sovereign.doctor import _anchor_is_the_runtime_database

    assert _anchor_is_the_runtime_database({}) is True
    assert (
        _anchor_is_the_runtime_database({"KESTREL_DB_BACKEND": "postgres"}) is True
    ), "no DSN -> the runtime is SQLite"
    assert (
        _anchor_is_the_runtime_database(
            {
                "KESTREL_DB_BACKEND": "postgres",
                "KESTREL_DATABASE_URL": "postgresql://h/db",
            }
        )
        is False
    )


def test_the_database_settings_are_read_from_the_project_env(tmp_path, monkeypatch):
    """The standard install keeps them in the project ``.env``, nowhere else.

    ``kestrel doctor`` and ``setup --check`` call ``diagnose(project_dir)``
    without loading that file, so a doctor that consulted only ``os.environ``
    saw an unset backend, concluded SQLite, and went on reporting the birth
    record as current governance — on precisely the hosts #2892 is about. The
    fix would have been invisible in production while every test that exported
    the variables stayed green.
    """
    from kestrel_sovereign.doctor import runtime_env

    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "KESTREL_DB_BACKEND=postgres\n"
        "KESTREL_DATABASE_URL=postgresql://durable.example/kestrel\n"
    )

    resolved = runtime_env(tmp_path)

    assert resolved["KESTREL_DB_BACKEND"] == "postgres"
    assert resolved["KESTREL_DATABASE_URL"] == "postgresql://durable.example/kestrel"


def test_the_project_env_outranks_an_exported_setting(tmp_path, monkeypatch):
    """The launcher's precedence, which is what the agents actually get.

    ``ProcessManager._load_env`` copies ``os.environ`` and then lets ``.env``
    overwrite it, so the file wins. This is the **opposite** of
    ``paths.load_project_env``'s ``setdefault``, and an earlier version of this
    test asserted that other direction — which would have doctor inspect the
    exported database while the agents it is diagnosing open the file's one.
    Diagnosing the wrong database is this issue's whole defect; getting there
    by copying the wrong precedence would just be a longer route to it.
    """
    from kestrel_sovereign.doctor import runtime_env

    (tmp_path / ".env").write_text("KESTREL_DATABASE_URL=postgresql://from-file/db\n")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://exported/db")

    assert runtime_env(tmp_path)["KESTREL_DATABASE_URL"] == "postgresql://from-file/db"


def test_doctor_and_the_launcher_resolve_identically(tmp_path, monkeypatch):
    """Not "the same rule" — the same function.

    Two copies of a precedence is how they drift, and a drift here means
    doctor certifies one database while ``kestrel start`` opens another.
    """
    from kestrel_sovereign.doctor import runtime_env
    from kestrel_sovereign.multi_agent.process_manager import ProcessManager

    (tmp_path / ".env").write_text(
        "KESTREL_DB_BACKEND=postgres\nKESTREL_DATABASE_URL=postgresql://from-file/db\n"
    )
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://exported/db")

    launcher = ProcessManager.__new__(ProcessManager)
    launcher.project_dir = tmp_path

    assert runtime_env(tmp_path) == launcher._load_env()


def test_diagnose_does_not_export_the_project_env(tmp_path, monkeypatch):
    """A diagnostic reports on the process; it does not become it.

    Loading ``.env`` into ``os.environ`` would leave every later command in the
    same process — ``setup``'s remaining steps, a test, an embedding CLI —
    running under settings the operator never exported.
    """
    _seed_ready(tmp_path)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    (tmp_path / ".env").write_text(
        (tmp_path / ".env").read_text() + "\nKESTREL_DB_BACKEND=postgres\n"
    )

    diagnose(tmp_path)

    assert "KESTREL_DB_BACKEND" not in os.environ


def test_asyncpg_environment_settings_are_folded_into_the_libpq_dsn(tmp_path):
    """libpq cannot be pointed at a copy of the launcher's environment.

    The effective values therefore have to become explicit connection-string
    parameters, where they outrank doctor's unrelated process environment.
    """
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    root_certificate = tmp_path / "ca.pem"
    root_certificate.write_text("test root certificate")
    effective = _doctor_postgres_dsn(
        "postgresql:///kestrel",
        {
            "PGHOST": "db.internal",
            "PGPORT": "6543",
            "PGUSER": "project_user",
            "PGPASSWORD": "project-password",
            "PGSSLMODE": "verify-full",
            "PGSSLROOTCERT": str(root_certificate),
        },
        tmp_path,
    )

    assert parse_dsn(effective) == {
        "dbname": "kestrel",
        "host": "db.internal",
        "options": "",
        "port": "6543",
        "user": "project_user",
        "password": "project-password",
        "sslmode": "verify-full",
        "sslrootcert": str(root_certificate),
        "target_session_attrs": "any",
        "gssencmode": "disable",
        "channel_binding": "disable",
        "connect_timeout": "5",
    }


def test_explicit_dsn_connection_parameters_outrank_the_environment(tmp_path):
    """Only absent settings may be filled from the spawned-agent env."""
    from urllib.parse import quote

    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
    )

    dsn_root_certificate = tmp_path / "dsn-ca.pem"
    dsn_root_certificate.write_text("test DSN root certificate")
    effective = _doctor_postgres_dsn(
        "postgresql://dsn_user:dsn-password@dsn.example:6543/dsn_db"
        "?sslmode=require&sslrootcert=" + quote(str(dsn_root_certificate), safe=""),
        {
            "PGHOST": "env.example",
            "PGPORT": "7777",
            "PGUSER": "env_user",
            "PGPASSWORD": "env-password",
            "PGDATABASE": "env_db",
            "PGSSLMODE": "verify-full",
            "PGSSLROOTCERT": "/env/ca.pem",
        },
        tmp_path,
    )
    parsed = parse_dsn(effective)

    assert parsed["host"] == "dsn.example"
    assert parsed["port"] == "6543"
    assert parsed["user"] == "dsn_user"
    assert parsed["password"] == "dsn-password"
    assert parsed["dbname"] == "dsn_db"
    assert parsed["sslmode"] == "require"
    assert parsed["sslrootcert"] == str(dsn_root_certificate)


def test_query_port_is_ignored_when_the_authority_names_a_host(tmp_path):
    """asyncpg resolves PGPORT while parsing an authority host.

    Its later query-port branch cannot replace that already-truthy value, so
    doctor must not let libpq dial the query parameter's different server.
    """
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        "postgresql://h/db?port=6543",
        {"PGPORT": "5433"},
        tmp_path,
    )

    parsed = parse_dsn(effective)
    assert parsed["host"] == "h"
    assert parsed["port"] == "5433"


def test_scheme_relative_dsn_preserves_the_database_name(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        "postgresql:db",
        {"USER": "runtime_user"},
        tmp_path,
    )

    assert parse_dsn(effective)["dbname"] == "db"


def test_queryless_uri_uses_the_guarded_asyncpg_parse_path(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    parsed = parse_dsn(_doctor_postgres_dsn("postgresql://u@h/db", {}, tmp_path))

    assert parsed["user"] == "u"
    assert parsed["dbname"] == "db"


@pytest.mark.parametrize(
    ("runtime_environment", "expected_user"),
    [
        ({"PGUSER": "postgres_runtime_user"}, "postgres_runtime_user"),
        ({"USER": "login_runtime_user"}, "login_runtime_user"),
    ],
    ids=("pguser", "login-name"),
)
def test_empty_query_user_falls_back_exactly_like_asyncpg(
    tmp_path, monkeypatch, runtime_environment, expected_user
):
    from asyncpg.connect_utils import _parse_connect_dsn_and_args
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    for name in ("PGUSER", "LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in runtime_environment.items():
        monkeypatch.setenv(name, value)
    _, asyncpg_params = _parse_connect_dsn_and_args(
        dsn="postgresql://h/db?user=",
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
    translated = parse_dsn(
        _doctor_postgres_dsn("postgresql://h/db?user=", runtime_environment, tmp_path)
    )

    assert asyncpg_params.user == expected_user
    assert translated["user"] == asyncpg_params.user


def test_blanked_spawned_login_names_do_not_reuse_the_parent_login(
    tmp_path, monkeypatch
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(name, "doctor_parent_user")
    monkeypatch.setattr(
        "kestrel_sovereign.doctor._system_account_user",
        lambda: "spawned_os_user",
    )
    spawned_env = {name: "" for name in ("LOGNAME", "USER", "LNAME", "USERNAME")}

    parsed = parse_dsn(_doctor_postgres_dsn("postgresql://h", spawned_env, tmp_path))

    assert parsed["user"] == "spawned_os_user"
    assert parsed["dbname"] == "spawned_os_user"


def test_windows_asyncpg_tls_defaults_are_frozen_from_userprofile(
    tmp_path, monkeypatch
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    monkeypatch.setattr("kestrel_sovereign.doctor._IS_WINDOWS", True)
    runtime_home = tmp_path / "runtime-home"
    tls_dir = runtime_home / ".postgresql"
    tls_dir.mkdir(parents=True)
    for filename in (
        "root.crt",
        "root.crl",
        "postgresql.crt",
        "postgresql.key",
    ):
        (tls_dir / filename).write_text(filename)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=verify-full",
            {
                "USERPROFILE": str(runtime_home),
                "PGPASSWORD": "",
                "PGSSLROOTCERT": "",
                "PGSSLCRL": "",
                "PGSSLCERT": "",
                "PGSSLKEY": "",
            },
            tmp_path,
        )
    )

    assert parsed["sslrootcert"] == str((tls_dir / "root.crt").resolve())
    assert parsed["sslcrl"] == str((tls_dir / "root.crl").resolve())
    assert parsed["sslcert"] == str((tls_dir / "postgresql.crt").resolve())
    assert parsed["sslkey"] == str((tls_dir / "postgresql.key").resolve())


def test_windows_missing_asyncpg_root_remains_explicit_for_verification(
    tmp_path, monkeypatch
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    monkeypatch.setattr("kestrel_sovereign.doctor._IS_WINDOWS", True)
    runtime_home = tmp_path / "runtime-home"
    runtime_home.mkdir()

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=verify-ca",
            {"USERPROFILE": str(runtime_home), "PGPASSWORD": ""},
            tmp_path,
        )
    )

    assert parsed["sslrootcert"] == str(
        (runtime_home / ".postgresql" / "root.crt").resolve()
    )


def test_bare_slash_preserves_asyncpg_empty_database(tmp_path):
    """Libpq must not replace asyncpg's explicit empty path from PGDATABASE."""
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h/",
            {"PGDATABASE": "must_not_be_selected"},
            tmp_path,
        )
    )

    assert parsed["dbname"] == ""


@pytest.mark.parametrize(
    ("pg_host", "pg_port", "expected_host", "expected_port"),
    [
        ("db.internal:6543", None, "db.internal", "6543"),
        (
            "db1:5433,db2,[::1]:5435,/tmp",
            "6543",
            "db1,db2,::1,/tmp",
            "5433,6543,5435,6543",
        ),
    ],
)
def test_pghost_uses_asyncpg_host_list_rules(
    tmp_path, pg_host, pg_port, expected_host, expected_port
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    env = {"PGHOST": pg_host, "USER": "runtime_user"}
    if pg_port is not None:
        env["PGPORT"] = pg_port

    parsed = parse_dsn(_doctor_postgres_dsn("postgresql:///db", env, tmp_path))

    assert parsed["host"] == expected_host
    assert parsed["port"] == expected_port


@pytest.mark.parametrize(
    ("dsn", "environment"),
    [
        ("postgresql://u@%40abstract/db", {}),
        ("postgresql:///db?user=u&host=@abstract", {}),
        ("postgresql:///db?user=u", {"PGHOST": "@abstract"}),
        ("postgresql:///db?user=u&host=h1,@abstract", {}),
    ],
    ids=("authority", "query", "environment", "host-list"),
)
def test_libpq_reserved_abstract_socket_hosts_fail_closed(tmp_path, dsn, environment):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match="reserves for Unix sockets") as exc:
        _doctor_postgres_dsn(dsn, environment, tmp_path)

    assert "@abstract" not in str(exc.value)


def test_asyncpg_treats_at_prefixed_hosts_as_tcp():
    from asyncpg.connect_utils import _parse_connect_dsn_and_args

    addresses, _ = _parse_connect_dsn_and_args(
        dsn="postgresql://u@%40abstract/db",
        host=None,
        port=None,
        user=None,
        password=None,
        passfile=None,
        database=None,
        ssl="disable",
        direct_tls=None,
        server_settings=None,
        target_session_attrs=None,
        krbsrvname=None,
        gsslib=None,
    )

    assert addresses == [("@abstract", 5432)]


def test_empty_pghost_uses_asyncpg_defaults(tmp_path):
    import sys

    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql:///db",
            {"PGHOST": "", "USER": "runtime_user"},
            tmp_path,
        )
    )

    expected_hosts = (
        ["localhost"]
        if sys.platform == "win32"
        else [
            "/run/postgresql",
            "/var/run/postgresql",
            "/tmp",
            "/private/tmp",
            "localhost",
        ]
    )
    assert parsed["host"] == ",".join(expected_hosts)
    assert parsed["port"] == ",".join("5432" for _ in expected_hosts)


@pytest.mark.parametrize(
    ("dsn", "pg_port", "expected_host", "expected_port"),
    [
        (
            "postgresql://u@h1:5433,h2/db",
            "6543",
            "h1,h2",
            "5433,6543",
        ),
        (
            "postgresql://u@[::1]:5433,%2Ftmp,h3/db",
            "6543",
            "::1,/tmp,h3",
            "5433,6543,6543",
        ),
        (
            "postgresql://u@h1:5433,h2/db",
            "6001,6002",
            "h1,h2",
            "5433,6002",
        ),
    ],
)
def test_pgport_fills_each_authority_host_without_a_port(
    tmp_path, dsn, pg_port, expected_host, expected_port
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    parsed = parse_dsn(_doctor_postgres_dsn(dsn, {"PGPORT": pg_port}, tmp_path))

    assert parsed["host"] == expected_host
    assert parsed["port"] == expected_port


def test_connection_files_are_resolved_from_the_agent_working_directory(
    tmp_path,
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    dsn_dir = project_dir / "dsn"
    dsn_dir.mkdir()
    for filename in ("root.crt", "root.crl", "client.key", "client.crt"):
        (dsn_dir / filename).write_text(filename)
    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=require"
            "&passfile=dsn%2F.pgpass"
            "&sslrootcert=dsn%2Froot.crt"
            "&sslcrl=dsn%2Froot.crl"
            "&sslkey=dsn%2Fclient.key"
            "&sslcert=dsn%2Fclient.crt",
            {
                "PGPASSFILE": "env/.pgpass",
                "PGSSLROOTCERT": "env/root.crt",
                "PGSSLCRL": "env/root.crl",
                "PGSSLKEY": "env/client.key",
                "PGSSLCERT": "env/client.crt",
            },
            project_dir,
        )
    )

    _assert_absent_doctor_passfile(parsed["passfile"], project_dir)
    for option, relative in {
        "sslrootcert": "dsn/root.crt",
        "sslcrl": "dsn/root.crl",
        "sslkey": "dsn/client.key",
        "sslcert": "dsn/client.crt",
    }.items():
        assert parsed[option] == str((project_dir / relative).resolve())


def test_sslrootcert_system_is_asyncpgs_relative_filename(tmp_path):
    """asyncpg 0.30 has no libpq ``system`` sentinel for this option."""
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    (tmp_path / "system").write_text("test root certificate")
    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=verify-full&sslrootcert=system",
            {},
            tmp_path,
        )
    )

    assert parsed["sslrootcert"] == str((tmp_path / "system").resolve())


def test_environment_connection_files_use_the_same_project_directory(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    env_dir = project_dir / "env"
    env_dir.mkdir()
    for filename in ("root.crt", "root.crl", "client.key", "client.crt"):
        (env_dir / filename).write_text(filename)
    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=require",
            {
                "PGPASSFILE": "env/.pgpass",
                "PGSSLROOTCERT": "env/root.crt",
                "PGSSLCRL": "env/root.crl",
                "PGSSLKEY": "env/client.key",
                "PGSSLCERT": "env/client.crt",
            },
            project_dir,
        )
    )

    _assert_absent_doctor_passfile(parsed["passfile"], project_dir)
    for option, relative in {
        "sslrootcert": "env/root.crt",
        "sslcrl": "env/root.crl",
        "sslkey": "env/client.key",
        "sslcert": "env/client.crt",
    }.items():
        assert parsed[option] == str((project_dir / relative).resolve())


@pytest.mark.parametrize(
    ("sslmode", "query_option", "environment", "expected_option"),
    [
        ("allow", "sslcert=private%2Fmissing.crt", {}, "sslcert"),
        (
            "require",
            "sslrootcert=private%2Fmissing-root.crt",
            {},
            "sslrootcert",
        ),
        ("require", "", {"PGSSLCRL": "private/missing.crl"}, "sslcrl"),
    ],
)
def test_explicit_missing_tls_files_fail_closed_like_asyncpg(
    tmp_path, sslmode, query_option, environment, expected_option
):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    query = f"?sslmode={sslmode}"
    if query_option:
        query += f"&{query_option}"

    with pytest.raises(ValueError) as raised:
        _doctor_postgres_dsn(
            "postgresql://u@h/db" + query,
            environment,
            tmp_path,
        )

    message = str(raised.value)
    assert expected_option in message
    assert str(tmp_path) not in message


def test_explicit_missing_sslkey_requires_an_explicit_client_certificate(
    tmp_path,
):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    certificate = tmp_path / "private" / "client.crt"
    certificate.parent.mkdir()
    certificate.write_text("client certificate placeholder")

    with pytest.raises(ValueError) as raised:
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=prefer"
            "&sslcert=private%2Fclient.crt"
            "&sslkey=private%2Fmissing.key",
            {},
            tmp_path,
        )

    message = str(raised.value)
    assert "sslkey" in message
    assert str(tmp_path) not in message


def test_lone_environment_missing_sslkey_is_tolerated_like_asyncpg(
    tmp_path, monkeypatch
):
    from asyncpg.connect_utils import _parse_connect_dsn_and_args
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    monkeypatch.delenv("PGSSLCERT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PGSSLKEY", "private/missing.key")
    monkeypatch.setattr(
        "asyncpg.connect_utils._dot_postgresql_path",
        lambda filename: tmp_path / "missing-defaults" / filename,
    )
    runtime_dsn = "postgresql://u@h/db?sslmode=prefer"

    _, asyncpg_params = _parse_connect_dsn_and_args(
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
    translated = parse_dsn(
        _doctor_postgres_dsn(
            runtime_dsn,
            {"PGSSLKEY": "private/missing.key"},
            tmp_path,
        )
    )

    assert asyncpg_params.ssl is not None
    assert translated["sslkey"] == str((tmp_path / "private" / "missing.key").resolve())


def _assert_absent_doctor_passfile(value: str, project_dir: Path) -> None:
    from kestrel_sovereign.doctor import _ABSENT_PASSFILE_SENTINEL

    if value == _ABSENT_PASSFILE_SENTINEL:
        return
    materialized = Path(value)
    assert project_dir.resolve() not in materialized.parents
    assert not materialized.exists()
    assert not materialized.parent.exists()


@pytest.mark.parametrize(
    "row",
    [
        "*:*:*:*:probe:x",
        r"*:*:*:*:pa\:ss",
    ],
    ids=("unescaped-password-colon", "escaped-password-colon"),
)
def test_single_host_passfile_uses_asyncpg_password_dialect(tmp_path, row):
    from urllib.parse import quote

    from asyncpg.connect_utils import _read_password_from_pgpass
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    passfile = tmp_path / "runtime.pgpass"
    passfile.write_text(row + "\n")
    passfile.chmod(0o600)
    expected = _read_password_from_pgpass(
        passfile=passfile,
        hosts=["h1"],
        ports=[5432],
        database="db",
        user="u",
    )

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h1:5432/db?passfile=" + quote(str(passfile), safe=""),
            {},
            tmp_path,
        )
    )

    assert parsed["password"] == expected
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


@pytest.mark.parametrize(
    ("passfile_rows", "expected"),
    [
        (
            [
                "h1:5432:db:u:first-host-secret",
                "h2:5433:db:u:second-host-secret",
            ],
            "first-host-secret",
        ),
        (
            [
                "other:5432:db:u:not-a-match",
                "h2:5433:db:u:first-matching-secret",
            ],
            "first-matching-secret",
        ),
    ],
    ids=("first-host", "first-matching-host"),
)
def test_multi_host_passfile_uses_asyncpg_selection_once(
    tmp_path, passfile_rows, expected
):
    """Libpq must not select a new credential after host fallback."""
    from urllib.parse import quote

    from asyncpg.connect_utils import _read_password_from_pgpass
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    passfile = tmp_path / "runtime.pgpass"
    passfile.write_text("\n".join(passfile_rows) + "\n")
    passfile.chmod(0o600)
    asyncpg_password = _read_password_from_pgpass(
        passfile=passfile,
        hosts=["h1", "h2"],
        ports=[5432, 5433],
        database="db",
        user="u",
    )

    effective = _doctor_postgres_dsn(
        "postgresql://u@h1:5432,h2:5433/db?passfile=" + quote(str(passfile), safe=""),
        {},
        tmp_path,
    )
    parsed = parse_dsn(effective)

    assert asyncpg_password == expected
    assert parsed["password"] == asyncpg_password
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )
    assert expected not in _safe(f"password={expected}", source)


def test_multi_host_passfile_with_no_match_cannot_fall_through_to_libpq(
    tmp_path,
):
    from urllib.parse import quote

    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    passfile = tmp_path / "runtime.pgpass"
    passfile.write_text("somewhere-else:5432:db:u:wrong-secret\n")
    passfile.chmod(0o600)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h1:5432,h2:5433/db?passfile="
            + quote(str(passfile), safe=""),
            {},
            tmp_path,
        )
    )

    assert "password" not in parsed
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


def test_multi_host_pgpassfile_environment_is_selected_once(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    passfile = tmp_path / "runtime.pgpass"
    passfile.write_text(
        "h1:5432:db:u:environment-first-secret\n"
        "h2:5433:db:u:environment-second-secret\n"
    )
    passfile.chmod(0o600)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h1:5432,h2:5433/db",
            {"PGPASSFILE": str(passfile)},
            tmp_path,
        )
    )

    assert parsed["password"] == "environment-first-secret"
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


def test_multi_host_default_pgpass_uses_the_spawned_home(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    runtime_home = tmp_path / "spawned-home"
    runtime_home.mkdir()
    passfile = runtime_home / ".pgpass"
    passfile.write_text(
        "h1:5432:db:u:default-first-secret\nh2:5433:db:u:default-second-secret\n"
    )
    passfile.chmod(0o600)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h1:5432,h2:5433/db",
            {"HOME": str(runtime_home)},
            tmp_path,
        )
    )

    assert parsed["password"] == "default-first-secret"
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


@pytest.mark.parametrize(
    "error",
    [
        ImportError("asyncpg compat unavailable"),
        OSError("Windows known-folder lookup failed"),
        AttributeError("ctypes Windows API unavailable"),
    ],
    ids=("import", "known-folder", "ctypes-api"),
)
def test_windows_passfile_discovery_failure_is_an_unreadable_finding(
    tmp_path, monkeypatch, error
):
    from asyncpg import compat as asyncpg_compat

    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", "{}")],
        node_owners=[("did:x", "did:x")],
    )
    monkeypatch.setattr("kestrel_sovereign.doctor._IS_WINDOWS", True)

    def fail_to_find_pg_home():
        raise error

    monkeypatch.setattr(asyncpg_compat, "get_pg_home_directory", fail_to_find_pg_home)

    result = _resolve_governance_source(
        db,
        {
            "KESTREL_DB_BACKEND": "postgres",
            "KESTREL_DATABASE_URL": "postgresql://u@h/db",
        },
        tmp_path,
    )

    assert isinstance(result, _UnreadableDB)
    assert "default passfile location could not be determined" in result.reason


def test_future_classified_translation_failure_is_contained(tmp_path, monkeypatch):
    from kestrel_sovereign import doctor as doctor_module

    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", "{}")],
        node_owners=[("did:x", "did:x")],
    )

    def fail_translation(*_args, **_kwargs):
        raise doctor_module._PostgresTranslationError(
            "future classified translation failure"
        )

    monkeypatch.setattr(doctor_module, "_doctor_postgres_dsn", fail_translation)

    result = doctor_module._resolve_governance_source(
        db,
        {
            "KESTREL_DB_BACKEND": "postgres",
            "KESTREL_DATABASE_URL": "postgresql://u@h/db",
        },
        tmp_path,
    )

    assert isinstance(result, _UnreadableDB)
    assert result.postgres_failure == "diagnostic_capability"
    assert "cannot construct an equivalent libpq diagnostic connection" in result.reason


@pytest.mark.parametrize(
    "dsn",
    (
        "postgresql://u@h1:5432/db",
        "postgresql://u@h1:5432,h2:5433/db",
    ),
    ids=("single-host", "multi-host"),
)
def test_empty_pgpassfile_does_not_fall_through_to_default(tmp_path, dsn):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    runtime_home = tmp_path / "spawned-home"
    runtime_home.mkdir()
    passfile = runtime_home / ".pgpass"
    passfile.write_text("h1:5432:db:u:must-not-be-selected\n")
    passfile.chmod(0o600)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            dsn,
            {"HOME": str(runtime_home), "PGPASSFILE": ""},
            tmp_path,
        )
    )

    assert "password" not in parsed
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


def test_empty_pgpassword_suppresses_passfile_on_single_host(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    runtime_home = tmp_path / "spawned-home"
    runtime_home.mkdir()
    passfile = runtime_home / ".pgpass"
    passfile.write_text("h1:5432:db:u:must-not-be-selected\n")
    passfile.chmod(0o600)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@h1:5432/db",
            {"HOME": str(runtime_home), "PGPASSWORD": ""},
            tmp_path,
        )
    )

    assert parsed["password"] == ""
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX socket fallback")
def test_default_host_pgpass_uses_asyncpg_auth_host_and_first_port(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    runtime_home = tmp_path / "spawned-home"
    runtime_home.mkdir()
    passfile = runtime_home / ".pgpass"
    passfile.write_text("localhost:5433:db:u:wrong-second-port-secret\n")
    passfile.chmod(0o600)

    parsed = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql:///db?user=u",
            {
                "HOME": str(runtime_home),
                "PGPORT": "5432,5433,5434,5435,5436",
            },
            tmp_path,
        )
    )

    assert "password" not in parsed
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


@pytest.mark.parametrize(
    ("row", "user"),
    [
        (r"h1:5432:db:u:pa\\ss", "u"),
        (r"h1:5432:db:u\\x:only-secret", r"u\x"),
        ("h1:5432:db:u:pa\vss", "u"),
        ("malformed\nh1:5432:db:u:secret", "u"),
        ("h1:5432:db:u:secret\nmalformed", "u"),
    ],
    ids=(
        "password-backslash",
        "user-backslash",
        "vertical-tab",
        "malformed-before-match",
        "match-before-malformed",
    ),
)
def test_pgpass_file_parsing_matches_asyncpg_exactly(tmp_path, row, user):
    from asyncpg.connect_utils import _read_password_from_pgpass

    from kestrel_sovereign.doctor import _read_asyncpg_passfile_password

    passfile = tmp_path / "runtime.pgpass"
    passfile.write_text(row + "\n")
    passfile.chmod(0o600)

    def asyncpg_call():
        return _read_password_from_pgpass(
            passfile=passfile,
            hosts=["h1", "h2"],
            ports=[5432, 5433],
            database="db",
            user=user,
        )

    def doctor_call():
        return _read_asyncpg_passfile_password(
            passfile, ["h1", "h2"], [5432, 5433], "db", user
        )

    if row.startswith("malformed"):
        with pytest.raises(ValueError):
            asyncpg_call()
        with pytest.raises(ValueError):
            doctor_call()
    else:
        asyncpg_result = asyncpg_call()
        assert doctor_call() == asyncpg_result
        if row.endswith("\nmalformed"):
            # asyncpg returns as soon as the first entry matches; it never
            # destructures the later malformed tuple. Preserve that order.
            assert asyncpg_result == "secret"


def test_empty_options_neutralizes_libpq_only_pgoptions(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        "postgresql://u@h/db",
        {"PGOPTIONS": "-c search_path=leaked_schema"},
        tmp_path,
    )

    assert parse_dsn(effective)["options"] == ""


@pytest.mark.parametrize(
    ("dsn_name", "expected"),
    (("gssencmode", "disable"), ("channel_binding", "disable")),
)
def test_libpq_compiled_defaults_are_always_asyncpg_equivalent(
    dsn_name,
    expected,
    tmp_path,
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _libpq_accepts_dsn_option,
    )

    parsed = parse_dsn(_doctor_postgres_dsn("postgresql://u@h/db", {}, tmp_path))

    if _libpq_accepts_dsn_option(dsn_name, expected):
        assert parsed[dsn_name] == expected
    else:
        assert dsn_name not in parsed


def test_missing_postgres_driver_has_a_driver_diagnostic(tmp_path, monkeypatch):
    import sys

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    monkeypatch.setitem(sys.modules, "psycopg2", None)

    with pytest.raises(ValueError, match="psycopg2 is not installed"):
        _doctor_postgres_dsn("postgresql://u@h/db", {}, tmp_path)


def test_libpq_only_pgrequiressl_cannot_steer_host_ssl_default(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        "postgresql://u@h/db",
        {"PGREQUIRESSL": "1"},
        tmp_path,
    )

    assert parse_dsn(effective)["sslmode"] == "prefer"
    unix_socket = parse_dsn(
        _doctor_postgres_dsn(
            "postgresql://u@%2Ftmp/db",
            {"PGREQUIRESSL": "1"},
            tmp_path,
        )
    )
    assert unix_socket["sslmode"] == "disable"


def test_hostless_dsn_states_asyncpg_connection_defaults(monkeypatch, tmp_path):
    import sys

    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    monkeypatch.setattr(
        "getpass.getuser",
        lambda: pytest.fail("ignored the spawned runtime's USER"),
    )
    effective = _doctor_postgres_dsn(
        "postgresql://",
        {"USER": "runtime_user"},
        tmp_path,
    )
    parsed = parse_dsn(effective)

    expected_host = (
        "localhost"
        if sys.platform == "win32"
        else "/run/postgresql,/var/run/postgresql,/tmp,/private/tmp,localhost"
    )
    assert parsed["host"] == expected_host
    assert parsed["port"] == ",".join("5432" for _ in expected_host.split(","))
    assert parsed["user"] == "runtime_user"
    assert parsed["dbname"] == "runtime_user"
    assert parsed["sslmode"] == "prefer"
    assert parsed["target_session_attrs"] == "any"
    assert parsed["options"] == ""
    assert parsed["connect_timeout"] == ("5" if sys.platform == "win32" else "2")


def test_libpq_service_file_without_a_service_name_is_inert(tmp_path):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    baseline = _doctor_postgres_dsn(
        "postgresql:///kestrel",
        {"USER": "runtime_user"},
        tmp_path,
    )
    with_service_file = _doctor_postgres_dsn(
        "postgresql:///kestrel",
        {
            "PGSERVICEFILE": "/libpq/only/service.conf",
            "USER": "runtime_user",
        },
        tmp_path,
    )

    assert with_service_file == baseline


def test_compiled_neutralizer_unknown_to_linked_libpq_is_not_emitted(
    monkeypatch, tmp_path
):
    import psycopg2
    from psycopg2.extensions import make_dsn as real_make_dsn
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    def reject_newer_option(dsn=None, **kwargs):
        if "gssencmode" in kwargs:
            raise psycopg2.ProgrammingError("unsupported by this libpq")
        return real_make_dsn(dsn, **kwargs)

    monkeypatch.setattr("psycopg2.extensions.make_dsn", reject_newer_option)
    effective = _doctor_postgres_dsn(
        "postgresql://u@h/db",
        {},
        tmp_path,
    )

    parsed = parse_dsn(effective)
    assert "gssencmode" not in parsed
    assert parsed["channel_binding"] == "disable"


def test_asyncpg_environment_option_unknown_to_linked_libpq_fails_closed(
    monkeypatch,
    tmp_path,
):
    import psycopg2
    from psycopg2.extensions import make_dsn as real_make_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    def reject_newer_option(dsn=None, **kwargs):
        if "sslnegotiation" in kwargs:
            raise psycopg2.ProgrammingError("unsupported by this libpq")
        return real_make_dsn(dsn, **kwargs)

    monkeypatch.setattr("psycopg2.extensions.make_dsn", reject_newer_option)
    with pytest.raises(ValueError, match="sslnegotiation"):
        _doctor_postgres_dsn(
            "postgresql://u@h/db",
            {"PGSSLMODE": "require", "PGSSLNEGOTIATION": "direct"},
            tmp_path,
        )


def test_asyncpg_query_option_unknown_to_linked_libpq_fails_closed(
    monkeypatch,
    tmp_path,
):
    """A stated runtime constraint must never disappear on older libpq."""
    import psycopg2
    from psycopg2.extensions import make_dsn as real_make_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    def reject_newer_option(dsn=None, **kwargs):
        if "sslnegotiation" in kwargs:
            raise psycopg2.ProgrammingError("unsupported by this libpq")
        return real_make_dsn(dsn, **kwargs)

    monkeypatch.setattr("psycopg2.extensions.make_dsn", reject_newer_option)
    with pytest.raises(ValueError, match="sslnegotiation"):
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslmode=require&sslnegotiation=direct",
            {},
            tmp_path,
        )


def test_unsupported_secret_connection_option_fails_without_leaking(
    monkeypatch, tmp_path
):
    import psycopg2
    from psycopg2.extensions import make_dsn as real_make_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    def reject_sslpassword(dsn=None, **kwargs):
        if "sslpassword" in kwargs:
            raise psycopg2.ProgrammingError("unsupported by this libpq")
        return real_make_dsn(dsn, **kwargs)

    monkeypatch.setattr("psycopg2.extensions.make_dsn", reject_sslpassword)
    with pytest.raises(ValueError, match="sslpassword") as exc_info:
        _doctor_postgres_dsn(
            "postgresql://u@h/db?sslpassword=super-secret",
            {},
            tmp_path,
        )

    assert "super-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("env_name", "option_name"),
    [
        ("PGSSLMODE", "sslmode"),
        ("PGSSLNEGOTIATION", "sslnegotiation"),
        ("PGTARGETSESSIONATTRS", "target_session_attrs"),
        ("PGGSSLIB", "gsslib"),
    ],
)
def test_empty_enum_environment_option_is_rejected_like_asyncpg(
    tmp_path, env_name, option_name
):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match=option_name):
        _doctor_postgres_dsn(
            "postgresql://u@h/db",
            {env_name: ""},
            tmp_path,
        )


@pytest.mark.parametrize(
    ("asyncpg_spelling", "libpq_spelling", "source"),
    [
        ("verify_ca", "verify-ca", "query"),
        ("verify_full", "verify-full", "query"),
        ("verify_ca", "verify-ca", "environment"),
        ("verify_full", "verify-full", "environment"),
    ],
)
def test_asyncpg_sslmode_aliases_are_normalized_for_libpq(
    tmp_path, asyncpg_spelling, libpq_spelling, source
):
    from asyncpg.connect_utils import SSLMode
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    root_certificate = tmp_path / "root.crt"
    root_certificate.write_text("test root certificate")
    if source == "query":
        runtime_dsn = (
            f"postgresql://u@h/db?sslmode={asyncpg_spelling}&sslrootcert=root.crt"
        )
        environment = {}
    else:
        runtime_dsn = "postgresql://u@h/db"
        environment = {
            "PGSSLMODE": asyncpg_spelling,
            "PGSSLROOTCERT": str(root_certificate),
        }

    parsed = parse_dsn(_doctor_postgres_dsn(runtime_dsn, environment, tmp_path))

    assert SSLMode.parse(asyncpg_spelling).name == asyncpg_spelling
    assert parsed["sslmode"] == libpq_spelling


@pytest.mark.parametrize(
    ("asyncpg_minimum", "asyncpg_maximum", "libpq_minimum", "libpq_maximum"),
    [
        ("TLSv1_2", "TLSv1_3", "TLSv1.2", "TLSv1.3"),
        (
            "MINIMUM_SUPPORTED",
            "MAXIMUM_SUPPORTED",
            "TLSv1",
            "TLSv1.3",
        ),
    ],
    ids=("version-members", "symbolic-bounds"),
)
@pytest.mark.parametrize("source", ["query", "environment"])
def test_asyncpg_tls_protocol_aliases_are_normalized_for_libpq(
    tmp_path,
    source,
    asyncpg_minimum,
    asyncpg_maximum,
    libpq_minimum,
    libpq_maximum,
):
    import ssl

    from asyncpg.connect_utils import _parse_tls_version
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    if source == "query":
        runtime_dsn = (
            "postgresql://u@h/db?sslmode=require"
            f"&ssl_min_protocol_version={asyncpg_minimum}"
            f"&ssl_max_protocol_version={asyncpg_maximum}"
        )
        environment = {}
    else:
        runtime_dsn = "postgresql://u@h/db?sslmode=require"
        environment = {
            "PGSSLMINPROTOCOLVERSION": asyncpg_minimum,
            "PGSSLMAXPROTOCOLVERSION": asyncpg_maximum,
        }

    parsed = parse_dsn(_doctor_postgres_dsn(runtime_dsn, environment, tmp_path))

    assert _parse_tls_version(asyncpg_minimum) is ssl.TLSVersion[asyncpg_minimum]
    assert _parse_tls_version(asyncpg_maximum) is ssl.TLSVersion[asyncpg_maximum]
    assert parsed["ssl_min_protocol_version"] == libpq_minimum
    assert parsed["ssl_max_protocol_version"] == libpq_maximum


@pytest.mark.parametrize("source", ["query", "environment"])
def test_disabled_ssl_omits_tls_protocol_settings_asyncpg_ignores(
    tmp_path, monkeypatch, source
):
    from asyncpg.connect_utils import _parse_connect_dsn_and_args
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    tls_environment = {
        "PGSSLMODE": "disable",
        "PGSSLMINPROTOCOLVERSION": "not-a-tls-version",
        "PGSSLMAXPROTOCOLVERSION": "also-not-a-tls-version",
    }
    for name in tls_environment:
        monkeypatch.delenv(name, raising=False)
    if source == "query":
        runtime_dsn = (
            "postgresql://u@h/db?sslmode=disable"
            "&ssl_min_protocol_version=not-a-tls-version"
            "&ssl_max_protocol_version=also-not-a-tls-version"
        )
        environment = {}
    else:
        runtime_dsn = "postgresql://u@h/db"
        environment = tls_environment
        for name, value in tls_environment.items():
            monkeypatch.setenv(name, value)

    _, asyncpg_params = _parse_connect_dsn_and_args(
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
    parsed = parse_dsn(_doctor_postgres_dsn(runtime_dsn, environment, tmp_path))

    assert asyncpg_params.ssl is False
    assert "ssl_min_protocol_version" not in parsed
    assert "ssl_max_protocol_version" not in parsed


@pytest.mark.parametrize(
    ("dsn", "env", "sslmode", "sslnegotiation"),
    [
        (
            "postgresql://u@h/db?sslmode=verify-full&sslnegotiation=direct",
            {},
            "verify-full",
            "direct",
        ),
        (
            "postgresql://u@h/db",
            {"PGSSLMODE": "disable", "PGSSLNEGOTIATION": "direct"},
            "disable",
            "postgres",
        ),
        (
            "postgresql://u@%2Ftmp/db?sslnegotiation=direct",
            {},
            "disable",
            "postgres",
        ),
        (
            "postgresql://u@h/db?sslmode=disable&sslnegotiation=direct",
            {},
            "disable",
            "postgres",
        ),
        (
            "postgresql://u@h/db?sslnegotiation=direct",
            {"PGSSLMODE": "disable"},
            "disable",
            "postgres",
        ),
        (
            "postgresql://u@h/db?sslmode=disable",
            {"PGSSLNEGOTIATION": "direct"},
            "disable",
            "postgres",
        ),
    ],
    ids=(
        "query-verify-full",
        "environment-disable",
        "unix-socket-default-disable",
        "query-disable",
        "query-direct-environment-disable",
        "query-disable-environment-direct",
    ),
)
def test_direct_tls_is_normalized_to_a_valid_libpq_pair(
    dsn,
    env,
    sslmode,
    sslnegotiation,
    tmp_path,
):
    """asyncpg accepts weak/disabled modes that libpq rejects with direct."""
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    parsed = parse_dsn(_doctor_postgres_dsn(dsn, env, tmp_path))

    assert parsed["sslmode"] == sslmode
    assert parsed["sslnegotiation"] == sslnegotiation


@pytest.mark.parametrize(
    ("dsn", "env", "sslmode"),
    [
        ("postgresql://u@h/db", {"PGSSLNEGOTIATION": "direct"}, "prefer"),
        (
            "postgresql://u@h/db",
            {"PGSSLMODE": "prefer", "PGSSLNEGOTIATION": "direct"},
            "prefer",
        ),
        ("postgresql://u@h/db?sslnegotiation=direct", {}, "prefer"),
        (
            "postgresql://u@h/db?sslmode=prefer&sslnegotiation=direct",
            {},
            "prefer",
        ),
        (
            "postgresql://u@h/db?sslmode=allow&sslnegotiation=direct",
            {},
            "allow",
        ),
        (
            "postgresql://u@h/db",
            {"PGSSLMODE": "allow", "PGSSLNEGOTIATION": "direct"},
            "allow",
        ),
    ],
)
def test_weak_direct_tls_fails_closed_when_libpq_cannot_represent_it(
    dsn, env, sslmode, tmp_path
):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match=f"sslmode='{sslmode}'.*direct TLS"):
        _doctor_postgres_dsn(dsn, env, tmp_path)


@pytest.mark.parametrize(
    ("query", "server_option"),
    [
        ("search_path=tenant", "-c search_path=tenant"),
        ("connect_timeout=30", "-c connect_timeout=30"),
        ("keepalives=1", "-c keepalives=1"),
        ("application_name=a%09b", "-c application_name=a\\\tb"),
    ],
)
def test_libpq_only_query_names_remain_asyncpg_server_settings(
    query,
    server_option,
    tmp_path,
):
    """Classification follows asyncpg, even when libpq knows the name."""
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        f"postgresql://u@h/db?{query}",
        {},
        tmp_path,
    )
    parsed = parse_dsn(effective)

    assert parsed["options"] == server_option
    option_name = query.partition("=")[0]
    if option_name == "connect_timeout":
        assert parsed["connect_timeout"] == "5"
    else:
        assert option_name not in parsed
    assert "%20" in effective
    assert "+" not in effective


@pytest.mark.parametrize(
    "query",
    [
        "bad%3Dname=translated-secret",
        "%00name=translated-secret",
        "=translated-secret",
    ],
    ids=("equals", "nul", "empty"),
)
def test_unrepresentable_startup_setting_names_fail_closed(query, tmp_path):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match="startup setting name") as exc_info:
        _doctor_postgres_dsn(f"postgresql://u@h/db?{query}", {}, tmp_path)

    assert "translated-secret" not in str(exc_info.value)


def test_equals_in_startup_setting_value_is_preserved(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        "postgresql://u@h/db?search_path=tenant%3Dblue",
        {},
        tmp_path,
    )

    assert parse_dsn(effective)["options"] == "-c search_path=tenant=blue"


def test_nul_in_startup_setting_value_fails_closed(tmp_path):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match="startup setting value"):
        _doctor_postgres_dsn(
            "postgresql://u@h/db?search_path=tenant%00hidden",
            {},
            tmp_path,
        )


@pytest.mark.parametrize(
    "query",
    [
        "options=-c%20statement_timeout%3D1000&search_path=tenant",
        "search_path=tenant&options=-c%20statement_timeout%3D1000",
    ],
    ids=("options-first", "direct-setting-first"),
)
def test_asyncpg_options_precede_direct_startup_settings(query, tmp_path):
    """PostgreSQL applies raw options first, independent of URI order."""
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    effective = _doctor_postgres_dsn(
        f"postgresql://u@h/db?{query}",
        {},
        tmp_path,
    )

    assert parse_dsn(effective)["options"] == (
        "-c statement_timeout=1000 -c search_path=tenant"
    )


def test_odd_trailing_options_backslash_cannot_consume_direct_setting(tmp_path):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match="cannot be combined losslessly"):
        _doctor_postgres_dsn(
            "postgresql://u@h/db?options=-c%20work_mem%3D4MB%5C&search_path=tenant",
            {},
            tmp_path,
        )


def test_a_folded_environment_password_stays_inside_error_redaction(tmp_path):
    """Moving PGPASSWORD into a logged string must not move it into logs."""
    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    effective = _doctor_postgres_dsn(
        "postgresql://project_user@db.internal/kestrel",
        {"PGPASSWORD": "project-password"},
        tmp_path,
    )
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )

    redacted = _safe(
        f'could not connect with password "project-password": {effective}',
        source,
    )
    assert "project-password" not in redacted
    assert effective not in redacted


def test_a_one_character_environment_password_does_not_shred_diagnostics(tmp_path):
    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    effective = _doctor_postgres_dsn(
        "postgresql://project_user@db.internal/kestrel",
        {"PGPASSWORD": "p"},
        tmp_path,
    )
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )

    redacted = _safe(
        "connection failed on port 55432: password authentication failed; credential=p",
        source,
    )

    assert "port 55432" in redacted
    assert "password authentication failed" in redacted
    assert "credential=<redacted>" in redacted


def test_bounded_dsn_redacts_password_and_sslpassword(tmp_path):
    """Libpq can echo decoded secrets from its reserialized bounded DSN."""
    from kestrel_sovereign.doctor import (
        _bounded_dsn,
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    effective = _doctor_postgres_dsn(
        "postgresql://project_user@db.internal/kestrel"
        "?password=database-secret&sslpassword=private-key-secret",
        {},
        tmp_path,
    )
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )
    bounded = _bounded_dsn(effective)

    redacted = _safe(
        "connection rejected sslpassword=private-key-secret "
        f"password=database-secret dsn={bounded}",
        source,
    )

    assert "database-secret" not in redacted
    assert "private-key-secret" not in redacted
    assert "sslpassword=<redacted>" in redacted


def test_bounded_conninfo_escaped_secret_forms_are_redacted(tmp_path):
    from urllib.parse import quote

    from kestrel_sovereign.doctor import (
        _bounded_dsn,
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    password = "database-leak'with\\escapes"
    sslpassword = "private-key-leak'with\\escapes"
    effective = _doctor_postgres_dsn(
        "postgresql://project_user@db.internal/kestrel?password="
        + quote(password, safe="")
        + "&sslpassword="
        + quote(sslpassword, safe=""),
        {},
        tmp_path,
    )
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )

    redacted = _safe(f"connection failed: {_bounded_dsn(effective)}", source)

    assert "database-leak" not in redacted
    assert "private-key-leak" not in redacted


def test_malformed_uri_redacts_encoded_and_decoded_password(tmp_path):
    from kestrel_sovereign.doctor import _GovernanceSource, _safe

    dsn = "postgresql://user:encoded%20secret@[bad/db"
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=dsn
    )

    redacted = _safe(
        "invalid URI password=encoded%20secret decoded=encoded secret",
        source,
    )

    assert "encoded%20secret" not in redacted
    assert "encoded secret" not in redacted


def test_multi_host_failure_redacts_the_individual_failed_host(tmp_path):
    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    effective = _doctor_postgres_dsn(
        "postgresql://runtime_user@first.internal,second.internal/db",
        {"PGPASSWORD": "secret"},
        tmp_path,
    )
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )

    redacted = _safe(
        'could not translate host name "second.internal" to address', source
    )

    assert "second.internal" not in redacted
    assert "<host>" in redacted


def test_synthetic_fallback_hosts_do_not_mangle_driver_messages(tmp_path, monkeypatch):
    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _fetch_postgres_rows_isolated,
        _PostgresProbeConnectionError,
        _translated_dsn_identity,
    )

    runtime_dsn = "postgresql:///db"
    env = {"USER": "runtime_user"}
    effective = _doctor_postgres_dsn(runtime_dsn, env, tmp_path)
    message = "localhost could not read unrelated file /tmp/root.crt"

    class Process:
        returncode = 0

        def communicate(self, payload=None, timeout=None):
            return json.dumps({"ok": False, "kind": "connection", "error": message}), ""

    monkeypatch.setattr(
        "kestrel_sovereign.doctor.subprocess.Popen",
        lambda *_args, **_kwargs: Process(),
    )

    with pytest.raises(_PostgresProbeConnectionError) as raised:
        _fetch_postgres_rows_isolated(
            effective,
            "SELECT 1",
            dsn_identity=_translated_dsn_identity(runtime_dsn, effective, env),
        )

    assert str(raised.value) == message


def test_explicit_socket_host_remains_inside_error_redaction(tmp_path):
    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
        _translated_dsn_identity,
    )

    runtime_dsn = "postgresql://runtime_user@%2Ftmp/db"
    effective = _doctor_postgres_dsn(runtime_dsn, {}, tmp_path)
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db",
        agent_did="did:x",
        dsn=effective,
        dsn_identity=_translated_dsn_identity(runtime_dsn, effective, {}),
    )

    assert _safe("socket /tmp failed", source) == "socket <host> failed"


def test_resolved_connection_file_paths_stay_inside_error_redaction(tmp_path):
    from kestrel_sovereign.doctor import (
        _doctor_postgres_dsn,
        _GovernanceSource,
        _safe,
    )

    root_certificate = tmp_path / "private/root.crt"
    root_certificate.parent.mkdir()
    root_certificate.write_text("test root certificate")
    effective = _doctor_postgres_dsn(
        "postgresql://project_user@db.internal/kestrel?sslmode=verify-full",
        {"PGSSLROOTCERT": "private/root.crt"},
        tmp_path,
    )
    source = _GovernanceSource(
        anchor_path=tmp_path / "k.db", agent_did="did:x", dsn=effective
    )
    resolved_path = str(root_certificate.resolve())

    redacted = _safe(f'root certificate file "{resolved_path}" does not exist', source)

    assert resolved_path not in redacted
    assert "<sslrootcert>" in redacted


# ---------------------------------------------------------------------------
# On a PostgreSQL host the anchor is the birth record, not the governance
# (#2892). Doctor has to read the database the agent is actually governed by,
# or it reports birth-time state as current — permanently flagging drift after
# any legitimate reanchor, and prescribing a repair that correctly answers
# "nothing to do". Two governance tools contradicting each other is how
# operators learn to ignore the one that cries wolf.
# ---------------------------------------------------------------------------


def _real_psycopg2_extensions():
    """Import the genuine ``psycopg2.extensions`` and keep it in ``sys.modules``.

    ``from psycopg2.extensions import make_dsn`` resolves out of
    ``sys.modules["psycopg2.extensions"]`` when that key is present, and only
    falls back to walking the parent package's ``__path__`` when it is not.
    Importing it before the parent is swapped for the double is what lets
    doctor's DSN handling run for real against a faked connection.
    """
    import psycopg2.extensions

    return psycopg2.extensions


class _FakePostgres:
    """Just enough psycopg2 to answer doctor's three queries.

    Records every statement and its parameters, because *which* rows a shared
    database returns is the thing that can go wrong: one PostgreSQL holds every
    local agent, so an unscoped read answers about whichever tenant came back
    first.

    ``extensions`` is the **real** submodule. A double that omitted it made
    ``from psycopg2.extensions import ...`` raise ``ModuleNotFoundError``
    ("'psycopg2' is not a package"), which the readers map to "database
    unreadable" — so every assertion below would have been satisfied by doctor
    quietly giving up, and the DSN it dials would never be exercised. Real
    psycopg2 is a package; the double has to be one too.
    """

    def __init__(self, rows_by_prefix: dict):
        self._rows_by_prefix = rows_by_prefix
        self.executed: list[tuple[str, tuple]] = []
        self.closed = False
        self.extensions = _real_psycopg2_extensions()

    # -- module surface ----------------------------------------------------
    def connect(self, dsn, **kwargs):
        # Real ``psycopg2.connect`` merges keywords into the DSN through
        # ``make_dsn``, where **the keyword wins**. The double has to merge the
        # same way: a ``connect(self, dsn)`` that simply rejected keywords
        # would kill a "passes connect_timeout as a keyword" regression with an
        # ``AttributeError`` about its own signature, proving nothing about
        # whose timeout survives.
        self.dsn = self.extensions.make_dsn(dsn, **kwargs) if kwargs else dsn
        return self

    # -- connection surface ------------------------------------------------
    def cursor(self):
        return self

    def close(self):
        self.closed = True

    # -- cursor surface (used as a context manager) ------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), tuple(params)))
        self._rows = next(
            (
                rows
                for prefix, rows in self._rows_by_prefix.items()
                if prefix in " ".join(sql.split())
            ),
            [],
        )

    def fetchall(self):
        return self._rows


def _postgres_host(monkeypatch, fake):
    from kestrel_sovereign import doctor as doctor_module

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable.example/kestrel")
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", fake)
    # Keep the subprocess boundary out of broad governance unit tests while
    # retaining the exact production worker, including private passfile
    # materialization.  Dedicated tests below exercise the real child.
    monkeypatch.setattr(
        doctor_module,
        "_fetch_postgres_rows_isolated",
        lambda dsn, sql, params=(), **_kwargs: (
            doctor_module._postgres_fetch_rows_in_process(dsn, sql, params, driver=fake)
        ),
    )


def test_libpq_service_environment_does_not_block_the_runtime_dsn(
    tmp_path,
    monkeypatch,
):
    """A shell's libpq recipe is not evidence that asyncpg is unreachable."""
    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    service_name = "private_libpq_only_recipe"
    monkeypatch.setenv("PGSERVICE", service_name)

    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert fake.executed, "doctor did not inspect the runtime database"
    assert os.environ["PGSERVICE"] == service_name
    assert service_name not in " ".join(report.ok + report.warn + report.fail)


def test_missing_tls_file_becomes_a_path_safe_postgres_finding(tmp_path, monkeypatch):
    from urllib.parse import quote

    _seed_matching_anchor(tmp_path, monkeypatch)
    missing_certificate = tmp_path / "private" / "missing-client.crt"
    runtime_dsn = "postgresql://runtime_user@db.internal/kestrel?sslcert=" + quote(
        str(missing_certificate), safe=""
    )
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", runtime_dsn)

    report = diagnose(tmp_path)

    findings = " ".join(report.fail)
    assert not report.ready
    assert "runtime PostgreSQL configuration is invalid" in findings
    assert "shared with the spawned asyncpg runtime" in findings
    assert "sslcert" in findings
    assert str(missing_certificate) not in findings


def test_non_utf8_passfile_becomes_a_path_safe_postgres_finding(tmp_path, monkeypatch):
    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    passfile = tmp_path / "private-runtime-passfile"
    passfile.write_bytes(b"host:5432:db:user:\xff\n")
    passfile.chmod(0o600)
    monkeypatch.setenv("PGPASSFILE", str(passfile))

    report = diagnose(tmp_path)

    findings = " ".join(report.fail)
    assert not report.ready
    assert not fake.executed
    assert "runtime PostgreSQL configuration is invalid" in findings
    assert "passfile cannot be read by asyncpg" in findings
    assert str(passfile) not in findings


def test_nul_connection_file_path_becomes_a_path_safe_postgres_finding(
    tmp_path, monkeypatch
):
    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    unsafe_path = "private\x00root-ca.pem"
    env_path = tmp_path / ".env"
    env_path.write_text(env_path.read_text() + f"\nPGSSLROOTCERT={unsafe_path}\n")

    report = diagnose(tmp_path)

    findings = " ".join(report.fail)
    assert not report.ready
    assert not fake.executed
    assert "runtime PostgreSQL configuration is invalid" in findings
    assert "sslrootcert file path cannot be resolved by asyncpg" in findings
    assert unsafe_path not in findings


def test_postgres_probe_child_strips_the_complete_libpq_namespace(monkeypatch):
    from kestrel_sovereign.doctor import _postgres_probe_env

    monkeypatch.setenv("PGSERVICE", "private_failed_recipe")
    monkeypatch.setenv("PGDATESTYLE", "invalid-runtime-divergence")
    monkeypatch.setenv("PGTOTALLY_FUTURE_LIBPQ_OPTION", "future")
    monkeypatch.setenv("pgservice_mixed_case", "windows-safe")
    monkeypatch.setenv("KESTREL_VISIBLE_TO_PROBE", "yes")
    monkeypatch.setenv("PYTHONPATH", "/operator/pythonpath")
    monkeypatch.setenv("PYTHONHOME", "/operator/pythonhome")
    monkeypatch.setenv("HOME", "/doctor/home")

    child_env = _postgres_probe_env("/spawned/agent/home")

    assert not any(name.upper().startswith("PG") for name in child_env)
    assert child_env["KESTREL_VISIBLE_TO_PROBE"] == "yes"
    assert "PYTHONPATH" not in child_env
    assert "PYTHONHOME" not in child_env
    assert child_env["HOME"] == "/spawned/agent/home"
    assert child_env["PYTHONIOENCODING"] == "utf-8"
    assert os.environ["HOME"] == "/doctor/home"
    assert os.environ["PGSERVICE"] == "private_failed_recipe"
    assert os.environ["PGDATESTYLE"] == "invalid-runtime-divergence"


def test_isolated_postgres_probe_uses_stdin_and_sanitized_environment(
    monkeypatch,
):
    from kestrel_sovereign.doctor import _fetch_postgres_rows_isolated

    captured = {}

    class Process:
        returncode = 0

        def communicate(self, payload=None, timeout=None):
            captured.update(payload=payload, timeout=timeout)
            return '{"ok": true, "rows": [[1]]}', ""

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()

    monkeypatch.setenv("PGSERVICE", "private-service-name")
    monkeypatch.setattr("kestrel_sovereign.doctor.subprocess.Popen", popen)

    rows = _fetch_postgres_rows_isolated(
        "postgresql://runtime_user:runtime_secret@db.internal/kestrel",
        "SELECT %s",
        (1,),
        postgres_home="/runtime/home",
    )

    assert rows == [(1,)]
    assert "runtime_secret" not in " ".join(captured["command"])
    assert "runtime_secret" in captured["payload"]
    assert not any(name.upper().startswith("PG") for name in captured["env"])
    assert captured["env"]["HOME"] == "/runtime/home"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["timeout"] == 10
    assert captured["command"][0] == __import__("sys").executable
    assert captured["command"][1] == "-P"
    assert captured["command"][2].endswith("_doctor_postgres_probe.py")
    assert "-m" not in captured["command"]
    assert os.environ["PGSERVICE"] == "private-service-name"


def test_isolated_probe_tolerates_non_utf8_stderr_with_valid_stdout(
    monkeypatch,
):
    from kestrel_sovereign.doctor import _fetch_postgres_rows_isolated

    def popen(_command, **kwargs):
        class Process:
            returncode = 0

            def communicate(self, payload=None, timeout=None):
                encoding = kwargs["encoding"]
                errors = kwargs.get("errors", "strict")
                return (
                    b'{"ok": true, "rows": [[1]]}'.decode(encoding, errors=errors),
                    b"libpq noise: \xff\xfe".decode(encoding, errors=errors),
                )

        return Process()

    monkeypatch.setattr("kestrel_sovereign.doctor.subprocess.Popen", popen)

    rows = _fetch_postgres_rows_isolated(
        "postgresql://u@h/db?connect_timeout=2", "SELECT 1"
    )

    assert rows == [(1,)]


def test_isolated_postgres_probe_kills_and_reaps_a_timed_out_child(
    monkeypatch,
):
    import subprocess

    from kestrel_sovereign.doctor import (
        _fetch_postgres_rows_isolated,
        _PostgresProbeError,
    )

    state = {"communicate_calls": 0, "killed": False}
    first_host = "first.private.example"
    second_host = "second.private.example"
    password = "partial-output-secret"

    class Process:
        returncode = None

        def communicate(self, payload=None, timeout=None):
            state["communicate_calls"] += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired(
                    "probe",
                    timeout,
                    output=(
                        '{"progress": "trying '
                        + second_host
                        + " with "
                        + password
                        + '"}'
                    ),
                    stderr=(
                        'could not translate host name "'
                        + first_host
                        + '" to address: deterministic failure'
                    ),
                )
            self.returncode = -9
            return "", "libpq retained its earlier address failure"

        def kill(self):
            state["killed"] = True

    monkeypatch.setattr(
        "kestrel_sovereign.doctor.subprocess.Popen",
        lambda *_args, **_kwargs: Process(),
    )

    with pytest.raises(_PostgresProbeError, match="terminated") as raised:
        _fetch_postgres_rows_isolated(
            "postgresql://runtime_user:"
            + password
            + "@"
            + first_host
            + ","
            + second_host
            + "/db?connect_timeout=2",
            "SELECT 1",
        )

    assert state == {"communicate_calls": 2, "killed": True}
    diagnostic = str(raised.value)
    assert "could not translate host name" in diagnostic
    assert "deterministic failure" in diagnostic
    assert "libpq retained its earlier address failure" in diagnostic
    assert first_host not in diagnostic
    assert second_host not in diagnostic
    assert password not in diagnostic
    assert "<host>" in diagnostic
    assert len(diagnostic) < 1200


def test_isolated_postgres_probe_kills_and_reaps_on_pipe_failure(monkeypatch):
    from kestrel_sovereign.doctor import (
        _fetch_postgres_rows_isolated,
        _PostgresProbeError,
    )

    state = {"killed": False, "waited": False}

    class Process:
        returncode = None

        def communicate(self, payload=None, timeout=None):
            raise OSError("broken diagnostic pipe")

        def kill(self):
            state["killed"] = True

        def wait(self):
            state["waited"] = True

    monkeypatch.setattr(
        "kestrel_sovereign.doctor.subprocess.Popen",
        lambda *_args, **_kwargs: Process(),
    )

    with pytest.raises(
        _PostgresProbeError, match="diagnostic process communication failed"
    ):
        _fetch_postgres_rows_isolated("postgresql://u@h/db", "SELECT 1")

    assert state == {"killed": True, "waited": True}


def test_isolated_postgres_probe_rejects_non_json_parameters():
    from kestrel_sovereign.doctor import (
        _fetch_postgres_rows_isolated,
        _PostgresProbeError,
    )

    with pytest.raises(_PostgresProbeError, match="not transportable"):
        _fetch_postgres_rows_isolated("postgresql://u@h/db", "SELECT %s", (object(),))


def test_real_isolated_postgres_probe_reports_an_unreachable_server(
    monkeypatch,
):
    from kestrel_sovereign.doctor import (
        _fetch_postgres_rows_isolated,
        _PostgresProbeError,
    )

    service_name = "private_service_must_not_reach_worker"
    monkeypatch.setenv("PGSERVICE", service_name)
    monkeypatch.setenv("PGSERVICEFILE", "/definitely/missing/service.conf")
    monkeypatch.setenv("PGTZ", "Definitely/Invalid")

    with pytest.raises(_PostgresProbeError) as raised:
        _fetch_postgres_rows_isolated(
            "postgresql://probe_user@127.0.0.1:1/probe_db"
            "?connect_timeout=2&sslmode=disable",
            "SELECT 1",
        )

    assert "isolated PostgreSQL diagnostic process" not in str(raised.value)
    assert service_name not in str(raised.value)
    assert "service" not in str(raised.value).lower()
    assert "127.0.0.1" in str(raised.value) or "connection" in str(raised.value).lower()


def test_absent_passfile_is_materialized_under_private_probe_custody(
    tmp_path, monkeypatch
):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import (
        _ABSENT_PASSFILE_SENTINEL,
        _doctor_postgres_dsn,
        _postgres_fetch_rows_in_process,
    )

    fake = _FakePostgres({"SELECT 1": [(1,)]})
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", fake)
    translated = _doctor_postgres_dsn(
        "postgresql://runtime_user@host.example/db", {}, tmp_path
    )

    assert _postgres_fetch_rows_in_process(translated, "SELECT 1", ()) == [(1,)]
    materialized = Path(parse_dsn(fake.dsn)["passfile"])
    assert str(materialized) != _ABSENT_PASSFILE_SENTINEL
    assert tmp_path.resolve() not in materialized.parents
    assert not materialized.parents[1].exists()


def test_windows_probe_hides_and_restores_libpq_appdata(monkeypatch):
    from kestrel_sovereign import _doctor_postgres_probe as worker

    fake = _FakePostgres({"SELECT 1": [(1,)]})
    original_connect = fake.connect
    captured = {}

    def connect(dsn):
        captured["appdata"] = os.environ["APPDATA"]
        captured["appdata_exists"] = Path(captured["appdata"]).exists()
        return original_connect(dsn)

    fake.connect = connect
    monkeypatch.setattr(worker, "_IS_WINDOWS", True)
    monkeypatch.setenv("APPDATA", "/operator/appdata")

    assert worker.fetch_rows_in_process(
        "postgresql://u@h/db",
        "SELECT 1",
        (),
        absent_passfile_sentinel="unused",
        driver=fake,
    ) == [(1,)]

    assert captured["appdata"].endswith("absent-appdata")
    assert captured["appdata_exists"] is False
    assert os.environ["APPDATA"] == "/operator/appdata"


def test_project_env_pg_settings_reach_the_driver_without_being_exported(
    tmp_path,
    monkeypatch,
):
    """Exercise the complete diagnose path, not only the URI helper."""
    from psycopg2.extensions import parse_dsn

    _seed_matching_anchor(tmp_path, monkeypatch)
    properties = json.dumps({"name": "Test", "constitution_hash": "a" * 64})
    fake = _FakePostgres(
        {
            "SELECT node_id, label, properties FROM graph_nodes": [
                ("did:test:Test", "Test", properties)
            ],
            "FROM graph_edges": [("a" * 64,)],
        }
    )
    _postgres_host(monkeypatch, fake)
    for name in (
        "KESTREL_DB_BACKEND",
        "KESTREL_DATABASE_URL",
        "PGHOST",
        "PGUSER",
        "PGPASSWORD",
        "PGSSLMODE",
    ):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        env_path.read_text()
        + "\nKESTREL_DB_BACKEND=postgres\n"
        + "KESTREL_DATABASE_URL=postgresql:///kestrel\n"
        + "PGHOST=project-db.internal\n"
        + "PGUSER=project_user\n"
        + "PGPASSWORD=project-password\n"
        + "PGSSLMODE=disable\n"
    )

    diagnose(tmp_path)

    parsed = parse_dsn(fake.dsn)
    assert parsed["host"] == "project-db.internal"
    assert parsed["user"] == "project_user"
    assert parsed["password"] == "project-password"
    assert parsed["sslmode"] == "disable"
    assert "PGPASSWORD" not in os.environ


def test_missing_login_name_is_an_unreadable_database_finding(tmp_path, monkeypatch):
    _seed_matching_anchor(tmp_path, monkeypatch)
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable.example/kestrel")
    for name in ("PGUSER", "LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(name, raising=False)

    def no_login_name():
        raise OSError("uid has no passwd entry")

    monkeypatch.setattr("kestrel_sovereign.doctor._system_account_user", no_login_name)

    report = diagnose(tmp_path)

    assert not report.ready
    assert any(
        "governance NOT verified" in message
        and "runtime PostgreSQL user could not be determined" in message
        for message in report.fail
    ), report.fail


def test_diagnose_resolves_connection_files_from_project_not_invocation_cwd(
    tmp_path, monkeypatch
):
    from psycopg2.extensions import parse_dsn

    _seed_matching_anchor(tmp_path, monkeypatch)
    properties = json.dumps({"name": "Test", "constitution_hash": "a" * 64})
    fake = _FakePostgres(
        {
            "SELECT node_id, label, properties FROM graph_nodes": [
                ("did:test:Test", "Test", properties)
            ],
            "FROM graph_edges": [("a" * 64,)],
        }
    )
    _postgres_host(monkeypatch, fake)
    for name in (
        "KESTREL_DB_BACKEND",
        "KESTREL_DATABASE_URL",
        "PGPASSFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        env_path.read_text()
        + "\nKESTREL_DB_BACKEND=postgres\n"
        + "KESTREL_DATABASE_URL=postgresql://u@db.internal/kestrel\n"
        + "PGPASSFILE=secrets/runtime.pgpass\n"
    )
    passfile = tmp_path / "secrets" / "runtime.pgpass"
    passfile.parent.mkdir()
    passfile.write_text("db.internal:5432:kestrel:u:relative-secret\n")
    passfile.chmod(0o600)
    invocation_dir = tmp_path / "nested" / "invocation"
    invocation_dir.mkdir(parents=True)
    monkeypatch.chdir(invocation_dir)

    diagnose(tmp_path)

    parsed = parse_dsn(fake.dsn)
    assert parsed["password"] == "relative-secret"
    _assert_absent_doctor_passfile(parsed["passfile"], tmp_path)


def test_on_postgres_the_drift_verdict_comes_from_the_runtime_database(
    tmp_path, monkeypatch
):
    """The anchor matches the canonical file; PostgreSQL does not.

    Reading the anchor reports "anchored to current file" — a clean bill of
    health for an agent that is governed by something else entirely.
    """
    stored = _seed_matching_anchor(tmp_path, monkeypatch)
    drifted = hashlib.sha256(b"what postgres actually holds").hexdigest()
    properties = json.dumps({"name": "Test", "constitution_hash": drifted})
    fake = _FakePostgres(
        {
            # Column lists differ between the two node reads; a double that
            # answers both with the same shape hands `properties` back where a
            # `node_id` belongs and the check silently degrades to "older
            # agent, no anchor". Keyed on the full projection for that reason.
            "SELECT properties FROM graph_nodes": [(properties,)],
            "SELECT node_id, label, properties FROM graph_nodes": [
                ("did:web:test", "Test", properties)
            ],
            "FROM graph_edges": [(drifted,)],
        }
    )
    _postgres_host(monkeypatch, fake)

    report = diagnose(tmp_path)

    assert any("constitution drift" in m and drifted[:12] in m for m in report.fail), (
        f"expected drift against the PostgreSQL hash; got {report.fail} / {report.ok}"
    )
    assert not any(stored[:12] in m for m in report.ok)


def test_the_runtime_read_is_scoped_to_this_agent(tmp_path, monkeypatch):
    """One PostgreSQL holds every local agent, so the ``LIMIT 1`` that is
    correct on a per-agent file would pick a neighbour by row order. The DID to
    scope by comes from the anchor, which is where identity is born on every
    backend (#2871, #2894)."""
    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres(
        {
            "SELECT properties FROM graph_nodes": [],
            "SELECT node_id, label, properties FROM graph_nodes": [],
            "FROM graph_edges": [],
        }
    )
    _postgres_host(monkeypatch, fake)

    diagnose(tmp_path)

    node_reads = [
        (sql, params) for sql, params in fake.executed if "FROM graph_nodes" in sql
    ]
    assert node_reads, "doctor never asked PostgreSQL anything"
    for sql, params in node_reads:
        assert "node_id = %s" in sql, sql
        assert "LIMIT 1" not in sql, sql
        assert params and params[0].startswith("did:"), params


def test_the_diagnostic_connection_is_bounded(tmp_path, monkeypatch):
    """Doctor is what an operator runs *when the database is unavailable*.

    A black-holed or firewalled endpoint does not refuse the connection — it
    drops the packets, and libpq then waits out the OS TCP timeout. Unbounded,
    the recovery diagnostic hangs for minutes on exactly the failure it exists
    to describe.
    """
    from psycopg2.extensions import parse_dsn

    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({"SELECT node_id, label, properties FROM graph_nodes": []})
    _postgres_host(monkeypatch, fake)

    diagnose(tmp_path)

    assert parse_dsn(fake.dsn)["connect_timeout"] == "5"


def test_an_agent_costs_one_postgres_connection(tmp_path, monkeypatch):
    """Both checks read one shared result, so the timeout is paid once.

    Resolving and reading per check meant an unreachable database cost two
    connection timeouts *per agent*: a ten-agent fleet waiting 100 seconds to
    be told the database is down, from the tool whose bound is five seconds and
    whose purpose is to answer quickly when the database is down.
    """
    _seed_matching_anchor(tmp_path, monkeypatch)
    properties = json.dumps({"name": "Test", "constitution_hash": "a" * 64})
    fake = _FakePostgres(
        {
            "SELECT node_id, label, properties FROM graph_nodes": [
                ("did:test:Test", "Test", properties)
            ],
            "FROM graph_edges": [("a" * 64,)],
        }
    )
    connects = []
    original_connect = fake.connect
    fake.connect = lambda dsn, **kw: (
        connects.append(dsn),
        original_connect(dsn, **kw),
    )[1]
    _postgres_host(monkeypatch, fake)

    diagnose(tmp_path)

    # "FROM graph_nodes", not "graph_nodes": the schema probe names the
    # table inside a to_regclass() literal and is not a read of the row.
    node_reads = [sql for sql, _ in fake.executed if "FROM graph_nodes" in sql]
    assert len(node_reads) == 1, f"agent node read {len(node_reads)}× : {node_reads}"


def test_asyncpg_connect_timeout_is_not_reclassified_as_libpq_timeout(
    tmp_path,
    monkeypatch,
):
    """asyncpg sends this query name as a server setting.

    Keeping it as libpq's connection timeout would let doctor connect while
    PostgreSQL rejects the runtime startup packet for the unknown GUC.
    Doctor's own five-second outage bound is a separate connection parameter.
    """
    from psycopg2.extensions import parse_dsn

    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({"SELECT node_id, label, properties FROM graph_nodes": []})
    _postgres_host(monkeypatch, fake)
    monkeypatch.setenv(
        "KESTREL_DATABASE_URL",
        "postgresql://durable.example/kestrel?connect_timeout=30",
    )

    diagnose(tmp_path)

    parsed = parse_dsn(fake.dsn)
    assert parsed["connect_timeout"] == "5"
    assert parsed["options"] == "-c connect_timeout=30"


def test_a_postgres_host_whose_anchor_names_no_agent_is_skipped_not_guessed(
    tmp_path, monkeypatch
):
    """Without a DID there is no tenant to scope to, and an unscoped read of a
    shared database would answer about someone else. Skipping says so; guessing
    would not."""
    _seed_with_anchored_constitution(
        tmp_path, constitution_text=b"# C\n", stored_hash=None
    )
    db_path = tmp_path / "agent_data" / "test" / "kestrel_prime.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM graph_nodes")
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)

    report = diagnose(tmp_path)

    assert any("names no agent" in m for m in report.warn), report.warn
    assert not fake.executed, "doctor queried PostgreSQL without a tenant"


def test_sqlite_hosts_never_reach_for_postgres(tmp_path, monkeypatch):
    """The control. Without it, a test suite that only ever runs on SQLite
    would pass with the dispatch wired backwards."""
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    stored = _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", fake)

    report = diagnose(tmp_path)

    assert not fake.executed
    assert any(stored[:12] in m for m in report.ok)


def test_an_unowned_row_in_the_anchor_is_not_ready(tmp_path, monkeypatch):
    """The SQLite mirror of the PostgreSQL case, and it is not pending anything.

    DID discovery reads the anchor unscoped, so a DID means the row is there.
    The ledger exists, so this is not a database awaiting the #2649 backfill.
    The witness is therefore missing or foreign — and the agent's own bound
    store cannot see its agent node either, so startup fails and ``add_node``
    will not overwrite a foreign-owned row to repair it. Warning let doctor
    exit Ready, having also skipped the edge and overlay checks.
    """
    _seed_matching_anchor(tmp_path, monkeypatch, witness_node=False)

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    assert any("is not owned by" in m for m in report.fail), report.fail
    # Named as what it is: reanchoring cannot fix an ownership ledger.
    assert any("not constitution drift" in m for m in report.fail)


def test_a_pre_migration_anchor_still_only_warns(tmp_path, monkeypatch):
    """Without the ledger there is nothing to be unowned by — that database is
    simply older than #2649, and boot backfills it."""
    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _legacy_graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc"}))],
    )
    source = _resolve_governance_source(db, {}, tmp_path)

    assert source.ownership_ledger is False
    assert not isinstance(_read_agent_node(source), _NoAgentNode)


def test_a_missing_witness_before_the_backfill_is_pending_not_broken(
    tmp_path,
    monkeypatch,
):
    """Table existence is not the same fact as backfill completion.

    Schema init creates ``graph_node_owners``; the #2649 backfill that fills
    it is recorded separately, and boot runs it exactly once. Before that
    marker exists, a row with no witness is repaired at the next start — so
    using "the tables exist" as the proxy turned a database awaiting its
    migration into a readiness failure.
    """
    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc"}))],
        node_owners=[],
        ownership_settled=False,
    )
    source = _resolve_governance_source(db, {}, tmp_path)

    assert source.ownership_ledger is True
    assert source.ownership_settled is False


def test_a_missing_witness_after_the_backfill_is_permanent(tmp_path):
    from kestrel_sovereign.doctor import _resolve_governance_source

    db = _graph_db(
        tmp_path / "k.db",
        nodes=[("did:x", "agent", "x", json.dumps({"constitution_hash": "abc"}))],
        node_owners=[],
        ownership_settled=True,
    )
    source = _resolve_governance_source(db, {}, tmp_path)

    assert source.ownership_settled is True


# ---------------------------------------------------------------------------
# Fail-closed paths. Each of these swallows an error by design, so each one is
# a place a wrong verdict can be manufactured silently — the failure mode this
# whole issue is about. They are asserted rather than assumed.
# ---------------------------------------------------------------------------


def test_a_failed_existence_probe_never_claims_the_runtime_is_empty(tmp_path):
    """ "Something is there, do not call this empty" is the safe direction.

    ``_row_physically_exists`` decides whether a bound read that found nothing
    means *absent* (replication repairs it) or *unowned* (it cannot). A probe
    that cannot run must not vote for the first, or a database it never
    inspected gets judged against the anchor.
    """
    from kestrel_sovereign.doctor import _GovernanceSource, _row_physically_exists

    missing = _GovernanceSource(
        anchor_path=tmp_path / "does-not-exist.db", agent_did="did:x"
    )

    assert _row_physically_exists(missing) is True


def test_an_unparseable_dsn_is_left_for_connect_to_report(tmp_path):
    """``_bounded_dsn`` must not swallow a malformed DSN into silence.

    Returning it untouched lets ``psycopg2.connect`` raise its own message,
    which the reader turns into a redacted ``_UnreadableDB``. Raising here
    instead would escape as a traceback out of a diagnostic.
    """
    from kestrel_sovereign.doctor import _bounded_dsn

    assert _bounded_dsn("not a dsn at all") == "not a dsn at all"


def test_an_unbounded_translated_dsn_is_rejected():
    """Only spawned-agent translation may choose the diagnostic timeout."""
    from kestrel_sovereign.doctor import _bounded_dsn

    with pytest.raises(ValueError, match="missing connect_timeout"):
        _bounded_dsn("postgresql://u:p@h:5445/db?sslmode=require")


def test_bounded_dsn_does_not_reread_parent_timeout_environment(monkeypatch):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _bounded_dsn

    monkeypatch.setenv("KESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS", "7")
    dsn = "postgresql://u:p@h/db?connect_timeout=23"

    assert parse_dsn(_bounded_dsn(dsn))["connect_timeout"] == "23"


def test_doctor_timeout_is_divided_across_libpq_hosts(tmp_path):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    translated = _doctor_postgres_dsn(
        "postgresql://u@h1,h2,h3/db",
        {"KESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS": "12"},
        tmp_path,
    )

    assert parse_dsn(translated)["connect_timeout"] == "4"


def test_worker_timeout_covers_every_floored_libpq_host():
    from kestrel_sovereign.doctor import _postgres_probe_timeout_seconds

    dsn = "postgresql://u@h1,h2,h3,h4,h5,h6/db?connect_timeout=2"

    assert _postgres_probe_timeout_seconds(dsn) == 17


@pytest.mark.parametrize("value", ["", " ", "\t \n"])
def test_blank_doctor_postgres_timeout_uses_the_default(tmp_path, value):
    from psycopg2.extensions import parse_dsn

    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    translated = _doctor_postgres_dsn(
        "postgresql://u@h/db",
        {"KESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS": value},
        tmp_path,
    )

    assert parse_dsn(translated)["connect_timeout"] == "5"


def test_blank_project_timeout_reaches_diagnose_as_the_default(tmp_path, monkeypatch):
    from psycopg2.extensions import parse_dsn

    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    env_path = tmp_path / ".env"
    env_path.write_text(
        env_path.read_text() + "\nKESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS=   \n"
    )

    report = diagnose(tmp_path)

    assert report.ready, f"ok={report.ok} warn={report.warn} fail={report.fail}"
    assert parse_dsn(fake.dsn)["connect_timeout"] == "5"


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_invalid_doctor_postgres_timeout_fails_as_configuration(tmp_path, value):
    from kestrel_sovereign.doctor import _doctor_postgres_dsn

    with pytest.raises(ValueError, match="KESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS"):
        _doctor_postgres_dsn(
            "postgresql://u@h/db",
            {"KESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS": value},
            tmp_path,
        )


def test_invalid_project_timeout_is_not_reported_as_a_database_outage(
    tmp_path, monkeypatch
):
    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    env_path = tmp_path / ".env"
    env_path.write_text(
        env_path.read_text()
        + "\nKESTREL_DOCTOR_POSTGRES_TIMEOUT_SECONDS=not-an-integer\n"
    )

    report = diagnose(tmp_path)

    findings = " ".join(report.fail)
    assert not report.ready
    assert not fake.executed
    assert "PostgreSQL doctor configuration is invalid" in findings
    assert "Runtime database reachability was not established" in findings
    assert "runtime database access with those settings will fail" not in findings


def test_libpq_direct_tls_limit_is_diagnostic_blindness_not_runtime_outage(
    tmp_path, monkeypatch
):
    from kestrel_sovereign import doctor as doctor_module

    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    monkeypatch.setenv(
        "KESTREL_DATABASE_URL",
        "postgresql://durable.example/kestrel?sslmode=require&sslnegotiation=direct",
    )
    accepts = doctor_module._libpq_accepts_dsn_option
    monkeypatch.setattr(
        doctor_module,
        "_libpq_accepts_dsn_option",
        lambda name, value: False if name == "sslnegotiation" else accepts(name, value),
    )

    report = diagnose(tmp_path)

    findings = " ".join(report.fail)
    assert not report.ready
    assert not fake.executed
    assert "cannot construct an equivalent libpq diagnostic connection" in findings
    assert "runtime connection option 'sslnegotiation'" in findings
    assert "Runtime database reachability was not established" in findings
    assert "runtime database access with those settings will fail" not in findings


def test_shared_invalid_sslmode_is_reported_as_runtime_configuration(
    tmp_path, monkeypatch
):
    _seed_matching_anchor(tmp_path, monkeypatch)
    fake = _FakePostgres({})
    _postgres_host(monkeypatch, fake)
    monkeypatch.setenv(
        "KESTREL_DATABASE_URL",
        "postgresql://durable.example/kestrel?sslmode=invalid-mode",
    )

    report = diagnose(tmp_path)

    findings = " ".join(report.fail)
    assert not report.ready
    assert not fake.executed
    assert "runtime PostgreSQL configuration is invalid" in findings
    assert "shared with the spawned asyncpg runtime" in findings
    assert "cannot open the configured database until it is fixed" in findings
    assert "Runtime database reachability was not established" not in findings


def test_the_emancipation_contract_survives_a_falsy_but_real_value():
    """``None`` means "no contract"; an empty string is a *corrupt* one.

    Collapsing them would render the canonical constitution for an agent whose
    receipt is damaged, and report it correctly anchored.
    """
    from kestrel_sovereign.doctor import _anchored_emancipation_contract

    assert _anchored_emancipation_contract({"emancipation_contract": ""}) == ""
    assert _anchored_emancipation_contract({}) is None


def test_postgres_failure_provenance_does_not_depend_on_reason_text():
    """Resolution can fail before a ``_GovernanceSource`` is built at all.

    The structured connection kind, not the word ``PostgreSQL``, makes this a
    runtime-impacting failure. Conversely, PostgreSQL-looking prose without
    provenance must not manufacture a reachability claim.
    """
    from kestrel_sovereign.doctor import DoctorReport, _report_unexamined, _UnreadableDB

    report = DoctorReport()
    sentinel = _UnreadableDB(
        reason="connection refused",
        postgres_failure="connection",
    )
    _report_unexamined("Test", sentinel.reason, sentinel, report)

    assert report.fail and not report.warn, (report.fail, report.warn)
    assert not report.ready
    assert "equivalent libpq connection failed" in report.fail[0]

    unknown = DoctorReport()
    unclassified = _UnreadableDB(reason="cannot read PostgreSQL")
    _report_unexamined("Test", unclassified.reason, unclassified, unknown)

    assert not unknown.fail
    assert unknown.warn


def test_an_edge_the_agent_does_not_own_is_reported_as_a_ledger_problem(
    tmp_path,
    monkeypatch,
):
    """ "No edge" and "an edge I cannot use" are different findings.

    The writer deletes an *ownerless* correct edge and re-creates it, so a
    forced reanchor really does repair that one. An edge witnessed by another
    tenant is refused outright by ``add_edge``. Reporting either as a missing
    edge promises a repair, and for the second the promise is false.
    """
    stored = _seed_matching_anchor(tmp_path, monkeypatch, witness_edge=False)

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    edge = [m for m in report.fail if "does not own it" in m]
    assert edge, f"fail={report.fail}"
    assert stored[:12] in edge[0]
    assert "safe-modes" in edge[0]
    # Not phrased as drift, which would send them to a plain reanchor.
    assert "anchor drift" not in edge[0]


def test_doctor_catches_a_crlf_smudged_semantic_checkout_before_boot(
    tmp_path,
    monkeypatch,
):
    """The registry mismatch that bricks agent boot must be a doctor failure.

    Redirect only *where* the audit reads; the classifier, the manifest, and
    the resource bytes are the real ones, so this fails if the diagnosis
    regresses.
    """
    import shutil
    from importlib import resources

    from kestrel_sovereign.doctor import DoctorReport, _check_semantic_registry
    from kestrel_sovereign.knowledge import registry as registry_module

    package_root = tmp_path / "kestrel_sovereign"
    semantic_root = package_root / "data" / "semantic"
    shutil.copytree(
        resources.files("kestrel_sovereign").joinpath("data", "semantic"),
        semantic_root,
    )
    audit = registry_module.audit_semantic_resources
    manifest = semantic_root / "registry.toml"
    monkeypatch.setattr(
        registry_module, "audit_semantic_resources", lambda: audit(manifest)
    )

    clean = DoctorReport()
    _check_semantic_registry(clean)
    assert clean.ready, f"fail={clean.fail}"
    assert any("all pinned resources verified" in m for m in clean.ok)

    pinned = registry_module.load_knowledge_registry(manifest).resources
    for path in {
        package_root.joinpath(*Path(resource.package_resource).parts)
        for resource in pinned
    }:
        content = path.read_bytes()
        path.write_bytes(content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    report = DoctorReport()
    _check_semantic_registry(report)

    assert not report.ready, f"ok={report.ok}"
    assert len(report.fail) == 1, report.fail
    assert f"{len(pinned)} semantic resource(s) fail their pin" in report.fail[0]
    assert "line-ending mismatch, not a corrupted resource" in report.fail[0]

    # Every affected path is named, not one example: 29 pins fail together and
    # a report that shows one leaves the operator repairing a single file while
    # the fleet stays unbootable.
    for resource in pinned:
        assert resource.package_resource in report.fail[0]
    # And the remedy it carries repairs all of them at once — the whole
    # directory, executed as commands proven in test_knowledge_registry.py.
    for command in registry_module.crlf_checkout_repair_commands():
        assert f"`{command}`" in report.fail[0]


def test_doctor_reports_an_unparseable_semantic_manifest_instead_of_crashing(
    tmp_path,
    monkeypatch,
):
    """A malformed manifest must be a readiness failure, not a traceback.

    ``kestrel doctor`` and ``setup --check`` are what an operator reaches for
    when the registry is broken; dying inside the TOML decoder is the one
    outcome that leaves them with nothing.
    """
    from kestrel_sovereign.doctor import DoctorReport, _check_semantic_registry
    from kestrel_sovereign.knowledge import registry as registry_module

    manifest = tmp_path / "kestrel_sovereign" / "data" / "semantic" / "registry.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("version = 1\n[resource.truncated\n", encoding="utf-8")
    audit = registry_module.audit_semantic_resources
    monkeypatch.setattr(
        registry_module, "audit_semantic_resources", lambda: audit(manifest)
    )

    report = DoctorReport()
    _check_semantic_registry(report)

    assert not report.ready, f"ok={report.ok}"
    assert len(report.fail) == 1, report.fail
    assert "semantic registry is unusable" in report.fail[0]
    assert "not valid TOML" in report.fail[0]
