"""Unit tests for kestrel_sovereign.doctor."""

from __future__ import annotations

from pathlib import Path

import toml
from cryptography.fernet import Fernet

from kestrel_sovereign.doctor import diagnose, format_report
from kestrel_sovereign.multi_agent.config import (
    HostConfig,
    LocalAgentConfig,
    MULTI_AGENT_CONFIG_FILENAME,
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


import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402

import pytest  # noqa: E402
from unittest.mock import patch  # noqa: E402

from kestrel_sovereign.doctor import (  # noqa: E402
    _NoAgentNode,
    _NoHashProperty,
    _UnreadableDB,
    _GovernanceSource,
    _anchored_constitution_hash,
    _anchored_emancipation_contract,
)

_TEST_DID = "did:test:Test"


def _anchor(db_path, agent_did: str = _TEST_DID) -> _GovernanceSource:
    """A source that reads the local anchor — the SQLite host's shape.

    The readers take a :class:`_GovernanceSource` rather than a path, because
    on a PostgreSQL host the anchor holds the *birth record* and the live
    governance is elsewhere (#2892). These helper-level tests are about the
    SQLite reading itself, so they say which database they mean rather than
    relying on a default.

    ``agent_did`` is required on both backends: the runtime's store is bound to
    it everywhere, so every governance read carries an ownership predicate.
    """
    return _GovernanceSource(anchor_path=db_path, agent_did=agent_did)


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
    multi_agent is already failing the check_multi_agent step elsewhere; we
    should not pile on additional drift noise."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
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
        m for m in (*report.warn, *report.fail, *report.ok)
        if _is_drift_msg(m)
    ]
    assert drift_msgs == []


def test_constitution_drift_skips_when_db_missing(tmp_path, monkeypatch):
    """Missing DB is already a fail in check_multi_agent; drift check should
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


def test_reads_the_hash_off_the_properties_it_was_given():
    assert _anchored_constitution_hash({"constitution_hash": "abc123"}) == "abc123"


@pytest.mark.parametrize(
    "properties",
    [
        None,                                  # corrupt/absent properties column
        {},                                     # older agent, never anchored
        {"constitution_hash": ""},              # present but empty
        {"constitution_hash": 12345},           # present but not a string
        "not a dict",                           # properties parsed to a scalar
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


from kestrel_sovereign.doctor import (  # noqa: E402
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
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    stored = hashlib.sha256(text).hexdigest()
    _seed_with_anchored_constitution(
        tmp_path, constitution_text=text, stored_hash=stored, **kwargs
    )
    return stored


def test_governed_by_edge_match_passes(tmp_path, monkeypatch):
    stored = _seed_matching_anchor(tmp_path, monkeypatch)
    report = diagnose(tmp_path)
    assert any(
        "governed_by edge targets the anchored constitution" in m
        and stored[:12] in m
        for m in report.ok
    )
    assert not any("anchor drift" in m for m in report.fail)


def test_governed_by_edge_mistargeted_fails(tmp_path, monkeypatch):
    """The 2026-07-18 incident: property + blob reanchored, edge still on
    the ancient anchor → proof 2 fails closed at boot."""
    ancient = hashlib.sha256(b"ancient constitution").hexdigest()
    stored = _seed_matching_anchor(
        tmp_path, monkeypatch, governed_by_target=ancient
    )
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
    stored = _seed_matching_anchor(
        tmp_path, monkeypatch, governed_by_target=None
    )
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
        m for m in report.fail
        if "cannot verify the governed_by governance edge" in m
    ]
    assert len(unverifiable) == 1
    assert "safe-mode" in unverifiable[0]
    assert (
        "kestrel constitution reanchor --agent-name Test --force"
        in unverifiable[0]
    )
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
        "governed_by edge targets the anchored constitution" in m
        and stored[:12] in m
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
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    _seed_with_anchored_constitution(
        tmp_path, constitution_text=text, stored_hash=None,
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
        m for m in report.fail
        if "anchored constitution overlay is missing from disk" in m
    ]
    assert len(missing) == 1
    assert anchored_hash[:12] in missing[0]


def test_overlay_malformed_anchor_with_overlay_present_fails(
    tmp_path, monkeypatch
):
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
        m for m in report.fail
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
    incomplete = [
        m for m in report.warn if "overlay anchor check incomplete" in m
    ]
    assert len(incomplete) == 1
    assert not any("overlay" in m.lower() for m in report.fail)
    assert report.ready, f"fail={report.fail}"


def test_overlay_silent_when_absent_and_unanchored(tmp_path, monkeypatch):
    """Normal agent: no overlay file, no anchor → zero overlay messages."""
    _seed_matching_anchor(tmp_path, monkeypatch)
    report = diagnose(tmp_path)
    assert not any(
        "overlay" in m for m in (*report.ok, *report.warn, *report.fail)
    )


def test_overlay_checked_for_legacy_agent_without_base_anchor(
    tmp_path, monkeypatch
):
    """#1722 legacy shape: anchored overlay but NO base constitution_hash.
    The overlay check must still run (the runtime audits it first and
    unconditionally)."""
    text = b"# constitution\n"
    canonical = _patch_canonical(tmp_path, text)
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
    overlay_text = b"# Overlay\n"
    overlay_hash = hashlib.sha256(overlay_text).hexdigest()
    _seed_with_anchored_constitution(
        tmp_path,
        constitution_text=text,
        stored_hash=None,
        overlay_anchor=overlay_hash,
    )
    (tmp_path / "agent_data" / "test" / "CONSTITUTION.md").write_bytes(
        overlay_text
    )
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
"""


def _graph_db(path, *, nodes=(), edges=(), node_owners=(), edge_owners=()):
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
        conn.executemany(
            "INSERT INTO graph_node_owners VALUES (?, ?)", node_owners
        )
        conn.executemany(
            "INSERT INTO graph_edge_owners VALUES (?, ?, ?, ?)", edge_owners
        )
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
    source = _resolve_governance_source(db, {})

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
    assert _resolve_governance_source(db, {}).ownership_ledger is True


# ---------------------------------------------------------------------------
# Driver errors reach report.warn, which operators read on a terminal and CI
# archives. libpq echoes the DSN it could not parse.
# ---------------------------------------------------------------------------


def test_an_unreachable_postgres_is_not_ready(tmp_path, monkeypatch):
    """"I did not check" must not print as "Ready".

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

    report = diagnose(tmp_path)

    assert not report.ready, f"ok={report.ok} warn={report.warn}"
    assert any("governance NOT verified" in m for m in report.fail), report.fail
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

    result = _resolve_governance_source(db, {})

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
            # Two columns, as the real probe asks: graph schema, then ledger.
            "SELECT to_regclass": [(True, True)],
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
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(canonical)
    )
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
        ('FATAL:  password authentication failed for user "kestrel_prod"',
         "kestrel_prod"),
        ('FATAL:  database "governance_prod" does not exist', "governance_prod"),
    ],
)
def test_routine_failures_do_not_leak_the_database_estate(
    tmp_path, message, leaked,
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
        dsn=(
            "postgresql://kestrel_prod:hunter2@db.internal:5432/governance_prod"
        ),
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
    write_toml(tmp_path / "kestrel.toml", {"llm": {
        "route_priority": ["openrouter:api"],
        "vendors": {"openrouter": {"routes": {"api": {
            "adapter": "OpenRouterAdapter",
            "api_key_env": "OPENROUTER_API_KEY",
            "management_api_key_env": "OPENROUTER_MANAGEMENT_API_KEY",
            "model": "auto",
        }}}},
    }}, deep_merge=False)
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
    assert _anchor_is_the_runtime_database(
        {
            "KESTREL_DB_BACKEND": "postgres",
            "KESTREL_DATABASE_URL": "postgresql://h/db",
        }
    ) is False


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
        "KESTREL_DB_BACKEND=postgres\n"
        "KESTREL_DATABASE_URL=postgresql://from-file/db\n"
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
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable.example/kestrel")
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", fake)


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

    assert any(
        "constitution drift" in m and drifted[:12] in m for m in report.fail
    ), f"expected drift against the PostgreSQL hash; got {report.fail} / {report.ok}"
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
        (sql, params)
        for sql, params in fake.executed
        if "FROM graph_nodes" in sql
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
    fake.connect = lambda dsn, **kw: (connects.append(dsn), original_connect(dsn, **kw))[1]
    _postgres_host(monkeypatch, fake)

    diagnose(tmp_path)

    # "FROM graph_nodes", not "graph_nodes": the schema probe names the
    # table inside a to_regclass() literal and is not a read of the row.
    node_reads = [sql for sql, _ in fake.executed if "FROM graph_nodes" in sql]
    assert len(node_reads) == 1, f"agent node read {len(node_reads)}× : {node_reads}"


def test_a_dsn_that_sets_its_own_timeout_keeps_it(tmp_path, monkeypatch):
    """``psycopg2.connect(dsn, connect_timeout=...)`` merges through
    ``make_dsn``, where the **keyword wins over the DSN**. Passing the default
    that way would overrule an operator who tuned a slow or distant link, and
    doctor would report a reachable database as unreadable — the cry-wolf
    failure this whole change is about, reintroduced by its own fix.
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

    assert parse_dsn(fake.dsn)["connect_timeout"] == "30"


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
    source = _resolve_governance_source(db, {})

    assert source.ownership_ledger is False
    assert not isinstance(_read_agent_node(source), _NoAgentNode)
