"""Doctrine-anchoring must survive the privacy graph boundary in volatile modes.

Regression for #2672 finding P3. The dispatcher anchors the doctrine bundle on
the agent identity node on the first full-constitution signal
(``_ensure_doctrine_bundle_anchored`` → ``anchor_doctrine_bundle`` →
``agent.storage.add_node(agent_node)``). That write carries the governance
fields ``doctrine_bundle_hash`` / ``doctrine_bundle_files`` /
``doctrine_bundle_anchored_at``.

An earlier #2672 revision omitted ``doctrine_bundle_files`` /
``doctrine_bundle_anchored_at`` from the control-plane schema, so in EPHEMERAL /
ISOLATED the write raised ``PrivacyViolationError``, the dispatcher swallowed it
as non-fatal, and the anchor never persisted — silently disabling doctrine-drift
detection for born-volatile agents.

This drives the REAL anchor primitive through a REAL privacy-enforcing wrapper in
a volatile mode and asserts the anchor persists on the durable agent node.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest

from kestrel_sovereign.agent.doctrine_bundle import (
    PROP_BUNDLE_ANCHORED_AT,
    PROP_BUNDLE_FILES,
    PROP_BUNDLE_HASH,
    anchor_doctrine_bundle,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


AGENT_ID = "did:test:doctrine-anchor"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_project(root: Path) -> Path:
    """Minimal project root with the DEFAULT_ANCHORED_PATHS doctrine files."""
    _write(root / "docs" / "principles" / "KESTREL_CONSTITUTION.md",
           "Article I. Identity rules.\n")
    _write(root / "docs" / "TORTOISE_DOCTRINE.md",
           "Slow is smooth. Smooth is fast.\n")
    _write(root / "AGENTS.md", "Agent conventions live here.\n")
    return root


class _Agent:
    """The minimal agent shape ``anchor_doctrine_bundle`` consumes."""

    def __init__(self, storage):
        self.agent_id = AGENT_ID
        self.storage = storage


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED])
async def test_doctrine_anchor_persists_in_volatile_mode(tmp_path, mode):
    """The doctrine bundle anchors (and PERSISTS) through the privacy wrapper in a
    volatile mode — the write carries ``doctrine_bundle_files`` /
    ``doctrine_bundle_anchored_at``, which the reconciled control-plane schema now
    admits via the trusted path (#2672 finding P3)."""
    project = _make_project(tmp_path / "project")

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        # Seed the agent identity node as a prior (persistent) stint would.
        await raw.add_node(
            GraphNode(
                node_id=AGENT_ID,
                node_type="agent",
                label="Kestrel",
                # A realistic 64-hex SHA-256 constitution hash — the per-field
                # validator (#2672 review P1) rejects a non-hash in this field.
                properties={
                    "name": "Kestrel",
                    "constitution_hash": "b" * 64,
                },
            )
        )

        wrapper = PrivacyEnforcingStorage(raw, mode)
        agent = _Agent(wrapper)

        # This is the primitive the dispatcher's first full-constitution signal
        # drives. It must NOT raise, and it must persist the anchor.
        snapshot = await anchor_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )

        stored = await raw.get_node(AGENT_ID)
        assert stored is not None
        # The anchor persisted: hash + file list + timestamp are all durable, so
        # the dispatcher's subsequent drift check has an anchored hash to compare.
        assert stored.properties.get(PROP_BUNDLE_HASH) == snapshot.hash
        assert stored.properties.get(PROP_BUNDLE_FILES) == list(snapshot.files)
        assert stored.properties.get(PROP_BUNDLE_ANCHORED_AT)


@pytest.mark.asyncio
async def test_doctrine_anchor_is_idempotent_in_volatile_mode(tmp_path):
    """A second anchor with the same bundle is a no-op and does not raise — the
    idempotent path also survives the volatile-mode graph boundary."""
    project = _make_project(tmp_path / "project")

    async with AsyncStorage(str(tmp_path / "kestrel.db"), agent_id=AGENT_ID) as raw:
        await raw.add_node(
            GraphNode(
                node_id=AGENT_ID,
                node_type="agent",
                label="Kestrel",
                properties={"name": "Kestrel"},
            )
        )
        wrapper = PrivacyEnforcingStorage(raw, PrivacyMode.EPHEMERAL)
        agent = _Agent(wrapper)

        first = await anchor_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )
        second = await anchor_doctrine_bundle(
            agent, project_root=project, bootstrap_files=OrderedDict()
        )
        assert first.hash == second.hash
        stored = await raw.get_node(AGENT_ID)
        assert stored.properties.get(PROP_BUNDLE_HASH) == first.hash
