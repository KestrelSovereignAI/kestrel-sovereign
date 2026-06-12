"""Per-agent CONSTITUTION.md overlay loading (#898).

When ``<agent_dir>/CONSTITUTION.md`` is present, KestrelAgent should populate
``self.constitution_text`` with its contents so feature-side grant lookups
(e.g. ``ComputerUseFeature._granted_capabilities``) see this agent's
Amendment IX checkboxes instead of falling through to the package
constitution.

The producer is ``KestrelAgent.__init__``; the consumer is the
existing lookup at ``feature.py:_granted_capabilities``. This test only
exercises the producer side — that the file is read, attribute is set,
and missing/malformed files are handled gracefully.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from unittest.mock import AsyncMock, MagicMock

import hashlib

from kestrel_sovereign.constitution.hierarchy import parse_amendment_ix_grants
from kestrel_sovereign.kestrel_agent import KestrelAgent


def _make_agent(storage_path: Path | None) -> KestrelAgent:
    """Construct a KestrelAgent without doing async storage init.

    ``KestrelAgent.__init__`` is purely synchronous attribute setting until
    the async ``initialize()`` method is called. We only need the synchronous
    attributes for these tests, so we construct directly without entering an
    event loop.
    """
    return KestrelAgent(
        did="did:test:overlay",
        privacy_mode="normal",
        storage_path=str(storage_path) if storage_path else None,
    )


class TestPerAgentOverlayLoading:
    def test_overlay_file_loaded_when_present(self, tmp_path):
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()
        overlay = agent_dir / "CONSTITUTION.md"
        overlay.write_text(
            "### Amendment IX\n"
            "\n"
            "#### Granted Capabilities\n"
            "- [x] shell_execution_host\n"
            "- [x] filesystem_read\n"
            "- [ ] filesystem_write\n",
            encoding="utf-8",
        )

        agent = _make_agent(agent_dir / "kestrel_prime.db")

        assert agent.constitution_text is not None
        # Round-trip through the parser the consumer uses; this is the
        # whole point of the producer change.
        grants = parse_amendment_ix_grants(agent.constitution_text)
        assert grants == frozenset({"shell_execution_host", "filesystem_read"})

    def test_no_overlay_file_means_attribute_is_none(self, tmp_path):
        agent_dir = tmp_path / "no_overlay_agent"
        agent_dir.mkdir()
        # No CONSTITUTION.md created.

        agent = _make_agent(agent_dir / "kestrel_prime.db")

        assert agent.constitution_text is None
        # The consumer will fall through to the package constitution path,
        # preserving pre-#898 behavior for agents without an overlay.

    def test_no_storage_path_means_no_overlay_lookup(self):
        agent = _make_agent(storage_path=None)
        assert agent.constitution_text is None

    def test_unreadable_overlay_warns_and_falls_through(self, tmp_path, caplog):
        if os.name == "nt":
            pytest.skip("chmod-based unreadable test isn't reliable on Windows")
        agent_dir = tmp_path / "unreadable_agent"
        agent_dir.mkdir()
        overlay = agent_dir / "CONSTITUTION.md"
        overlay.write_text("### Amendment IX\n- [x] shell_execution_host\n")
        os.chmod(overlay, 0o000)
        try:
            with caplog.at_level(logging.WARNING):
                agent = _make_agent(agent_dir / "kestrel_prime.db")
            assert agent.constitution_text is None
            assert any(
                "per-agent constitution overlay" in rec.getMessage()
                for rec in caplog.records
            ), f"Expected a warning naming the overlay file; got {caplog.records}"
        finally:
            os.chmod(overlay, 0o644)

    def test_malformed_overlay_loads_but_yields_no_grants(self, tmp_path):
        # Garbage/wrong-section text loads fine but parser returns empty.
        # This is intentional — keeps the loader robust while the parser's
        # strictness ensures typos don't widen permissions.
        agent_dir = tmp_path / "malformed_agent"
        agent_dir.mkdir()
        overlay = agent_dir / "CONSTITUTION.md"
        overlay.write_text("This is not a constitution.\n")

        agent = _make_agent(agent_dir / "kestrel_prime.db")

        assert agent.constitution_text == "This is not a constitution.\n"
        assert parse_amendment_ix_grants(agent.constitution_text) == frozenset()

    def test_unchecked_box_is_not_a_grant(self, tmp_path):
        # The constitution explicitly says "only ``[x]`` (lowercase x)
        # counts; ``[ ]``, ``[X]``, and any other variant are intentionally
        # treated as ungranted so a typo never widens permissions."
        # Producer-side test: typo'd grants don't sneak through.
        agent_dir = tmp_path / "typo_agent"
        agent_dir.mkdir()
        overlay = agent_dir / "CONSTITUTION.md"
        overlay.write_text(
            "### Amendment IX\n"
            "- [X] shell_execution_host\n"        # capital X — not a grant
            "- [ ] filesystem_read\n"             # empty — not a grant
            "- [x ] filesystem_write\n"           # extra space — not a grant
            "- [x] shell_execution_sandboxed\n",  # only this counts
        )
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        grants = parse_amendment_ix_grants(agent.constitution_text)
        assert grants == frozenset({"shell_execution_sandboxed"})


class TestComputerUseFeaturePicksUpOverlay:
    """End-to-end at the lookup level: ``feature.py:_granted_capabilities``
    uses ``agent.constitution_text`` ONLY when the overlay is integrity-verified
    against its anchor (#1722) — an unverified overlay's grants are ignored so a
    file written next to the agent DB can't self-grant host shell.
    """

    def test_verified_overlay_grants_take_precedence_over_package(self, tmp_path):
        from kestrel_sovereign.features.computer_use.feature import (
            ComputerUseFeature,
        )

        agent_dir = tmp_path / "emma_like"
        agent_dir.mkdir()
        (agent_dir / "CONSTITUTION.md").write_text(
            "### Amendment IX\n- [x] shell_execution_host\n"
        )

        agent = _make_agent(agent_dir / "kestrel_prime.db")
        # Anchored/verified overlay (set by verify_constitution_overlay in the
        # real flow) → grants honored.
        agent.constitution_overlay_verified = True
        feature = ComputerUseFeature(agent=agent)
        granted = feature._granted_capabilities()
        assert "shell_execution_host" in granted

    def test_unverified_overlay_grants_are_ignored(self, tmp_path):
        """#1722: the self-grant vector. An overlay present but NOT anchored
        (verified defaults False) must NOT grant dangerous capabilities."""
        from kestrel_sovereign.features.computer_use.feature import (
            ComputerUseFeature,
        )

        agent_dir = tmp_path / "attacker_overlay"
        agent_dir.mkdir()
        (agent_dir / "CONSTITUTION.md").write_text(
            "### Amendment IX\n- [x] shell_execution_host\n"
        )
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent.constitution_overlay_verified is False  # default
        feature = ComputerUseFeature(agent=agent)
        granted = feature._granted_capabilities()
        # The unverified overlay's host-shell grant is withheld. (The package
        # constitution doesn't grant shell_execution_host either.)
        assert "shell_execution_host" not in granted

    def test_verified_empty_overlay_is_authoritative_not_package(self, tmp_path):
        """#1722 codex r2: a VERIFIED overlay that grants nothing must NARROW
        capabilities (return empty), not fall through to the packaged
        constitution's grants."""
        from kestrel_sovereign.features.computer_use.feature import (
            ComputerUseFeature,
        )

        agent_dir = tmp_path / "narrowing"
        agent_dir.mkdir()
        # Valid Amendment IX section that intentionally grants NOTHING.
        (agent_dir / "CONSTITUTION.md").write_text(
            "### Amendment IX\n#### Granted Capabilities\n- [ ] shell_execution_host\n"
        )
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        agent.constitution_overlay_verified = True
        feature = ComputerUseFeature(agent=agent)
        granted = feature._granted_capabilities()
        # Authoritative empty → deny-all, regardless of what the package grants.
        assert granted == frozenset()


class TestOverlayAnchorVerification:
    """``ConstitutionMixin.verify_constitution_overlay`` decision matrix (#1722).

    The overlay sha is computed at load in ``__init__``; the anchor lives in the
    identity node. We mock ``storage.get_node`` to drive each case.
    """

    def _agent_with_overlay(self, tmp_path, text="### Amendment IX\n- [x] shell_execution_host\n"):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(exist_ok=True)
        (agent_dir / "CONSTITUTION.md").write_text(text, encoding="utf-8")
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        return agent, hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _node(self, props):
        n = MagicMock()
        n.properties = dict(props)
        return n

    def _set_storage(self, agent, node):
        storage = MagicMock()
        storage.get_node = AsyncMock(return_value=node)
        storage.add_node = AsyncMock()
        agent.storage = storage
        return storage

    @pytest.mark.asyncio
    async def test_present_and_anchor_matches_verifies(self, tmp_path):
        agent, sha = self._agent_with_overlay(tmp_path)
        self._set_storage(agent, self._node({"constitution_overlay_hash": sha}))
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is True and agent.constitution_overlay_verified is True

    @pytest.mark.asyncio
    async def test_present_but_unanchored_fails_closed(self, tmp_path):
        agent, _ = self._agent_with_overlay(tmp_path)
        self._set_storage(agent, self._node({}))  # no anchor
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is False and agent.constitution_overlay_verified is False
        assert "not anchored" in msg.lower()

    @pytest.mark.asyncio
    async def test_present_but_mutated_fails_closed(self, tmp_path):
        agent, _ = self._agent_with_overlay(tmp_path)
        self._set_storage(agent, self._node({"constitution_overlay_hash": "deadbeef"}))
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is False and agent.constitution_overlay_verified is False
        assert "modified" in msg.lower()

    @pytest.mark.asyncio
    async def test_anchored_but_overlay_removed_fails_closed(self, tmp_path):
        # No overlay file → sha is None, but an anchor exists → tampering.
        agent_dir = tmp_path / "removed"
        agent_dir.mkdir()
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent._constitution_overlay_sha is None
        self._set_storage(agent, self._node({"constitution_overlay_hash": "abc123"}))
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is False and "missing" in msg.lower()

    @pytest.mark.asyncio
    async def test_no_overlay_no_anchor_is_ok(self, tmp_path):
        agent_dir = tmp_path / "plain"
        agent_dir.mkdir()
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        self._set_storage(agent, self._node({}))
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is True and agent.constitution_overlay_verified is False

    @pytest.mark.asyncio
    async def test_live_mutation_detected_on_reverify(self, tmp_path):
        """#1722 P2: the audit re-reads the overlay from disk, so a file mutated
        WHILE the agent runs flips verification to failed (not stuck on the
        __init__ hash)."""
        agent, sha = self._agent_with_overlay(tmp_path)
        self._set_storage(agent, self._node({"constitution_overlay_hash": sha}))
        ok, _ = await agent.verify_constitution_overlay()
        assert ok is True and agent.constitution_overlay_verified is True
        # Attacker rewrites the overlay at runtime to add a grant.
        (tmp_path / "agent" / "CONSTITUTION.md").write_text(
            "### Amendment IX\n- [x] shell_execution_host\n- [x] filesystem_write\n",
            encoding="utf-8",
        )
        ok2, msg = await agent.verify_constitution_overlay()
        assert ok2 is False and agent.constitution_overlay_verified is False
        assert "modified" in msg.lower()

    @pytest.mark.asyncio
    async def test_live_removal_detected_on_reverify(self, tmp_path):
        """An anchored overlay deleted at runtime is detected as tampering."""
        agent, sha = self._agent_with_overlay(tmp_path)
        self._set_storage(agent, self._node({"constitution_overlay_hash": sha}))
        assert (await agent.verify_constitution_overlay())[0] is True
        (tmp_path / "agent" / "CONSTITUTION.md").unlink()
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is False and "missing" in msg.lower()
        assert agent.constitution_text is None

    @pytest.mark.asyncio
    async def test_non_utf8_overlay_fails_closed(self, tmp_path):
        """#1722 codex r3: an overlay rewritten to non-UTF-8 bytes must fail
        closed (hash mismatch), not raise UnicodeDecodeError past the check."""
        agent, sha = self._agent_with_overlay(tmp_path)
        self._set_storage(agent, self._node({"constitution_overlay_hash": sha}))
        assert (await agent.verify_constitution_overlay())[0] is True
        # Attacker writes invalid UTF-8.
        (tmp_path / "agent" / "CONSTITUTION.md").write_bytes(b"\xff\xfe\x00bad")
        ok, msg = await agent.verify_constitution_overlay()
        assert ok is False and agent.constitution_overlay_verified is False
        assert agent.constitution_text is None

    @pytest.mark.asyncio
    async def test_anchor_constitution_overlay_persists_hash(self, tmp_path):
        agent, sha = self._agent_with_overlay(tmp_path)
        node = self._node({})
        storage = self._set_storage(agent, node)
        ok, msg = await agent.anchor_constitution_overlay()
        assert ok is True
        assert node.properties["constitution_overlay_hash"] == sha
        storage.add_node.assert_awaited()
        assert agent.constitution_overlay_verified is True
