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
    must use ``agent.constitution_text`` when set, regardless of what's in
    the package constitution. This is the consumer side that #898 unblocks.
    """

    def test_overlay_grants_take_precedence_over_package(self, tmp_path):
        from kestrel_sovereign.features.computer_use.feature import (
            ComputerUseFeature,
        )

        agent_dir = tmp_path / "emma_like"
        agent_dir.mkdir()
        (agent_dir / "CONSTITUTION.md").write_text(
            "### Amendment IX\n- [x] shell_execution_host\n"
        )

        agent = _make_agent(agent_dir / "kestrel_prime.db")
        feature = ComputerUseFeature(agent=agent)
        # Don't await initialize; we only need the lookup helper to read
        # ``self.agent.constitution_text``, which __init__ already populated.
        granted = feature._granted_capabilities()
        assert "shell_execution_host" in granted
