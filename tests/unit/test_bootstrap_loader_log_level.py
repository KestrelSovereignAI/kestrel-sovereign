"""Log-level regression for #659.

``DEFAULT_BOOTSTRAP_FILES`` is a list of **optional** context files
(SOUL.md, AGENTS.md, IDENTITY.md, ...). A fresh agent has none of them,
so logging a WARNING per missing file produced a wall of red on first
run. These tests lock in the correct severities:

  * Missing optional file → DEBUG (expected + harmless).
  * One INFO summary naming what loaded + how many were absent.
  * Failed-to-read (file exists but corrupt) → still WARNING.
  * Budget exhausted → still WARNING.
"""
from __future__ import annotations

import logging
from pathlib import Path

from kestrel_sovereign.features.bootstrap.loader import BootstrapLoader


class TestMissingOptionalFilesAreDebugNotWarning:

    def test_fresh_agent_produces_no_warnings(self, tmp_path: Path, caplog):
        """Empty agent_data dir → every default file missing → DEBUG log
        lines only. The summary INFO still fires so ops has signal that
        bootstrap ran."""
        agent_dir = tmp_path / "fresh_agent"
        agent_dir.mkdir()

        loader = BootstrapLoader(agent_data_path=str(agent_dir))
        with caplog.at_level(logging.DEBUG, logger=loader.__module__):
            loader.load()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings == [], (
            f"A fresh agent produced {len(warnings)} WARNING lines from "
            f"optional-file lookups — should be DEBUG. See #659. "
            f"Messages: {[w.getMessage() for w in warnings]}"
        )

        # Confirm the DEBUG lines actually fired (the lookups ran)
        debug_msgs = [
            r.getMessage() for r in caplog.records if r.levelname == "DEBUG"
        ]
        assert any("not found in any search path" in m for m in debug_msgs), (
            "Expected DEBUG 'not found' lines for each missing optional file."
        )

    def test_summary_info_reports_missing_count(self, tmp_path: Path, caplog):
        """Fresh agent → one INFO summary line with the missing count so
        users see at a glance what bootstrap did without reading DEBUG."""
        agent_dir = tmp_path / "fresh_agent"
        agent_dir.mkdir()

        loader = BootstrapLoader(agent_data_path=str(agent_dir))
        with caplog.at_level(logging.INFO, logger=loader.__module__):
            loader.load()

        info_msgs = [
            r.getMessage() for r in caplog.records if r.levelname == "INFO"
        ]
        summary = next((m for m in info_msgs if m.startswith("Bootstrap:")), None)
        assert summary is not None, f"No 'Bootstrap:' summary INFO line. Got: {info_msgs}"
        assert "0 loaded" in summary
        assert "not present" in summary, (
            f"Summary should name the missing count. Got: {summary!r}"
        )

    def test_partial_load_still_logs_summary(self, tmp_path: Path, caplog):
        """Agent with some files → INFO 'Loaded bootstrap file' per hit
        plus one summary. No WARNINGs for the files that aren't present."""
        agent_dir = tmp_path / "partial_agent"
        agent_dir.mkdir()
        (agent_dir / "SOUL.md").write_text("# I am Claw.\n")
        (agent_dir / "IDENTITY.md").write_text("did:...\n")

        loader = BootstrapLoader(agent_data_path=str(agent_dir))
        with caplog.at_level(logging.DEBUG, logger=loader.__module__):
            loader.load()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings == [], (
            f"Partial-load produced {len(warnings)} WARNINGs — should be "
            f"0. Messages: {[w.getMessage() for w in warnings]}"
        )

        info_msgs = [
            r.getMessage() for r in caplog.records if r.levelname == "INFO"
        ]
        summary = next((m for m in info_msgs if m.startswith("Bootstrap:")), None)
        assert summary is not None
        assert "2 loaded" in summary
        assert "SOUL.md" in summary or "IDENTITY.md" in summary
        assert "not present" in summary  # some were still absent

    def test_unreadable_file_still_warns(self, tmp_path: Path, caplog, monkeypatch):
        """File present but can't be read → WARNING stays. We're only
        softening MISSING optional files, not actual failures."""
        agent_dir = tmp_path / "broken_agent"
        agent_dir.mkdir()
        soul_path = agent_dir / "SOUL.md"
        soul_path.write_text("real content")

        def boom(*_args, **_kwargs):
            raise OSError("disk error")

        monkeypatch.setattr(Path, "read_text", boom)

        loader = BootstrapLoader(agent_data_path=str(agent_dir))
        with caplog.at_level(logging.WARNING, logger=loader.__module__):
            loader.load()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "Failed to load" in w.getMessage() and "SOUL.md" in w.getMessage()
            for w in warnings
        ), (
            f"Unreadable file should still produce a WARNING. Got: "
            f"{[w.getMessage() for w in warnings]}"
        )
