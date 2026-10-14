"""STRATEGY.yaml edits made while the agent runs must survive its writes (#2273).

The feature reads STRATEGY.yaml once at start. Every decision it recorded then
dumped that whole in-memory copy back over the file, so an edit an operator
made meanwhile was silently erased. On 2026-07-08 an approved
``morning_signal_config.scan_repos`` was added to Emma's STRATEGY.yaml by hand,
and Emma had to avoid every strategy write until a restart re-read it.

A write now goes ahead only over the file the feature last loaded or wrote. A
changed file is re-read and only the call's own change is applied to it; a
file that cannot be read, or that no longer has room for the change, is left
untouched and the write refused. The file is replaced atomically, so a failed
write cannot leave a truncated STRATEGY.yaml.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory import feature as feature_module
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    LEDGER_FILENAME,
)

FEATURE_LOGGER = "kestrel_sovereign.features.strategic_memory.feature"

_BRIEF = {
    "version": 1,
    "vision": "Ship the sovereign agent.",
    "milestones": [{"name": "M1", "status": "in_progress"}],
    "stakeholders": [],
    "decisions": [
        {
            "date": "2026-07-01",
            "session": "1",
            "decision": "keep STRATEGY.yaml canonical",
            "rationale": "it travels with the agent",
            "impact": "",
        }
    ],
}


async def _feature(tmp_path: Path, brief=None) -> StrategicMemoryFeature:
    """A feature started over ``tmp_path`` the way an agent starts it."""
    if brief is not None:
        (tmp_path / "STRATEGY.yaml").write_text(yaml.dump(brief), encoding="utf-8")
    agent = MagicMock()
    agent.agent_id = "did:test:2273"
    agent.agent_data_dir = str(tmp_path)
    agent.storage = None
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    return feature


def _strategy(tmp_path: Path) -> Path:
    return tmp_path / "STRATEGY.yaml"


def _edit_on_disk(tmp_path: Path, edit) -> None:
    """Change STRATEGY.yaml the way an operator does: outside the feature."""
    path = _strategy(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")


def _on_disk(tmp_path: Path) -> dict:
    return yaml.safe_load(_strategy(tmp_path).read_text(encoding="utf-8"))


def _decision_texts(data: dict) -> list:
    return [d["decision"] for d in data.get("decisions") or []]


def _staging_files(tmp_path: Path) -> list:
    return [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------------------
# External edits survive later writes
# ---------------------------------------------------------------------------


class TestExternalEditsSurviveWrites:
    @pytest.mark.asyncio
    async def test_a_top_level_key_added_after_start_survives_a_decision(
        self, tmp_path, caplog
    ):
        """The issue's regression case: the scan_repos key Emma had to protect."""
        feature = await _feature(tmp_path, _BRIEF)
        _edit_on_disk(
            tmp_path,
            lambda d: d.update(
                {"morning_signal_config": {"scan_repos": ["owner/repo"]}}
            ),
        )

        with caplog.at_level(logging.INFO, logger=FEATURE_LOGGER):
            result = await feature.strategy_add_decision(
                decision="adopt durable leases", rationale="cron drops them"
            )

        assert result.status is ToolResultStatus.OK
        assert result.data["persisted"] is True
        assert result.data["reloaded_from_disk"] is True
        on_disk = _on_disk(tmp_path)
        assert on_disk["morning_signal_config"] == {"scan_repos": ["owner/repo"]}, (
            "the key added on disk after start must survive the write"
        )
        assert _decision_texts(on_disk) == [
            "keep STRATEGY.yaml canonical",
            "adopt durable leases",
        ]
        assert on_disk["vision"] == _BRIEF["vision"]
        # The in-memory brief is now the file, not the copy read at start.
        assert feature._data["morning_signal_config"] == {
            "scan_repos": ["owner/repo"]
        }
        assert any(
            "changed on disk" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        ), "a reload-and-merge must be logged at INFO with its reason"

    @pytest.mark.asyncio
    async def test_an_edit_to_an_existing_entry_survives_a_blocker_and_the_next_write(
        self, tmp_path
    ):
        """An edited entry stays edited, and the new blocker is recorded too.

        ``strategy_add_blocker`` writes STRATEGY_LEDGER.yaml, not STRATEGY.yaml
        (#2954), so it must leave the edit alone; the brief write that follows
        it is the one that used to put the stale entry back.
        """
        feature = await _feature(tmp_path, _BRIEF)

        def edit(data):
            data["milestones"][0]["status"] = "done"
            data["decisions"][0]["rationale"] = "corrected by the operator"

        _edit_on_disk(tmp_path, edit)

        blocker = await feature.strategy_add_blocker(
            issue="#7", title="waiting on infra"
        )
        assert blocker.status is ToolResultStatus.OK
        assert _on_disk(tmp_path)["milestones"][0]["status"] == "done"

        decision = await feature.strategy_add_decision(
            decision="ship M2 next", rationale="M1 is done"
        )

        assert decision.status is ToolResultStatus.OK
        assert decision.data["persisted"] is True
        on_disk = _on_disk(tmp_path)
        assert on_disk["milestones"][0]["status"] == "done"
        assert on_disk["decisions"][0]["rationale"] == "corrected by the operator"
        assert _decision_texts(on_disk) == [
            "keep STRATEGY.yaml canonical",
            "ship M2 next",
        ]
        ledger = yaml.safe_load(
            (tmp_path / LEDGER_FILENAME).read_text(encoding="utf-8")
        )
        assert [b["title"] for b in ledger[BLOCKERS_KEY]] == ["waiting on infra"]

    @pytest.mark.asyncio
    async def test_a_decision_added_on_disk_is_kept_beside_the_agents_own(
        self, tmp_path
    ):
        """Both writers append to the same list; neither row may be lost."""
        feature = await _feature(tmp_path, _BRIEF)
        _edit_on_disk(
            tmp_path,
            lambda d: d["decisions"].append(
                {"date": "2026-07-08", "decision": "added by the operator"}
            ),
        )

        result = await feature.strategy_add_decision(
            decision="added by the agent", rationale="r"
        )

        assert result.data["persisted"] is True
        assert _decision_texts(_on_disk(tmp_path)) == [
            "keep STRATEGY.yaml canonical",
            "added by the operator",
            "added by the agent",
        ]

    @pytest.mark.asyncio
    async def test_a_file_created_at_start_is_protected_too(self, tmp_path):
        """The default template the feature writes is also a file it last wrote."""
        feature = await _feature(tmp_path)
        assert _strategy(tmp_path).exists()
        _edit_on_disk(tmp_path, lambda d: d.update({"vision": "Set by hand."}))

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.data["persisted"] is True
        on_disk = _on_disk(tmp_path)
        assert on_disk["vision"] == "Set by hand."
        assert _decision_texts(on_disk) == ["d"]


# ---------------------------------------------------------------------------
# An unchanged file is written exactly as before
# ---------------------------------------------------------------------------


class TestUnchangedDiskWritesAsBefore:
    @pytest.mark.asyncio
    async def test_a_transient_read_failure_does_not_log_an_edit_that_never_happened(
        self, tmp_path, caplog
    ):
        """The reload runs, but the file is unchanged: no "changed on disk" line."""
        feature = await _feature(tmp_path, _BRIEF)
        real_read = feature_module._read_strategy_bytes
        reads = []

        def flaky_read(path):
            reads.append(path)
            if len(reads) == 1:
                raise OSError(5, "Input/output error")
            return real_read(path)

        with caplog.at_level(logging.INFO, logger=FEATURE_LOGGER):
            with patch.object(feature_module, "_read_strategy_bytes", flaky_read):
                result = await feature.strategy_add_decision(
                    decision="d", rationale="r"
                )

        assert result.status is ToolResultStatus.OK
        assert _decision_texts(_on_disk(tmp_path))[-1] == "d"
        assert not any("changed on disk" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_two_writes_in_a_row_raise_no_false_conflict(self, tmp_path, caplog):
        feature = await _feature(tmp_path, _BRIEF)

        with caplog.at_level(logging.INFO, logger=FEATURE_LOGGER):
            first = await feature.strategy_add_decision(decision="one", rationale="r")
            second = await feature.strategy_add_decision(decision="two", rationale="r")

        for result in (first, second):
            assert result.status is ToolResultStatus.OK
            assert result.data["persisted"] is True
            assert "reloaded_from_disk" not in result.data
        assert not any("changed on disk" in r.getMessage() for r in caplog.records)
        assert _decision_texts(_on_disk(tmp_path)) == [
            "keep STRATEGY.yaml canonical",
            "one",
            "two",
        ]

    @pytest.mark.asyncio
    async def test_the_file_holds_the_same_bytes_the_old_write_produced(
        self, tmp_path
    ):
        feature = await _feature(tmp_path, _BRIEF)

        await feature.strategy_add_decision(decision="one", rationale="r")

        expected = yaml.dump(
            feature._data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )
        assert _strategy(tmp_path).read_bytes() == expected.encode("utf-8")
        assert _staging_files(tmp_path) == []


# ---------------------------------------------------------------------------
# A file that cannot take the write is left untouched
# ---------------------------------------------------------------------------


class TestRefusedWritesLeaveTheFileUntouched:
    @pytest.mark.asyncio
    async def test_an_unparseable_file_is_refused_not_overwritten(self, tmp_path):
        feature = await _feature(tmp_path, _BRIEF)
        broken = b"vision: [unclosed\nmorning_signal_config: {scan_repos: x\n"
        _strategy(tmp_path).write_bytes(broken)

        result = await feature.strategy_add_decision(
            decision="never written", rationale="the file is mid-edit"
        )

        assert result.status is ToolResultStatus.ERROR
        assert result.data["persisted"] is False
        assert result.data["recorded"] is False
        assert result.data["conflict"] is True
        assert str(_strategy(tmp_path)) in result.error
        assert _strategy(tmp_path).read_bytes() == broken
        assert "never written" not in _decision_texts(feature._data), (
            "a refused write records nothing, in memory either"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "broken",
        [
            pytest.param(b"vision: Ship it\ndue: 2026-02-30\n", id="impossible-date"),
            pytest.param(b"vision: !!timestamp 'soon'\n", id="malformed-timestamp"),
            pytest.param(b"count: !!int 'many'\n", id="malformed-int"),
            pytest.param(b"x: " + b"[" * 5000 + b"]" * 5000 + b"\n", id="deep-nesting"),
            pytest.param(b"vision: \xff\xfe\n", id="not-utf8"),
        ],
    )
    async def test_a_file_the_loader_raises_on_is_refused_not_raised(
        self, tmp_path, broken
    ):
        """PyYAML's safe loader raises more than YAMLError on hand edits.

        ValueError, AttributeError and RecursionError must become the same
        refusal as a syntax error, not escape the tool.
        """
        feature = await _feature(tmp_path, _BRIEF)
        _strategy(tmp_path).write_bytes(broken)

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        assert str(_strategy(tmp_path)) in result.error
        assert _strategy(tmp_path).read_bytes() == broken

    @pytest.mark.asyncio
    async def test_once_repaired_the_next_write_keeps_the_repair(self, tmp_path):
        feature = await _feature(tmp_path, _BRIEF)
        _strategy(tmp_path).write_bytes(b"vision: [unclosed\n")
        await feature.strategy_add_decision(decision="refused", rationale="r")

        repaired = dict(_BRIEF, vision="Repaired by hand.")
        _strategy(tmp_path).write_text(yaml.dump(repaired), encoding="utf-8")
        result = await feature.strategy_add_decision(decision="later", rationale="r")

        assert result.data["persisted"] is True
        on_disk = _on_disk(tmp_path)
        assert on_disk["vision"] == "Repaired by hand."
        assert _decision_texts(on_disk) == ["keep STRATEGY.yaml canonical", "later"]

    @pytest.mark.asyncio
    async def test_a_file_unparseable_at_start_is_never_overwritten(self, tmp_path):
        """A failed load used to leave ``{}`` in memory and write it over the file."""
        broken = b"decisions: [unclosed\n"
        _strategy(tmp_path).write_bytes(broken)
        feature = await _feature(tmp_path)

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["persisted"] is False
        assert _strategy(tmp_path).read_bytes() == broken

    @pytest.mark.asyncio
    async def test_a_file_that_is_no_longer_a_mapping_is_refused(self, tmp_path):
        feature = await _feature(tmp_path, _BRIEF)
        replaced = b"- just\n- a list\n"
        _strategy(tmp_path).write_bytes(replaced)

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.ERROR
        assert "not a mapping" in result.error
        assert _strategy(tmp_path).read_bytes() == replaced

    @pytest.mark.asyncio
    async def test_a_change_that_no_longer_applies_is_refused(self, tmp_path):
        """The edit turned ``decisions`` into a mapping: there is nowhere to append."""
        feature = await _feature(tmp_path, _BRIEF)
        _edit_on_disk(
            tmp_path, lambda d: d.update({"decisions": {"note": "reorganised"}})
        )
        before = _strategy(tmp_path).read_bytes()

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        assert "'decisions' section is a dict" in result.error
        assert _strategy(tmp_path).read_bytes() == before

    @pytest.mark.asyncio
    async def test_a_section_malformed_at_start_is_refused_while_unrepaired(
        self, tmp_path
    ):
        malformed = dict(_BRIEF, decisions={"note": "not a list"})
        feature = await _feature(tmp_path, malformed)
        before = _strategy(tmp_path).read_bytes()

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["recorded"] is False
        assert "'decisions' section is a dict" in result.error
        assert _strategy(tmp_path).read_bytes() == before
        assert feature._data["decisions"] == {"note": "not a list"}

    @pytest.mark.asyncio
    async def test_a_section_malformed_at_start_and_repaired_needs_no_restart(
        self, tmp_path
    ):
        """The copy read at start has no room for the decision; the file does now."""
        feature = await _feature(tmp_path, dict(_BRIEF, decisions={"note": "x"}))
        _edit_on_disk(tmp_path, lambda d: d.update({"decisions": []}))

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.OK
        assert result.data["reloaded_from_disk"] is True
        assert _decision_texts(_on_disk(tmp_path)) == ["d"]

    @pytest.mark.asyncio
    async def test_a_removed_file_is_not_recreated_from_the_stale_copy(
        self, tmp_path
    ):
        feature = await _feature(tmp_path, _BRIEF)
        _strategy(tmp_path).unlink()

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.ERROR
        assert "was removed" in result.error
        assert not _strategy(tmp_path).exists()

    @pytest.mark.asyncio
    async def test_an_edit_landing_during_the_reload_is_not_overwritten(
        self, tmp_path
    ):
        """A second edit between the re-read and the write wins; this call refuses.

        The refusal must not blame "the in-memory copy": what it declined to
        write was the re-read file plus this change.
        """
        feature = await _feature(tmp_path, _BRIEF)
        _edit_on_disk(tmp_path, lambda d: d.update({"first": "edit"}))
        real_read = feature_module._read_strategy_bytes
        reads = []

        def read_with_a_racing_edit(path):
            reads.append(path)
            if len(reads) == 3:  # the write's own check, after the re-read
                _edit_on_disk(tmp_path, lambda d: d.update({"second": "edit"}))
            return real_read(path)

        with patch.object(
            feature_module, "_read_strategy_bytes", read_with_a_racing_edit
        ):
            result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert len(reads) == 3
        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        assert "in-memory copy" not in result.error
        on_disk = _on_disk(tmp_path)
        assert on_disk["second"] == "edit"
        assert "d" not in _decision_texts(on_disk)

    @pytest.mark.asyncio
    async def test_a_save_with_no_change_to_reapply_refuses_an_edited_file(
        self, tmp_path
    ):
        """``_save()`` alone (migration, template) must never overwrite an edit."""
        feature = await _feature(tmp_path, _BRIEF)
        _edit_on_disk(tmp_path, lambda d: d.update({"added": "by hand"}))
        before = _strategy(tmp_path).read_bytes()

        outcome = feature._save()

        assert outcome.persisted is False
        assert outcome.refused is True
        assert "changed on disk" in outcome.error
        assert _strategy(tmp_path).read_bytes() == before


# ---------------------------------------------------------------------------
# A failed write cannot leave a partial file
# ---------------------------------------------------------------------------


class TestFailedWritesLeaveTheOriginalIntact:
    @pytest.mark.asyncio
    async def test_a_write_that_fails_part_way_leaves_the_original(self, tmp_path):
        feature = await _feature(tmp_path, _BRIEF)
        original = _strategy(tmp_path).read_bytes()

        def half_write(self, data, *args, **kwargs):
            with open(self, "w", encoding="utf-8") as handle:
                handle.write(data[: len(data) // 2])
            raise OSError(28, "No space left on device")

        with patch.object(Path, "write_text", half_write):
            result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["persisted"] is False
        assert "No space left on device" in result.error
        assert _strategy(tmp_path).read_bytes() == original
        assert _staging_files(tmp_path) == []

    @pytest.mark.asyncio
    async def test_a_failed_replace_leaves_the_original(self, tmp_path):
        feature = await _feature(tmp_path, _BRIEF)
        original = _strategy(tmp_path).read_bytes()

        with patch.object(os, "replace", side_effect=OSError("device busy")):
            result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.data["persisted"] is False
        assert "device busy" in result.error
        assert _strategy(tmp_path).read_bytes() == original
        assert _staging_files(tmp_path) == []

    @pytest.mark.asyncio
    async def test_a_failed_write_is_no_conflict_for_the_next_one(self, tmp_path):
        """The file still holds what was last written, so the next write proceeds,
        carrying the change the failed one held in memory (the PARTIAL contract)."""
        feature = await _feature(tmp_path, _BRIEF)
        with patch.object(os, "replace", side_effect=OSError("device busy")):
            await feature.strategy_add_decision(decision="held", rationale="r")

        result = await feature.strategy_add_decision(decision="next", rationale="r")

        assert result.status is ToolResultStatus.OK
        assert "reloaded_from_disk" not in result.data
        assert _decision_texts(_on_disk(tmp_path)) == [
            "keep STRATEGY.yaml canonical",
            "held",
            "next",
        ]

    @pytest.mark.asyncio
    async def test_a_write_that_fails_after_a_reload_keeps_memory_current(
        self, tmp_path
    ):
        """Memory keeps the reloaded file plus this change, not the stale copy."""
        feature = await _feature(tmp_path, _BRIEF)
        _edit_on_disk(tmp_path, lambda d: d.update({"added": "by hand"}))

        with patch.object(os, "replace", side_effect=OSError("device busy")):
            failed = await feature.strategy_add_decision(decision="held", rationale="r")

        assert failed.status is ToolResultStatus.PARTIAL
        assert failed.data["reloaded_from_disk"] is True
        assert feature._data["added"] == "by hand"
        assert "held" in _decision_texts(feature._data)

        result = await feature.strategy_add_decision(decision="next", rationale="r")

        assert result.status is ToolResultStatus.OK
        on_disk = _on_disk(tmp_path)
        assert on_disk["added"] == "by hand"
        assert _decision_texts(on_disk) == [
            "keep STRATEGY.yaml canonical",
            "held",
            "next",
        ]


# ---------------------------------------------------------------------------
# The replacement keeps what the in-place write kept
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
class TestReplacementKeepsTheFilesShape:
    @pytest.mark.asyncio
    async def test_the_files_permission_bits_are_kept(self, tmp_path):
        feature = await _feature(tmp_path, _BRIEF)
        _strategy(tmp_path).chmod(0o640)

        await feature.strategy_add_decision(decision="d", rationale="r")

        assert _strategy(tmp_path).stat().st_mode & 0o777 == 0o640

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root may write a read-only file, in place or not",
    )
    async def test_a_read_only_file_is_not_replaced(self, tmp_path):
        """The in-place write failed on a read-only file; replacing it must too.

        ``os.replace`` only needs the directory to be writable, so without the
        check it would swap a file the operator made read-only.
        """
        feature = await _feature(tmp_path, _BRIEF)
        original = _strategy(tmp_path).read_bytes()
        _strategy(tmp_path).chmod(0o444)
        try:
            result = await feature.strategy_add_decision(decision="d", rationale="r")

            assert result.status is ToolResultStatus.PARTIAL
            assert result.data["persisted"] is False
            assert _strategy(tmp_path).read_bytes() == original
            assert _strategy(tmp_path).stat().st_mode & 0o777 == 0o444
            assert _staging_files(tmp_path) == []
        finally:
            _strategy(tmp_path).chmod(0o644)

    @pytest.mark.asyncio
    async def test_a_symlinked_file_is_written_through_not_replaced(self, tmp_path):
        target_dir = tmp_path / "checkout"
        target_dir.mkdir()
        target = target_dir / "STRATEGY.yaml"
        target.write_text(yaml.dump(_BRIEF), encoding="utf-8")
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "STRATEGY.yaml").symlink_to(target)
        feature = await _feature(agent_dir)

        result = await feature.strategy_add_decision(decision="d", rationale="r")

        assert result.data["persisted"] is True
        assert (agent_dir / "STRATEGY.yaml").is_symlink()
        assert "d" in _decision_texts(
            yaml.safe_load(target.read_text(encoding="utf-8"))
        )
