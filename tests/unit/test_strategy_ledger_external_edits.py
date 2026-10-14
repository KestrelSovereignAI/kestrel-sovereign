"""STRATEGY_LEDGER.yaml edits made while the agent runs must survive its writes (#3636).

The ledger is read once at start, and every pattern/blocker tool used to dump
that whole in-memory copy back over the file -- written in place, so a failed
write could also leave it truncated. An operator's edit to a pattern or a
blocker was silently erased by the next ``strategy_add_pattern``, the same
damage #2273 fixed for STRATEGY.yaml.

A write now goes ahead only over the bytes the ledger last loaded or wrote. A
changed file is re-read and only the calling tool's own change is applied to
it; a file that cannot be read, or that no longer takes the change, is left
untouched and the write refused. The file is replaced atomically.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory import feature as feature_module
from kestrel_sovereign.features.strategic_memory import ledger as ledger_module
from kestrel_sovereign.features.strategic_memory.atomic_file import (
    file_digest,
    read_file_bytes,
    replace_atomically,
)
from kestrel_sovereign.features.strategic_memory.blocker_reconcile import (
    RECONCILIATION_KEY,
)
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    LEDGER_FILENAME,
    PATTERNS_KEY,
    LedgerChangeConflict,
    StrategyLedger,
    pattern_row_id,
)

LEDGER_LOGGER = "kestrel_sovereign.features.strategic_memory.ledger"

_LEDGER = {
    "version": 1,
    PATTERNS_KEY: [
        {
            "id": "pat_old",
            "pattern": "Retries hide flaky dependencies",
            "source": "#100",
            "implication": "fix the dependency",
            "recorded_at": "2026-07-01",
        },
        {
            "id": "pat_keep",
            "pattern": "Small PRs merge faster",
            "source": "#101",
            "implication": "split work",
            "recorded_at": "2026-07-02",
        },
    ],
    BLOCKERS_KEY: [
        {
            "id": "blk_infra",
            "issue": "#7",
            "title": "waiting on infra",
            "severity": "high",
            "owner": "ops",
            "repo": "",
            "blocked_since": "2026-07-03",
            "notes": "",
        },
        {
            "id": "blk_review",
            "issue": "#8",
            "title": "waiting on review",
            "severity": "medium",
            "owner": "me",
            "repo": "",
            "blocked_since": "2026-07-04",
            "notes": "",
        },
    ],
}


def _ledger_path(tmp_path: Path) -> Path:
    return tmp_path / LEDGER_FILENAME


async def _feature(tmp_path: Path, ledger=_LEDGER, strategy=None):
    """A feature started over ``tmp_path`` the way an agent starts it."""
    if ledger is not None:
        _ledger_path(tmp_path).write_text(
            yaml.dump(ledger, sort_keys=False), encoding="utf-8"
        )
    (tmp_path / "STRATEGY.yaml").write_text(
        yaml.dump(strategy or {"version": 1, "vision": "v"}), encoding="utf-8"
    )
    agent = MagicMock()
    agent.agent_id = "did:test:3636"
    agent.agent_data_dir = str(tmp_path)
    agent.storage = None
    feature = StrategicMemoryFeature(agent)
    await feature.initialize()
    return feature


def _edit_on_disk(tmp_path: Path, edit) -> None:
    """Change the ledger the way an operator does: outside the feature."""
    path = _ledger_path(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")


def _on_disk(tmp_path: Path) -> dict:
    return yaml.safe_load(_ledger_path(tmp_path).read_text(encoding="utf-8"))


def _by_id(rows) -> dict:
    return {row["id"]: row for row in rows}


def _staging_files(tmp_path: Path) -> list:
    return [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def _old_dump(data) -> bytes:
    """Exactly what the in-place write used to put in the file."""
    return yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    ).encode("utf-8")


_HAND_PATTERN = {
    "id": "pat_by_hand",
    "pattern": "Operators record lessons too",
    "source": "operator",
    "implication": "keep their rows",
    "recorded_at": "2026-07-08",
}

_HAND_BLOCKER = {
    "id": "blk_by_hand",
    "issue": "#9",
    "title": "added by the operator",
    "severity": "low",
    "owner": "ops",
    "repo": "",
    "blocked_since": "2026-07-08",
    "notes": "",
}


# ---------------------------------------------------------------------------
# External edits survive later writes
# ---------------------------------------------------------------------------


class TestExternalEditsSurviveWrites:
    @pytest.mark.asyncio
    async def test_a_pattern_added_on_disk_survives_strategy_add_pattern(
        self, tmp_path, caplog
    ):
        """The issue's regression case."""
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d[PATTERNS_KEY].append(dict(_HAND_PATTERN)))

        with caplog.at_level(logging.INFO, logger=LEDGER_LOGGER):
            result = await feature.strategy_add_pattern(
                pattern="Agents learn too", source="#3636", implication="i"
            )

        assert result.status is ToolResultStatus.OK
        assert result.data["persisted"] is True
        assert result.data["reloaded_from_disk"] is True
        patterns = _by_id(_on_disk(tmp_path)[PATTERNS_KEY])
        assert patterns["pat_by_hand"] == _HAND_PATTERN, (
            "the pattern added on disk after load must survive the write"
        )
        assert result.data["pattern_id"] in patterns
        assert list(patterns) == [
            "pat_old", "pat_keep", "pat_by_hand", result.data["pattern_id"],
        ]
        # The in-memory ledger is now the file, not the copy read at start.
        assert "pat_by_hand" in _by_id(feature._ledger.patterns)
        assert any(
            "changed on disk" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        ), "a reload-and-reapply must be logged at INFO with its reason"

    @pytest.mark.asyncio
    async def test_a_blocker_added_on_disk_survives_resolving_another(
        self, tmp_path
    ):
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d[BLOCKERS_KEY].append(dict(_HAND_BLOCKER)))

        result = await feature.strategy_resolve_blocker(
            issue="blk_review", resolution="reviewed"
        )

        assert result.status is ToolResultStatus.OK
        assert result.data["reloaded_from_disk"] is True
        assert result.data["blocker_id"] == "blk_review"
        blockers = _by_id(_on_disk(tmp_path)[BLOCKERS_KEY])
        assert blockers["blk_by_hand"] == _HAND_BLOCKER
        assert blockers["blk_review"]["resolved_at"]
        assert blockers["blk_review"]["resolution"] == "reviewed"
        assert not blockers["blk_infra"].get("resolved_at")

    @pytest.mark.asyncio
    async def test_supersede_after_an_edit_reapplies_only_the_supersede(
        self, tmp_path
    ):
        """Every other row keeps the file's version, including edited fields."""
        feature = await _feature(tmp_path)

        def edit(data):
            rows = _by_id(data[PATTERNS_KEY])
            rows["pat_keep"]["implication"] = "corrected by the operator"
            rows["pat_old"]["source"] = "#100 (verified)"
            data[PATTERNS_KEY].append(dict(_HAND_PATTERN))
            _by_id(data[BLOCKERS_KEY])["blk_infra"]["notes"] = "ETA Friday"

        _edit_on_disk(tmp_path, edit)
        expected = _on_disk(tmp_path)

        result = await feature.strategy_supersede_pattern(
            pattern_id="pat_old", reason="no longer true", superseded_by="pat_keep"
        )

        assert result.status is ToolResultStatus.OK
        assert result.data["reloaded_from_disk"] is True
        superseded = _by_id(expected[PATTERNS_KEY])["pat_old"]
        superseded.update(
            superseded_at=result.data["pattern"]["superseded_at"],
            superseded_reason="no longer true",
            superseded_by="pat_keep",
        )
        assert _on_disk(tmp_path) == expected, (
            "the write must be the file as edited plus the supersede, nothing else"
        )
        assert result.data["pattern"]["source"] == "#100 (verified)", (
            "the result describes the row as the file now holds it"
        )

    @pytest.mark.asyncio
    async def test_a_blocker_survives_strategy_add_blocker(self, tmp_path):
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d[BLOCKERS_KEY].append(dict(_HAND_BLOCKER)))

        result = await feature.strategy_add_blocker(issue="#10", title="new one")

        assert result.status is ToolResultStatus.OK
        blockers = _by_id(_on_disk(tmp_path)[BLOCKERS_KEY])
        assert blockers["blk_by_hand"] == _HAND_BLOCKER
        assert blockers[result.data["blocker_id"]]["title"] == "new one"

    @pytest.mark.asyncio
    async def test_an_unknown_top_level_key_added_on_disk_survives(self, tmp_path):
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d.update({"operator_notes": "keep me"}))

        result = await feature.strategy_add_pattern(pattern="p")

        assert result.data["persisted"] is True
        assert _on_disk(tmp_path)["operator_notes"] == "keep me"

    @pytest.mark.asyncio
    async def test_a_row_written_by_hand_without_an_id_gets_one(self, tmp_path):
        """The re-read normalizes exactly as a load does."""
        feature = await _feature(tmp_path)
        _edit_on_disk(
            tmp_path,
            lambda d: d[PATTERNS_KEY].append({"pattern": "No id written"}),
        )

        result = await feature.strategy_add_pattern(pattern="p")

        assert result.data["persisted"] is True
        rows = [
            r for r in _on_disk(tmp_path)[PATTERNS_KEY]
            if r["pattern"] == "No id written"
        ]
        assert rows[0]["id"] == pattern_row_id({"pattern": "No id written"})
        assert not feature._ledger.needs_save

    @pytest.mark.asyncio
    async def test_a_pattern_the_operator_also_wrote_gets_its_own_id(self, tmp_path):
        """The id is minted against the file's rows, not the stale copy."""
        feature = await _feature(tmp_path)
        same_text = "Both of us noticed this"
        _edit_on_disk(
            tmp_path,
            lambda d: d[PATTERNS_KEY].append(
                {"id": pattern_row_id({"pattern": same_text}), "pattern": same_text}
            ),
        )

        result = await feature.strategy_add_pattern(pattern=same_text)

        assert result.data["persisted"] is True
        ids = [r["id"] for r in _on_disk(tmp_path)[PATTERNS_KEY]]
        assert len(ids) == len(set(ids)), "two rows must not share an address"
        operators_id = pattern_row_id({"pattern": same_text})
        assert result.data["pattern_id"] == f"{operators_id}-2"
        assert result.data["pattern_id"] in ids
        assert f"(id {result.data['pattern_id']})" in result.confirmation

    @pytest.mark.asyncio
    async def test_reconcile_after_an_edit_keeps_it_and_a_hand_resolution(
        self, tmp_path
    ):
        """A blocker the operator resolved meanwhile keeps their resolution."""
        feature = await _feature(
            tmp_path,
            strategy={
                "version": 1,
                "morning_signal_config": {"scan_repos": ["owner/repo"]},
            },
        )

        real_check = feature_module.check_blockers

        async def check_then_edit(*args, **kwargs):
            # The GitHub check runs before the lock; the edit lands after it.
            report = await real_check(*args, **kwargs)

            def edit(data):
                rows = _by_id(data[BLOCKERS_KEY])
                rows["blk_infra"]["resolved_at"] = "2026-07-09"
                rows["blk_infra"]["resolution"] = "by hand"
                data[BLOCKERS_KEY].append(dict(_HAND_BLOCKER, issue="owner/repo#9"))

            _edit_on_disk(tmp_path, edit)
            return report

        with patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "get_github_token",
            return_value="t",
        ), patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "github_api_get",
            new=AsyncMock(return_value={"state": "closed"}),
        ), patch.object(feature_module, "check_blockers", check_then_edit):
            result = await feature.strategy_reconcile_blockers(apply="yes")

        assert result.status is ToolResultStatus.OK
        assert result.data["reloaded_from_disk"] is True
        assert result.data["resolved_ids"] == ["blk_review"]
        on_disk = _on_disk(tmp_path)
        blockers = _by_id(on_disk[BLOCKERS_KEY])
        assert blockers["blk_infra"]["resolution"] == "by hand"
        assert blockers["blk_review"]["resolved_at"]
        assert "blk_by_hand" in blockers
        assert on_disk[RECONCILIATION_KEY]

    @pytest.mark.asyncio
    async def test_reconcile_does_not_resolve_a_row_now_naming_another_issue(
        self, tmp_path
    ):
        """GitHub's verdict was about the issue the stale copy named.

        The operator corrected blk_review from #8 to #80 on disk; the check
        read the copy loaded at start and found #8 closed. That says nothing
        about #80.
        """
        feature = await _feature(
            tmp_path,
            strategy={
                "version": 1,
                "morning_signal_config": {"scan_repos": ["owner/repo"]},
            },
        )
        _edit_on_disk(
            tmp_path,
            lambda d: _by_id(d[BLOCKERS_KEY])["blk_review"].update(issue="#80"),
        )
        states = {7: "open", 8: "closed", 80: "open"}

        async def lookup(path, token, **kwargs):
            return {"state": states[int(path.rsplit("/", 1)[-1])]}

        with patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "get_github_token",
            return_value="t",
        ), patch(
            "kestrel_sovereign.features.strategic_memory.blocker_reconcile."
            "github_api_get",
            new=lookup,
        ):
            result = await feature.strategy_reconcile_blockers(apply="yes")

        assert result.data["closed_count"] == 1, "precondition: the stale #8 closed"
        assert result.status is ToolResultStatus.OK
        assert result.data["resolved_ids"] == []
        row = _by_id(_on_disk(tmp_path)[BLOCKERS_KEY])["blk_review"]
        assert row["issue"] == "#80"
        assert not row.get("resolved_at")

    @pytest.mark.asyncio
    async def test_a_reloaded_write_confirms_the_files_rows_for_projection(
        self, tmp_path
    ):
        """The snapshot handed to the assertion producer is the file as written."""
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d[PATTERNS_KEY].append(dict(_HAND_PATTERN)))
        projected = []

        async def capture(snapshot):
            projected.append(snapshot)
            return {}

        with patch.object(feature, "_project_ledger_assertions", capture):
            await feature.strategy_add_pattern(pattern="p")

        (snapshot,) = projected
        assert snapshot.confirmed_persisted
        assert "pat_by_hand" in _by_id(snapshot.data[PATTERNS_KEY])
        assert dict(snapshot.data) == _on_disk(tmp_path)


# ---------------------------------------------------------------------------
# An unchanged file is written exactly as before
# ---------------------------------------------------------------------------


class TestUnchangedDiskWritesAsBefore:
    @pytest.mark.asyncio
    async def test_the_file_holds_the_same_bytes_the_old_write_produced(
        self, tmp_path, caplog
    ):
        feature = await _feature(tmp_path)

        with caplog.at_level(logging.INFO, logger=LEDGER_LOGGER):
            first = await feature.strategy_add_pattern(pattern="one")
            second = await feature.strategy_resolve_blocker(issue="#7")
            third = await feature.strategy_supersede_pattern(pattern_id="pat_old")

        for result in (first, second, third):
            assert result.status is ToolResultStatus.OK
            assert result.data["persisted"] is True
            assert "reloaded_from_disk" not in result.data
            assert "conflict" not in result.data
        assert _ledger_path(tmp_path).read_bytes() == _old_dump(feature._ledger.data)
        assert not any("changed on disk" in r.getMessage() for r in caplog.records)
        assert _staging_files(tmp_path) == []

    @pytest.mark.asyncio
    async def test_a_new_ledger_is_created_by_its_first_write(self, tmp_path):
        feature = await _feature(tmp_path, ledger=None)
        assert not _ledger_path(tmp_path).exists()

        result = await feature.strategy_add_pattern(pattern="first")

        assert result.status is ToolResultStatus.OK
        assert "reloaded_from_disk" not in result.data
        assert [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]] == ["first"]

    @pytest.mark.asyncio
    async def test_a_transient_read_failure_does_not_log_an_edit_that_never_happened(
        self, tmp_path, caplog
    ):
        """The check's read fails once; the reload finds the file unchanged."""
        feature = await _feature(tmp_path)
        real_read = ledger_module.read_file_bytes
        reads = []

        def flaky_read(path):
            reads.append(path)
            if len(reads) == 1:
                raise OSError(5, "Input/output error")
            return real_read(path)

        with caplog.at_level(logging.INFO, logger=LEDGER_LOGGER):
            with patch.object(ledger_module, "read_file_bytes", flaky_read):
                result = await feature.strategy_add_pattern(pattern="p")

        assert result.status is ToolResultStatus.OK
        assert "reloaded_from_disk" not in result.data
        assert "p" in [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]]
        assert not any("changed on disk" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_transient_read_failure_still_writes_a_held_change(
        self, tmp_path
    ):
        """The file did not change, so nothing held in memory is dropped."""
        feature = await _feature(tmp_path)
        with patch.object(os, "replace", side_effect=OSError("device busy")):
            held = await feature.strategy_add_pattern(pattern="held")
        assert held.status is ToolResultStatus.PARTIAL
        real_read = ledger_module.read_file_bytes
        reads = []

        def flaky_read(path):
            reads.append(path)
            if len(reads) == 1:
                raise OSError(5, "Input/output error")
            return real_read(path)

        with patch.object(ledger_module, "read_file_bytes", flaky_read):
            result = await feature.strategy_add_pattern(pattern="next")

        assert result.status is ToolResultStatus.OK
        assert [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]][-2:] == [
            "held", "next",
        ]

    @pytest.mark.asyncio
    async def test_a_new_ledgers_first_write_survives_a_transient_read_failure(
        self, tmp_path
    ):
        """No file then, no file now: that is not a removal to refuse."""
        feature = await _feature(tmp_path, ledger=None)
        real_read = ledger_module.read_file_bytes
        reads = []

        def flaky_read(path):
            reads.append(path)
            if len(reads) == 1:
                raise OSError(5, "Input/output error")
            return real_read(path)

        with patch.object(ledger_module, "read_file_bytes", flaky_read):
            result = await feature.strategy_add_pattern(pattern="first")

        assert result.status is ToolResultStatus.OK
        assert [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]] == ["first"]


# ---------------------------------------------------------------------------
# A file that cannot take the write is left untouched
# ---------------------------------------------------------------------------


class TestRefusedWritesLeaveTheFileUntouched:
    @pytest.mark.asyncio
    async def test_an_unparseable_ledger_is_refused_not_overwritten(self, tmp_path):
        feature = await _feature(tmp_path)
        before = feature._ledger.persisted_snapshot
        broken = b"patterns_learned: [unclosed\nblockers: {x\n"
        _ledger_path(tmp_path).write_bytes(broken)

        result = await feature.strategy_add_pattern(pattern="never written")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["persisted"] is False
        assert result.data["recorded"] is False
        assert result.data["conflict"] is True
        assert "could not be parsed as YAML" in result.error
        assert str(_ledger_path(tmp_path)) in result.error
        assert _ledger_path(tmp_path).read_bytes() == broken
        in_memory = [p["pattern"] for p in feature._ledger.patterns]
        assert "never written" not in in_memory, (
            "a refused write records nothing, in memory either"
        )
        assert feature._ledger.persisted_snapshot is before
        assert feature._ledger.readable, (
            "the in-memory ledger is still the last good read; a later repair "
            "needs no restart"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "broken",
        [
            pytest.param(b"- just\n- a list\n", id="not-a-mapping"),
            pytest.param(b"patterns_learned: {a: 1}\n", id="section-not-a-list"),
            pytest.param(b"blockers: [{since: 2026-02-30}]\n", id="impossible-date"),
            pytest.param(b"x: " + b"[" * 5000 + b"]" * 5000 + b"\n", id="deep-nesting"),
            pytest.param(b"patterns_learned: \xff\xfe\n", id="not-utf8"),
        ],
    )
    async def test_every_unreadable_shape_is_refused(self, tmp_path, broken):
        feature = await _feature(tmp_path)
        _ledger_path(tmp_path).write_bytes(broken)

        for call in (
            lambda: feature.strategy_add_pattern(pattern="p"),
            lambda: feature.strategy_add_blocker(issue="#1", title="t"),
            lambda: feature.strategy_supersede_pattern(pattern_id="pat_old"),
            lambda: feature.strategy_resolve_blocker(issue="#7"),
        ):
            result = await call()
            assert result.status is ToolResultStatus.ERROR
            assert result.data["conflict"] is True
            assert _ledger_path(tmp_path).read_bytes() == broken
        assert feature._ledger.data == _LEDGER

    @pytest.mark.asyncio
    async def test_once_repaired_the_next_write_keeps_the_repair(self, tmp_path):
        feature = await _feature(tmp_path)
        _ledger_path(tmp_path).write_bytes(b"patterns_learned: [unclosed\n")
        refused = await feature.strategy_add_pattern(pattern="refused")
        assert refused.status is ToolResultStatus.ERROR

        repaired = dict(_LEDGER, operator_notes="repaired by hand")
        _ledger_path(tmp_path).write_text(yaml.dump(repaired), encoding="utf-8")
        result = await feature.strategy_add_pattern(pattern="later")

        assert result.data["persisted"] is True
        on_disk = _on_disk(tmp_path)
        assert on_disk["operator_notes"] == "repaired by hand"
        assert [p["pattern"] for p in on_disk[PATTERNS_KEY]][-1] == "later"
        assert "refused" not in [p["pattern"] for p in on_disk[PATTERNS_KEY]]

    @pytest.mark.asyncio
    async def test_a_removed_ledger_is_not_recreated_from_the_stale_copy(
        self, tmp_path
    ):
        feature = await _feature(tmp_path)
        _ledger_path(tmp_path).unlink()

        result = await feature.strategy_add_pattern(pattern="p")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        assert "was removed" in result.error
        assert not _ledger_path(tmp_path).exists()

    @pytest.mark.asyncio
    async def test_resolving_a_blocker_the_operator_already_resolved_is_refused(
        self, tmp_path
    ):
        """The copy read at start still calls it active; the file does not."""
        feature = await _feature(tmp_path)

        def edit(data):
            row = _by_id(data[BLOCKERS_KEY])["blk_infra"]
            row["resolved_at"] = "2026-07-09"
            row["resolution"] = "by hand"

        _edit_on_disk(tmp_path, edit)
        before = _ledger_path(tmp_path).read_bytes()

        result = await feature.strategy_resolve_blocker(issue="#7", resolution="mine")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        assert result.data["resolved"] is False
        assert result.data["removed_count"] == 0
        assert "no active blocker matches #7" in result.error
        assert _ledger_path(tmp_path).read_bytes() == before
        assert not _by_id(feature._ledger.blockers)["blk_infra"].get("resolved_at"), (
            "the in-memory change is undone along with the write"
        )

    @pytest.mark.asyncio
    async def test_an_issue_that_now_names_another_row_is_refused(self, tmp_path):
        """Resolving by issue key re-matches; two rows is a refusal, not a guess."""
        feature = await _feature(tmp_path)
        _edit_on_disk(
            tmp_path,
            lambda d: d[BLOCKERS_KEY].append(dict(_HAND_BLOCKER, issue="#7")),
        )
        before = _ledger_path(tmp_path).read_bytes()

        result = await feature.strategy_resolve_blocker(issue="#7")

        assert result.status is ToolResultStatus.ERROR
        assert "2 active blockers now match #7" in result.error
        assert _ledger_path(tmp_path).read_bytes() == before

    @pytest.mark.asyncio
    async def test_an_issue_now_naming_only_a_different_row_is_refused(self, tmp_path):
        """The row the call resolved is gone from the active set; its issue key
        now names an operator's new row, which this call never saw."""
        feature = await _feature(tmp_path)

        def edit(data):
            row = _by_id(data[BLOCKERS_KEY])["blk_infra"]
            row["resolved_at"] = "2026-07-09"
            data[BLOCKERS_KEY].append(dict(_HAND_BLOCKER, issue="#7"))

        _edit_on_disk(tmp_path, edit)
        before = _ledger_path(tmp_path).read_bytes()

        result = await feature.strategy_resolve_blocker(issue="#7")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        assert "#7 now matches blocker blk_by_hand, not blk_infra" in result.error
        assert _ledger_path(tmp_path).read_bytes() == before

    @pytest.mark.asyncio
    async def test_superseding_a_pattern_removed_on_disk_is_refused(self, tmp_path):
        feature = await _feature(tmp_path)
        _edit_on_disk(
            tmp_path,
            lambda d: d.update(
                {PATTERNS_KEY: [r for r in d[PATTERNS_KEY] if r["id"] != "pat_old"]}
            ),
        )
        before = _ledger_path(tmp_path).read_bytes()

        result = await feature.strategy_supersede_pattern(pattern_id="pat_old")

        assert result.status is ToolResultStatus.ERROR
        assert result.data["superseded"] is False
        assert result.data["pattern"] is None
        assert result.data["conflict"] is True
        assert "No pattern found with id: pat_old" in result.error
        assert _ledger_path(tmp_path).read_bytes() == before
        assert not _by_id(feature._ledger.patterns)["pat_old"].get("superseded_at")

    @pytest.mark.asyncio
    async def test_an_edit_landing_during_the_reload_is_not_overwritten(
        self, tmp_path
    ):
        """A second edit between the re-read and the write wins; this call refuses."""
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d.update({"first": "edit"}))
        real_read = ledger_module.read_file_bytes
        reads = []

        def read_with_a_racing_edit(path):
            reads.append(path)
            if len(reads) == 3:  # the write's own check, after the re-read
                _edit_on_disk(tmp_path, lambda d: d.update({"second": "edit"}))
            return real_read(path)

        with patch.object(ledger_module, "read_file_bytes", read_with_a_racing_edit):
            result = await feature.strategy_add_pattern(pattern="p")

        assert len(reads) == 3
        assert result.status is ToolResultStatus.ERROR
        assert result.data["conflict"] is True
        on_disk = _on_disk(tmp_path)
        assert on_disk["second"] == "edit"
        assert "p" not in [r["pattern"] for r in on_disk[PATTERNS_KEY]]
        assert "p" not in [r["pattern"] for r in feature._ledger.patterns]

    @pytest.mark.asyncio
    async def test_a_save_with_no_change_to_reapply_refuses_an_edited_file(
        self, tmp_path
    ):
        """``save()`` alone (migration, normalization) must never overwrite an edit."""
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d.update({"added": "by hand"}))
        before = _ledger_path(tmp_path).read_bytes()

        error = feature._ledger.save()

        assert error is not None
        assert "changed on disk" in error
        assert _ledger_path(tmp_path).read_bytes() == before

    def test_a_ledger_that_failed_to_load_still_refuses_with_its_load_error(
        self, tmp_path
    ):
        _ledger_path(tmp_path).write_bytes(b"patterns_learned: [unclosed\n")
        ledger = StrategyLedger(_ledger_path(tmp_path))
        ledger.load()

        error = ledger.save()

        assert "could not be parsed as YAML" in error
        assert "repair or move it, then restart" in error
        assert _ledger_path(tmp_path).read_bytes() == b"patterns_learned: [unclosed\n"


# ---------------------------------------------------------------------------
# A failed write cannot leave a partial file
# ---------------------------------------------------------------------------


class TestFailedWritesLeaveTheOriginalIntact:
    @pytest.mark.asyncio
    async def test_a_write_that_fails_part_way_leaves_the_original(self, tmp_path):
        feature = await _feature(tmp_path)
        original = _ledger_path(tmp_path).read_bytes()

        def half_write(self, data, *args, **kwargs):
            with open(self, "w", encoding="utf-8") as handle:
                handle.write(data[: len(data) // 2])
            raise OSError(28, "No space left on device")

        with patch.object(Path, "write_text", half_write):
            result = await feature.strategy_add_pattern(pattern="p")

        assert result.status is ToolResultStatus.PARTIAL
        assert result.data["persisted"] is False
        assert "No space left on device" in result.error
        assert _ledger_path(tmp_path).read_bytes() == original
        assert _staging_files(tmp_path) == []

    @pytest.mark.asyncio
    async def test_a_failed_replace_leaves_the_original(self, tmp_path):
        feature = await _feature(tmp_path)
        original = _ledger_path(tmp_path).read_bytes()

        with patch.object(os, "replace", side_effect=OSError("device busy")):
            result = await feature.strategy_resolve_blocker(issue="#7")

        assert result.status is ToolResultStatus.PARTIAL
        assert "device busy" in result.error
        assert _ledger_path(tmp_path).read_bytes() == original
        assert _staging_files(tmp_path) == []

    @pytest.mark.asyncio
    async def test_a_failed_write_is_no_conflict_for_the_next_one(self, tmp_path):
        """The file still holds what was last written, so the next write
        proceeds, carrying the change the failed one held in memory."""
        feature = await _feature(tmp_path)
        with patch.object(os, "replace", side_effect=OSError("device busy")):
            await feature.strategy_add_pattern(pattern="held")

        result = await feature.strategy_add_pattern(pattern="next")

        assert result.status is ToolResultStatus.OK
        assert "reloaded_from_disk" not in result.data
        assert [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]][-2:] == [
            "held", "next",
        ]

    @pytest.mark.asyncio
    async def test_an_edit_drops_a_held_change_and_says_so(self, tmp_path, caplog):
        """Only this call's change is re-applied; the held one is reported lost."""
        feature = await _feature(tmp_path)
        with patch.object(os, "replace", side_effect=OSError("device busy")):
            await feature.strategy_add_pattern(pattern="held")
        _edit_on_disk(tmp_path, lambda d: d[PATTERNS_KEY].append(dict(_HAND_PATTERN)))

        with caplog.at_level(logging.WARNING, logger=LEDGER_LOGGER):
            result = await feature.strategy_add_pattern(pattern="next")

        assert result.status is ToolResultStatus.OK
        patterns = [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]]
        assert _HAND_PATTERN["pattern"] in patterns and "next" in patterns
        assert "held" not in patterns
        assert any(
            "earlier change was dropped" in r.getMessage()
            and r.levelno == logging.WARNING
            for r in caplog.records
        ), "a held change the reload discards must not vanish silently"

    @pytest.mark.asyncio
    async def test_a_held_change_once_written_is_not_reported_dropped(
        self, tmp_path, caplog
    ):
        feature = await _feature(tmp_path)
        with patch.object(os, "replace", side_effect=OSError("device busy")):
            await feature.strategy_add_pattern(pattern="held")
        carried = await feature.strategy_add_pattern(pattern="carries it")
        assert carried.status is ToolResultStatus.OK
        _edit_on_disk(tmp_path, lambda d: d.update({"edited": True}))

        with caplog.at_level(logging.WARNING, logger=LEDGER_LOGGER):
            result = await feature.strategy_add_pattern(pattern="after the edit")

        assert result.data["reloaded_from_disk"] is True
        assert "held" in [r["pattern"] for r in _on_disk(tmp_path)[PATTERNS_KEY]]
        assert not any("dropped" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_write_that_fails_after_a_reload_keeps_memory_current(
        self, tmp_path
    ):
        """Memory keeps the reloaded file plus this change, not the stale copy."""
        feature = await _feature(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d[PATTERNS_KEY].append(dict(_HAND_PATTERN)))

        with patch.object(os, "replace", side_effect=OSError("device busy")):
            failed = await feature.strategy_add_pattern(pattern="held")

        assert failed.status is ToolResultStatus.PARTIAL
        assert failed.data["reloaded_from_disk"] is True
        in_memory = [r["pattern"] for r in feature._ledger.patterns]
        assert _HAND_PATTERN["pattern"] in in_memory and "held" in in_memory

        result = await feature.strategy_add_pattern(pattern="next")

        assert result.status is ToolResultStatus.OK
        assert "reloaded_from_disk" not in result.data
        on_disk = _by_id(_on_disk(tmp_path)[PATTERNS_KEY])
        assert on_disk["pat_by_hand"] == _HAND_PATTERN
        assert [r["pattern"] for r in on_disk.values()][-2:] == ["held", "next"]


# ---------------------------------------------------------------------------
# The replacement keeps what the in-place write kept
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
class TestReplacementKeepsTheFilesShape:
    @pytest.mark.asyncio
    async def test_the_files_permission_bits_are_kept(self, tmp_path):
        feature = await _feature(tmp_path)
        _ledger_path(tmp_path).chmod(0o640)

        await feature.strategy_add_pattern(pattern="p")

        assert _ledger_path(tmp_path).stat().st_mode & 0o777 == 0o640

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root may write a read-only file, in place or not",
    )
    async def test_a_read_only_ledger_is_not_replaced(self, tmp_path):
        feature = await _feature(tmp_path)
        original = _ledger_path(tmp_path).read_bytes()
        _ledger_path(tmp_path).chmod(0o444)
        try:
            result = await feature.strategy_add_pattern(pattern="p")

            assert result.status is ToolResultStatus.PARTIAL
            assert _ledger_path(tmp_path).read_bytes() == original
            assert _staging_files(tmp_path) == []
        finally:
            _ledger_path(tmp_path).chmod(0o644)

    @pytest.mark.asyncio
    async def test_a_symlinked_ledger_is_written_through_not_replaced(self, tmp_path):
        target_dir = tmp_path / "checkout"
        target_dir.mkdir()
        target = target_dir / LEDGER_FILENAME
        target.write_text(yaml.dump(_LEDGER), encoding="utf-8")
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        _ledger_path(agent_dir).symlink_to(target)
        feature = await _feature(agent_dir, ledger=None)

        result = await feature.strategy_add_pattern(pattern="p")

        assert result.data["persisted"] is True
        assert _ledger_path(agent_dir).is_symlink()
        assert "p" in [
            r["pattern"]
            for r in yaml.safe_load(target.read_text(encoding="utf-8"))[PATTERNS_KEY]
        ]


# ---------------------------------------------------------------------------
# StrategyLedger.commit, directly
# ---------------------------------------------------------------------------


class TestCommit:
    def _ledger(self, tmp_path) -> StrategyLedger:
        _ledger_path(tmp_path).write_text(yaml.dump(_LEDGER), encoding="utf-8")
        ledger = StrategyLedger(_ledger_path(tmp_path))
        ledger.load()
        return ledger

    def test_a_change_the_memory_copy_cannot_take_is_retried_on_the_file(
        self, tmp_path
    ):
        """A row added on disk can be acted on without a restart."""
        ledger = self._ledger(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d[PATTERNS_KEY].append(dict(_HAND_PATTERN)))

        def supersede_hand_row(book: StrategyLedger):
            _, row = book.find("pat_by_hand")
            if row is None:
                raise LedgerChangeConflict("no such row")
            return book.supersede_pattern(row, reason="r")

        commit = ledger.commit(supersede_hand_row)

        assert commit.persisted and commit.reloaded
        assert _by_id(_on_disk(tmp_path)[PATTERNS_KEY])["pat_by_hand"]["superseded_at"]

    def test_a_change_no_version_of_the_file_takes_is_refused(self, tmp_path):
        ledger = self._ledger(tmp_path)
        before = _ledger_path(tmp_path).read_bytes()

        def never_applies(book: StrategyLedger):
            book.data[PATTERNS_KEY].append({"id": "half-applied"})
            raise LedgerChangeConflict("does not apply")

        commit = ledger.commit(never_applies)

        assert commit.refused and not commit.persisted
        assert "does not apply" in commit.error
        assert _ledger_path(tmp_path).read_bytes() == before
        assert ledger.data == _LEDGER, "a partly applied change is undone"

    def test_a_replaced_save_does_not_inherit_an_earlier_refusal(self, tmp_path):
        """A save stubbed to fail is a failed write, not a conflict to reload."""
        ledger = self._ledger(tmp_path)
        _edit_on_disk(tmp_path, lambda d: d.update({"edited": True}))
        assert ledger.save() is not None  # records a refusal
        ledger.save = lambda: "disk full (injected)"

        commit = ledger.commit(lambda book: book.add_pattern("p"))

        assert commit.error == "disk full (injected)"
        assert not commit.refused and not commit.reloaded
        assert "p" in [r["pattern"] for r in ledger.patterns]


class TestAtomicFile:
    def test_a_missing_file_has_no_bytes_and_no_digest(self, tmp_path):
        assert read_file_bytes(tmp_path / "absent") is None
        assert file_digest(None) is None
        assert file_digest(b"x") != file_digest(b"y")

    def test_a_replace_holds_exactly_the_content(self, tmp_path):
        path = tmp_path / "f.yaml"
        path.write_text("old\n", encoding="utf-8")

        replace_atomically(path, "new\r\nline\n")

        assert path.read_bytes() == b"new\r\nline\n"
        assert _staging_files(tmp_path) == []
