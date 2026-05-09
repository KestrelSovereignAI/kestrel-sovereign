"""Unit tests for SkillsFeature and Skill serialization."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.tools.result import ToolResultStatus
import pytest_asyncio

from kestrel_sovereign.features.skills.feature import (
    CANDIDATE_INSIGHT_TYPES,
    CLAIM_STALENESS_SECONDS,
    MIN_CANDIDATE_CONFIDENCE,
    SKILL_NODE_TYPE,
    SkillsFeature,
)
from kestrel_sovereign.features.skills.models import (
    Skill,
    normalize_title,
    skill_id_from_title,
)


# =============================================================================
# Skill model
# =============================================================================


class TestSkillModel:

    def test_roundtrip_markdown(self):
        s = Skill(
            id="skill_abc123",
            title="Fix flaky auth test by seeding the keyring",
            trigger="auth tests fail intermittently in CI",
            steps=[
                "Seed KEYRING_BACKEND=keyring.backends.null.Keyring",
                "Clear the test runner's cache",
                "Re-run the test with -p no:cacheprovider",
            ],
            verification="Run the auth suite 3 times in a row — all should pass",
            tags=["testing", "ci", "auth"],
            source_insight_id="ins-1",
            source_session_id="sess-42",
            confidence=0.85,
        )
        text = s.to_markdown()
        parsed = Skill.from_markdown(text)
        assert parsed.id == s.id
        assert parsed.title == s.title
        assert parsed.trigger == s.trigger
        assert parsed.steps == s.steps
        assert parsed.verification == s.verification
        assert parsed.tags == s.tags
        assert parsed.source_insight_id == s.source_insight_id
        assert parsed.confidence == s.confidence

    def test_markdown_quotes_special_characters(self):
        s = Skill(
            id="skill_x",
            title='A title with a colon: and a "quote"',
            trigger="t",
            steps=["a"],
            verification="v",
        )
        text = s.to_markdown()
        # Title must round-trip through YAML quoting
        parsed = Skill.from_markdown(text)
        assert parsed.title == 'A title with a colon: and a "quote"'

    def test_from_dict_handles_missing_optional_fields(self):
        d = {
            "id": "skill_a",
            "title": "T",
            "trigger": "when",
            "steps": ["x"],
            "verification": "v",
        }
        s = Skill.from_dict(d)
        assert s.tags == []
        assert s.source_insight_id is None
        assert s.confidence == 0.5


class TestHelpers:

    def test_normalize_title_collapses_non_alnum(self):
        assert normalize_title("Fix flaky AUTH test!!!") == "fix_flaky_auth_test"
        assert normalize_title("   ") == ""

    def test_skill_id_is_stable(self):
        a = skill_id_from_title("Fix flaky auth test")
        b = skill_id_from_title("fix flaky AUTH test")
        assert a == b

    def test_skill_id_differs_on_different_titles(self):
        a = skill_id_from_title("Fix flaky auth test")
        b = skill_id_from_title("Fix flaky database test")
        assert a != b


# =============================================================================
# Fixtures
# =============================================================================


def _make_mock_db():
    db = MagicMock()
    db.execute = AsyncMock()
    db.fetchall = AsyncMock(return_value=[])
    db.fetchone = AsyncMock(return_value=None)
    return db


def _make_mock_agent(tmp_path: Path, db=None):
    agent = MagicMock()
    agent.did = "did:test:skills-agent"
    agent.agent_id = agent.did
    agent.features = {}

    agent.storage = MagicMock()
    agent.storage.db = db or _make_mock_db()
    agent.storage.add_node = AsyncMock()
    agent.storage.delete_node = AsyncMock()
    agent.storage.get_nodes_by_type = AsyncMock(return_value=[])

    # Skills feature resolves agent data dir via bootstrap_service.
    agent.bootstrap_service = MagicMock()
    agent.bootstrap_service.agent_data_path = tmp_path
    return agent


@pytest_asyncio.fixture
async def feature(tmp_path):
    agent = _make_mock_agent(tmp_path)
    f = SkillsFeature(agent)
    await f.initialize()
    return f


# =============================================================================
# SkillsFeature — tool registration + tool_description
# =============================================================================


class TestToolRegistration:

    @pytest.mark.asyncio
    async def test_all_tools_present(self, feature):
        names = {t.name for t in feature.get_tools()}
        assert names >= {
            "skill_list",
            "skill_show",
            "skill_extract_candidates",
            "skill_save",
            "skill_delete",
        }

    @pytest.mark.asyncio
    async def test_skills_dir_is_created(self, feature, tmp_path):
        assert (tmp_path / "skills").exists()


# =============================================================================
# skill_extract_candidates
# =============================================================================


class TestExtractCandidates:

    @pytest.mark.asyncio
    async def test_returns_high_confidence_actionable_insights(self, feature):
        # Mock the DB to return two candidates
        feature._db.fetchall = AsyncMock(return_value=[
            ("ins-1", "sess-1", "pattern", "User often needs retry guidance",
             "Retry guidance pattern observed", 0.9, 1, "Always suggest exponential backoff for retries"),
            ("ins-2", "sess-1", "improvement", "Memory recall too slow",
             "Queries take > 2s", 0.8, 1, "Add vector search index for memory retrieval"),
        ])
        envelope = await feature.skill_extract_candidates(min_confidence=0.7, limit=10)
        assert envelope.data["count"] == 2
        assert envelope.data["candidates"][0]["insight_id"] == "ins-1"
        assert envelope.data["candidates"][0]["title"] == "Always suggest exponential backoff for retries"

    @pytest.mark.asyncio
    async def test_dedups_against_existing_skills_on_disk(self, feature, tmp_path):
        # Pre-create a skill file
        existing = Skill(
            id=skill_id_from_title("Add vector search index for memory retrieval"),
            title="Add vector search index for memory retrieval",
            trigger="t", steps=["s"], verification="v",
        )
        (tmp_path / "skills" / f"{existing.id}.md").write_text(
            existing.to_markdown(), encoding="utf-8"
        )

        feature._db.fetchall = AsyncMock(return_value=[
            ("ins-2", "sess-1", "improvement", "Memory recall too slow",
             "Queries take > 2s", 0.8, 1, "Add vector search index for memory retrieval"),
        ])
        envelope = await feature.skill_extract_candidates()
        assert envelope.data["count"] == 0

    @pytest.mark.asyncio
    async def test_dedups_against_graph_nodes(self, feature):
        node = MagicMock()
        node.label = "Use exponential backoff for retries"
        feature.agent.storage.get_nodes_by_type = AsyncMock(return_value=[node])

        feature._db.fetchall = AsyncMock(return_value=[
            ("ins-3", "sess-1", "pattern", "retries",
             "", 0.9, 1, "Use exponential backoff for retries"),
        ])
        envelope = await feature.skill_extract_candidates()
        assert envelope.data["count"] == 0

    @pytest.mark.asyncio
    async def test_skips_insights_without_suggested_action(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("ins-x", "s", "pattern", "vague", "", 0.9, 1, ""),
        ])
        envelope = await feature.skill_extract_candidates()
        assert envelope.data["count"] == 0

    @pytest.mark.asyncio
    async def test_query_respects_min_confidence(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        await feature.skill_extract_candidates(min_confidence=0.9, limit=5)
        params = feature._db.fetchall.call_args[0][1]
        # agent_id, min_confidence, *types, limit
        assert params[0] == "did:test:skills-agent"
        assert params[1] == 0.9
        # Tuple order: agent_id, min_conf, then each candidate type, then limit
        assert set(params[2:2 + len(CANDIDATE_INSIGHT_TYPES)]) == set(CANDIDATE_INSIGHT_TYPES)
        assert params[-1] == 5


# =============================================================================
# skill_save
# =============================================================================


class TestSkillSave:

    @pytest.mark.asyncio
    async def test_promotes_insight_to_skill_file_and_graph(self, feature, tmp_path):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-1", "sess-1", "improvement", "Memory recall too slow",
            "Queries take > 2s", 0.85, 1, "Add vector search index for memory retrieval",
        ))
        envelope = await feature.skill_save(
            insight_id="ins-1",
            steps_json=json.dumps([
                "Create HNSW index on embeddings column",
                "Backfill existing rows",
                "Switch retrieval to the new index",
            ]),
            verification="p99 recall latency < 200ms on 10k rows",
            tags_json='["memory", "performance"]',
        )
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["skill_id"].startswith("skill_")

        # File exists and is parseable
        skill_path = tmp_path / "skills" / f"{envelope.data['skill_id']}.md"
        assert skill_path.exists()
        parsed = Skill.from_markdown(skill_path.read_text(encoding="utf-8"))
        assert parsed.title == "Add vector search index for memory retrieval"
        assert len(parsed.steps) == 3

        # Graph node also written
        feature.agent.storage.add_node.assert_awaited_once()
        node = feature.agent.storage.add_node.await_args[0][0]
        assert node.node_type == SKILL_NODE_TYPE

    @pytest.mark.asyncio
    async def test_rejects_duplicate_title(self, feature, tmp_path):
        # Plant an existing skill
        existing = Skill(
            id=skill_id_from_title("Use exponential backoff for retries"),
            title="Use exponential backoff for retries",
            trigger="t", steps=["s"], verification="v",
        )
        (tmp_path / "skills" / f"{existing.id}.md").write_text(
            existing.to_markdown(), encoding="utf-8"
        )

        feature._db.fetchone = AsyncMock(return_value=(
            "ins-a", "s", "pattern", "t", "d", 0.9, 1, "Use exponential backoff for retries",
        ))
        envelope = await feature.skill_save(
            insight_id="ins-a",
            steps_json='["step 1"]',
            verification="v",
        )
        assert envelope.status is not ToolResultStatus.OK
        assert "already exists" in envelope.error.lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_steps(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-b", "s", "pattern", "t", "d", 0.9, 1, "Some suggestion",
        ))
        envelope = await feature.skill_save(
            insight_id="ins-b", steps_json="[]", verification="v"
        )
        assert envelope.status is not ToolResultStatus.OK
        assert "step" in envelope.error.lower()

    @pytest.mark.asyncio
    async def test_rejects_malformed_steps_json(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-c", "s", "pattern", "t", "d", 0.9, 1, "Some suggestion",
        ))
        envelope = await feature.skill_save(
            insight_id="ins-c", steps_json="not json", verification="v"
        )
        assert envelope.status is not ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_rejects_non_string_steps(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-d", "s", "pattern", "t", "d", 0.9, 1, "Some suggestion",
        ))
        envelope = await feature.skill_save(
            insight_id="ins-d", steps_json="[1, 2, 3]", verification="v"
        )
        assert envelope.status is not ToolResultStatus.OK
        assert "steps_json" in envelope.error.lower()

    @pytest.mark.asyncio
    async def test_missing_insight(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        envelope = await feature.skill_save(
            insight_id="nope",
            steps_json='["x"]',
            verification="v",
        )
        assert envelope.status is not ToolResultStatus.OK
        assert "not found" in envelope.error.lower()


# =============================================================================
# skill_list + skill_show + skill_delete
# =============================================================================


class TestListShowDelete:

    @pytest.mark.asyncio
    async def test_list_roundtrips(self, feature, tmp_path):
        s = Skill(
            id="skill_roundtrip",
            title="Roundtrip test",
            trigger="when testing",
            steps=["step a"],
            verification="v",
            tags=["t"],
        )
        (tmp_path / "skills" / f"{s.id}.md").write_text(s.to_markdown(), encoding="utf-8")

        envelope = await feature.skill_list()
        assert envelope.data["count"] == 1
        assert envelope.data["skills"][0]["title"] == "Roundtrip test"

    @pytest.mark.asyncio
    async def test_show_returns_skill(self, feature, tmp_path):
        s = Skill(
            id="skill_show_1",
            title="Shown",
            trigger="t", steps=["x"], verification="v",
        )
        (tmp_path / "skills" / f"{s.id}.md").write_text(s.to_markdown(), encoding="utf-8")
        envelope = await feature.skill_show(skill_id="skill_show_1")
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["skill"]["title"] == "Shown"

    @pytest.mark.asyncio
    async def test_show_missing(self, feature):
        envelope = await feature.skill_show(skill_id="skill_missing")
        assert envelope.status is not ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_delete_removes_file_and_node(self, feature, tmp_path):
        s = Skill(
            id="skill_del",
            title="Del", trigger="t", steps=["x"], verification="v",
        )
        path = tmp_path / "skills" / f"{s.id}.md"
        path.write_text(s.to_markdown(), encoding="utf-8")

        envelope = await feature.skill_delete(skill_id="skill_del")
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["removed_file"] is True
        assert envelope.data["removed_node"] is True
        assert not path.exists()
        feature.agent.storage.delete_node.assert_awaited_once_with("skill_del")

    @pytest.mark.asyncio
    async def test_delete_missing(self, feature):
        # Make the graph delete raise so removed_node stays False
        feature.agent.storage.delete_node = AsyncMock(side_effect=Exception("no node"))
        envelope = await feature.skill_delete(skill_id="skill_nope")
        assert envelope.status is not ToolResultStatus.OK


# =============================================================================
# Module-level constants used by callers
# =============================================================================


class TestModuleConstants:

    def test_min_candidate_confidence_is_sensible(self):
        assert 0.0 < MIN_CANDIDATE_CONFIDENCE <= 1.0

    def test_candidate_types_match_insight_enum(self):
        # The extractor only accepts actionable pattern/improvement insights.
        # Successes and failures are observations, not reusable procedures.
        assert "pattern" in CANDIDATE_INSIGHT_TYPES
        assert "improvement" in CANDIDATE_INSIGHT_TYPES


# =============================================================================
# Input validation
# =============================================================================


class TestInputValidation:

    @pytest.mark.asyncio
    async def test_min_confidence_rejects_out_of_range(self, feature):
        for bad in (-0.1, 1.5, 2.0):
            envelope = await feature.skill_extract_candidates(min_confidence=bad)
            assert envelope.status is not ToolResultStatus.OK
            assert "[0.0, 1.0]" in envelope.error

    @pytest.mark.asyncio
    async def test_min_confidence_rejects_non_numeric(self, feature):
        envelope = await feature.skill_extract_candidates(min_confidence="high")
        assert envelope.status is not ToolResultStatus.OK
        assert "numeric" in envelope.error.lower()

    @pytest.mark.asyncio
    async def test_limit_rejects_out_of_range(self, feature):
        for bad in (0, -1, 10_000):
            envelope = await feature.skill_extract_candidates(limit=bad)
            assert envelope.status is not ToolResultStatus.OK
            assert "[1, 500]" in envelope.error

    @pytest.mark.asyncio
    async def test_limit_rejects_non_numeric(self, feature):
        envelope = await feature.skill_extract_candidates(limit="lots")
        assert envelope.status is not ToolResultStatus.OK


# =============================================================================
# Atomic writes + write-time collision guard
# =============================================================================


class TestAtomicWriteAndCollision:

    @pytest.mark.asyncio
    async def test_write_is_atomic_via_temp_rename(self, feature, tmp_path, monkeypatch):
        """Simulate a crash during finalization (the second os.link call)
        and prove no partial .md file is left, and the claim is cleaned up
        since this writer owns it."""
        from kestrel_sovereign.features.skills import feature as mod

        feature._db.fetchone = AsyncMock(return_value=(
            "ins-a", "sess", "pattern", "t", "d", 0.85, 1, "Atomic write skill",
        ))

        real_link = os.link
        link_count = [0]

        def _boom_on_finalize(src, dst):
            link_count[0] += 1
            if link_count[0] == 2:
                # Second link call is finalization — simulate crash.
                raise OSError("simulated crash during finalization")
            return real_link(src, dst)

        monkeypatch.setattr(mod.os, "link", _boom_on_finalize)

        envelope = await feature.skill_save(
            insight_id="ins-a",
            steps_json='["step a"]',
            verification="v",
        )
        assert envelope.status is not ToolResultStatus.OK

        skills_dir = tmp_path / "skills"
        # No .md at the target, and no leftover tmp or claim files.
        assert list(skills_dir.glob("*.md")) == []
        assert list(skills_dir.glob("*.md.tmp.*")) == []
        assert list(skills_dir.glob("*.md.claim")) == []

        # Restore and confirm a clean save works afterwards.
        monkeypatch.setattr(mod.os, "link", real_link)
        envelope = await feature.skill_save(
            insight_id="ins-a",
            steps_json='["step a"]',
            verification="v",
        )
        assert envelope.status is ToolResultStatus.OK

    @pytest.mark.asyncio
    async def test_tmp_path_is_per_writer_not_deterministic(self, feature, tmp_path, monkeypatch):
        """Two writers targeting the same skill must not collide on the tmp
        path. The remaining race (both pass path.exists, both os.replace) is
        documented as last-writer-wins and out of scope here."""
        from kestrel_sovereign.features.skills import feature as mod

        observed_tmps: list[str] = []
        real_write_text = mod.Path.write_text

        def _capture_write(self, *args, **kwargs):
            observed_tmps.append(self.name)
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(mod.Path, "write_text", _capture_write)

        feature._db.fetchone = AsyncMock(return_value=(
            "ins-p1", "s", "pattern", "t", "d", 0.85, 1, "Parallel title",
        ))

        r1 = await feature.skill_save(
            insight_id="ins-p1",
            steps_json='["s1"]',
            verification="v1",
        )
        assert r1.status is ToolResultStatus.OK

        # Remove the persisted file so the second save passes the collision
        # guard; the point of the test is the tmp-name uniqueness, not the
        # final-file race.
        (tmp_path / "skills" / f"{r1.data['skill_id']}.md").unlink()

        r2 = await feature.skill_save(
            insight_id="ins-p1",
            steps_json='["s2"]',
            verification="v2",
        )
        assert r2.status is ToolResultStatus.OK

        # Both writes used distinct tmp filenames — no clobber possible.
        tmp_names = [n for n in observed_tmps if ".tmp." in n]
        assert len(tmp_names) == 2
        assert tmp_names[0] != tmp_names[1]

    @pytest.mark.asyncio
    async def test_collision_guard_refuses_overwrite(self, feature, tmp_path):
        """If preflight dedup misses (e.g. a file present but preflight
        didn't see it), the write-time existence check must refuse to
        clobber an existing skill file."""
        title = "Pre-existing skill"
        existing_id = skill_id_from_title(title)
        existing_path = tmp_path / "skills" / f"{existing_id}.md"
        # Write an unparseable file so preflight's normalized-title set
        # doesn't include this title — simulating dedup drift.
        existing_path.write_text("--- existing content ---", encoding="utf-8")

        feature.agent.storage.get_nodes_by_type = AsyncMock(return_value=[])
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-collide", "s", "pattern", "t", "d", 0.85, 1, title,
        ))

        envelope = await feature.skill_save(
            insight_id="ins-collide",
            steps_json='["s"]',
            verification="v",
        )
        assert envelope.status is not ToolResultStatus.OK
        # File was NOT overwritten
        assert "existing content" in existing_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_concurrent_save_exactly_one_wins(self, tmp_path):
        """Two real threads attempt os.link on the same claim path
        simultaneously. Exactly one must win; the other gets an error."""
        from kestrel_sovereign.features.skills.models import Skill

        agent = _make_mock_agent(tmp_path)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(exist_ok=True)

        barrier = threading.Barrier(2, timeout=5)
        results: list[dict] = [None, None]  # type: ignore[list-item]

        def _writer(idx: int):
            feat = SkillsFeature(agent)
            # Manually set the internals without async initialize.
            feat._db = agent.storage.db
            feat._skills_dir = skills_dir

            skill = Skill(
                id="skill_race",
                title=f"Race writer {idx}",
                trigger="t", steps=["s"], verification="v",
                source_insight_id=f"ins-{idx}",
                confidence=0.9,
            )
            barrier.wait()
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(feat._save_skill(skill))
                results[idx] = {"won": True}
            except (FileExistsError, OSError) as e:
                results[idx] = {"won": False, "error": str(e)}
            finally:
                loop.close()

        t0 = threading.Thread(target=_writer, args=(0,))
        t1 = threading.Thread(target=_writer, args=(1,))
        t0.start(); t1.start()
        t0.join(timeout=10); t1.join(timeout=10)

        wins = [r for r in results if r and r["won"]]
        losses = [r for r in results if r and not r["won"]]
        assert len(wins) == 1, f"Expected exactly one winner: {results}"
        assert len(losses) == 1, f"Expected exactly one loser: {results}"
        # The final file must exist and be valid.
        assert (skills_dir / "skill_race.md").exists()
        # No claim or tmp files left behind.
        assert list(skills_dir.glob("*.claim")) == []
        assert list(skills_dir.glob("*.tmp.*")) == []

    @pytest.mark.asyncio
    async def test_loser_cannot_delete_winners_claim(self, tmp_path):
        """Force the interleaving where the loser's cleanup runs after the
        winner's os.link succeeded. The winner's claim must survive."""
        from kestrel_sovereign.features.skills import feature as mod

        agent = _make_mock_agent(tmp_path)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(exist_ok=True)

        claim_path = skills_dir / "skill_interleave.md.claim"

        winner_linked = threading.Event()
        loser_may_cleanup = threading.Event()

        real_link = os.link

        def _controlled_link(src, dst):
            """Intercept os.link to synchronize the two writers."""
            if str(dst).endswith(".claim"):
                try:
                    real_link(src, dst)
                    # Winner succeeded — signal the loser may now try cleanup.
                    winner_linked.set()
                except FileExistsError:
                    # Loser — wait until test explicitly lets us proceed
                    # to the cleanup phase.
                    winner_linked.wait(timeout=5)
                    loser_may_cleanup.set()
                    raise
            else:
                real_link(src, dst)

        results: list[dict] = [None, None]  # type: ignore[list-item]
        barrier = threading.Barrier(2, timeout=5)

        def _writer(idx: int):
            from kestrel_sovereign.features.skills.models import Skill

            feat = SkillsFeature(agent)
            feat._db = agent.storage.db
            feat._skills_dir = skills_dir

            skill = Skill(
                id="skill_interleave",
                title=f"Interleave writer {idx}",
                trigger="t", steps=["s"], verification="v",
                source_insight_id=f"ins-{idx}",
                confidence=0.9,
            )
            barrier.wait()
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(feat._save_skill(skill))
                results[idx] = {"won": True}
            except (FileExistsError, OSError) as e:
                results[idx] = {"won": False, "error": str(e)}
            finally:
                loop.close()

        import unittest.mock
        with unittest.mock.patch.object(mod.os, "link", side_effect=_controlled_link):
            t0 = threading.Thread(target=_writer, args=(0,))
            t1 = threading.Thread(target=_writer, args=(1,))
            t0.start(); t1.start()
            t0.join(timeout=10); t1.join(timeout=10)

        # Exactly one winner, one loser.
        wins = [r for r in results if r and r["won"]]
        losses = [r for r in results if r and not r["won"]]
        assert len(wins) == 1, f"Expected one winner: {results}"
        assert len(losses) == 1, f"Expected one loser: {results}"

        # The critical assertion: the final file exists (winner's work
        # survived the loser's cleanup).
        assert (skills_dir / "skill_interleave.md").exists()

    @pytest.mark.asyncio
    async def test_finalization_refuses_overwrite_via_link(self, feature, tmp_path, monkeypatch):
        """Finalization uses os.link (not os.replace), so if the target file
        appeared between preflight and finalization, the save fails instead
        of silently overwriting."""
        from kestrel_sovereign.features.skills import feature as mod

        real_link = os.link

        created_claim = threading.Event()

        def _sneaky_link(src, dst):
            """After claim succeeds, plant the final file before finalization."""
            real_link(src, dst)
            if str(dst).endswith(".claim"):
                created_claim.set()
                # Simulate another process creating the final file.
                final = Path(str(dst).replace(".md.claim", ".md"))
                final.write_text("planted by another process", encoding="utf-8")

        monkeypatch.setattr(mod.os, "link", _sneaky_link)

        feature._db.fetchone = AsyncMock(return_value=(
            "ins-sneak", "s", "pattern", "t", "d", 0.85, 1, "Sneaky overwrite test",
        ))

        envelope = await feature.skill_save(
            insight_id="ins-sneak",
            steps_json='["step"]',
            verification="v",
        )
        assert envelope.status is not ToolResultStatus.OK
        # The original planted file is preserved, not overwritten.
        skill_id = skill_id_from_title("Sneaky overwrite test")
        final = tmp_path / "skills" / f"{skill_id}.md"
        assert "planted by another process" in final.read_text(encoding="utf-8")


# =============================================================================
# Stale claim reclamation
# =============================================================================


class TestStaleClaimReclamation:

    @pytest.mark.asyncio
    async def test_stale_claim_is_reclaimed(self, feature, tmp_path, monkeypatch):
        """A claim file older than the staleness threshold is removed and
        the new writer can proceed."""
        from kestrel_sovereign.features.skills import feature as mod

        feature._db.fetchone = AsyncMock(return_value=(
            "ins-stale", "s", "pattern", "t", "d", 0.85, 1, "Stale claim skill",
        ))

        skill_id = skill_id_from_title("Stale claim skill")
        claim_path = tmp_path / "skills" / f"{skill_id}.md.claim"
        claim_path.write_text("orphaned claim", encoding="utf-8")

        # Make the claim appear old enough to reclaim.
        stale_mtime = time.time() - CLAIM_STALENESS_SECONDS - 10
        os.utime(claim_path, (stale_mtime, stale_mtime))

        envelope = await feature.skill_save(
            insight_id="ins-stale",
            steps_json='["step"]',
            verification="v",
        )
        assert envelope.status is ToolResultStatus.OK
        # Final file written, no leftover claim.
        final = tmp_path / "skills" / f"{skill_id}.md"
        assert final.exists()
        assert not claim_path.exists()

    @pytest.mark.asyncio
    async def test_fresh_claim_blocks_new_writer(self, feature, tmp_path):
        """A claim file that is NOT stale blocks the new writer."""
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-fresh", "s", "pattern", "t", "d", 0.85, 1, "Fresh claim skill",
        ))

        skill_id = skill_id_from_title("Fresh claim skill")
        claim_path = tmp_path / "skills" / f"{skill_id}.md.claim"
        claim_path.write_text("active claim", encoding="utf-8")
        # Freshly created — not stale.

        envelope = await feature.skill_save(
            insight_id="ins-fresh",
            steps_json='["step"]',
            verification="v",
        )
        assert envelope.status is not ToolResultStatus.OK
        assert "concurrent" in envelope.error.lower() or "in progress" in envelope.error.lower()
        # Claim file was NOT removed by the blocked writer.
        assert claim_path.exists()
        assert claim_path.read_text(encoding="utf-8") == "active claim"


# =============================================================================
# Partial-failure behavior
# =============================================================================


class TestPartialFailures:

    @pytest.mark.asyncio
    async def test_graph_add_failure_does_not_fail_save(self, feature, tmp_path):
        """Graph is best-effort. A graph write error must not fail the save —
        the file is the primary record."""
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-g", "s", "pattern", "t", "d", 0.85, 1, "Survives graph failure",
        ))
        feature.agent.storage.add_node = AsyncMock(side_effect=Exception("graph down"))

        envelope = await feature.skill_save(
            insight_id="ins-g",
            steps_json='["step"]',
            verification="v",
        )
        assert envelope.status is ToolResultStatus.OK

        skill_path = tmp_path / "skills" / f"{envelope.data['skill_id']}.md"
        assert skill_path.exists()
        assert "Survives graph failure" in skill_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_delete_reports_partial_removal(self, feature, tmp_path):
        """File present but graph already lost it: delete now surfaces
        as PARTIAL (since #1061 wave 30) — the file went, but a stale
        graph node will surface in associative recall, so the LLM
        must not claim a clean removal.
        """
        s = Skill(
            id="skill_partial",
            title="Partial",
            trigger="t", steps=["x"], verification="v",
        )
        (tmp_path / "skills" / f"{s.id}.md").write_text(s.to_markdown(), encoding="utf-8")

        feature.agent.storage.delete_node = AsyncMock(side_effect=Exception("not in graph"))
        envelope = await feature.skill_delete(skill_id="skill_partial")
        assert envelope.status is ToolResultStatus.PARTIAL
        assert envelope.data["removed_file"] is True
        assert envelope.data["removed_node"] is False
        assert "graph node" in envelope.error and "associative recall" in envelope.error

    @pytest.mark.asyncio
    async def test_delete_graph_only_no_skills_dir_returns_ok(self, feature):
        """Graph-only agent (no skills directory configured) — a
        successful graph node deletion must NOT trip the asymmetric-
        PARTIAL branch, which would call ``_skill_path`` and raise
        AssertionError. Codex round 1 of #1130 caught this.

        A "no file_attempted, only graph_attempted" delete is
        symmetric by construction — the graph layer is the only one
        that could apply.
        """
        feature._skills_dir = None  # graph-only agent
        feature.agent.storage.delete_node = AsyncMock(return_value=None)
        envelope = await feature.skill_delete(skill_id="skill_graph_only")
        assert envelope.status is ToolResultStatus.OK
        assert envelope.data["removed_file"] is False
        assert envelope.data["removed_node"] is True

    @pytest.mark.asyncio
    async def test_list_skips_unparseable_files(self, feature, tmp_path):
        """A human-edited file that broke the parser must not poison listing."""
        good = Skill(
            id="skill_good",
            title="Good skill",
            trigger="t", steps=["x"], verification="v",
        )
        (tmp_path / "skills" / f"{good.id}.md").write_text(good.to_markdown(), encoding="utf-8")
        (tmp_path / "skills" / "skill_bad.md").write_text(
            "just a plain note, not a skill\n", encoding="utf-8"
        )

        envelope = await feature.skill_list()
        titles = [s["title"] for s in envelope.data["skills"]]
        assert "Good skill" in titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
