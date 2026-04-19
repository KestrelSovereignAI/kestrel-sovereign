"""Unit tests for SkillsFeature and Skill serialization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sovereign.features.skills.feature import (
    CANDIDATE_INSIGHT_TYPES,
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
        result = await feature.skill_extract_candidates(min_confidence=0.7, limit=10)
        assert result["count"] == 2
        assert result["candidates"][0]["insight_id"] == "ins-1"
        assert result["candidates"][0]["title"] == "Always suggest exponential backoff for retries"

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
        result = await feature.skill_extract_candidates()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_dedups_against_graph_nodes(self, feature):
        node = MagicMock()
        node.label = "Use exponential backoff for retries"
        feature.agent.storage.get_nodes_by_type = AsyncMock(return_value=[node])

        feature._db.fetchall = AsyncMock(return_value=[
            ("ins-3", "sess-1", "pattern", "retries",
             "", 0.9, 1, "Use exponential backoff for retries"),
        ])
        result = await feature.skill_extract_candidates()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_skips_insights_without_suggested_action(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[
            ("ins-x", "s", "pattern", "vague", "", 0.9, 1, ""),
        ])
        result = await feature.skill_extract_candidates()
        assert result["count"] == 0

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
        result = await feature.skill_save(
            insight_id="ins-1",
            steps_json=json.dumps([
                "Create HNSW index on embeddings column",
                "Backfill existing rows",
                "Switch retrieval to the new index",
            ]),
            verification="p99 recall latency < 200ms on 10k rows",
            tags_json='["memory", "performance"]',
        )
        assert result["success"] is True
        assert result["skill_id"].startswith("skill_")

        # File exists and is parseable
        skill_path = tmp_path / "skills" / f"{result['skill_id']}.md"
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
        result = await feature.skill_save(
            insight_id="ins-a",
            steps_json='["step 1"]',
            verification="v",
        )
        assert result["success"] is False
        assert "already exists" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_empty_steps(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-b", "s", "pattern", "t", "d", 0.9, 1, "Some suggestion",
        ))
        result = await feature.skill_save(
            insight_id="ins-b", steps_json="[]", verification="v"
        )
        assert result["success"] is False
        assert "step" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_malformed_steps_json(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-c", "s", "pattern", "t", "d", 0.9, 1, "Some suggestion",
        ))
        result = await feature.skill_save(
            insight_id="ins-c", steps_json="not json", verification="v"
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rejects_non_string_steps(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(
            "ins-d", "s", "pattern", "t", "d", 0.9, 1, "Some suggestion",
        ))
        result = await feature.skill_save(
            insight_id="ins-d", steps_json="[1, 2, 3]", verification="v"
        )
        assert result["success"] is False
        assert "steps_json" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_insight(self, feature):
        feature._db.fetchone = AsyncMock(return_value=None)
        result = await feature.skill_save(
            insight_id="nope",
            steps_json='["x"]',
            verification="v",
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()


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

        result = await feature.skill_list()
        assert result["count"] == 1
        assert result["skills"][0]["title"] == "Roundtrip test"

    @pytest.mark.asyncio
    async def test_show_returns_skill(self, feature, tmp_path):
        s = Skill(
            id="skill_show_1",
            title="Shown",
            trigger="t", steps=["x"], verification="v",
        )
        (tmp_path / "skills" / f"{s.id}.md").write_text(s.to_markdown(), encoding="utf-8")
        result = await feature.skill_show(skill_id="skill_show_1")
        assert result["success"] is True
        assert result["skill"]["title"] == "Shown"

    @pytest.mark.asyncio
    async def test_show_missing(self, feature):
        result = await feature.skill_show(skill_id="skill_missing")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_delete_removes_file_and_node(self, feature, tmp_path):
        s = Skill(
            id="skill_del",
            title="Del", trigger="t", steps=["x"], verification="v",
        )
        path = tmp_path / "skills" / f"{s.id}.md"
        path.write_text(s.to_markdown(), encoding="utf-8")

        result = await feature.skill_delete(skill_id="skill_del")
        assert result["success"] is True
        assert result["removed_file"] is True
        assert result["removed_node"] is True
        assert not path.exists()
        feature.agent.storage.delete_node.assert_awaited_once_with("skill_del")

    @pytest.mark.asyncio
    async def test_delete_missing(self, feature):
        # Make the graph delete raise so removed_node stays False
        feature.agent.storage.delete_node = AsyncMock(side_effect=Exception("no node"))
        result = await feature.skill_delete(skill_id="skill_nope")
        assert result["success"] is False


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
            result = await feature.skill_extract_candidates(min_confidence=bad)
            assert result["success"] is False
            assert "[0.0, 1.0]" in result["error"]

    @pytest.mark.asyncio
    async def test_min_confidence_rejects_non_numeric(self, feature):
        result = await feature.skill_extract_candidates(min_confidence="high")
        assert result["success"] is False
        assert "numeric" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_limit_rejects_out_of_range(self, feature):
        for bad in (0, -1, 10_000):
            result = await feature.skill_extract_candidates(limit=bad)
            assert result["success"] is False
            assert "[1, 500]" in result["error"]

    @pytest.mark.asyncio
    async def test_limit_rejects_non_numeric(self, feature):
        result = await feature.skill_extract_candidates(limit="lots")
        assert result["success"] is False


# =============================================================================
# Atomic writes + write-time collision guard
# =============================================================================


class TestAtomicWriteAndCollision:

    @pytest.mark.asyncio
    async def test_write_is_atomic_via_temp_rename(self, feature, tmp_path, monkeypatch):
        """Simulate a crash mid-rename and prove no partial .md file is left."""
        from kestrel_sovereign.features.skills import feature as mod

        feature._db.fetchone = AsyncMock(return_value=(
            "ins-a", "sess", "pattern", "t", "d", 0.85, 1, "Atomic write skill",
        ))

        real_replace = mod.os.replace

        def _boom(src, dst):
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(mod.os, "replace", _boom)

        result = await feature.skill_save(
            insight_id="ins-a",
            steps_json='["step a"]',
            verification="v",
        )
        assert result["success"] is False

        skills_dir = tmp_path / "skills"
        # No .md at the target, and no per-writer .tmp.* files left behind.
        assert list(skills_dir.glob("*.md")) == []
        assert list(skills_dir.glob("*.md.tmp.*")) == []

        # Restore and confirm a clean save works afterwards.
        monkeypatch.setattr(mod.os, "replace", real_replace)
        result = await feature.skill_save(
            insight_id="ins-a",
            steps_json='["step a"]',
            verification="v",
        )
        assert result["success"] is True

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
        assert r1["success"] is True

        # Remove the persisted file so the second save passes the collision
        # guard; the point of the test is the tmp-name uniqueness, not the
        # final-file race.
        (tmp_path / "skills" / f"{r1['skill_id']}.md").unlink()

        r2 = await feature.skill_save(
            insight_id="ins-p1",
            steps_json='["s2"]',
            verification="v2",
        )
        assert r2["success"] is True

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

        result = await feature.skill_save(
            insight_id="ins-collide",
            steps_json='["s"]',
            verification="v",
        )
        assert result["success"] is False
        # File was NOT overwritten
        assert "existing content" in existing_path.read_text(encoding="utf-8")


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

        result = await feature.skill_save(
            insight_id="ins-g",
            steps_json='["step"]',
            verification="v",
        )
        assert result["success"] is True

        skill_path = tmp_path / "skills" / f"{result['skill_id']}.md"
        assert skill_path.exists()
        assert "Survives graph failure" in skill_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_delete_reports_partial_removal(self, feature, tmp_path):
        """File present but graph already lost it: delete still succeeds
        and signals partial state via flags."""
        s = Skill(
            id="skill_partial",
            title="Partial",
            trigger="t", steps=["x"], verification="v",
        )
        (tmp_path / "skills" / f"{s.id}.md").write_text(s.to_markdown(), encoding="utf-8")

        feature.agent.storage.delete_node = AsyncMock(side_effect=Exception("not in graph"))
        result = await feature.skill_delete(skill_id="skill_partial")
        assert result["success"] is True
        assert result["removed_file"] is True
        assert result["removed_node"] is False

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

        result = await feature.skill_list()
        titles = [s["title"] for s in result["skills"]]
        assert "Good skill" in titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
