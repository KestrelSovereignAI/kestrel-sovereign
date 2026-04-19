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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
