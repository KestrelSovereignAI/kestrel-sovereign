"""Identity export discovers real SDK tools independently of A2A wire DTOs."""

from types import SimpleNamespace

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.identity.exporter import IdentityExporter
from kestrel_sovereign.identity.identity_package import SkillRecord
from kestrel_sovereign.identity.importer import IdentityImporter
from tests.unit.test_identity_import_hardening import (
    graph_db as graph_db,  # noqa: PLC0414
)


class ExportFeature(Feature):
    @property
    def tool_description(self):
        return "An exportable runtime tool"

    async def initialize(self):
        pass

    @tool(
        name="export_runtime_skill",
        description="Inspect the registered runtime capability",
        category=ToolCategory.DATA_ACCESS,
    )
    async def inspect_capability(self, target: str) -> ToolResult:
        return ToolResult.ok(confirmation=target)


async def test_export_includes_uninvoked_runtime_tool_with_its_complete_schema(
    graph_db,
):
    agent = SimpleNamespace(did="did:test:runtime-skill-export")
    feature = ExportFeature(agent)
    agent.features = {"export": feature}
    schema = feature.get_tools()[0].schema

    records = await IdentityExporter(graph_db, agent.did, agent=agent)._get_skills()

    assert len(records) == 1, "Identity export dropped an uninvoked runtime tool"
    record = records[0]
    assert record.skill_id == record.skill_name == schema.name
    assert record.skill_type == "data_access"
    assert record.times_used == 0 and record.last_used is None
    assert record.proficiency == 0.5
    assert record.configuration == schema.to_dict()
    assert record.configuration["parameters"][0]["name"] == "target"
    assert "version" not in record.configuration
    assert SkillRecord.from_dict(record.to_dict()) == record


async def test_runtime_schema_merges_owned_graph_usage_and_retains_graph_only_skills(
    graph_db,
):
    agent = SimpleNamespace(did="did:test:runtime-skill-export")
    feature = ExportFeature(agent)
    agent.features = {"export": feature}
    importer = IdentityImporter(graph_db)
    prior_use = "2026-10-20T15:10:00+00:00"
    await importer._import_skills(
        agent.did,
        [
            {
                "skill_id": "export_runtime_skill",
                "skill_name": "Old name",
                "skill_type": "old_category",
                "times_used": 27,
                "last_used": prior_use,
                "configuration": {"old": True},
            },
            {
                "skill_id": "learned_capability",
                "skill_name": "Retained knowledge",
                "skill_type": "knowledge_domain",
                "times_used": 4,
                "configuration": {"domain": "navigation"},
            },
        ],
    )
    await importer._import_skills(
        "did:test:other-agent",
        [
            {
                "skill_id": "export_runtime_skill",
                "skill_name": "Other agent's capability",
                "times_used": 99,
            }
        ],
    )

    records = await IdentityExporter(graph_db, agent.did, agent=agent)._get_skills()

    assert len(records) == 2, "Export duplicated the imported runtime skill"
    by_id = {record.skill_id: record for record in records}
    runtime = by_id["export_runtime_skill"]
    assert runtime.skill_name == "export_runtime_skill"
    assert runtime.skill_type == "data_access"
    assert runtime.configuration == feature.get_tools()[0].schema.to_dict()
    assert runtime.times_used == 27 and runtime.last_used == prior_use
    assert runtime.proficiency == 0.77
    retained = by_id["learned_capability"]
    assert retained.skill_type == "knowledge_domain"
    assert retained.configuration == {"domain": "navigation"}
    assert retained.times_used == 4
