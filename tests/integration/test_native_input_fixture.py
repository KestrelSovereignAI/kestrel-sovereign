"""Core-only integration input exercises the public native fixture end to end."""
import pytest

from kestrel_sovereign.features.base import Feature, tool
from tests.utils.native_tool_effects import drive_native


@pytest.mark.asyncio
async def test_native_fixture_owns_original_storage_and_effect(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:test:integration-native-fixture")
    database = agent._raw_storage.db
    await database.execute("CREATE TABLE native_fixture_effects (id INTEGER PRIMARY KEY)")

    class NativeFixtureProbe(Feature):
        @property
        def tool_description(self):
            return "Record an owned integration effect"

        async def initialize(self):
            pass

        @tool(name="record_fixture", description="Write one original native effect")
        async def record_fixture(self) -> dict:
            await self.agent._raw_storage.db.execute("INSERT INTO native_fixture_effects VALUES (1)")
            return {"written": True}

    feature = NativeFixtureProbe(agent)
    agent.features[feature.name] = feature
    result = await drive_native(agent, lambda: agent.execute_named_tool(
        "record_fixture", {}, session_id="native-fixture", source="integration",
    ))

    assert result == {"success": True, "tool": "record_fixture", "result": {"written": True}}
    assert agent.tool_effect_store.database is database
    assert await database.fetchone("SELECT COUNT(*) FROM native_fixture_effects") == (1,)
    assert await database.fetchone("SELECT state FROM tool_effect_claims") == ("RETURNED_OK",)
    assert await database.fetchone("SELECT state FROM tool_effect_operations") == ("CLOSED",)
    assert not agent._runtime_publication_ready.is_set()
