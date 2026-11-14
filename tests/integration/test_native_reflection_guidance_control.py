"""Original prompt consumers preserve native guidance faults before generation.

A real booted Core feature port faults after SQLite I/O; no Reflection package,
provider output or guidance text is fabricated. All resource payers are NONE.
"""
from uuid import uuid4

import pytest
from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.llm.service import PolicyDeniedError
from tests.shared.booted_agent import booted_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("carrier", ["direct", "group", "timeout"])
async def test_original_prompt_consumer_preserves_guidance_control(tmp_path, monkeypatch, streaming, carrier):
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))
    (tmp_path / "kestrel.toml").write_text('[llm]\nroute_priority=[]\n')
    defaults = PayerPolicy.host_env_default()
    policy = PayerPolicy(**{field: PayerSpec(vendor=getattr(defaults, field).vendor, kind=PayerKind.NONE)
                           for field in ("llm", "storage", "compute", "tools", "comms")})
    async with booted_agent(tmp_path, name="OwnedGuidanceControl",
                            provenance="test:1990-original-guidance-consumer", features=set(),
                            payer_policy=policy) as agent:
        try:
            agent._get_privacy_transition_lock().require_native_scope(None)
        except ExecutionAuthorityError as native:
            original = native
        if carrier == "group":
            original = ExceptionGroup("Original guidance authority", [original])
        elif carrier == "timeout":
            try:
                raise TimeoutError("Guidance authority carrier") from original
            except TimeoutError as carried:
                original = carried
        reads = []
        class FaultingGuidanceFeature(Feature):
            @property
            def tool_description(self):
                return "Owned guidance fault port"

            async def initialize(self):
                await self.agent._raw_storage.db.fetchone("SELECT 1")

            async def get_active_guidance(self):
                await self.agent._raw_storage.db.fetchone("SELECT 1")
                reads.append(True)
                raise original
        agent.features["ReflectionFeature"] = FaultingGuidanceFeature(agent)
        try:
            terminal = None
            try:
                if streaming:
                    async for _chunk in agent.process_input_streaming("Owned guidance boundary", session_id=uuid4().hex):
                        pass
                else:
                    await agent.process_input("Owned guidance boundary", session_id=uuid4().hex)
            except (ExecutionAuthorityError, ExceptionGroup, TimeoutError, PolicyDeniedError) as error:
                terminal = error
            assert reads, "Original prompt did not invoke the guidance boundary"
            assert terminal is original, "Original prompt consumer erased native guidance control"
        finally:
            agent.features.pop("ReflectionFeature")
