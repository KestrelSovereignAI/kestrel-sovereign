"""Real feature caller binding and private diagnostics for the same-row step."""

import json

import pytest
from kestrel_sdk.operator import OperatorAuthorizationError

from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.security.access_log import (
    private_capability_tool_name,
    redact_private_capability_tool_calls,
)
from tests.unit.visual_computer.test_host import AUTH, KEY, intent, path
from tests.unit.visual_computer.test_host import (
    visual_host as visual_host,  # noqa: PLC0414 - pytest fixture reexport
)


@pytest.mark.asyncio
async def test_step_feature_binds_original_host_and_does_not_replan(
    visual_host, monkeypatch
):
    from kestrel_sovereign.visual_computer.inference import VisualInference

    client, agents, _ = visual_host
    response = await client.post(path() + "/requests", headers=AUTH, json=intent())
    sid = response.json()["session_id"]
    feature = agents["kite"].features["VisualComputerFeature"]
    service = agents["kite"].llm_service = object()
    calls = []

    async def plan(self, context, session_id, prompt, **kwargs):
        assert self.runtime is feature.runtime and self.llm_service is service
        owner = kwargs["_step_owner"]
        assert owner.proposal_action_id() == "host-stable"
        stored = await self.runtime.repository.action(
            self.runtime.scope(context), session_id, "host-stable"
        )
        assert stored.intent.prompt == prompt
        calls.append((context, kwargs))
        return {"action_request": None, "answer": "Finished"}

    monkeypatch.setattr(VisualInference, "plan", plan)
    with caller_context_scope(CallerContext.sovereign(credential=KEY)):
        answer = await feature.visual_computer_step(
            sid, "host-stable", "PRIVATE-HUMAN-TEXT"
        )
        assert (
            await feature.visual_computer_step(sid, "host-stable", "PRIVATE-HUMAN-TEXT")
            == answer
        )
    assert len(calls) == 1 and answer["receipt"] is None
    assert calls[0][0].acting_agent_id == "did:test:kite"
    with (
        caller_context_scope(CallerContext.authenticated("peer")),
        pytest.raises(OperatorAuthorizationError),
    ):
        await feature.visual_computer_step(sid, "foreign", "PRIVATE-HUMAN-TEXT")
    assert len(calls) == 1
    assert private_capability_tool_name("visual_computer_step")


def test_step_classifier_redacts_actual_prompt_request_result():
    private = "PRIVATE-HUMAN-TEXT"
    calls = [
        {
            "id": "one",
            "type": "function",
            "function": {
                "name": "visual_computer_step",
                "arguments": json.dumps({"prompt": private}),
            },
        },
        {
            "role": "tool",
            "tool_call_id": "one",
            "content": json.dumps(
                {"answer": private, "action_request": {"action": {"text": private}}}
            ),
        },
    ]
    safe = redact_private_capability_tool_calls(calls)
    assert private not in json.dumps(safe)
    assert private in json.dumps(calls)
