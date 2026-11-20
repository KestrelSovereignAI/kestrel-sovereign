"""Live acceptance for #4198 on an isolated agent with synthetic input.

Skipped unless ``KESTREL_LIVE_TESTS=1`` and ``OPENAI_API_KEY`` are set: it
makes a handful of paid OpenAI calls. The agent is a fresh test identity on
temporary storage; nothing else is touched.

1. A native turn with an exact ``openai:api/gpt-6.2-sol`` override is asked
   to call ``state_of_mind`` once. The tool's result carries a value minted
   when the tool runs. The turn must reach the Responses API, run the tool,
   send its output back and answer with the value.
2. The next turn in the same session has no override. It must be served by
   the configured default on chat completions, with the value in the
   session's history, and answer with it. A fresh session's history must not
   hold the value.

``KESTREL_LIVE_EVIDENCE`` names a file the observed evidence is written to:
model ids the provider returned, response ids, tool dispatch rows, whether
each reply carried the value, and the first 300 characters of each reply.
The input is synthetic, so the replies carry no private content.
"""

from __future__ import annotations

import functools
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx
import openai
import pytest

from kestrel_sovereign.llm.provider_served import PROVIDER_SERVED_KEY
from tests.llm._live import require_live_provider

SOL = os.environ.get("OPENAI_SOL_SMOKE_MODEL", "gpt-6.2-sol")
DEFAULT = os.environ.get("OPENAI_DEFAULT_SMOKE_MODEL", "gpt-5.5")
MEMORIES_RE = re.compile(r"<memories>.*?</memories>", re.DOTALL)


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Forwards to the real transport and keeps each request's path and body."""

    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.requests: List[tuple] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        self.requests.append((request.url.path, json.loads(body) if body else {}))
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _session_text(messages: List[Dict[str, Any]]) -> str:
    texts = []
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        texts.append(MEMORIES_RE.sub("", text or ""))
    return "\n".join(texts)


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
async def test_live_sol_override_tool_turn_then_default_turn_in_same_session(tmp_path, monkeypatch):
    require_live_provider("openai")
    from kestrel_sdk.tools import ToolResult

    from kestrel_sovereign.features.state_of_mind import StateOfMindFeature
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry
    from tests.shared.booted_agent import booted_agent

    minted: List[str] = []
    real_tool = StateOfMindFeature.get_state_of_mind

    @functools.wraps(real_tool)
    async def state_of_mind_with_fresh_value(self):
        result = await real_tool(self)
        if not minted:
            minted.append(f"probe-{uuid.uuid4().hex[:12]}")
        return ToolResult.ok(
            confirmation=f"{result.confirmation}\nProbe value: {minted[0]}",
            data={**(result.data or {}), "probe": minted[0]},
        )

    monkeypatch.setattr(StateOfMindFeature, "get_state_of_mind", state_of_mind_with_fresh_value)
    transport = _RecordingTransport()
    async with booted_agent(
        tmp_path, name="SolLive", provenance="test:openai_sol_live",
        features={"StateOfMindFeature"},
    ) as agent:
        service = agent.llm_service
        info = ProviderRegistry({})._build_route(
            "openai", "api", {"is_cloud": True},
            {"adapter": "OpenAIAdapter", "api_key": os.environ["OPENAI_API_KEY"],
             "model": DEFAULT, "supports_embeddings": False},
        )
        [provider] = service._convert_providers_format([info])
        provider["client"] = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            http_client=httpx.AsyncClient(transport=transport, timeout=180),
            max_retries=0,
        )
        monkeypatch.setattr(service, "providers", [provider])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": [provider["name"]],
        })
        monkeypatch.setattr(service, "_mandate_preference", {
            "vendor": "openai", "route": "api", "model": DEFAULT,
        })
        configured = dict(service._mandate_preference)
        session_id = f"sol-live-{uuid.uuid4().hex[:8]}"

        first = await agent.process_input(
            "Call the state_of_mind tool exactly once, then reply with the "
            "Probe value it reports, verbatim, and nothing else.",
            session_id=session_id, model_override=f"openai:api/{SOL}",
        )
        after_override = dict(service._mandate_preference)
        turn_one_requests = len(transport.requests)
        # Cross-turn history carries the earlier turn's reply text, not its
        # tool call rows, so ask for the value that conversation holds.
        question = ("Earlier in this conversation you replied with a Probe value. "
                    "Repeat that exact value only, or reply NONE if this "
                    "conversation has no earlier Probe value.")
        second = await agent.process_input(question, session_id=session_id)
        control = await agent.process_input(
            question, session_id=f"{session_id}-fresh", include_memories=False,
        )
        calls = await agent.observability_store.query_llm_calls(limit=100)
        dispatches = await agent.observability_store.backend.fetch_all(
            "SELECT tool_name, session_id, turn_id, result_status "
            "FROM a2a_tool_dispatches ORDER BY rowid",
        )

    [value] = minted
    generations = [(path, body) for path, body in transport.requests
                   if path in ("/v1/responses", "/v1/chat/completions")]
    turn_one = generations[:sum(1 for path, _ in transport.requests[:turn_one_requests]
                                if path in ("/v1/responses", "/v1/chat/completions"))]
    rest = generations[len(turn_one):]
    outputs = [item for _, body in turn_one for item in body.get("input", [])
               if item.get("type") == "function_call_output"]
    evidence = {
        "sol_model": SOL,
        "default_model": DEFAULT,
        "session_id": session_id,
        "turn_one_paths": [path for path, _ in turn_one],
        "turn_one_tool_output_carries_value": any(value in json.dumps(item) for item in outputs),
        "turn_one_replayed_reasoning_items": sum(
            1 for _, body in turn_one for item in body.get("input", [])
            if item.get("type") == "reasoning"
        ),
        "turn_one_reply_carries_value": value in first,
        "configured_default_unchanged": after_override == configured,
        "later_paths_and_models": [(path, body.get("model")) for path, body in rest],
        "turn_two_session_history_carries_value": bool(rest) and value in _session_text(rest[0][1]["messages"]),
        "turn_two_reply_carries_value": value in second,
        "control_session_history_carries_value": len(rest) > 1 and value in _session_text(rest[1][1]["messages"]),
        "control_reply_carries_value": value in control,
        "reply_excerpts": {"turn_one": first[:300], "turn_two": second[:300],
                           "control": control[:300]},
        "llm_calls": [
            {"session_id": call.session_id, "model": call.model, "success": call.success,
             PROVIDER_SERVED_KEY: call.metadata.get(PROVIDER_SERVED_KEY)}
            for call in reversed(calls) if call.model in (SOL, DEFAULT)
        ],
        "tool_dispatches": [
            dict(row) if isinstance(row, dict)
            else dict(zip(("tool_name", "session_id", "turn_id", "result_status"), row))
            for row in dispatches
        ],
    }
    target = os.environ.get("KESTREL_LIVE_EVIDENCE")
    if target:
        Path(target).write_text(json.dumps(evidence, indent=2, default=str))

    assert evidence["turn_one_paths"] and set(evidence["turn_one_paths"]) == {"/v1/responses"}
    assert evidence["turn_one_tool_output_carries_value"]
    assert evidence["turn_one_reply_carries_value"]
    assert evidence["configured_default_unchanged"]
    assert evidence["later_paths_and_models"] == [
        ("/v1/chat/completions", DEFAULT), ("/v1/chat/completions", DEFAULT),
    ]
    assert evidence["turn_two_session_history_carries_value"]
    assert evidence["turn_two_reply_carries_value"]
    assert not evidence["control_session_history_carries_value"]
    assert not evidence["control_reply_carries_value"]
    served = [call[PROVIDER_SERVED_KEY] for call in evidence["llm_calls"]]
    assert all(entry and entry["response_id"] and entry["model"] for entry in served)
    assert any(entry["api"] == "responses" and entry["model"].startswith(SOL) for entry in served)
    assert any(entry["api"] == "chat.completions" and entry["model"].startswith(DEFAULT)
               for entry in served)
