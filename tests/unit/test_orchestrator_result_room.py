"""#3740: the orchestrator tells a tool batch how much of the turn it has.

Before each model call the orchestrator prunes the turn to the model's context
window, stubbing tool results oldest first. A batch whose results do not fit
beside what the turn keeps loses its first result before the model reads it.
``OrchestratorResultRoom`` is the room the prune leaves a batch: these tests
hold it to the prune's own arithmetic, check that both tool loops publish it
to the batch they run, and that a subagent's tools, which the turn's prune
never sees, are not given it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.llm.response import LLMResponse as SdkLLMResponse
from kestrel_sdk.llm.response import ToolCall as SdkToolCall
from kestrel_sdk.tools.base import ToolCategory, ToolSchema

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.base import (
    Feature,
    OrchestratorResultRoom,
    bind_orchestrator_result_room,
    orchestrator_result_room,
)
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

#: A window small enough that a few results press on it.
CONTEXT_TOKENS = 4_000
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]


def _host():
    return SimpleNamespace(llm_service=SimpleNamespace(_context_limit=CONTEXT_TOKENS))


def _call(call_id: str, offset: int) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": {"path": "a.md", "offset": offset}},
    }


def _turn():
    """A turn mid-read: replayed history, an earlier batch and its (large)
    result, and the assistant message making a batch of two calls."""
    history = [
        {"role": "user", "content": "earlier question " * 20},
        {"role": "assistant", "content": "earlier answer " * 20},
    ]
    messages = [
        {"role": "system", "content": "system prompt " * 100},
        *history,
        {"role": "user", "content": "read a.md, all of it"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", 0)]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 3_000},
        {
            "role": "assistant",
            "content": "reading on",
            "tool_calls": [_call("c2", 3_000), _call("c3", 6_000)],
        },
    ]
    return messages, frozenset(id(m) for m in history)


def _room(messages, history_ids, calls=2) -> OrchestratorResultRoom:
    return OrchestratorEngineMixin._tool_batch_result_room(
        _host(), messages, TOOLS, history_ids=history_ids,
        has_user_message=True, model=None, calls=calls,
    )


def _prune(messages, history_ids):
    protected_prefix, history_len = OrchestratorEngineMixin._prefix_bounds(
        messages, history_ids, True
    )
    return OrchestratorEngineMixin._prune_orchestrator_messages(
        _host(), list(messages), TOOLS,
        protected_prefix=protected_prefix, history_len=history_len,
    )


def _with_results(messages, sizes, first_id=2):
    return messages + [
        {"role": "tool", "tool_call_id": f"c{first_id + i}", "content": "y" * size}
        for i, size in enumerate(sizes)
    ]


def test_a_batch_that_fits_its_room_reaches_the_model_whole():
    messages, history_ids = _turn()
    room = _room(messages, history_ids)
    assert room.calls == 2 and room.chars > 1_000

    sizes = [room.chars // 2, room.chars - room.chars // 2]
    pruned = _prune(_with_results(messages, sizes), history_ids)

    assert [len(m["content"]) for m in pruned[-2:]] == sizes
    # The room was made by reclaiming everything older: the history and the
    # earlier result.
    assert pruned[1]["content"] == "read a.md, all of it"
    assert pruned[-4]["content"].startswith("[Result truncated: was 3000 chars")


def test_one_character_past_the_room_loses_the_batchs_first_result():
    messages, history_ids = _turn()
    room = _room(messages, history_ids)

    sizes = [room.chars // 2, room.chars - room.chars // 2 + 1]
    pruned = _prune(_with_results(messages, sizes), history_ids)

    assert pruned[-2]["content"].startswith("[Result truncated: was")
    assert len(pruned[-1]["content"]) == sizes[1]


def test_a_later_response_has_the_room_once_this_batch_is_pruned():
    """A later response's text, calls and results fit ``later_chars``: it
    reaches the model whole whatever this batch's results were."""
    messages, history_ids = _turn()
    room = _room(messages, history_ids)
    turn = _with_results(messages, [room.chars // 2, room.chars - room.chars // 2])

    later_call = {
        "role": "assistant",
        "content": "and on",
        "tool_calls": [_call("c4", 9_000)],
    }
    spent = len("and on") + len('{"path": "a.md", "offset": 9000}')
    later = _with_results(turn + [later_call], [room.later_chars - spent], first_id=4)
    pruned = _prune(later, history_ids)

    assert len(pruned[-1]["content"]) == room.later_chars - spent
    assert room.later_chars == room.chars - 2 * 200


def test_the_room_counts_the_tool_schemas_and_the_window():
    messages, history_ids = _turn()
    room = _room(messages, history_ids)

    wider = OrchestratorEngineMixin._tool_batch_result_room(
        SimpleNamespace(llm_service=SimpleNamespace(_context_limit=CONTEXT_TOKENS * 2)),
        messages, TOOLS, history_ids=history_ids,
        has_user_message=True, model=None, calls=2,
    )
    more_tools = OrchestratorEngineMixin._tool_batch_result_room(
        _host(), messages, TOOLS * 2, history_ids=history_ids,
        has_user_message=True, model=None, calls=2,
    )

    assert wider.chars > room.chars > more_tools.chars


def test_a_share_divides_the_room_among_the_batchs_calls():
    assert OrchestratorResultRoom(chars=9_000, calls=3, later_chars=0).share() == 3_000
    assert OrchestratorResultRoom(chars=9_000, calls=0, later_chars=0).share() == 9_000


# ---------------------------------------------------------------------------
# Both tool loops publish the room to the batch they run
# ---------------------------------------------------------------------------


def _loop_agent(seen: list):
    agent = MagicMock()
    agent.features = {}
    agent._direct_tools = {}
    agent.llm_service = MagicMock()
    agent.llm_service._context_limit = CONTEXT_TOKENS
    agent._build_assistant_tool_history_msg = MagicMock(
        side_effect=lambda response: {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"arguments": tc.arguments}} for tc in response.tool_calls
            ],
        }
    )

    async def _batch(*_args, **_kwargs):
        seen.append(orchestrator_result_room())

    agent._execute_tool_batch = AsyncMock(side_effect=_batch)
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(agent)
    )
    agent._tool_batch_result_room = (
        OrchestratorEngineMixin._tool_batch_result_room.__get__(agent)
    )
    agent._build_all_tools = MagicMock(return_value=TOOLS)
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._known_tool_names = MagicMock(return_value=set())
    agent._prune_orchestrator_messages = MagicMock(
        side_effect=lambda msgs, _tools, **_kw: msgs
    )
    agent.is_request_cancelled = MagicMock(return_value=False)
    return agent


def _two_calls():
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id="c1", name="read", arguments={"offset": 0}),
            ToolCall(id="c2", name="read", arguments={"offset": 10}),
        ],
    )


def _expected_room(user_message: str) -> OrchestratorResultRoom:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": user_message},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"arguments": {"offset": 0}}},
                {"function": {"arguments": {"offset": 10}}},
            ],
        },
    ]
    return OrchestratorEngineMixin._tool_batch_result_room(
        _host(), messages, TOOLS, history_ids=frozenset(),
        has_user_message=True, model=None, calls=2,
    )


@pytest.mark.asyncio
async def test_the_non_streaming_loop_publishes_the_room_to_its_batch():
    seen: list = []
    agent = _loop_agent(seen)
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="done", tool_calls=None)
    )

    await OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)(
        response=_two_calls(), feature_tools=TOOLS, system_prompt="sys",
        force_local_only=False, effective_model="m", user_message="read it",
        session_id="s-1",
    )

    assert seen == [_expected_room("read it")]
    assert orchestrator_result_room() is None


@pytest.mark.asyncio
async def test_the_streaming_loop_publishes_the_room_to_its_batch():
    seen: list = []
    agent = _loop_agent(seen)

    async def _stream(*_args, **_kwargs):
        yield LLMResponse(content="done", tool_calls=None)

    agent.llm_service.stream_with_tool_detection = _stream

    async for _chunk in OrchestratorEngineMixin._handle_orchestrator_response_streaming.__get__(
        agent
    )(
        response=_two_calls(), feature_tools=TOOLS, system_prompt="sys",
        force_local_only=False, effective_model="m", user_message="read it",
        tool_events=[], tool_results=[], session_id="s-1",
    ):
        pass

    assert seen == [_expected_room("read it")]
    assert orchestrator_result_room() is None


# ---------------------------------------------------------------------------
# A subagent's tools answer to the subagent's loop, not the turn's prune
# ---------------------------------------------------------------------------


class _RoomProbe:
    name = "probe"

    def __init__(self):
        self.schema = ToolSchema(
            name="probe", description="records the room", category=ToolCategory.UTILITY,
        )
        self.seen = "not run"

    async def execute(self, **_kwargs):
        self.seen = orchestrator_result_room()
        return {"success": True}


class _ProbeFeature(Feature):
    @property
    def tool_description(self) -> str:
        return "probe"

    async def initialize(self):
        return None

    def get_tools(self):
        return [self._probe]


class _LoopLLM:
    async def generate(self, **_kwargs):
        return SdkLLMResponse(
            content=None, tool_calls=[SdkToolCall(id="c1", name="probe", arguments={})]
        )

    async def generate_with_messages(self, **_kwargs):
        return SdkLLMResponse(content="done", tool_calls=None)


class _InlineLLM:
    async def generate(self, **kwargs):
        await kwargs["tool_executor"]("probe", {})
        return SdkLLMResponse(content="done", tool_calls=None)

    async def generate_with_messages(self, **_kwargs):
        return SdkLLMResponse(content="done", tool_calls=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [_LoopLLM, _InlineLLM], ids=["loop", "inline"])
async def test_a_subagents_tools_are_not_given_the_turns_room(llm):
    probe = _RoomProbe()
    agent = SimpleNamespace(
        llm_service=llm(), did="did:test:probe", hooks_manager=None,
        get_turn_bound_session_id=lambda: "s-1",
    )
    feature = _ProbeFeature(agent)
    feature._probe = probe

    room = OrchestratorResultRoom(chars=5_000, calls=1, later_chars=4_800)
    with bind_orchestrator_result_room(room):
        envelope = await feature.execute_as_subagent(task="probe")
        assert orchestrator_result_room() is room

    assert envelope["success"] is True, envelope
    assert probe.seen is None


def test_the_binding_is_released_on_exit():
    room = OrchestratorResultRoom(chars=1, calls=1, later_chars=0)
    with bind_orchestrator_result_room(room):
        assert orchestrator_result_room() is room
    assert orchestrator_result_room() is None
