"""#4198: a real agent turn's Responses continuation is append-only.

A booted agent runs genuine turns on an ``openai:api`` route whose client is
the real ``openai`` SDK over :class:`tests.shared.openai_responses_wire.ResponsesWire`.
Its ``StateOfMindFeature`` is offered as the feature dispatcher
``state_of_mind_feature``; a dispatch runs the feature's own subagent (its own
Responses chain) and then mounts ``state_of_mind`` as a direct tool, the
registry's real progressive discovery. A queued Auto-mode change makes the
turn's first request carry a genuine inline operator notice. The tool's body
is replaced so it reports a fresh value the prompt and history never hold;
its dispatch is the agent's own.

Every assertion about what the provider received reads the requests exactly
as the SDK serialized them, never the agent's message list.

Checked against OpenAI's documented contract only
(https://developers.openai.com/api/docs/guides/keep-prefix-unchanged): no
live upstream request is made here.
"""

from __future__ import annotations

import contextlib
import functools
import json
import re
import uuid
from typing import Any, Dict, List

import pytest

from kestrel_sdk.tools import ToolResult
from kestrel_sovereign.features.state_of_mind import StateOfMindFeature
from kestrel_sovereign.llm.service import LLMService
from tests.shared.openai_responses_wire import (
    CHAT_MODEL,
    CHAT_SERVED,
    MODEL,
    SERVED,
    ChatReply,
    Reply,
    ResponsesWire,
    StallingWire,
    call,
    input_types,
    message,
    reasoning,
    tool_names,
    wire_items,
)

#: The agent's configured default; a turn's exact override names another.
CONFIGURED = "gpt-6.2-sol-2026-11-01"
OVERRIDE = f"openai:api/{MODEL}"
DISPATCHER = "state_of_mind_feature"
DIRECT = "state_of_mind"
#: The provider's own argument text: compact, with escaped non-ASCII, so a
#: re-encoded replay (``{"task": "résumé ..."}``) differs from it.
DISPATCH_ARGUMENTS = '{"task":"r\\u00e9sum\\u00e9 of the current state"}'
DIRECT_ARGUMENTS = "{ }"
QUESTION = "What value did you report earlier?"
TOKEN = re.compile(r"tok-[0-9a-f]{12}")

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("isolated_process_rate_limiter")]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _outputs(body: Dict[str, Any]) -> List[str]:
    return [item["output"] for item in body["input"] if item.get("type") == "function_call_output"]


def _token_in(text: str) -> str:
    found = TOKEN.search(text)
    return found.group(0) if found else ""


def _main(body: Dict[str, Any]) -> bool:
    return DISPATCHER in tool_names(body)


def _rid() -> str:
    return f"resp_{uuid.uuid4().hex[:10]}"


def _discovery_script(body: Dict[str, Any], _index: int) -> Reply:
    """The model's side of one dispatch, one discovered call and one answer."""
    outputs = _outputs(body)
    if not _main(body):
        if not outputs:
            return Reply([reasoning("rs_sub_1", "ENC-SUB-1"), call("fc_sub_1", "call_sub_1", DIRECT, "{}")], _rid())
        return Reply([message("msg_sub", f"The state report holds {_token_in(' '.join(outputs))}.")], _rid())
    if not outputs:
        return Reply([
            reasoning("rs_main_1", "ENC-MAIN-1"),
            message("msg_pre", "Checking the state first.", phase="commentary"),
            call("fc_main_1", "call_main_1", DISPATCHER, DISPATCH_ARGUMENTS),
        ], _rid())
    if len(outputs) == 1:
        return Reply([reasoning("rs_main_2", "ENC-MAIN-2"), call("fc_main_2", "call_main_2", DIRECT, DIRECT_ARGUMENTS)], _rid())
    return Reply([reasoning("rs_main_3", "ENC-MAIN-3"), message("msg_final", f"The value is {_token_in(' '.join(outputs))}.")], _rid())


def _answer_script(body: Dict[str, Any], _index: int) -> Reply:
    """A question turn: the model knows the value only if it was sent."""
    seen = _token_in(json.dumps(body))
    text = f"It was {seen}." if seen else "I do not know of any value."
    return Reply([reasoning("rs_q", "ENC-QUESTION"), message("msg_q", text)], _rid())


def _openai_route(agent, wire: ResponsesWire, model: str = CONFIGURED) -> dict:
    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    info = ProviderRegistry({})._build_route(
        "openai", "api", {"is_cloud": True},
        {"adapter": "OpenAIAdapter", "api_key": "test-key", "model": model},
    )
    [provider] = agent.llm_service._convert_providers_format([info])
    provider["client"] = wire.client()
    return provider


@contextlib.asynccontextmanager
async def _agent(tmp_path, monkeypatch, wire: ResponsesWire, *, name: str, configured: str = CONFIGURED):
    from tests.shared.booted_agent import booted_agent

    async with booted_agent(
        tmp_path, name=name, provenance="test:openai_responses_continuation",
        features={"StateOfMindFeature"},
    ) as agent:
        service = agent.llm_service
        provider = _openai_route(agent, wire, configured)
        monkeypatch.setattr(service, "providers", [provider])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": [provider["name"]],
        })
        monkeypatch.setattr(service, "_mandate_preference", {})
        yield agent


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def fresh_token(monkeypatch):
    """``state_of_mind`` reports a value the prompt and history never hold."""
    token = f"tok-{uuid.uuid4().hex[:12]}"

    @functools.wraps(StateOfMindFeature.get_state_of_mind)
    async def report(self) -> ToolResult:
        return ToolResult.ok(confirmation=f"state report {token}", data={"token": token})

    monkeypatch.setattr(StateOfMindFeature, "get_state_of_mind", report)
    return token


@pytest.fixture
def llm_rows(monkeypatch) -> List[Dict[str, Any]]:
    """Every attempt the service records, as handed to its ``llm_calls`` writer."""
    rows: List[Dict[str, Any]] = []
    original = LLMService._log_llm_call

    @functools.wraps(original)
    async def record(self, **row):
        rows.append(row)
        return await original(self, **row)

    monkeypatch.setattr(LLMService, "_log_llm_call", record)
    return rows


def _served(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row["metadata"]["provider_served"] for row in rows
        if "provider_served" in (row.get("metadata") or {})
    ]


async def _turn(agent, path: str, prompt: str, session_id: str, model_override=None) -> str:
    if path == "stream":
        from kestrel_sovereign.agent.streaming import is_only_sentinels

        chunks = [
            chunk async for chunk in agent.process_input_streaming(
                prompt, session_id=session_id,
                request_id=f"{session_id}-{uuid.uuid4().hex[:6]}",
                model_override=model_override,
            )
        ]
        return "".join(c for c in chunks if isinstance(c, str) and not is_only_sentinels(c))
    return await agent.process_input(prompt, session_id=session_id, model_override=model_override)


def _assert_prefix(earlier: bytes, later: bytes) -> None:
    """``later`` begins with every item ``earlier`` sent, as sent."""
    sent = wire_items(earlier)
    assert wire_items(later)[: len(sent)] == sent
    assert wire_items(later, "tools") == wire_items(earlier, "tools")
    for key in ("include", "store", "reasoning", "text"):
        assert json.loads(later).get(key) == json.loads(earlier).get(key)


PATHS = pytest.mark.parametrize("path", ["invoke", "stream"])


# ---------------------------------------------------------------------------
# Serialized-prefix equality: operator notice and real progressive discovery
# ---------------------------------------------------------------------------


@PATHS
async def test_discovery_and_operator_notice_keep_the_serialized_prefix(
    tmp_path, monkeypatch, fresh_token, llm_rows, path,
):
    wire = ResponsesWire(_discovery_script)
    async with _agent(tmp_path, monkeypatch, wire, name="Prefix") as agent:
        agent.operator_signal_producer.enqueue_auto_mode("session")
        reply = await _turn(agent, path, "How are you?", f"prefix-{path}", OVERRIDE)

    assert not wire.unexpected
    assert fresh_token in reply
    raw = wire.raw()
    main = [body for body in raw if _main(json.loads(body))]
    assert len(main) == 3, [input_types(json.loads(body)) for body in raw]
    first, after_dispatch, after_direct = (json.loads(body) for body in main)

    # The first request carries the operator notice, after the user turn.
    roles = [(item.get("type"), item.get("role")) for item in first["input"]]
    assert roles == [("message", "system"), ("message", "user"), ("message", "system")]
    assert "Operator context: auto-mode" in first["input"][2]["content"]
    assert DISPATCHER in tool_names(first) and DIRECT not in tool_names(first)

    # Each continuation replays its predecessor unchanged and only appends.
    _assert_prefix(main[0], main[1])
    _assert_prefix(main[1], main[2])

    # The first reply's output, in order and as emitted, then the tool output,
    # then the tool the dispatch exposed, declared by a developer item.
    gained = after_dispatch["input"][len(first["input"]):]
    assert [item["type"] for item in gained] == [
        "reasoning", "message", "function_call", "function_call_output", "additional_tools",
    ]
    assert gained[0]["encrypted_content"] == "ENC-MAIN-1"
    assert gained[1]["phase"] == "commentary"
    assert gained[2]["arguments"] == DISPATCH_ARGUMENTS
    assert gained[3]["call_id"] == "call_main_1"
    assert gained[4]["role"] == "developer"
    assert [tool["name"] for tool in gained[4]["tools"]] == [DIRECT]
    assert DIRECT not in tool_names(after_dispatch)
    # The registry withdrew the dispatcher; its declaration stays, and the
    # model is restricted to what the registry offers now.
    choice = after_dispatch["tool_choice"]
    allowed = [tool["name"] for tool in choice["tools"]]
    assert (choice["type"], choice["mode"]) == ("allowed_tools", "auto")
    assert DISPATCHER in tool_names(after_dispatch) and DISPATCHER not in allowed
    assert DIRECT in allowed

    # The discovered tool's declaration is replayed where it was first sent,
    # once, and the direct call's own argument text is replayed exactly.
    later = after_direct["input"][len(after_dispatch["input"]):]
    assert [item["type"] for item in later] == ["reasoning", "function_call", "function_call_output"]
    assert later[1]["arguments"] == DIRECT_ARGUMENTS
    assert input_types(after_direct).count("additional_tools") == 1
    assert fresh_token in later[2]["output"]

    # The provider's own identification of each call is on its llm_calls row.
    served = _served(llm_rows)
    assert all(entry["model"] == SERVED and entry["response_id"].startswith("resp_")
               for entry in served)
    statuses = [entry["continuation"]["status"] for entry in served]
    # The main chain and the subagent's own chain each start fresh once.
    assert sorted(statuses) == ["continued"] * 3 + ["fresh"] * 2
    continued = [entry for entry in _served(llm_rows) if entry["continuation"].get("additional_tools")]
    assert [entry["continuation"]["additional_tools"] for entry in continued] == [[DIRECT]]


@PATHS
async def test_a_reply_served_by_another_model_ends_the_chain(
    tmp_path, monkeypatch, fresh_token, llm_rows, path,
):
    """The policy on a reply from a mismatched model: it is the provider's
    answer and the turn acts on it (its tool call runs), but its chain ends,
    the substitution is recorded, and the next request replays nothing."""
    replies = iter([
        Reply([reasoning("rs_a", "ENC-A"), call("fc_a", "call_a", DIRECT, "{}")], "resp_a"),
        Reply([reasoning("rs_b", "ENC-B"), call("fc_b", "call_b", DIRECT, "{}")], "resp_b",
              model="gpt-6.2-sol-2026-12-01"),
        Reply([message("msg_c", "Done.")], "resp_c"),
    ])
    wire = ResponsesWire(lambda body, index: next(replies))
    ran: List[str] = []
    reporting = StateOfMindFeature.get_state_of_mind

    @functools.wraps(reporting)
    async def counted(self):
        ran.append("state_of_mind")
        return await reporting(self)

    monkeypatch.setattr(StateOfMindFeature, "get_state_of_mind", counted)
    async with _agent(tmp_path, monkeypatch, wire, name="Mismatch") as agent:
        feature = next(f for f in agent.features.values() if isinstance(f, StateOfMindFeature))
        agent._register_explored_feature_tools(feature)
        await _turn(agent, path, "Report twice.", f"mismatch-{path}", OVERRIDE)
    # Both replies' tool calls ran, the mismatched one's included.
    assert ran == ["state_of_mind", "state_of_mind"]

    first, second, third = wire.raw()
    _assert_prefix(first, second)
    assert input_types(json.loads(third)) == ["message", "message"]
    assert b"ENC-" not in third
    served = _served(llm_rows)
    assert served[1]["continuation"]["chain_ended"] == "served_model_mismatch"
    assert served[1]["model"] == "gpt-6.2-sol-2026-12-01"
    # The next request is told the chain was retired and why, never that
    # there was no snapshot. The mismatched reply's tool call ran: its result
    # is one of the four tool items (two calls, two results) it left out.
    assert served[2]["continuation"] == {
        "status": "boundary", "reason": "retired",
        "retired_reason": "served_model_mismatch", "retired_response_id": "resp_b",
        "omitted_tool_items": 4,
    }
    assert served[1]["continuation"]["expected_model"] == SERVED
    assert served[1]["snapshot"] == {"stored": False, "reason": "served_model_mismatch"}


# ---------------------------------------------------------------------------
# Between turns: a snapshot expires with its turn
# ---------------------------------------------------------------------------


def _text_turn_script(body: Dict[str, Any], _index: int) -> Reply:
    if QUESTION in json.dumps(body):
        return _answer_script(body, _index)
    return Reply([reasoning("rs_t", "ENC-TEXT"), message("msg_t", "Noted.")], _rid())


def _tool_turn_script(body: Dict[str, Any], index: int) -> Reply:
    if QUESTION in json.dumps(body):
        return _answer_script(body, index)
    return _discovery_script(body, index)


@PATHS
@pytest.mark.parametrize("first_turn", ["tool", "text"])
@pytest.mark.parametrize("second_override", [None, OVERRIDE], ids=["configured", "override"])
async def test_the_next_turn_replays_nothing_from_the_last(
    tmp_path, monkeypatch, fresh_token, llm_rows, path, first_turn, second_override,
):
    script = _tool_turn_script if first_turn == "tool" else _text_turn_script
    wire = ResponsesWire(script)
    session = f"between-{path}-{first_turn}"
    async with _agent(tmp_path, monkeypatch, wire, name="Between") as agent:
        preference = agent.llm_service.get_model_preference()
        await _turn(agent, path, "How are you?", session, OVERRIDE)
        turn_one = len(wire.raw())
        rows_one = len(llm_rows)
        answer = await _turn(agent, path, QUESTION, session, second_override)
        assert agent.llm_service.get_model_preference() == preference

    next_turn = wire.raw()[turn_one]
    body = json.loads(next_turn)
    assert set(input_types(body)) == {"message"}
    assert b"ENC-" not in next_turn.replace(b"ENC-QUESTION", b"")
    assert body["model"] == (MODEL if second_override else CONFIGURED)
    assert json.loads(wire.raw()[0])["model"] == MODEL
    assert _served(llm_rows[rows_one:])[0]["continuation"] == {"status": "fresh"}

    # What each turn's rows record is what each call actually used: the
    # model it asked for (the override, then the configured default unless
    # overridden again) and the model and response id the provider served.
    chat_rows = [row for row in llm_rows if "provider_served" in (row.get("metadata") or {})]
    assert [
        (row["metadata"]["provider_served"]["model"], row["metadata"]["provider_served"]["response_id"])
        for row in chat_rows
    ] == wire.issued
    requested = [row["model"] for row in chat_rows]
    # Each row's requested model is the model its request actually carried.
    assert requested == [json.loads(body)["model"] for body in wire.raw()]
    first_turn_rows = sum(1 for row in llm_rows[:rows_one] if row in chat_rows)
    assert requested[:first_turn_rows] == [MODEL] * first_turn_rows
    assert requested[first_turn_rows:] == [MODEL if second_override else CONFIGURED] * (
        len(requested) - first_turn_rows
    )
    if first_turn == "tool":
        # The value reached the next turn only as the earlier reply's text.
        assert fresh_token in next_turn.decode()
        assert fresh_token in answer


# ---------------------------------------------------------------------------
# Negative control: an independent agent never receives the value
# ---------------------------------------------------------------------------


@PATHS
async def test_an_independent_agent_never_receives_the_tool_value(
    tmp_path, monkeypatch, fresh_token, record_property, path,
):
    wire = ResponsesWire(_tool_turn_script)
    async with _agent(tmp_path / "positive", monkeypatch, wire, name="Positive") as agent:
        await _turn(agent, path, "How are you?", "positive", OVERRIDE)
        sent = len(wire.raw())
        same_session = await _turn(agent, path, QUESTION, "positive")
        recalled = len(wire.raw())
        other_session = await _turn(agent, path, QUESTION, "another-session")
    positive = b"".join(wire.raw()[sent:recalled])
    assert fresh_token.encode() in positive
    assert fresh_token in same_session
    # Kestrel's long-term memory may legitimately recall it across sessions
    # of the same agent; record what this host did rather than assume.
    record_property("same_host_fresh_session_request_has_value",
                    fresh_token.encode() in b"".join(wire.raw()[recalled:]))
    record_property("same_host_fresh_session_answer_has_value", fresh_token in other_session)

    control = ResponsesWire(_tool_turn_script)
    async with _agent(tmp_path / "control", monkeypatch, control, name="Control") as agent:
        answer = await _turn(agent, path, QUESTION, "control")
    assert control.raw(), "the control made no provider request"
    assert all(fresh_token.encode() not in body for body in control.requests_bytes())
    assert fresh_token not in answer


# ---------------------------------------------------------------------------
# Evidence: before a failure is raised, and through a disconnect
# ---------------------------------------------------------------------------


async def _failed_turn(agent, path: str, session: str) -> None:
    try:
        await _turn(agent, path, "How are you?", session, OVERRIDE)
    except Exception:  # noqa: BLE001 - the failed attempt's row is the subject
        pass


@PATHS
async def test_a_failed_response_records_usage_and_served_model_before_raising(
    tmp_path, monkeypatch, llm_rows, path,
):
    wire = ResponsesWire(lambda body, index: Reply(
        [], "resp_failed", status="failed",
        error={"code": "server_error", "message": "The model could not finish."},
    ))
    async with _agent(tmp_path, monkeypatch, wire, name="Failed") as agent:
        await _failed_turn(agent, path, f"failed-{path}")

    [row] = [row for row in llm_rows if (row.get("metadata") or {}).get("provider_served")]
    assert row["success"] is False
    assert row["metadata"]["provider_served"] == {
        "api": "responses", "model": SERVED, "response_id": "resp_failed",
        "continuation": {"status": "fresh"},
    }
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_input_tokens"]) == (100, 30, 20)
    assert "server_error" in row["error_message"]
    assert "The model could not finish." in row["error_message"]


@PATHS
async def test_a_disconnect_keeps_the_identity_response_created_named(
    tmp_path, monkeypatch, llm_rows, path,
):
    wire = ResponsesWire(lambda body, index: Reply([], "resp_cut", disconnect_after_created=True))
    async with _agent(tmp_path, monkeypatch, wire, name="Cut") as agent:
        await _failed_turn(agent, path, f"cut-{path}")

    rows = [row for row in llm_rows if (row.get("metadata") or {}).get("provider_served")]
    assert rows, [row.get("metadata") for row in llm_rows]
    for row in rows:
        assert row["success"] is False
        assert row["metadata"]["provider_served"]["model"] == SERVED
        assert row["metadata"]["provider_served"]["response_id"] == "resp_cut"


# ---------------------------------------------------------------------------
# Context budget: the replay never restores what the tool loop pruned
# ---------------------------------------------------------------------------

STORY_PROMPT = "First, tell me a long story."
REPORT_PROMPT = "Second, report the current state."
STORY = "STORY-" + "a long answer the tool loop will later shed. " * 70


def _budget_script(body: Dict[str, Any], index: int) -> Reply:
    text = json.dumps(body)
    if REPORT_PROMPT not in text:
        return Reply([reasoning(f"rs_s{index}", "ENC-STORY"), message(f"msg_s{index}", STORY)], _rid())
    outputs = _outputs(body)
    if not outputs:
        return Reply([
            reasoning(f"rs_r{index}", f"ENC-REPORT-{index}"),
            call(f"fc_r{index}", f"call_r{index}", DIRECT, "{}"),
        ], _rid())
    return Reply([message(f"msg_r{index}", f"The value is {_token_in(' '.join(outputs))}.")], _rid())


def _message_chars(messages: List[Dict[str, Any]]) -> int:
    """The tool loop's own measure (``_prune_orchestrator_messages``)."""
    total = 0
    for item in messages:
        total += len(item.get("content", "") or "")
        for tool_call in item.get("tool_calls", []):
            total += len(json.dumps(tool_call.get("function", {}).get("arguments", {})))
    return total


@pytest.fixture
def squeezed_prune(monkeypatch) -> List[int]:
    """The real prune, with a budget that fits the turn but not its history.

    Only a pass that carries replayed history is squeezed; the budget then
    holds everything but that history, so the real prune must shed it.
    Returns how many messages each squeezed pass shed.
    """
    import math

    from kestrel_sovereign import kestrel_agent
    from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin

    original = OrchestratorEngineMixin._prune_orchestrator_messages
    shed: List[int] = []

    @functools.wraps(original)
    def prune(self, messages, tools, context_limit=None, protected_prefix=2, history_len=0, model=None):
        if not history_len or context_limit is not None:
            return original(self, messages, tools, context_limit, protected_prefix, history_len, model)
        kept = messages[:1] + messages[1 + history_len:]
        need = _message_chars(kept) + sum(len(json.dumps(tool)) for tool in tools or ()) + 200
        limit = math.ceil(need / (3.5 * (1 - kestrel_agent.CONTEXT_RESERVE_FRACTION)))
        pruned = original(self, messages, tools, limit, protected_prefix, history_len, model)
        shed.append(len(messages) - len(pruned))
        return pruned

    monkeypatch.setattr(OrchestratorEngineMixin, "_prune_orchestrator_messages", prune)
    return shed


@PATHS
async def test_a_continuation_never_restores_history_the_tool_loop_pruned(
    tmp_path, monkeypatch, fresh_token, llm_rows, squeezed_prune, path,
):
    wire = ResponsesWire(_budget_script)
    session = f"budget-{path}"
    async with _agent(tmp_path, monkeypatch, wire, name="Budget") as agent:
        feature = next(f for f in agent.features.values() if isinstance(f, StateOfMindFeature))
        agent._register_explored_feature_tools(feature)
        await _turn(agent, path, STORY_PROMPT, session, OVERRIDE)
        turn_one = len(wire.raw())
        rows_one = len(llm_rows)
        reply = await _turn(agent, path, REPORT_PROMPT, session, OVERRIDE)

    story = STORY.encode()
    first, after_tool, *rest = wire.raw()[turn_one:]
    # Preconditions: the turn's first request replays the earlier answer, and
    # the real prune shed it before the continuation.
    assert story in first
    assert squeezed_prune and squeezed_prune[0] >= 2

    # The continuation does not restore it: its chain ends at a recorded boundary.
    assert story not in after_tool
    assert b"ENC-" not in after_tool and b"function_call" not in after_tool
    served = _served(llm_rows[rows_one:])
    boundary = served[1]["continuation"]
    assert (boundary["status"], boundary["reason"]) == ("boundary", "prefix_pruned")

    # The boundary reply starts a chain that fits, and the turn completes.
    assert rest and all(story not in body for body in rest)
    assert served[-1]["continuation"]["status"] == "continued"
    assert fresh_token in reply


# ---------------------------------------------------------------------------
# A watchdog timeout and a disconnect after the terminal event (m, n)
# ---------------------------------------------------------------------------


@PATHS
async def test_timeout_keeps_created_identity(tmp_path, monkeypatch, llm_rows, path):
    """The inactivity watchdog cancels a stream that went silent after
    ``response.created``: the failed attempt's row still names the model and
    response id the provider had already given."""
    from kestrel_sovereign.agent.orchestrator_engine import LLMCallWatchdog

    # Long enough that the in-process wire has always delivered
    # ``response.created`` first, however loaded the host; the stream then
    # stays silent, so only the watchdog ends it.
    monkeypatch.setattr(LLMCallWatchdog, "for_call", classmethod(
        lambda cls, *args, **kwargs: cls(1.0, attempt_timeout=lambda route: 1.0)))
    wire = StallingWire("resp_stall")
    async with _agent(tmp_path, monkeypatch, wire, name="Timeout") as agent:
        await _failed_turn(agent, path, f"timeout-{path}")

    rows = [row for row in llm_rows if row["provider"] == "openai:api" and row["model"] == MODEL]
    assert rows, llm_rows
    for row in rows:
        assert row["success"] is False
        served = row["metadata"].get("provider_served", {})
        assert (served.get("model"), served.get("response_id")) == (SERVED, "resp_stall"), row
        # No terminal response arrived: the usage is unknown, never zero.
        assert (row["input_tokens"], row["output_tokens"]) == (None, None)
        assert row["usage_available"] is False


@PATHS
@pytest.mark.parametrize("usage", ["reported", "none_reported"])
async def test_a_disconnect_after_the_terminal_event_records_its_usage(
    tmp_path, monkeypatch, llm_rows, path, usage,
):
    """The stream broke after the terminal event: the failed attempt's row
    keeps the usage that event reported, or records none (unknown, never
    zero or an estimate) when it reported none."""
    wire = ResponsesWire(lambda body, index: Reply(
        [message("msg", "Done.")], "resp_terminal", disconnect_after_terminal=True,
        **({} if usage == "reported" else {"usage": None}),
    ))
    async with _agent(tmp_path, monkeypatch, wire, name="Terminal") as agent:
        await _failed_turn(agent, path, f"terminal-{path}-{usage}")

    rows = [row for row in llm_rows if (row.get("metadata") or {}).get("provider_served")]
    assert rows, [row.get("metadata") for row in llm_rows]
    for row in rows:
        assert row["success"] is False
        assert row["metadata"]["provider_served"]["response_id"] == "resp_terminal"
        counts = (row["input_tokens"], row["output_tokens"], row["cache_read_input_tokens"])
        if usage == "reported":
            assert counts == (100, 30, 20)
            assert row["usage_available"] is True
        else:
            assert counts == (None, None, None)
            assert row["usage_available"] is False


@PATHS
async def test_terminal_stall_keeps_reported_usage(tmp_path, monkeypatch, llm_rows, path):
    """The terminal response reported its usage and the stream then went
    silent before its clean end, so the inactivity watchdog cancelled it
    (#4205). The failed attempt's row keeps the usage the provider reported
    and the identity it gave; ``test_timeout_keeps_created_identity`` is the
    same stall before any terminal response, whose usage stays unknown."""
    from kestrel_sovereign.agent.orchestrator_engine import LLMCallWatchdog

    monkeypatch.setattr(LLMCallWatchdog, "for_call", classmethod(
        lambda cls, *args, **kwargs: cls(1.0, attempt_timeout=lambda route: 1.0)))
    wire = ResponsesWire(lambda body, index: Reply(
        [message("msg", "Done.")], "resp_terminal_stall", stall_after_terminal=True,
    ))
    async with _agent(tmp_path, monkeypatch, wire, name="TerminalStall") as agent:
        await _failed_turn(agent, path, f"terminal-stall-{path}")

    assert wire.silent and all(body.closed for body in wire.silent)
    rows = [row for row in llm_rows if row["provider"] == "openai:api" and row["model"] == MODEL]
    assert rows, llm_rows
    for row in rows:
        assert row["success"] is False
        served = row["metadata"]["provider_served"]
        assert (served["model"], served["response_id"]) == (SERVED, "resp_terminal_stall")
        counts = (row["input_tokens"], row["output_tokens"], row["cache_read_input_tokens"])
        assert counts == (100, 30, 20), counts
        assert row["usage_available"] is True


# ---------------------------------------------------------------------------
# A Sol tool turn, then the configured chat-completions default (#4205)
# ---------------------------------------------------------------------------


def _chat_answer(body: Dict[str, Any], _index: int) -> ChatReply:
    """The default model knows the value only if the request carried it."""
    seen = _token_in(json.dumps(body))
    text = f"It was {seen}." if seen else "I do not know of any value."
    return ChatReply(text, f"chatcmpl-{uuid.uuid4().hex[:10]}")


@PATHS
async def test_a_sol_tool_turn_then_the_chat_default_in_one_session(
    tmp_path, monkeypatch, fresh_token, llm_rows, path,
):
    """Turn one asks for ``gpt-6.2-sol`` and runs a real tool call on the
    Responses API; turn two, in the same session, is served by the agent's
    configured non-Sol default on chat completions. Turn two replays nothing
    opaque: the value reaches it only as turn one's answer in the session's
    history. Every row records the model and response id its own reply named,
    and the configured preference is unchanged."""
    wire = ResponsesWire(_tool_turn_script, chat_script=_chat_answer)
    session = f"sol-then-default-{path}"
    async with _agent(tmp_path, monkeypatch, wire, name="SolThenDefault", configured=CHAT_MODEL) as agent:
        preference = agent.llm_service.get_model_preference()
        first = await _turn(agent, path, "How are you?", session, OVERRIDE)
        responses_sent = len(wire.raw())
        rows_one = len(llm_rows)
        answer = await _turn(agent, path, QUESTION, session)
        assert agent.llm_service.get_model_preference() == preference

    assert not wire.unexpected
    assert fresh_token in first
    # Turn one's own tool call ran on the Responses API.
    turn_one = [json.loads(body) for body in wire.raw()]
    assert any(
        fresh_token in output for body in turn_one for output in _outputs(body)
    ), "turn one never ran its tool"
    assert all(body["model"] == MODEL for body in turn_one)
    # Turn two never touched the Responses API.
    assert len(wire.raw()) == responses_sent
    chats = wire.raw("/v1/chat/completions")
    assert chats, "turn two was not served by the chat-completions default"
    for raw in chats:
        body = json.loads(raw)
        assert body["model"] == CHAT_MODEL
        assert b"ENC-" not in raw and b"encrypted_content" not in raw
        assert b'"function_call"' not in raw and b'"reasoning"' not in raw
        assert all(m["role"] != "tool" and not m.get("tool_calls") for m in body["messages"])
    question = [json.loads(raw) for raw in chats if QUESTION.encode() in raw]
    assert question
    carriers = [
        m["role"] for body in question for m in body["messages"]
        if fresh_token in json.dumps(m.get("content"))
    ]
    assert carriers and set(carriers) == {"assistant"}, carriers
    assert fresh_token in answer

    served_rows = [row for row in llm_rows if "provider_served" in (row.get("metadata") or {})]
    assert [
        (row["metadata"]["provider_served"].get("model"), row["metadata"]["provider_served"].get("response_id"))
        for row in served_rows
    ] == wire.issued
    turn_two = [row for row in llm_rows[rows_one:] if row in served_rows]
    assert turn_two
    for row in turn_two:
        served = row["metadata"]["provider_served"]
        assert (row["model"], served["api"], served["model"]) == (CHAT_MODEL, "chat_completions", CHAT_SERVED)
        assert served["continuation"] == {"status": "fresh"}
        assert row["success"] is True
    for row in served_rows[: len(served_rows) - len(turn_two)]:
        assert row["model"] == MODEL and row["metadata"]["provider_served"]["api"] == "responses"


@PATHS
async def test_a_cancelled_attempt_is_not_recorded_as_failed_after_its_usage_arrived(
    tmp_path, monkeypatch, llm_rows, path,
):
    """The caller cancels the turn after the terminal response reported its
    usage and before the stream ended. The adapter's cancellation carries
    that usage (``test_every_exit_keeps_what_the_provider_reported``), but a
    cancelled attempt is not a failed one on either path (#3938): the
    cancellation is re-raised and no row is written for the attempt."""
    import asyncio

    wire = ResponsesWire(lambda body, index: Reply(
        [message("msg", "Done.")], "resp_cancelled", stall_after_terminal=True,
    ))
    async with _agent(tmp_path, monkeypatch, wire, name="Cancelled") as agent:
        turn = asyncio.create_task(_turn(agent, path, "How are you?", f"cancelled-{path}", OVERRIDE))

        async def silent():
            while not wire.silent:
                await asyncio.sleep(0.01)
            await wire.silent[0].stalled.wait()

        try:
            await asyncio.wait_for(silent(), timeout=10)
        finally:
            turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    assert wire.silent[0].closed
    assert [row for row in llm_rows if row["model"] == MODEL] == []


@pytest.mark.parametrize("path, failure", [
    ("invoke", "protocol_error"),
    ("stream", "protocol_error"),
    ("stream", "transport_after_usage"),
])
async def test_a_failed_chat_default_turn_records_what_its_reply_reported(
    tmp_path, monkeypatch, llm_rows, path, failure,
):
    """On the configured chat-completions default, the reply named its model
    and id and reported its usage before the attempt failed: with tool
    arguments that are not JSON, or with a stream that broke after its usage
    chunk (#4205). The failed attempt's row keeps all of it, never the
    requested alias in place of the served model."""
    usage = {
        "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20},
    }
    ending = (
        {"tool_arguments": '{"key": '} if failure == "protocol_error"
        else {"disconnect_after_usage": True}
    )
    wire = ResponsesWire(
        lambda body, index: Reply([], "unused"),
        chat_script=lambda body, index: ChatReply(
            "Partial.", f"chatcmpl-failed-{index}", usage=usage, **ending,
        ),
    )
    async with _agent(tmp_path, monkeypatch, wire, name="ChatFailed", configured=CHAT_MODEL) as agent:
        with contextlib.suppress(Exception):
            await _turn(agent, path, "How are you?", f"chat-failed-{path}-{failure}")

    assert not wire.unexpected and not wire.raw(), "the chat default reached the Responses API"
    rows = [row for row in llm_rows if row["model"] == CHAT_MODEL]
    assert rows, llm_rows
    assert [
        (row["metadata"]["provider_served"]["model"], row["metadata"]["provider_served"]["response_id"])
        for row in rows
    ] == wire.issued
    for row in rows:
        assert row["success"] is False
        served = row["metadata"]["provider_served"]
        assert (served["api"], served["model"]) == ("chat_completions", CHAT_SERVED)
        assert served["continuation"] == {"status": "fresh"}
        counts = (row["input_tokens"], row["output_tokens"], row["cache_read_input_tokens"])
        assert counts == (100, 30, 20), counts
        assert row["usage_available"] is True


@PATHS
async def test_a_cancelled_chat_default_attempt_is_not_recorded_as_failed(
    tmp_path, monkeypatch, llm_rows, path,
):
    """The caller cancels a turn on the configured chat-completions default
    once its reply went silent: on a stream, after the usage chunk and before
    ``[DONE]``, so the adapter's cancellation carries that usage
    (``test_every_chat_exit_keeps_what_the_provider_reported``). As on the
    Responses route, a cancelled attempt is not a failed one (#3938, #4205):
    the cancellation is re-raised and no row is written for the attempt."""
    import asyncio

    wire = ResponsesWire(
        lambda body, index: Reply([], "unused"),
        chat_script=lambda body, index: ChatReply("Done.", "chatcmpl-cancelled", stall_after_usage=True),
    )
    async with _agent(tmp_path, monkeypatch, wire, name="ChatCancelled", configured=CHAT_MODEL) as agent:
        turn = asyncio.create_task(_turn(agent, path, "How are you?", f"chat-cancelled-{path}"))

        async def silent():
            while not wire.silent:
                await asyncio.sleep(0.01)
            await wire.silent[0].stalled.wait()

        try:
            await asyncio.wait_for(silent(), timeout=10)
        finally:
            turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    assert not wire.raw(), "the chat default reached the Responses API"
    [request] = wire.bodies("/v1/chat/completions")
    # Only a stream had sent its usage chunk before going silent.
    assert bool(request.get("stream")) is (path == "stream")
    assert wire.silent[0].closed
    assert [row for row in llm_rows if row["model"] == CHAT_MODEL] == []
