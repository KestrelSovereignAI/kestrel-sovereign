"""The subagent's gate bounds a request from its bytes, not tiktoken (#4068).

On 2026-11-13 tiktoken's ``count`` measured a subagent request at 663,779
tokens. The provider counted it at 1,158,906 and rejected it, after the gate
had passed it against an 850,000-token budget. ``TokenCounter.upper_bound``
bounds the same request at ``ceil(wire_bytes / bpt_min) + overhead``, with
both values ratcheted from the counts providers report.

Each test is named for its row in the #4068 binding scope (#4067 comment
6870146659, amended by #4068 comment 6883985733). No test makes a live call,
starts a host, or runs a provider tokenizer.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from kestrel_sdk.llm.response import LLMResponse
from kestrel_sovereign.agent import token_counter as tc
from kestrel_sovereign.features.base import Feature, SubagentContextBudgetExceeded
from tests.utils.native_tool_effects import drive_native

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "token_bound"

CLAUDE = "claude-fable-5-2"
PLAN = "anthropic:plan/claude-fable-5-2"
#: A Claude-family model that exists only here: a 4,000-token window, so the
#: gate's budget is 3,400 tokens, about 6,800 bytes.
SMALL = "claude-test-4068-small"
LIMITS = {CLAUDE: 1_000_000, PLAN: 1_000_000, SMALL: 4_000}

#: The 2026-11-13 payload and the provider's count of it.
ROW1_WIRE_BYTES = 2_616_316
PROVIDER_COUNT = 1_158_906
BUDGET_1M = 850_000

HOST = SimpleNamespace(
    did="did:test:host-a", storage=None,
    _get_privacy_policy_epoch=lambda: "normal:host-a:0",
)


@pytest.fixture(autouse=True)
def isolated_ratchet(monkeypatch):
    """A fresh ratchet and event count per test, and pinned context limits.

    The unit suite already points the context-limit cache (and so the ratchet
    file beside it) into the test's temporary directory.
    """
    monkeypatch.setattr(tc, "_ratchets", {})
    monkeypatch.setattr(tc, "_TOKEN_BOUND_EVENTS", Counter())
    real = tc.TokenCounter.resolved_context_limit

    def pinned(self):
        return LIMITS.get(self.model, real(self))

    monkeypatch.setattr(tc.TokenCounter, "resolved_context_limit", pinned)
    monkeypatch.setenv("KESTREL_CONTINUATION_CHECK", "regex")
    monkeypatch.delenv("KESTREL_CONTINUATION_CHECK_DECISION_MODEL", raising=False)


def _restart():
    """What a process restart does to the ratchet: memory is gone, the file stays."""
    tc._ratchets.clear()


class _StubFeature(Feature):
    tool_description = "stub"

    async def initialize(self) -> bool:
        return True

    def get_tools(self):
        return []


@pytest.fixture
def gate():
    """A feature whose gate is exercised without a host."""
    return _StubFeature(agent=MagicMock())


@pytest_asyncio.fixture
async def feature(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:test:token-bound-4068")
    feature = _StubFeature(agent=agent)
    agent.features[feature.name] = feature
    return feature


def _spy_gate(feature, monkeypatch):
    """Record every gate call and what it decided."""
    real = feature._subagent_context_overflow
    calls = []

    def spy(*args, **kwargs):
        check = real(*args, **kwargs)
        calls.append(SimpleNamespace(args=args, kwargs=kwargs, check=check))
        return check

    monkeypatch.setattr(feature, "_subagent_context_overflow", spy)
    return calls


def _wire(part) -> int:
    return len(json.dumps(part, ensure_ascii=True, separators=(",", ":"), default=str).encode())


def _cache():
    path = tc.token_bound_cache_file()
    return json.loads(path.read_text()) if path.exists() else None


def _bound(model, messages, system=None, tools=None, **kwargs):
    bound = tc.get_token_counter(model).upper_bound(system, messages, tools, **kwargs)
    assert isinstance(bound, tc.TokenBound), bound
    return bound


def _user(chars: int):
    return [{"role": "user", "content": "x" * chars}]


def _response(total, **kwargs):
    response = LLMResponse(content="done", input_tokens=total, **kwargs)
    response.provider = "anthropic:plan"
    return response


# ---------------------------------------------------------------------------
# Row 1: the 2026-11-13 payload
# ---------------------------------------------------------------------------


_REPO_MAP_WORDS = (
    "memory", "consolidation", "service", "orchestrator", "engine", "signal",
    "dispatcher", "privacy", "storage", "identity", "constitution", "scheduler",
    "workflow", "feature", "registry", "context", "manager", "observability",
    "session", "conversation", "attachment", "artifact", "repository", "native",
)


def _repo_map_text(n_bytes: int) -> str:
    """Path-dense, ASCII text shaped like docs/audit/REPO_MAP.md.

    Word-built like the real map, so tiktoken counts it near the 664k tokens
    it gave the 2026-11-13 payload, under the 850k budget.
    """
    words = _REPO_MAP_WORDS
    lines, size, i = [], 0, 0
    while size < n_bytes:
        a, b = words[i % len(words)], words[(i // len(words)) % len(words)]
        line = (
            f"- **kestrel_sovereign/features/{a}/{b}_{a}.py** - The {a} {b} "
            f"implementation for durable agents.\n"
            f"  - `class {a.title()}{b.title()}`; `def {b}_{a}(agent, request)`; "
            f"`def describe_{a}(agent)`\n"
        )
        lines.append(line)
        size += len(line)
        i += 1
    return "".join(lines)[:n_bytes]


@functools.lru_cache(maxsize=1)
def _row1_messages_json() -> str:
    """The subagent array after one ``fs_read`` of REPO_MAP, at exactly 2,616,316 wire bytes."""
    text = _repo_map_text(1_280_000)

    def build(pad: int):
        result = {"success": True, "confirmation": text,
                  "data": {"path": "docs/audit/REPO_MAP.md", "content": text}, "pad": "a" * pad}
        return [
            {"role": "system", "content": "EXECUTION MODE: you are the ComputerUse subagent."},
            {"role": "user", "content": "Task: read docs/audit/REPO_MAP.md"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "fs_read",
                             "arguments": json.dumps({"path": "docs/audit/REPO_MAP.md"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps(result)},
        ]

    unpadded = _bound(CLAUDE, build(0)).wire_bytes
    return json.dumps(build(ROW1_WIRE_BYTES - unpadded))


def _row1_messages():
    return json.loads(_row1_messages_json())


def test_row_01_the_2026_11_13_payload_is_bounded_above_the_provider_count_and_refused(gate):
    messages = _row1_messages()
    bound = _bound(CLAUDE, messages)
    assert bound.wire_bytes == ROW1_WIRE_BYTES
    assert bound.tokens == 1_308_158
    assert bound.tokens >= PROVIDER_COUNT
    assert gate._subagent_context_budget(CLAUDE) == BUDGET_1M
    # What the gate measured before #4068: under the budget, so it was sent.
    assert tc.get_token_counter(CLAUDE).count(json.dumps(messages, default=str)) < BUDGET_1M
    check = gate._subagent_context_overflow(messages, CLAUDE, None, system=None)
    assert check.refusal is not None
    assert "exceeded its context budget" in check.refusal


# ---------------------------------------------------------------------------
# Rows 2-4: a request that cannot be measured is refused
# ---------------------------------------------------------------------------


def test_row_02_a_measuring_error_refuses_naming_it_with_a_warning(gate, monkeypatch, caplog):
    def broken(self, *args, **kwargs):
        raise RuntimeError("measurement exploded")

    monkeypatch.setattr(tc.TokenCounter, "upper_bound", broken)
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.features.base"):
        check = gate._subagent_context_overflow(_user(10), CLAUDE, None, system=None)
    assert check.refusal is not None
    assert "RuntimeError" in check.refusal
    assert any(
        record.levelno == logging.WARNING and "RuntimeError" in record.getMessage()
        for record in caplog.records
    )
    assert tc.token_bound_event_counts()["subagent_gate_measure_error"] == 1


def test_row_03_a_block_type_with_no_table_entry_is_refused_by_name(gate):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is in this picture?"},
        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
    ]}]
    measured = tc.get_token_counter(CLAUDE).upper_bound(None, messages, None)
    assert isinstance(measured, tc.MeasurementRefusal)
    assert "'image'" in measured.reason
    check = gate._subagent_context_overflow(messages, CLAUDE, None, system=None)
    assert check.refusal is not None and "'image'" in check.refusal


def test_a_block_type_with_a_recorded_cost_is_bounded_with_it(monkeypatch):
    """The table's other side: an entry makes the type measurable, at its cost."""
    monkeypatch.setattr(tc, "RECORDED_BLOCK_TOKEN_COSTS", {"image": 1_600})
    messages = [{"role": "user", "content": [{"type": "image", "source": {}}]}]
    with_image = _bound(CLAUDE, messages)
    assert with_image.measured_tokens == -(-with_image.wire_bytes // 2) + 1_600


def test_row_04_a_part_that_does_not_serialize_is_refused(gate):
    circular: dict = {"role": "user", "content": "loop"}
    circular["self"] = circular
    measured = tc.get_token_counter(CLAUDE).upper_bound(None, [circular], None)
    assert isinstance(measured, tc.MeasurementRefusal)
    assert "does not serialize" in measured.reason
    check = gate._subagent_context_overflow([circular], CLAUDE, None, system=None)
    assert check.refusal is not None


# ---------------------------------------------------------------------------
# Rows 5-6: the bytes a sample is fed with
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_05_the_system_prompt_is_in_the_gate_bytes_and_the_sample_bytes(
    feature, monkeypatch,
):
    calls = _spy_gate(feature, monkeypatch)
    sent = {}

    class Service:
        async def generate(self, **kwargs):
            sent.update(kwargs)
            bound = calls[0].check.bound
            # Below the sample floor and above the bound: it lands as overhead,
            # stored with its sample.
            return _response(bound.tokens + 300)

    feature.agent.llm_service = Service()
    result = await drive_native(feature.agent, lambda: feature.execute_as_subagent(
        task="read one record", model_override=PLAN,
    ))
    assert result["success"] is True, result
    gated = calls[0]
    assert gated.kwargs["system"] == sent["system_prompt"]
    bound = gated.check.bound
    assert bound.wire_bytes >= _wire(sent["system_prompt"]) + _wire(
        [{"role": "user", "content": sent["user_prompt"]}]
    )
    sample = _cache()["overhead"]["anthropic:plan"]["sample"]
    assert sample["wire_bytes"] == bound.wire_bytes


@pytest.mark.asyncio
async def test_row_06_a_response_feeds_with_the_bytes_captured_before_the_send(gate):
    messages = _user(20_000)
    check = gate._subagent_context_overflow(messages, PLAN, None, system=None)
    captured = check.bound.wire_bytes

    async def send():
        # The request left with the array as it was; it grows afterwards.
        messages.append({"role": "assistant", "content": "y" * 50_000})
        return _response(check.bound.tokens + 400)

    await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    cache = _cache()
    assert cache is not None, "the count was not learned against the bytes that were sent"
    assert cache["overhead"]["anthropic:plan"]["sample"]["wire_bytes"] == captured
    assert _bound(PLAN, messages).wire_bytes > captured


# ---------------------------------------------------------------------------
# Rows 7-11: the ratchet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_07_a_sample_above_the_floor_raises_the_next_bound_across_a_restart(gate):
    messages = _user(400_000)
    check = gate._subagent_context_overflow(messages, PLAN, None, system=None)
    before = check.bound.tokens
    total = before + 50_000            # above the 100,000 floor and above the bound

    async def send():
        return _response(total - 30_000, cache_read_input_tokens=20_000,
                         cache_creation_input_tokens=10_000)

    await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    after = _bound(PLAN, messages).tokens
    assert after >= total > before
    _restart()
    assert _bound(PLAN, messages).tokens == after
    entry = _cache()["bpt_min"]["claude"]
    assert set(entry["sample"]) == {"date", "model", "route", "wire_bytes", "total_input"}
    assert entry["sample"]["total_input"] == total
    assert "x" * 100 not in tc.token_bound_cache_file().read_text()


@pytest.mark.asyncio
async def test_row_08_a_sample_below_the_floor_raises_overhead_not_bpt_min(gate):
    messages = _user(20_000)
    check = gate._subagent_context_overflow(messages, PLAN, None, system=None)

    async def send():
        return _response(check.bound.tokens + 500)

    await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    after = _bound(PLAN, messages)
    assert after.bytes_per_token == tc.STARTING_BYTES_PER_TOKEN
    assert after.overhead == 500
    assert after.tokens == check.bound.tokens + 500


@pytest.mark.asyncio
async def test_row_09_an_excess_above_the_cap_raises_overhead_to_the_cap_and_is_counted(
    gate, caplog,
):
    messages = _user(20_000)
    check = gate._subagent_context_overflow(messages, PLAN, None, system=None)
    cap = int(LIMITS[PLAN] * tc.OVERHEAD_CAP_FRACTION)

    async def send():
        return _response(check.bound.tokens + cap + 5_000)

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.agent.token_counter"):
        await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    after = _bound(PLAN, messages)
    assert after.bytes_per_token == tc.STARTING_BYTES_PER_TOKEN
    assert after.overhead == cap
    assert tc.token_bound_event_counts()["unlearned_excess"] == 1
    assert any(
        record.levelno == logging.WARNING and "not learned" in record.getMessage()
        and "anthropic:plan" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_row_10_a_sample_while_persistent_writes_are_not_allowed_is_never_written(gate):
    host = SimpleNamespace(
        did="did:test:host-a",
        storage=SimpleNamespace(allows_persistent_writes=lambda: False),
        _get_privacy_policy_epoch=lambda: "ephemeral:host-a:1",
    )
    ephemeral = tc.PrivacyEpoch("did:test:host-a", "ephemeral:host-a:1")
    messages = _user(20_000)
    check = gate._subagent_context_overflow(
        messages, PLAN, None, system=None, privacy_epoch=ephemeral,
    )

    async def send():
        return _response(check.bound.tokens + 500)

    await gate._subagent_measured_send(check, send, _dispatch_host=host)
    cache = _cache()
    assert cache is None or "anthropic:plan" not in cache.get("overhead", {})
    # Held in memory for its epoch, and dropped at the next transition.
    assert _bound(PLAN, messages, privacy_epoch=ephemeral).overhead == 500
    next_epoch = tc.PrivacyEpoch("did:test:host-a", "normal:host-a:2")
    assert _bound(PLAN, messages, privacy_epoch=next_epoch).overhead == 0
    assert _bound(PLAN, messages, privacy_epoch=ephemeral).overhead == 0


@pytest.mark.parametrize("damage", ["deleted", "corrupt"])
def test_row_11_an_unreadable_cache_means_the_starting_values_and_the_gate_still_refuses(
    gate, damage, caplog,
):
    path = tc.token_bound_cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bpt_min": {"claude": {"value": 1.5}}}))
    assert _bound(CLAUDE, _user(10_000)).bytes_per_token == 1.5
    if damage == "deleted":
        path.unlink()
    else:
        path.write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.agent.token_counter"):
        assert _bound(CLAUDE, _user(10_000)).bytes_per_token == tc.STARTING_BYTES_PER_TOKEN
    assert any("starting values" in record.getMessage() for record in caplog.records)
    check = gate._subagent_context_overflow(_row1_messages(), CLAUDE, None, system=None)
    assert check.refusal is not None


# ---------------------------------------------------------------------------
# Rows 12-15
# ---------------------------------------------------------------------------

PROSE = (
    "The committee met on Tuesday to review the harbour plan. After a long "
    "discussion of the budget, the members agreed to postpone the vote until "
    "the engineers had finished their survey of the northern breakwater, which "
    "had been damaged in the spring storms. "
) * 20


def test_row_12_count_truncate_and_fits_are_unchanged_on_prose():
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(PROSE)
    counter = tc.get_token_counter(CLAUDE)
    assert counter.count(PROSE) == len(tokens)
    assert counter.truncate_to_tokens(PROSE, 40) == encoding.decode(tokens[:40])
    limit = LIMITS[CLAUDE]
    assert counter.fits_in_context(PROSE, reserved_tokens=limit - len(tokens)) is True
    assert counter.fits_in_context(PROSE, reserved_tokens=limit - len(tokens) + 1) is False


def test_row_13_an_emoji_fixture_is_bounded_at_or_above_its_utf8_bytes():
    text = (FIXTURES / "emoji.txt").read_text(encoding="utf-8")
    bound = _bound(CLAUDE, [{"role": "user", "content": text}])
    assert bound.tokens >= len(text.encode("utf-8"))


class RecordMissing(Exception):
    """The six fixture classes have not been counted by the provider yet."""


@pytest.mark.xfail(strict=True, raises=RecordMissing, reason="#4085: counts not recorded yet")
def test_row_14_the_bound_is_at_or_above_each_recorded_class_count():
    from scripts.record_token_bound_counts import FIXTURE_CLASSES, RECORD_PATH, fixture_request

    if not RECORD_PATH.exists():
        raise RecordMissing(f"{RECORD_PATH.name} is not committed (#4085)")
    record = json.loads(RECORD_PATH.read_text())
    assert set(record["classes"]) == set(FIXTURE_CLASSES)
    for name, counted in record["classes"].items():
        request = fixture_request(name)
        bound = _bound(record["model"], request["messages"], request["system"], request["tools"])
        assert bound.wire_bytes == counted["wire_bytes"], name
        assert bound.tokens >= counted["provider_count"], name


def test_row_15_a_refusal_log_line_carries_the_bound_the_old_count_and_the_budget(
    gate, caplog,
):
    tiktoken = pytest.importorskip("tiktoken")
    messages = _user(10_000)
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.features.base"):
        check = gate._subagent_context_overflow(messages, SMALL, None, system=None)
    assert check.refusal is not None
    old = len(tiktoken.get_encoding("cl100k_base").encode(json.dumps(messages, default=str)))
    line = next(r.getMessage() for r in caplog.records if "refused: bound=" in r.getMessage())
    assert f"bound={check.bound.tokens} " in line
    assert f"count={old} " in line
    assert f"budget={int(LIMITS[SMALL] * 0.85)} " in line
    assert SMALL in line
    assert tc.token_bound_event_counts()["subagent_gate_over_budget"] == 1


def test_a_broken_count_is_reported_on_the_line_and_the_refusal_stands(gate, monkeypatch, caplog):
    def broken(self, text):
        raise RuntimeError("tiktoken exploded")

    monkeypatch.setattr(tc.TokenCounter, "count", broken)
    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.features.base"):
        check = gate._subagent_context_overflow(_user(10_000), SMALL, None, system=None)
    assert check.refusal is not None
    assert any("count=unavailable (RuntimeError)" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Rows 16-18: the sends outside the loop are gated on the request they send
# ---------------------------------------------------------------------------


class _Call:
    def __init__(self, name, id="c1"):
        self.name, self.id, self.arguments = name, id, {}


class _Resp:
    def __init__(self, tool_calls=None, content="", model=SMALL):
        self.tool_calls = tool_calls or []
        self.content = content
        self.model = model


NARRATION = "y" * 8_000 + " Let me check the file now."
SCHEMA = [{"type": "function", "function": {
    "name": "small_tool", "description": "d",
    "parameters": {"type": "object", "properties": {}}}}]


def _small_tool():
    tool = MagicMock()
    tool.name = "small_tool"
    return tool


@pytest.mark.asyncio
async def test_row_16_an_over_budget_first_request_is_refused_and_generate_is_never_called(
    feature,
):
    sends = []

    class Service:
        async def generate(self, **kwargs):
            sends.append(kwargs)
            return _response(10)

    feature.agent.llm_service = Service()
    result = await drive_native(feature.agent, lambda: feature.execute_as_subagent(
        task="z" * 20_000, model_override=SMALL,
    ))
    assert sends == [], "the first request was sent over budget"
    assert result["success"] is False
    assert "exceeded its context budget" in result["error"]


@pytest.mark.asyncio
async def test_row_17_the_repair_before_the_loop_is_refused_with_its_repair_prompt(feature):
    sends = []

    async def generate_with_messages(**kwargs):
        sends.append(kwargs)
        return _Resp(content="done")

    feature.agent.llm_service.generate_with_messages = generate_with_messages
    without = feature._subagent_context_overflow(
        [{"role": "system", "content": "sp"}, {"role": "user", "content": "Task: x"}],
        SMALL, SCHEMA, system=None,
    )
    assert without.refusal is None, "the array must fit without the repair, or this proves nothing"
    with pytest.raises(SubagentContextBudgetExceeded):
        await drive_native(feature.agent, lambda: feature._handle_feature_tool_calls(
            _Resp(content=NARRATION), tools=SCHEMA, system_prompt="sp", user_prompt="Task: x",
            max_iterations=3, model_override=SMALL, runtime_tools=[_small_tool()],
        ))
    assert sends == [], "the repair was sent over budget"


@pytest.mark.asyncio
async def test_row_18_the_repair_after_the_loop_is_refused_with_its_narration_and_prompt(
    feature,
):
    sends = []

    async def generate_with_messages(**kwargs):
        sends.append(kwargs)
        return _Resp(content=NARRATION)

    async def small(*, tool_name, args, tools_by_name, **kwargs):
        return {"payload": "ok"}

    feature.agent.llm_service.generate_with_messages = generate_with_messages
    feature._execute_subagent_tool = small
    with pytest.raises(SubagentContextBudgetExceeded):
        await drive_native(feature.agent, lambda: feature._handle_feature_tool_calls(
            _Resp(tool_calls=[_Call("small_tool")]), tools=SCHEMA, system_prompt="sp",
            user_prompt="Task: x", max_iterations=3, model_override=SMALL,
            runtime_tools=[_small_tool()],
        ))
    # The continuation fit and was sent; the repair carrying the narration was not.
    assert len(sends) == 1


# ---------------------------------------------------------------------------
# Row 19: what a route adds in this process is counted
# ---------------------------------------------------------------------------

ADDITION_SYSTEM = "You execute tasks."
ADDITION_MESSAGES = [
    {"role": "system", "content": ADDITION_SYSTEM},
    {"role": "user", "content": "Task: read one record."},
]
ADDITION_TOOLS = [{"type": "function", "function": {
    "name": "read_record", "description": "Read one record",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"]}}}]


@pytest.mark.asyncio
async def test_row_19_the_plan_route_identity_and_cache_markers_are_counted():
    from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter
    from tests.utils.anthropic_client import anthropic_client

    client = anthropic_client(MagicMock(
        content=[MagicMock(type="text", text="ok")], stop_reason="end_turn",
        usage=MagicMock(input_tokens=10, output_tokens=1)))
    model = "claude-sonnet-4-5-20250929"
    await ClaudeMaxAdapter().get_response(
        client=client, model=model, messages=ADDITION_MESSAGES, tools=ADDITION_TOOLS)
    built = client.messages.stream.call_args.kwargs
    built_bytes = sum(_wire(built[name]) for name in ("system", "messages", "tools"))
    measured = _bound(f"anthropic:plan/{model}", ADDITION_MESSAGES, tools=ADDITION_TOOLS)
    assert measured.wire_bytes >= built_bytes


def test_row_19_the_gpt5_behaviour_contract_is_counted():
    from kestrel_sovereign.llm import codex_adapter

    model = "gpt-5.5"
    instructions, turn_input, _ = codex_adapter._extract_instructions_and_input(ADDITION_MESSAGES)
    adapter = codex_adapter.CodexAdapter.__new__(codex_adapter.CodexAdapter)
    built_bytes = (
        _wire(adapter.contribute_system_prompt(model, instructions))
        + _wire(codex_adapter._convert_tools_to_codex_dynamic_tools(ADDITION_TOOLS))
        + _wire(codex_adapter._build_turn_input(turn_input, fresh_thread=True))
    )
    measured = _bound(f"openai:plan/{model}", ADDITION_MESSAGES, tools=ADDITION_TOOLS)
    assert measured.wire_bytes >= built_bytes


# ---------------------------------------------------------------------------
# Rows 20-21: sends left unmeasured for the subagent admission PR
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#4067 PR 10 (subagent admission): Codex inline continuations are "
           "sent inside the app-server, where no gate runs",
)
@pytest.mark.asyncio
async def test_row_20_every_codex_inline_continuation_is_measured(feature, monkeypatch):
    calls = _spy_gate(feature, monkeypatch)
    route_sends = []

    class Service:
        async def generate(self, **kwargs):
            # The Codex app-server: the first request, then one more after
            # each tool it runs inline, all within this one call.
            route_sends.extend(["first", "after-inline-tool"])
            response = _response(10)
            response.executed_tool_calls = [{"name": "small_tool"}]
            return response

    feature.agent.llm_service = Service()
    result = await drive_native(feature.agent, lambda: feature.execute_as_subagent(
        task="read one record", model_override=PLAN,
    ))
    if result.get("success") is not True or not calls:
        raise RuntimeError(f"precondition: the run did not reach the route: {result!r}")
    assert len(calls) >= len(route_sends), "a send the route made was never measured"


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#4067 PR 10 (subagent admission): the decision check's request is "
           "rendered and its model chosen inside LLMService.decide",
)
@pytest.mark.asyncio
async def test_row_21_the_continuation_decision_request_is_measured(feature, monkeypatch):
    from kestrel_sovereign.turn_completion import CONTINUATION_QUESTION

    monkeypatch.setenv("KESTREL_CONTINUATION_CHECK", "decision")
    calls = _spy_gate(feature, monkeypatch)
    decided = []

    async def decide(request, **kwargs):
        decided.append(len(calls))
        return SimpleNamespace(
            answers={CONTINUATION_QUESTION: SimpleNamespace(p_true=0.0)},
            thresholds={CONTINUATION_QUESTION: 0.5}, route="fake", model="fake")

    feature.agent.llm_service.decide = decide
    await drive_native(feature.agent, lambda: feature._handle_feature_tool_calls(
        _Resp(content="Let me check the file now."), tools=SCHEMA, system_prompt="sp",
        user_prompt="Task: x", max_iterations=3, model_override=SMALL,
        runtime_tools=[_small_tool()],
    ))
    if not decided:
        raise RuntimeError("precondition: the decision check never ran")
    assert decided[0] > 0, "the decision request was sent without a gate measuring it"


# ---------------------------------------------------------------------------
# Supporting behaviour
# ---------------------------------------------------------------------------


def test_the_count_inside_a_prompt_is_too_long_rejection_is_read_through_its_causes():
    from kestrel_sovereign.llm.service import LLMServiceError

    provider = ValueError("Error code: 400 - prompt is too long: 1,158,906 tokens > 1000000 maximum")
    wrapped = LLMServiceError("All providers failed")
    wrapped.__cause__ = provider
    assert tc.rejected_prompt_tokens(wrapped) == PROVIDER_COUNT
    assert tc.rejected_prompt_tokens(LLMServiceError("rate limited")) is None


@pytest.mark.asyncio
async def test_a_prompt_is_too_long_rejection_feeds_and_propagates(gate):
    from kestrel_sovereign.llm.service import LLMServiceError

    messages = _user(400_000)
    check = gate._subagent_context_overflow(messages, PLAN, None, system=None)
    total = check.bound.tokens + 60_000

    async def send():
        raise LLMServiceError(f"Request rejected: prompt is too long: {total} tokens > 1000000 maximum")

    with pytest.raises(LLMServiceError):
        await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    assert _bound(PLAN, messages).tokens >= total


@pytest.mark.asyncio
async def test_a_response_from_inline_tools_is_not_a_sample(gate):
    messages = _user(20_000)
    check = gate._subagent_context_overflow(messages, PLAN, None, system=None)

    async def send():
        response = _response(check.bound.tokens + 500)
        response.executed_tool_calls = [{"name": "t"}]
        return response

    await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    assert _cache() is None


def test_families_and_routes():
    assert tc.model_family("anthropic:plan/claude-fable-5-2") == "claude"
    assert tc.model_family("gpt-4o") == "tiktoken"
    assert tc.model_family("gemini-2.5-pro") == "gemini"
    assert tc.model_route("anthropic:plan/claude-fable-5-2") == "anthropic:plan"
    assert tc.model_route("gemma2:9b") == ""


def test_an_unknown_route_takes_the_largest_learned_overhead():
    path = tc.token_bound_cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"overhead": {"a:x": {"value": 40}, "b:y": {"value": 90},
                                             "bad": {"value": -1}}}))
    assert _bound(CLAUDE, _user(100)).overhead == 90
    assert _bound(PLAN, _user(100)).overhead == 0


def test_a_tiktoken_family_is_bounded_by_its_tiktoken_count_plus_overhead():
    tiktoken = pytest.importorskip("tiktoken")
    messages = _user(1_000)
    bound = _bound("gpt-4o", messages)
    encoding = tiktoken.encoding_for_model("gpt-4o")
    assert bound.bytes_per_token is None
    assert bound.tokens == len(encoding.encode(json.dumps(
        messages, ensure_ascii=True, separators=(",", ":"))))


# ---------------------------------------------------------------------------
# Review findings: rejections on the first send, co-hosted agents, and
# concurrent writers of the ratchet file
# ---------------------------------------------------------------------------


def _too_long(count: int) -> str:
    return f"prompt is too long: {count} tokens > 1000000 maximum"


def _rejecting_service(count_of):
    """A real ``LLMService`` whose one route, ``anthropic:plan``, rejects every request.

    Its adapter raises the Anthropic SDK's own 400 naming ``count_of()``
    tokens, so the service's route loop wraps it and raises what it raises in
    production. Discovery is skipped: the route and the model are explicit.
    """
    import anthropic
    import httpx

    from kestrel_sovereign.llm.service import LLMService

    class Rejecting:
        def create_messages(self, user_prompt=None, system_prompt=None, **_):
            return [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        async def get_response(self, **_):
            message = _too_long(count_of())
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            response = httpx.Response(400, request=request, json={
                "type": "error", "error": {"type": "invalid_request_error", "message": message}})
            raise anthropic.BadRequestError(
                f"Error code: 400 - {message}", response=response, body=None)

    service = LLMService()
    service.providers = [{
        "name": "anthropic:plan", "vendor": "anthropic", "route": "plan",
        "adapter": Rejecting(), "client": object(), "model": CLAUDE,
        "is_cloud": True, "is_local": False,
    }]
    service._ensure_models_discovered = AsyncMock(return_value=None)
    return service


@pytest.mark.asyncio
async def test_a_first_request_every_route_rejects_feeds_its_count_through_the_service(
    feature, monkeypatch,
):
    """The first ``generate`` raises ``LLMAllProvidersFailedError``, not an ``LLMServiceError``."""
    from kestrel_sovereign.llm.error_handling import LLMAllProvidersFailedError

    calls = _spy_gate(feature, monkeypatch)
    service = _rejecting_service(lambda: calls[-1].check.bound.tokens + 500)
    raised = []
    real_generate = service.generate

    async def generate(**kwargs):
        try:
            return await real_generate(**kwargs)
        except BaseException as error:
            raised.append(error)
            raise

    service.generate = generate
    feature.agent.llm_service = service
    try:
        result = await drive_native(feature.agent, lambda: feature.execute_as_subagent(
            task="Task: read one record.", model_override=PLAN,
        ))
    finally:
        await service.close()

    assert len(raised) == 1 and isinstance(raised[0], LLMAllProvidersFailedError), raised
    assert result["success"] is False
    first = calls[0]
    epoch = tc.PrivacyEpoch(feature.agent.did, feature.agent._get_privacy_policy_epoch())
    after = _bound(PLAN, first.args[0], system=first.kwargs["system"], tools=first.args[2],
                   privacy_epoch=epoch)
    assert after.overhead == 500, "the route's rejection count was not fed"


@pytest.mark.asyncio
async def test_a_rejection_every_route_raised_propagates_unchanged(gate):
    service = _rejecting_service(lambda: 1_158_906)
    messages = [{"role": "user", "content": "Task: read one record."}]
    check = gate._subagent_context_overflow(messages, PLAN, None, system="sp")
    raised = []

    async def send():
        try:
            return await service.generate(system_prompt="sp", user_prompt=messages[0]["content"],
                                          model_override=PLAN)
        except BaseException as error:
            raised.append(error)
            raise

    try:
        with pytest.raises(Exception) as info:
            await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    finally:
        await service.close()
    assert info.value is raised[0]


def _wrapped_route_error(text: str):
    """A route's failure as the service's error decorator raises it."""
    from kestrel_sovereign.llm.error_handling import LLMProviderError

    try:
        try:
            raise ValueError(text)
        except ValueError as raw:
            raise LLMProviderError("unknown", text, raw) from raw
    except LLMProviderError as wrapped:
        return wrapped


@pytest.mark.asyncio
async def test_an_aggregate_rejection_is_fed_under_the_route_that_counted_it(gate):
    from kestrel_sovereign.llm.error_handling import LLMAllProvidersFailedError
    from kestrel_sovereign.llm.service import LLMServiceError

    messages = _user(20_000)
    check = gate._subagent_context_overflow(messages, CLAUDE, None, system=None)
    assert check.bound.route == "", "a bare model id names no route, or this proves nothing"
    total = check.bound.tokens + 700
    aggregate = LLMAllProvidersFailedError({
        "anthropic:api": LLMServiceError("Route anthropic:api skipped: vendor not authorized"),
        "anthropic:plan": _wrapped_route_error(f"Error code: 400 - {_too_long(total)}"),
    })

    async def send():
        raise aggregate

    with pytest.raises(LLMAllProvidersFailedError) as info:
        await gate._subagent_measured_send(check, send, _dispatch_host=HOST)
    assert info.value is aggregate
    overhead = _cache()["overhead"]
    assert overhead["anthropic:plan"]["value"] == 700
    assert overhead["anthropic:plan"]["sample"]["total_input"] == total
    assert "" not in overhead and "anthropic:api" not in overhead
    assert tc.rejected_prompt_samples(aggregate) == [("anthropic:plan", total)]


def _agent_host(did: str, epoch: str, *, persist: bool):
    return SimpleNamespace(
        did=did,
        storage=SimpleNamespace(allows_persistent_writes=lambda: persist),
        _get_privacy_policy_epoch=lambda: epoch,
    )


@pytest.mark.asyncio
async def test_another_agents_measurement_neither_drops_nor_applies_a_held_sample(gate):
    """Co-hosted agents share the ratchet; each one's volatile samples stay its own."""
    agent_a = _agent_host("did:test:agent-a", "ephemeral:a:1", persist=False)
    agent_b = _agent_host("did:test:agent-b", "isolated:b:1", persist=False)
    agent_c = _agent_host("did:test:agent-c", "normal:c:0", persist=True)
    messages = _user(20_000)

    def gate_for(host):
        return gate._subagent_gate_send(
            messages, PLAN, None, system=None, route=None, _dispatch_host=host,
        )

    async def learn(host, extra):
        check = gate_for(host)

        async def send():
            return _response(check.bound.tokens + extra)

        await gate._subagent_measured_send(check, send, _dispatch_host=host)

    await learn(agent_a, 500)
    assert gate_for(agent_c).bound.overhead == 0, "agent C applied agent A's held sample"
    assert gate_for(agent_b).bound.overhead == 0, "agent B applied agent A's held sample"
    await learn(agent_b, 300)
    assert gate_for(agent_a).bound.overhead == 500, "another agent's measurement dropped A's sample"
    assert gate_for(agent_b).bound.overhead == 300

    # A's transition drops A's samples and nobody else's.
    agent_a._get_privacy_policy_epoch = lambda: "normal:a:2"
    assert gate_for(agent_a).bound.overhead == 0
    assert gate_for(agent_b).bound.overhead == 300
    cache = _cache()
    assert cache is None or "anthropic:plan" not in cache.get("overhead", {})


def test_a_held_sample_needs_a_privacy_epoch_object():
    with pytest.raises(TypeError, match="PrivacyEpoch"):
        tc.get_token_counter(PLAN).upper_bound(None, _user(10), None, privacy_epoch="ephemeral:a:1")


_BPT_SAMPLE_BYTES = 400_000


def _sample_total(section: str, bound, value):
    """The total input count that makes a ratchet learn ``value`` for ``bound``."""
    if section == "bpt_min":
        return round(bound.wire_bytes / value)
    return bound.tokens + value


@pytest.mark.parametrize("section, key, theirs, mine", [
    ("bpt_min", "claude", 1.5, 1.8),
    ("overhead", "anthropic:plan", 900, 500),
])
def test_a_writer_keeps_the_stricter_value_another_process_published_after_it_read(
    monkeypatch, section, key, theirs, mine,
):
    """Another host commits a stricter value between this one's read and its write."""
    messages = _user(_BPT_SAMPLE_BYTES if section == "bpt_min" else 20_000)
    bound = _bound(PLAN, messages)
    their_total = _sample_total(section, bound, theirs)
    real_sample_record = tc._sample_record
    raced = []

    def other_process_commits_first(*args):
        if not raced:
            raced.append(True)
            # A second host: its own ratchet, reading and writing the same file.
            mine_in_memory = dict(tc._ratchets)
            tc._ratchets.clear()
            tc.record_input_sample(bound, their_total, persist=True, privacy_epoch=None)
            tc._ratchets.clear()
            tc._ratchets.update(mine_in_memory)
        return real_sample_record(*args)

    monkeypatch.setattr(tc, "_sample_record", other_process_commits_first)
    tc.record_input_sample(
        bound, _sample_total(section, bound, mine), persist=True, privacy_epoch=None,
    )
    assert raced, "the other writer never ran inside this one's write"
    entry = _cache()[section][key]
    assert entry["value"] == pytest.approx(theirs), "a weaker value replaced a stricter one"
    assert entry["sample"]["total_input"] == their_total, "the value lost its own sample"
    _restart()
    after = _bound(PLAN, messages)
    if section == "bpt_min":
        assert after.bytes_per_token == pytest.approx(theirs)
    else:
        assert after.overhead == theirs


def test_a_writer_never_replaces_the_file_while_another_holds_its_lock(monkeypatch, caplog):
    from kestrel_sovereign.private_storage import lock_file_descriptor, unlock_file_descriptor

    monkeypatch.setattr(tc, "TOKEN_BOUND_LOCK_TIMEOUT_SECONDS", 0.05)
    path = tc.token_bound_cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + tc.TOKEN_BOUND_LOCK_SUFFIX)
    messages = _user(20_000)
    bound = _bound(PLAN, messages)
    holder = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    token = lock_file_descriptor(holder)
    try:
        with caplog.at_level(logging.WARNING, logger=tc.logger.name):
            tc.record_input_sample(bound, bound.tokens + 500, persist=True, privacy_epoch=None)
        assert not path.exists(), "the file was replaced while another writer held its lock"
        assert any("Could not persist" in record.getMessage() for record in caplog.records)
        assert _bound(PLAN, messages).overhead == 500, "the value does not apply in this process"
    finally:
        unlock_file_descriptor(holder, token)
        os.close(holder)
    tc.record_input_sample(bound, bound.tokens + 800, persist=True, privacy_epoch=None)
    assert _cache()["overhead"]["anthropic:plan"]["value"] == 800
