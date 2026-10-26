"""Clipped tool output can be read back whole (#3740).

A tool result over ``MAX_TOOL_RESULT_CHARS`` reaches the model as a 2,000 +
500 character preview. Before #3740 the middle was discarded and no tool could
bring it back: a 13.7k-character design doc read through ``read_github_file``
lost §3–§10, and routing the read through a subagent only clipped the
subagent's answer the same way. The orchestrator now keeps the whole result in
the agent's ``ToolOutputStore``, the preview names it by a handle, and
``read_tool_output`` pages it in chunks that each fit under the cap.

The acceptance case is driven through the real ``KestrelAgent`` dispatch path:
a 50,000-character repository file comes back clipped, and the agent reads
every character of it, chunk by chunk, each chunk saying where it sits.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.agent import orchestrator_engine
from kestrel_sovereign.agent.orchestrator_engine import (
    PREVIEW_HEAD_CHARS,
    _build_persisted_preview,
)
from kestrel_sovereign.agent.tool_output_store import (
    DEFAULT_RETAIN_CHARS,
    DEFAULT_RETAIN_ENTRIES,
    RETAIN_CHARS_ENV,
    RETAIN_ENTRIES_ENV,
    TOOL_OUTPUT_HANDLE_PREFIX,
    Retention,
    ToolOutputScope,
    ToolOutputStore,
    ToolOutputUnavailable,
    tool_output_scope,
)
from kestrel_sovereign.features.security.feature import default_permission_for_feature
from kestrel_sovereign.features.security.permissions import PermissionLevel
from kestrel_sovereign.features.tool_output.feature import ToolOutputFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURE_MODULES
from kestrel_sovereign.privacy import PrivacyMode
from tests.shared.booted_agent import booted_agent

#: The default orchestrator cap, pinned so the bound does not move with the
#: environment.
RESULT_CAP = 8000

READ_CALL_RE = re.compile(
    r'read_tool_output\(handle="(?P<handle>tool_output:[0-9a-f]+)", '
    r"offset=(?P<offset>\d+)\)"
)


def _store(**sizes: int) -> ToolOutputStore:
    return ToolOutputStore(
        max_chars=sizes.get("max_chars", 1000),
        max_entries=sizes.get("max_entries", 8),
    )


def _scope(session: Optional[str] = "s1", mode: Optional[str] = "normal",
           policy: Optional[str] = None) -> ToolOutputScope:
    """A scope whose policy is its mode's first epoch unless named."""
    return ToolOutputScope(
        session_id=session,
        privacy_mode=mode,
        privacy_policy=policy if policy is not None else f"{mode}:instance:0",
    )


def _retain(store: ToolOutputStore, text: str, **scope: Optional[str]) -> str:
    retention = store.retain(text, tool_name="read_github_file", scope=_scope(**scope))
    assert retention.handle is not None, retention.reason
    return retention.handle


def _read(store: ToolOutputStore, handle: Any, **scope: Optional[str]):
    return store.read(handle, scope=_scope(**scope))


# --- the store ------------------------------------------------------------------


def test_a_retained_output_reads_back_whole_under_its_handle():
    store = _store()
    handle = _retain(store, "x" * 300)

    assert re.fullmatch(r"tool_output:[0-9a-f]{24}", handle)
    entry = _read(store, handle)
    assert entry.text == "x" * 300
    assert entry.tool_name == "read_github_file"
    assert store.retained_chars == 300
    assert _retain(store, "y") != handle


def test_the_entry_cap_evicts_the_least_recently_read():
    store = _store(max_entries=2)
    first = _retain(store, "a")
    second = _retain(store, "b")
    _read(store, first)

    third = _retain(store, "c")

    assert _read(store, first).text == "a"
    assert _read(store, third).text == "c"
    with pytest.raises(ToolOutputUnavailable, match="evicted"):
        _read(store, second)
    assert len(store) == 2


def test_the_character_budget_evicts_until_the_new_output_fits():
    store = _store(max_chars=10)
    first = _retain(store, "a" * 6)
    second = _retain(store, "b" * 6)

    with pytest.raises(ToolOutputUnavailable):
        _read(store, first)
    assert _read(store, second).text == "b" * 6
    assert store.retained_chars == 6


def test_an_output_over_the_whole_budget_is_refused_without_evicting_anything():
    store = _store(max_chars=10)
    kept = _retain(store, "k" * 8)

    retention = store.retain("z" * 11, tool_name="fs_read", scope=_scope())

    assert retention.handle is None
    assert "10-character tool-output retention budget" in retention.reason
    assert _read(store, kept).text == "k" * 8


def test_another_conversation_cannot_read_a_handle():
    store = _store()
    handle = _retain(store, "secret", session="owner-session")

    with pytest.raises(ToolOutputUnavailable, match="another conversation"):
        _read(store, handle, session="other-session")
    with pytest.raises(ToolOutputUnavailable, match="another conversation"):
        _read(store, handle, session=None)
    assert _read(store, handle, session="owner-session").text == "secret"


def test_a_sessionless_output_is_readable_only_where_no_session_is_bound():
    store = _store()
    handle = _retain(store, "unattended", session=None)

    with pytest.raises(ToolOutputUnavailable, match="another conversation"):
        _read(store, handle, session="chat")
    assert _read(store, handle, session=None).text == "unattended"


def test_a_privacy_mode_change_makes_a_handle_unreadable_for_good():
    store = _store()
    handle = _retain(store, "produced while ephemeral", mode="ephemeral")

    with pytest.raises(ToolOutputUnavailable, match="ephemeral.*now in normal"):
        _read(store, handle, mode="normal")
    with pytest.raises(ToolOutputUnavailable, match="not retained"):
        _read(store, handle, mode="ephemeral")
    assert store.retained_chars == 0


def test_a_handle_is_not_readable_after_a_round_trip_back_to_its_mode():
    """EPHEMERAL -> NORMAL -> EPHEMERAL: the mode matches, the policy does not.

    Leaving EPHEMERAL closed the session the output belonged to, so returning
    to EPHEMERAL must not bring it back, even with no read in between.
    """
    store = _store()
    handle = _retain(store, "ephemeral output", mode="ephemeral",
                     policy="ephemeral:instance:1")

    with pytest.raises(
        ToolOutputUnavailable,
        match="changed since, even though it is ephemeral again",
    ):
        _read(store, handle, mode="ephemeral", policy="ephemeral:instance:3")
    with pytest.raises(ToolOutputUnavailable, match="not retained"):
        _read(store, handle, mode="ephemeral", policy="ephemeral:instance:1")
    assert store.retained_chars == 0


def test_clear_forgets_every_output():
    store = _store()
    handles = [_retain(store, "a" * 10), _retain(store, "b" * 20, session=None)]

    assert store.clear() == 2

    assert len(store) == 0 and store.retained_chars == 0
    for handle in handles:
        with pytest.raises(ToolOutputUnavailable, match="privacy-mode change"):
            _read(store, handle)
    assert _read(store, _retain(store, "c")).text == "c"


def test_the_scope_names_the_agents_privacy_policy_epoch():
    """Both ends bind to the policy every applied transition advances."""
    agent = SimpleNamespace(
        privacy_mode=PrivacyMode.EPHEMERAL,
        get_turn_bound_session_id=lambda: "chat-1",
        _get_privacy_policy_epoch=lambda: "ephemeral:abc:4",
    )
    assert tool_output_scope(agent) == ToolOutputScope(
        session_id="chat-1", privacy_mode="ephemeral", privacy_policy="ephemeral:abc:4"
    )

    without_epoch = SimpleNamespace(privacy_mode=PrivacyMode.NORMAL)
    assert tool_output_scope(without_epoch) == ToolOutputScope(
        session_id=None, privacy_mode="normal", privacy_policy="normal"
    )


@pytest.mark.parametrize(
    "handle",
    [None, 7, "", "tool_output:", "tool_output:XYZ", "abc", "tool_output:" + "A" * 24],
)
def test_something_that_is_not_a_handle_is_refused_by_name(handle):
    store = _store()
    _retain(store, "x")

    with pytest.raises(ToolOutputUnavailable, match="not a tool-output handle"):
        _read(store, handle)


def test_the_environment_sizes_the_store():
    default = ToolOutputStore.from_environment({})
    assert (default.max_chars, default.max_entries) == (
        DEFAULT_RETAIN_CHARS,
        DEFAULT_RETAIN_ENTRIES,
    )

    sized = ToolOutputStore.from_environment(
        {RETAIN_CHARS_ENV: " 500 ", RETAIN_ENTRIES_ENV: "3"}
    )
    assert (sized.max_chars, sized.max_entries) == (500, 3)


@pytest.mark.parametrize("name", [RETAIN_CHARS_ENV, RETAIN_ENTRIES_ENV])
@pytest.mark.parametrize("raw", ["0", "-5", "many", "1.5"])
def test_a_malformed_size_fails_naming_the_variable(name, raw):
    with pytest.raises(ValueError, match=name):
        ToolOutputStore.from_environment({name: raw})


# --- the preview -----------------------------------------------------------------


def test_the_preview_names_the_handle_and_where_the_omitted_characters_start():
    body = "H" * 3000 + "M" * 5000 + "T" * 1000
    handle = TOOL_OUTPUT_HANDLE_PREFIX + "0" * 24

    preview = _build_persisted_preview(body, "read_github_file", len(body), Retention(handle))

    calls = READ_CALL_RE.findall(preview)
    assert calls == [(handle, str(PREVIEW_HEAD_CHARS))] * 2
    assert "6,500 chars omitted; read them with" in preview
    assert preview.index(handle) < preview.index("Preview (first")


def test_a_preview_whose_output_was_not_retained_says_it_cannot_be_read_back():
    body = "x" * 9000

    preview = _build_persisted_preview(
        body, "fs_read", len(body), Retention(None, "it is larger than the budget")
    )

    assert "were not retained (it is larger than the budget)" in preview
    assert "cannot be read back" in preview
    assert "read_tool_output" not in preview


# --- the whole dispatch path -----------------------------------------------------


class _AllowAllHooks:
    async def execute_hooks(self, event: HookEvent, hook_input: HookInput) -> HookOutput:
        return HookOutput()

    async def execute_hooks_parallel(self, event: HookEvent, hook_input: HookInput):
        return None


class _AuditStore:
    async def log_decision(self, **row):
        return None


class _RepoFileTool:
    """A ``read_github_file``-shaped tool: one path in, the whole file out."""

    name = "read_repo_file"

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.schema = ToolSchema(
            name=self.name,
            description="read a repository file",
            category=ToolCategory.DATA_ACCESS,
            parameters=[ToolParameter(name="path", type="string", description="path")],
        )

    async def execute(self, **kwargs):
        content = self.files[kwargs["path"]]
        return ToolResult.ok(
            f"Read {kwargs['path']}", data={"path": kwargs["path"], "content": content}
        ).to_dict()


@pytest.fixture
def result_cap(monkeypatch):
    """Pin the cap the orchestrator applies and the one chunks are sized to."""
    import kestrel_sovereign.kestrel_agent as kestrel_agent

    monkeypatch.setattr(kestrel_agent, "MAX_TOOL_RESULT_CHARS", RESULT_CAP)
    monkeypatch.setattr(orchestrator_engine, "MAX_TOOL_RESULT_CHARS", RESULT_CAP)
    return RESULT_CAP


@pytest.fixture
def agent(result_cap):
    """A real ``KestrelAgent`` whose dispatch path and reader are both real."""
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        agent = KestrelAgent(did="did:test:3740")
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_error = AsyncMock()
    agent.observability_store.log_tool_dispatch = AsyncMock(return_value="d-1")
    agent.hooks_manager = _AllowAllHooks()
    agent.features = {
        "SecurityFeature": SimpleNamespace(permission_store=_AuditStore()),
    }
    reader = ToolOutputFeature(agent)
    agent.features[reader.name] = reader
    agent._register_explored_feature_tools(reader)
    return agent


async def _dispatch(agent, tool_name: str, arguments: dict, features=None) -> str:
    messages: list = []
    await agent._dispatch_tool_call(
        SimpleNamespace(id=f"tc-{tool_name}", name=tool_name, arguments=arguments),
        features or {},
        {tool_name},
        messages,
        0,
        None,
    )
    (message,) = messages
    return message["content"]


def _repo_file(chars: int) -> str:
    """Text that JSON-escaping expands: quotes, backslashes, newlines, emoji."""
    unit = 'line "quoted" \\path\\ — naïve café 🐢\n'
    return (unit * (chars // len(unit) + 1))[:chars]


async def _read_all(agent, handle: str, offset: int) -> tuple[str, list[dict]]:
    """Page a handle from ``offset`` to its end through the dispatch path."""
    text, chunks = "", []
    next_offset: Optional[int] = offset
    while next_offset is not None:
        content = await _dispatch(
            agent, "read_tool_output", {"handle": handle, "offset": next_offset}
        )
        assert not content.startswith("[Large tool output"), "a chunk was clipped"
        assert len(content) <= RESULT_CAP
        chunk = json.loads(content)["data"]
        assert chunk["offset"] == next_offset
        assert chunk["length"] == len(chunk["content"]) > 0
        chunks.append(chunk)
        text += chunk["content"]
        next_offset = chunk["next_offset"]
        assert len(chunks) < 200, "paging did not converge"
    return text, chunks


@pytest.mark.asyncio
async def test_a_50k_repo_file_is_read_back_every_character_in_ranged_chunks(agent):
    document = _repo_file(50_000)
    agent.register_dynamic_tools("github", [_RepoFileTool({"docs/BIG.md": document})])

    preview = await _dispatch(agent, "read_repo_file", {"path": "docs/BIG.md"})

    assert preview.startswith("[Large tool output — ")
    ((handle, offset),) = set(READ_CALL_RE.findall(preview))
    head = preview.split(f"Preview (first {PREVIEW_HEAD_CHARS} chars):\n", 1)[1]
    head = head[: int(offset)]

    rest, chunks = await _read_all(agent, handle, int(offset))

    whole = head + rest
    total = chunks[0]["total_chars"]
    assert len(whole) == total
    assert all(chunk["total_chars"] == total for chunk in chunks)
    assert [c["offset"] + c["length"] for c in chunks[:-1]] == [
        c["offset"] for c in chunks[1:]
    ]
    assert chunks[-1]["next_offset"] is None and chunks[-1]["truncated"] is False
    assert len(chunks) > 1
    assert json.loads(whole)["data"]["content"] == document


@pytest.mark.asyncio
async def test_a_subagent_answer_over_the_cap_can_be_read_back(agent):
    answer = _repo_file(12_500)
    researcher = SimpleNamespace(
        name="ResearchFeature",
        tool_name="researcher",
        execute_as_subagent=AsyncMock(return_value=answer),
        get_tools=lambda: [],
    )
    agent._explored_features[researcher.tool_name] = True

    preview = await _dispatch(
        agent, "researcher", {"task": "summarise"}, features={"researcher": researcher}
    )

    ((handle, offset),) = set(READ_CALL_RE.findall(preview))
    rest, _ = await _read_all(agent, handle, 0)
    assert json.loads(rest) == answer
    assert int(offset) == PREVIEW_HEAD_CHARS


@pytest.mark.asyncio
async def test_a_handle_is_read_only_in_the_conversation_that_produced_it(agent):
    agent.register_dynamic_tools("github", [_RepoFileTool({"a.md": _repo_file(20_000)})])
    agent.get_turn_bound_session_id = lambda: "conversation-a"
    preview = await _dispatch(agent, "read_repo_file", {"path": "a.md"})
    ((handle, _),) = set(READ_CALL_RE.findall(preview))

    agent.get_turn_bound_session_id = lambda: "conversation-b"
    refused = json.loads(await _dispatch(agent, "read_tool_output", {"handle": handle}))

    assert refused["success"] is False
    assert "another conversation" in refused["error"]

    agent.get_turn_bound_session_id = lambda: "conversation-a"
    _, chunks = await _read_all(agent, handle, 0)
    assert chunks[-1]["next_offset"] is None


@pytest.mark.asyncio
async def test_a_handle_produced_in_one_privacy_mode_is_not_read_in_another(agent):
    agent.register_dynamic_tools("github", [_RepoFileTool({"a.md": _repo_file(20_000)})])
    agent._privacy_mode = PrivacyMode.EPHEMERAL
    preview = await _dispatch(agent, "read_repo_file", {"path": "a.md"})
    ((handle, _),) = set(READ_CALL_RE.findall(preview))

    agent._privacy_mode = PrivacyMode.NORMAL
    refused = json.loads(await _dispatch(agent, "read_tool_output", {"handle": handle}))

    assert refused["success"] is False
    assert "privacy mode ephemeral" in refused["error"]


@pytest.mark.asyncio
async def test_a_privacy_round_trip_releases_retained_output(
    tmp_path, result_cap, monkeypatch
):
    """EPHEMERAL -> NORMAL -> EPHEMERAL on a booted agent, with no read between.

    Leaving EPHEMERAL closes and purges its session, so the round trip must not
    bring that session's clipped output back. Before the fix the handle read
    whole again on return, because the store compared only the mode's name.
    """
    # The repo-file tool goes through the real approval gate; a test instance
    # passes it only with the operator's non-interactive opt-in, read at boot.
    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    async with booted_agent(
        tmp_path, name="ToolOutput", provenance="test-3740", features=set()
    ) as agent:
        agent.register_dynamic_tools(
            "github", [_RepoFileTool({"a.md": _repo_file(20_000)})]
        )
        entered = await agent.set_privacy_mode_with_effects(PrivacyMode.EPHEMERAL)
        assert entered.applied and agent.privacy_mode is PrivacyMode.EPHEMERAL

        preview = await _dispatch(agent, "read_repo_file", {"path": "a.md"})
        ((handle, _),) = set(READ_CALL_RE.findall(preview))
        assert len(agent.tool_output_store) == 1

        left = await agent.set_privacy_mode_with_effects(PrivacyMode.NORMAL)
        assert left.applied and agent.privacy_mode is PrivacyMode.NORMAL
        assert len(agent.tool_output_store) == 0
        assert agent.tool_output_store.retained_chars == 0

        back = await agent.set_privacy_mode_with_effects(PrivacyMode.EPHEMERAL)
        assert back.applied and agent.privacy_mode is PrivacyMode.EPHEMERAL

        refused = json.loads(
            await _dispatch(agent, "read_tool_output", {"handle": handle})
        )
        assert refused["success"] is False
        assert "privacy-mode change released it" in refused["error"]


@pytest.mark.asyncio
async def test_a_refused_privacy_transition_keeps_retained_output(agent):
    """Only a transition that applies releases anything."""
    agent.register_dynamic_tools("github", [_RepoFileTool({"a.md": _repo_file(20_000)})])
    preview = await _dispatch(agent, "read_repo_file", {"path": "a.md"})
    ((handle, _),) = set(READ_CALL_RE.findall(preview))
    agent._recorded_privacy_state = AsyncMock(return_value=None)

    refused = await agent._apply_privacy_mode_locked(PrivacyMode.ISOLATED)

    assert refused.applied is False
    _, chunks = await _read_all(agent, handle, 0)
    assert chunks[-1]["next_offset"] is None


@pytest.mark.asyncio
async def test_an_output_over_the_retention_budget_is_named_as_unrecoverable(agent):
    agent.tool_output_store = ToolOutputStore(max_chars=10_000, max_entries=4)
    agent.register_dynamic_tools("github", [_RepoFileTool({"a.md": _repo_file(20_000)})])

    preview = await _dispatch(agent, "read_repo_file", {"path": "a.md"})

    assert "were not retained (it is larger than the 10,000-character" in preview
    assert READ_CALL_RE.search(preview) is None
    assert len(agent.tool_output_store) == 0


@pytest.mark.asyncio
async def test_a_result_under_the_cap_is_neither_clipped_nor_retained(agent):
    agent.register_dynamic_tools("github", [_RepoFileTool({"a.md": "short"})])

    content = await _dispatch(agent, "read_repo_file", {"path": "a.md"})

    assert json.loads(content)["data"]["content"] == "short"
    assert len(agent.tool_output_store) == 0


# --- read_tool_output's own arguments ------------------------------------------


@pytest.fixture
async def retained(agent):
    """A handle to a retained 20k-character output, and its whole text."""
    agent.register_dynamic_tools("github", [_RepoFileTool({"a.md": _repo_file(20_000)})])
    preview = await _dispatch(agent, "read_repo_file", {"path": "a.md"})
    ((handle, _),) = set(READ_CALL_RE.findall(preview))
    whole, _ = await _read_all(agent, handle, 0)
    return handle, whole


@pytest.mark.asyncio
async def test_length_bounds_a_chunk(agent, retained):
    handle, whole = retained
    reader = agent.features["ToolOutputFeature"]

    result = await reader.read_tool_output(handle, offset=100, length=50)

    assert result.data["content"] == whole[100:150]
    assert result.data["next_offset"] == 150
    assert "characters 100–150 of" in result.confirmation


@pytest.mark.asyncio
async def test_reading_at_the_end_returns_an_empty_final_chunk(agent, retained):
    handle, whole = retained
    reader = agent.features["ToolOutputFeature"]

    result = await reader.read_tool_output(handle, offset=len(whole))

    assert result.data["content"] == ""
    assert result.data["next_offset"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments, error",
    [
        ({"offset": -1}, "offset must be in"),
        ({"offset": "soon"}, "offset must be an integer"),
        ({"length": 0}, "length must be in"),
        ({"offset": 10**9}, "is past the end of"),
    ],
)
async def test_bad_ranges_are_refused(agent, retained, arguments, error):
    handle, _ = retained
    reader = agent.features["ToolOutputFeature"]

    result = await reader.read_tool_output(handle, **arguments)

    assert result.error is not None and error in result.error


@pytest.mark.asyncio
async def test_an_agent_without_a_store_says_so():
    reader = ToolOutputFeature(SimpleNamespace())

    result = await reader.read_tool_output(TOOL_OUTPUT_HANDLE_PREFIX + "0" * 24)

    assert "keeps no tool-output store" in result.error


# --- always present ------------------------------------------------------------


def test_the_reader_is_mandatory_promoted_and_allowed():
    """Wherever the cap clips, the reader is loaded, visible and ungated."""
    assert (
        MANDATORY_FEATURE_MODULES["ToolOutputFeature"]
        == "kestrel_sovereign.features.tool_output.feature"
    )
    assert ToolOutputFeature(SimpleNamespace()).promote_tools_on_startup is True
    assert default_permission_for_feature("ToolOutputFeature") is PermissionLevel.ALLOW


def test_a_constructed_agent_keeps_a_store():
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        agent = KestrelAgent(did="did:test:3740-store")

    assert isinstance(agent.tool_output_store, ToolOutputStore)
