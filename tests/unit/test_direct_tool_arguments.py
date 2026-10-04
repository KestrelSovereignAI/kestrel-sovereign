"""Direct tools take only the arguments they declare (#3396).

On 2026-09-29 a ``shell`` call that also passed ``context`` died before
anything ran: ``ComputerUseFeature.shell() got an unexpected keyword argument
'context'``. A feature's subagent dispatcher takes ``task`` and ``context``,
so callers pass them to direct tools by habit. The call's
``capture_output=true`` died with it, and truncated inline output was the
only visible fallback.

The ruling: in the direct-tool dispatcher, drop ``task``/``context`` when the
tool's schema does not accept them, and refuse any other argument it does not
accept with an error that lists the tool's parameters and suggests a close
match. A misspelt argument is never dropped, because that would run the tool
with a default. "Accepts" is the schema's own answer, not membership of
``properties``: an MCP schema that leaves ``additionalProperties`` open, or
whose ``patternProperties`` match the name, accepts ``task``/``context`` as
tool data, and they reach the tool unchanged.

Both direct-tool paths are driven: the chat loop's ``_dispatch_direct_tool``
and ``execute_named_tool``, which the codex inline executor uses. The shell
cases run the real ``ComputerUseFeature`` tool and a real subprocess, because
the claim is that a capture artifact exists.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kestrel_sdk.hooks.base import HookEvent, HookInput, HookOutput
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.agent.direct_tool_arguments import (
    GENERIC_ORCHESTRATION_ARGUMENTS,
    normalize_direct_tool_arguments,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyConfig
from kestrel_sovereign.security.tool_audit import (
    ACTION_TOOL_VALIDATION,
    REJECTION_FEATURE_NAME,
)


# --- tools -----------------------------------------------------------------


class _SchemaTool:
    """An AgentTool whose schema lists its parameters (the SDK shape)."""

    def __init__(self, name: str, *parameter_names: str):
        self.name = name
        self.schema = ToolSchema(
            name=name,
            description="test tool",
            category=ToolCategory.SYSTEM,
            parameters=[
                ToolParameter(name=p, type="string", description=p)
                for p in parameter_names
            ],
        )
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True}


class _JsonSchemaTool:
    """An MCP-style tool: ``schema.parameters`` is a JSON-Schema object."""

    def __init__(self, name: str, parameters: dict):
        self.name = name
        self.schema = SimpleNamespace(parameters=parameters)
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True}


def _object_schema(**keywords) -> dict:
    return {"type": "object", "properties": {"url": {"type": "string"}}, **keywords}


#: JSON-Schema shapes that accept ``context`` without listing it in
#: ``properties``. The review of #3396 found the first fix dropping it here.
SCHEMAS_THAT_ACCEPT_CONTEXT = {
    "additionalProperties absent": _object_schema(),
    "additionalProperties true": _object_schema(additionalProperties=True),
    "additionalProperties a schema": _object_schema(
        additionalProperties={"type": "string"}
    ),
    "patternProperties matches": _object_schema(
        additionalProperties=False,
        patternProperties={"^context$": {"type": "string"}},
    ),
}


# --- the rules ---------------------------------------------------------------


def test_the_generic_arguments_are_exactly_task_and_context():
    assert GENERIC_ORCHESTRATION_ARGUMENTS == {"task", "context"}


def test_undeclared_task_and_context_are_dropped():
    tool = _SchemaTool("get_github_issue", "owner", "repo", "number")

    args, error = normalize_direct_tool_arguments(
        tool.name,
        tool,
        {"owner": "o", "repo": "r", "number": 7, "task": "t", "context": "c"},
    )

    assert error is None
    assert args == {"owner": "o", "repo": "r", "number": 7}


def test_a_tool_that_declares_context_receives_it():
    tool = _SchemaTool("summarize", "text", "context")

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"text": "x", "context": "keep me", "task": "drop me"}
    )

    assert error is None
    assert args == {"text": "x", "context": "keep me"}


def test_declared_arguments_pass_through_as_the_same_mapping():
    tool = _SchemaTool("shell", "command", "timeout")
    supplied = {"command": "ls"}

    args, error = normalize_direct_tool_arguments(tool.name, tool, supplied)

    assert error is None
    assert args is supplied


def test_an_unknown_argument_lists_the_parameters_and_suggests_the_match():
    tool = _SchemaTool("get_pull_request", "owner", "repo", "pull_number")
    supplied = {"owner": "o", "repo": "r", "pr_number": 12}

    args, error = normalize_direct_tool_arguments(tool.name, tool, supplied)

    assert error == (
        "get_pull_request does not accept argument 'pr_number' "
        "(did you mean 'pull_number'?). "
        "Valid parameters: owner, repo, pull_number."
    )
    # Refused, not dropped: the caller gets back exactly what it sent.
    assert args == supplied


def test_an_abbreviation_is_suggested_over_a_lookalike():
    """By spelling similarity alone ``cmd`` is nearer ``cwd``."""
    tool = _SchemaTool("shell", "command", "timeout", "cwd", "capture_output")

    _, error = normalize_direct_tool_arguments(tool.name, tool, {"cmd": "ls"})

    assert "'cmd' (did you mean 'command'?)" in error


def test_a_parameter_already_supplied_is_not_suggested():
    tool = _SchemaTool("get_pull_request", "owner", "repo", "pull_number")

    _, error = normalize_direct_tool_arguments(
        tool.name, tool, {"pull_number": 1, "pr_number": 1}
    )

    assert "did you mean" not in error
    assert "Valid parameters: owner, repo, pull_number." in error


def test_an_unknown_argument_is_refused_even_beside_a_generic_one():
    tool = _SchemaTool("shell", "command", "capture_output")

    _, error = normalize_direct_tool_arguments(
        tool.name, tool, {"command": "ls", "capture": True, "context": "c"}
    )

    assert error == (
        "shell does not accept argument 'capture' "
        "(did you mean 'capture_output'?). "
        "Valid parameters: command, capture_output."
    )


def test_several_unknown_arguments_are_all_named():
    tool = _SchemaTool("shell", "command", "timeout")

    _, error = normalize_direct_tool_arguments(
        tool.name, tool, {"command": "ls", "timout": 5, "zzz": 1}
    )

    assert error.startswith(
        "shell does not accept arguments 'timout' (did you mean 'timeout'?), 'zzz'."
    )


def test_a_tool_with_no_parameters_says_so():
    tool = _SchemaTool("heartbeat_check")

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"verbose": True, "task": "t"}
    )

    assert error == (
        "heartbeat_check does not accept argument 'verbose'. "
        "Valid parameters: none (call it with no arguments)."
    )
    assert normalize_direct_tool_arguments(tool.name, tool, {"task": "t"}) == ({}, None)


def test_a_pathological_argument_name_is_bounded():
    tool = _SchemaTool("shell", "command")

    _, error = normalize_direct_tool_arguments(tool.name, tool, {"x" * 5000: 1})

    assert len(error) < 300


def test_a_non_string_argument_name_is_refused_not_raised():
    """JSON keys are strings, but an in-process caller's need not be; it must
    get the refusal rather than a ``TypeError`` from the suggestion search."""
    tool = _SchemaTool("shell", "command")

    _, error = normalize_direct_tool_arguments(tool.name, tool, {1: "x"})

    assert error == "shell does not accept argument 1. Valid parameters: command."


@pytest.mark.parametrize(
    "schema",
    [
        _object_schema(),
        _object_schema(additionalProperties=True),
        _object_schema(additionalProperties={"type": "string"}),
    ],
    ids=["absent", "true", "schema"],
)
def test_an_open_json_schema_passes_every_argument_unchanged(schema):
    """Unless ``additionalProperties`` is ``false``, JSON Schema accepts every
    name, ``task`` and ``context`` included. The MCP server validates them."""
    tool = _JsonSchemaTool("mcp__fetch__fetch", schema)
    supplied = {"url": "u", "max_length": 5, "context": "c", "task": "t"}

    args, error = normalize_direct_tool_arguments(tool.name, tool, supplied)

    assert error is None
    assert args is supplied


def test_a_closed_json_schema_refuses_unknown_arguments():
    tool = _JsonSchemaTool(
        "mcp__fetch__fetch",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"uri": "u", "task": "t"}
    )

    assert error == (
        "mcp__fetch__fetch does not accept argument 'uri' (did you mean 'url'?). "
        "Valid parameters: url."
    )


def _patterned_schema(*patterns: str) -> dict:
    return {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "patternProperties": {pattern: {"type": "string"} for pattern in patterns},
        "additionalProperties": False,
    }


def test_a_closed_schema_accepts_names_its_patterns_match():
    """``patternProperties`` admit names beyond ``properties`` even when
    ``additionalProperties`` is false. ``context`` matches no pattern here, so
    the schema does not accept it, and it is dropped."""
    tool = _JsonSchemaTool("mcp__api__call", _patterned_schema("^x-"))

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"path": "/", "x-trace": "1", "context": "c"}
    )

    assert error is None
    assert args == {"path": "/", "x-trace": "1"}


def test_a_pattern_that_matches_a_generic_name_passes_it_through():
    tool = _JsonSchemaTool("mcp__api__call", _patterned_schema("^context$"))

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"path": "/", "context": "c", "task": "t"}
    )

    assert error is None
    assert args == {"path": "/", "context": "c"}


@pytest.mark.parametrize(
    ("pattern", "kept"),
    [
        ("^context$", {"context"}),
        ("^cont", {"context"}),
        ("text$", {"context"}),
        # JSON Schema patterns are unanchored.
        ("ontex", {"context"}),
        ("as", {"task"}),
        ("t", {"context", "task"}),
        ("^con$", set()),
        ("^$", set()),
        ("", {"context", "task"}),
    ],
)
def test_a_literal_pattern_is_matched_as_json_schema_defines(pattern, kept):
    tool = _JsonSchemaTool("mcp__api__call", _patterned_schema(pattern))

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"path": "/", "context": "c", "task": "t"}
    )

    assert error is None
    assert set(args) == {"path"} | kept


def test_a_pattern_schema_leaves_other_names_to_the_tool():
    """Only ``task``/``context`` are compared with patterns, so any other
    name outside ``properties`` reaches the tool, which validates it."""
    tool = _JsonSchemaTool("mcp__api__call", _patterned_schema("^x-"))

    args, error = normalize_direct_tool_arguments(
        tool.name, tool, {"pth": "/", "x-trace": "1", "context": "c"}
    )

    assert error is None
    assert args == {"pth": "/", "x-trace": "1"}


def test_a_pattern_is_never_run_as_a_regular_expression():
    """``^[0-9]+$`` matches neither name, but finding that out means running
    the server's pattern, and ``(?:x?){4000000000}`` exhausts memory on
    ``task``. A pattern that is not a literal may match, so both names pass
    through for the server to validate."""
    tool = _JsonSchemaTool("mcp__api__call", _patterned_schema("^x-", "^[0-9]+$"))
    supplied = {"path": "/", "context": "c", "task": "t"}

    assert normalize_direct_tool_arguments(tool.name, tool, supplied) == (
        supplied,
        None,
    )


@pytest.mark.parametrize(
    "pattern_properties",
    [
        {r"^\p{L}+$": {}},
        {"(?:x?){4000000000}": {}},
        {"^(a+)+$": {}},
        {"^context\\$": {}},
        {7: {}},
        ["^x-"],
    ],
    ids=[
        "unicode-property",
        "repeat-bomb",
        "backtracking",
        "escaped-anchor",
        "non-string-pattern",
        "not-a-mapping",
    ],
)
def test_patterns_this_check_does_not_evaluate_leave_every_argument_alone(
    pattern_properties,
):
    """A pattern that is not a literal, or patterns that cannot be read, may
    match any name, so nothing is dropped or refused on their account."""
    tool = _JsonSchemaTool(
        "mcp__api__call",
        {
            "properties": {"path": {}},
            "patternProperties": pattern_properties,
            "additionalProperties": False,
        },
    )
    supplied = {"path": "/", "context": "c", "zzz": 1}

    assert normalize_direct_tool_arguments(tool.name, tool, supplied) == (
        supplied,
        None,
    )


class _ImpersonatesTask:
    """An in-process key that hashes and compares like ``"task"``."""

    def __hash__(self):
        return hash("task")

    def __eq__(self, other):
        return other == "task"


class _UncomparableKey:
    def __eq__(self, other):
        raise TypeError("not comparable")

    __hash__ = object.__hash__


def test_a_key_that_merely_equals_task_is_not_pattern_matched():
    tool = _JsonSchemaTool("mcp__api__call", _patterned_schema("^x-"))
    supplied = {_ImpersonatesTask(): 1}

    assert normalize_direct_tool_arguments(tool.name, tool, supplied) == (
        supplied,
        None,
    )


def test_an_uncomparable_key_is_refused_not_raised():
    tool = _SchemaTool("shell", "command")
    key = _UncomparableKey()

    _, error = normalize_direct_tool_arguments(tool.name, tool, {key: 1})

    assert error.startswith("shell does not accept argument ")
    assert error.endswith("Valid parameters: command.")


def test_a_flood_of_unknown_arguments_is_summarized():
    tool = _SchemaTool("shell", "command")
    supplied = {f"zz{i}": i for i in range(5000)}

    _, error = normalize_direct_tool_arguments(tool.name, tool, supplied)

    assert "and 4990 more." in error
    assert "'zz9'" in error and "'zz10'" not in error
    assert len(error) < 500


@pytest.mark.parametrize(
    "tool",
    [
        object(),
        SimpleNamespace(schema=None),
        SimpleNamespace(schema=SimpleNamespace(parameters="nonsense")),
        SimpleNamespace(schema=SimpleNamespace(parameters=[{"name": "dict-shaped"}])),
        SimpleNamespace(
            schema=SimpleNamespace(
                parameters={"properties": {1: {}}, "additionalProperties": False}
            )
        ),
    ],
)
def test_a_tool_whose_parameters_are_unknowable_is_left_alone(tool):
    supplied = {"anything": 1, "context": "c"}

    assert normalize_direct_tool_arguments("t", tool, supplied) == (supplied, None)


# --- the dispatch paths -------------------------------------------------------


class _AllowAllHooks:
    def __init__(self):
        self.pre_calls: list[dict] = []
        self.post_calls: list[dict] = []

    async def execute_hooks(self, event: HookEvent, hook_input: HookInput) -> HookOutput:
        self.pre_calls.append(dict(hook_input.tool_input or {}))
        return HookOutput()

    async def execute_hooks_parallel(self, event: HookEvent, hook_input: HookInput):
        self.post_calls.append(dict(hook_input.tool_input or {}))


class _AuditStore:
    def __init__(self):
        self.rows: list[dict] = []

    async def log_decision(self, **row):
        self.rows.append(row)


class _ObservabilityStore:
    def __init__(self):
        self.responses: list[dict] = []

    async def log_tool_response(self, **kwargs):
        self.responses.append(kwargs)


class _Orchestrator(OrchestratorEngineMixin):
    """The orchestrator surface the two direct-tool paths read."""

    def __init__(self, *, direct_tools=None, features=None):
        self._direct_tools = dict(direct_tools or {})
        self._tool_to_feature: dict = {}
        self.audit = _AuditStore()
        self.features = {
            "SecurityFeature": SimpleNamespace(permission_store=self.audit),
            **(features or {}),
        }
        self.hooks_manager = _AllowAllHooks()
        self.observability_store = _ObservabilityStore()


async def _dispatch_direct(agent, tool_name, args, **kwargs):
    return await agent._dispatch_direct_tool(
        SimpleNamespace(id="call-1", name=tool_name),
        tool_name,
        args,
        0.0,
        "evt-3396",
        session_id="s-3396",
        **kwargs,
    )


class _ApprovalQueue:
    async def request_approval(self, feature_name, tool_name, tool_args, timeout):
        return True, "once"


class _ComputerUseAgent:
    def __init__(self):
        self.privacy_config = PrivacyConfig(computer_access=True)
        self.granted_capabilities = frozenset(
            {"shell_execution_sandboxed", "shell_execution_host"}
        )
        self.did = "did:test:agent"
        self.features = {
            "security": SimpleNamespace(approval_queue=_ApprovalQueue())
        }

    def get_feature(self, name):
        return self.features.get(name)


@pytest.fixture()
async def computer_use(tmp_path: Path) -> ComputerUseFeature:
    (tmp_path / "captures").mkdir()
    feature = ComputerUseFeature(_ComputerUseAgent())
    feature._cfg = {
        "enabled": True,
        "backend": "local",
        "allowed_paths": [str(tmp_path)],
        "auto_approved_binaries": ["echo"],
        "audit_log_path": str(tmp_path / "audit.jsonl"),
        "capture_dir": str(tmp_path / "captures"),
    }
    await feature.initialize()
    return feature


def _shell_tool(feature: ComputerUseFeature):
    return next(t for t in feature.get_tools() if t.name == "shell")


def _assert_captured(result: dict, captures: Path, text: str) -> None:
    assert result.get("status") == "ok", result
    stdout_path = Path(result["data"]["stdout_path"])
    assert stdout_path.parent == captures
    assert stdout_path.read_text().strip() == text
    assert Path(result["data"]["manifest_path"]).is_file()


SHELL_WITH_HABITUAL_KWARGS = {
    "command": "echo captured-3396",
    "capture_output": True,
    "context": "verify the CI log",
    "task": "fetch the failing run",
}


@pytest.mark.asyncio
async def test_chat_path_shell_with_context_still_writes_the_capture(
    computer_use, tmp_path: Path
):
    agent = _Orchestrator(direct_tools={"shell": _shell_tool(computer_use)})

    result = await _dispatch_direct(agent, "shell", dict(SHELL_WITH_HABITUAL_KWARGS))

    _assert_captured(result, tmp_path / "captures", "captured-3396")
    # Hooks and the audit see what ran, not the habitual extras.
    assert agent.hooks_manager.pre_calls == [
        {"command": "echo captured-3396", "capture_output": True}
    ]
    assert agent.audit.rows == []


@pytest.mark.asyncio
async def test_inline_executor_path_shell_with_context_still_writes_the_capture(
    computer_use, tmp_path: Path
):
    """``execute_named_tool`` is the codex inline executor's door."""
    agent = _Orchestrator(features={"ComputerUseFeature": computer_use})

    result = await agent.execute_named_tool(
        "shell", dict(SHELL_WITH_HABITUAL_KWARGS), session_id="s-3396",
        source="codex_app_server",
    )

    _assert_captured(result, tmp_path / "captures", "captured-3396")
    assert agent.hooks_manager.pre_calls == [
        {"command": "echo captured-3396", "capture_output": True}
    ]


@pytest.mark.asyncio
async def test_chat_path_refuses_an_unknown_argument_before_anything_runs():
    tool = _SchemaTool("get_pull_request", "owner", "repo", "pull_number")
    agent = _Orchestrator(direct_tools={tool.name: tool})
    dispatch_meta: dict = {}
    tool_events: list = []

    result = await _dispatch_direct(
        agent,
        tool.name,
        {"owner": "o", "repo": "r", "pr_number": 12, "context": "c"},
        dispatch_meta=dispatch_meta,
        tool_events=tool_events,
        streaming=True,
    )

    expected = (
        "get_pull_request does not accept argument 'pr_number' "
        "(did you mean 'pull_number'?). Valid parameters: owner, repo, pull_number."
    )
    assert result == {"success": False, "error": expected}
    assert tool.calls == []
    assert agent.hooks_manager.pre_calls == []
    assert dispatch_meta == {
        "status": "error",
        "error_class": "ToolArgumentError",
        "error_message": expected,
    }
    assert tool_events == [{"type": "error", "tool": tool.name, "error": expected}]
    assert agent.observability_store.responses[0]["success"] is False
    # Refused before PRE_TOOL_USE, so the guardrail writes the audit row (#2929).
    [row] = agent.audit.rows
    assert row["feature_name"] == REJECTION_FEATURE_NAME
    assert row["action"] == ACTION_TOOL_VALIDATION
    assert row["tool_name"] == tool.name
    assert expected in row["args_summary"]


@pytest.mark.asyncio
async def test_chat_path_still_reports_a_tool_that_raises():
    """The failure bookkeeping the refusal shares with the exception path."""

    class _Raises(_SchemaTool):
        async def execute(self, **kwargs):
            raise TimeoutError("took too long")

    tool = _Raises("slow", "n")
    agent = _Orchestrator(direct_tools={tool.name: tool})
    dispatch_meta: dict = {}

    result = await _dispatch_direct(agent, tool.name, {"n": "1"}, dispatch_meta=dispatch_meta)

    assert result == {"success": False, "error": "took too long"}
    assert dispatch_meta == {
        "status": "timeout",
        "error_class": "TimeoutError",
        "error_message": "took too long",
    }
    assert agent.audit.rows == []


@pytest.mark.asyncio
async def test_inline_executor_path_refuses_an_unknown_argument():
    tool = _SchemaTool("get_pull_request", "owner", "repo", "pull_number")
    feature = SimpleNamespace(name="GitHubFeature", get_tools=lambda: [tool])
    agent = _Orchestrator(features={"GitHubFeature": feature})

    result = await agent.execute_named_tool(
        tool.name, {"owner": "o", "repo": "r", "pr_number": 12},
        session_id="s-3396", source="codex_app_server",
    )

    assert result == {
        "success": False,
        "error": (
            "Invalid tool arguments: get_pull_request does not accept argument "
            "'pr_number' (did you mean 'pull_number'?). "
            "Valid parameters: owner, repo, pull_number."
        ),
    }
    assert tool.calls == []
    assert agent.hooks_manager.pre_calls == []
    assert [row["action"] for row in agent.audit.rows] == [ACTION_TOOL_VALIDATION]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "schema",
    list(SCHEMAS_THAT_ACCEPT_CONTEXT.values()),
    ids=list(SCHEMAS_THAT_ACCEPT_CONTEXT),
)
@pytest.mark.parametrize("path", ["chat", "inline_executor"])
async def test_context_a_json_schema_accepts_reaches_the_tool_unchanged(path, schema):
    """The review of the first #3396 fix: dropping ``context`` whenever
    ``properties`` omits it silently stripped tool data from an MCP server
    whose schema accepts the name some other way."""
    tool = _JsonSchemaTool("mcp__api__call", schema)
    agent = _Orchestrator(direct_tools={tool.name: tool})
    supplied = {"url": "u", "context": "the server's own context field"}

    if path == "chat":
        result = await _dispatch_direct(agent, tool.name, dict(supplied))
    else:
        result = await agent.execute_named_tool(
            tool.name, dict(supplied), session_id="s-3396", source="mcp",
        )

    assert result == {"success": True}
    assert tool.calls == [supplied]
    assert agent.hooks_manager.pre_calls == [supplied]
    assert agent.audit.rows == []


# --- the whole chat loop --------------------------------------------------------


@pytest.fixture()
def chat_agent():
    """A real ``KestrelAgent`` wired for ``_dispatch_tool_call``."""
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        agent = KestrelAgent(did="did:test:3396")
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_error = AsyncMock()
    agent.observability_store.log_tool_dispatch = AsyncMock(return_value="d-1")
    agent.hooks_manager = _AllowAllHooks()
    agent.audit = _AuditStore()
    agent.features = {
        "SecurityFeature": SimpleNamespace(permission_store=agent.audit),
    }
    return agent


async def _chat_dispatch(agent, tool_name: str, arguments: dict) -> tuple[Any, list]:
    messages: list = []
    result = await agent._dispatch_tool_call(
        SimpleNamespace(id="tc-3396", name=tool_name, arguments=arguments),
        {},
        {tool_name},
        messages,
        0,
        None,
    )
    return result, messages


def _github_feature(*tools):
    return SimpleNamespace(
        name="GitHubFeature",
        tool_name="github",
        enabled=True,
        get_tools=lambda: list(tools),
    )


@pytest.mark.asyncio
async def test_a_promoted_tool_called_with_task_runs_without_it(chat_agent):
    """The ``get_github_issue(task=...)`` case from the report."""
    tool = _SchemaTool("get_github_issue", "owner", "repo", "number")
    chat_agent.features["GitHubFeature"] = _github_feature(tool)
    chat_agent.register_dynamic_tools("github", [tool])

    result, _ = await _chat_dispatch(
        chat_agent,
        tool.name,
        {"owner": "o", "repo": "r", "number": "7", "task": "read the issue"},
    )

    assert result == {"success": True}
    assert tool.calls == [{"owner": "o", "repo": "r", "number": "7"}]


@pytest.mark.asyncio
async def test_an_unpromoted_tool_refuses_an_unknown_argument(chat_agent):
    """The registered-but-unpromoted branch reaches the same check."""
    tool = _SchemaTool("get_pull_request", "owner", "repo", "pull_number")
    chat_agent.features["GitHubFeature"] = _github_feature(tool)

    result, messages = await _chat_dispatch(
        chat_agent, tool.name, {"owner": "o", "repo": "r", "pr_number": "12"}
    )

    assert result["success"] is False
    assert "(did you mean 'pull_number'?)" in result["error"]
    assert tool.calls == []
    assert "Valid parameters: owner, repo, pull_number." in messages[-1]["content"]
    assert [row["action"] for row in chat_agent.audit.rows] == [ACTION_TOOL_VALIDATION]
