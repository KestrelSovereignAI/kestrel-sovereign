"""Same-turn allowlist bounces (#2935).

#2929 reported tools that answered one call and were refused as "not in the
known tool allowlist" on a later call in the same turn. #2934 derived the
allowlist from the live registry; these tests cover what it left open:

* the one mid-turn path that still SHRINKS the derived set — LRU eviction,
  triggered by exploring a feature mid-turn, unmounting a dynamically-mounted
  tool (MCP) that no feature's ``get_tools()`` carries;
* the evidence every rejection now carries, so a future bounce says what the
  allowlist held at that instant instead of leaving it to inference;
* the codex inline executor's route, ``execute_named_tool``, which reaches the
  same tools without going through ``_dispatch_tool_call``.
"""

import asyncio
import contextvars
import copy
import logging
import pickle
import time
from contextlib import asynccontextmanager

import pytest

from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.agent.tool_allowlist import (
    MAX_TRACKED_TURNS,
    KnownToolNames,
    registry_generation,
    tool_name_set_digest,
)
from kestrel_sovereign.features.security.args_summary import remask_summary
from kestrel_sovereign.security.tool_audit import (
    ACTION_TOOL_RESOLUTION,
    ACTION_TOOL_VALIDATION,
)
from kestrel_sovereign.telemetry import bind_current_turn_id, turn_span_scope
from tests.unit.test_feature_runtime_lifecycle import _agent, _DispatchFeature
from tests.unit.test_tool_allowlist_registry import (  # noqa: F401
    _attach_security_feature,
    _make_feature,
    _make_tool,
    _tool_call,
    agent,
    dispatch_agent,
)


@asynccontextmanager
async def _live_turn(target, turn_id):
    """Run the body inside a real live turn: the lifecycle's own scope sets
    the task's turn id and the agent's live-turn mirror."""
    async with target._active_turn_scope(turn_id, turn_id, time.monotonic()):
        yield


async def _dispatch(target, name, arguments=None, *, iteration=0, known=None):
    """One loop iteration's dispatch: the allowlist is re-derived first,
    exactly as both orchestrator loops do at the top of every iteration."""
    messages = []
    result = await target._dispatch_tool_call(
        _tool_call(name, arguments),
        target._visible_features_by_tool_name(),
        target._known_tool_names() if known is None else known,
        messages,
        iteration,
        None,
    )
    return result, messages


def _evidence_fields(summary: str) -> dict:
    """Parse the ``key=value`` evidence out of a refusal's audit summary."""
    evidence = next(
        part for part in summary.split(" | ") if part.startswith("allowlist ")
    )
    return dict(
        token.split("=", 1) for token in evidence.split()[1:]
    )


@pytest.fixture
def mcp_agent(dispatch_agent):  # noqa: F811 - the imported fixture
    """An agent with one mounted MCP tool and a feature whose exploration
    overflows a two-slot direct-tool cap, evicting the oldest owner."""
    dispatch_agent.MAX_DIRECT_TOOLS = 2
    dispatch_agent.mcp_read = _make_tool("mcp__files__read")
    dispatch_agent.register_dynamic_tools("mcp:files", [dispatch_agent.mcp_read])
    dispatch_agent.features = {
        "MemoryFeature": _make_feature(
            "memory_feature", [_make_tool("recall_a"), _make_tool("recall_b")]
        ),
    }
    return dispatch_agent


# =============================================================================
# Step 1: the mid-turn shrink path, reproduced and closed
# =============================================================================

class TestMidTurnEviction:
    @pytest.mark.asyncio
    async def test_exploration_does_not_unmount_a_tool_this_turn_used(
        self, mcp_agent,
    ):
        """The reproduction: call a mounted MCP tool, explore a feature (which
        overflows MAX_DIRECT_TOOLS and evicts the oldest owner), call the MCP
        tool again — all in one turn, with no configuration change.

        On the code #2934 left, the exploration evicted ``mcp:files`` and the
        second call was refused as not in the allowlist."""
        async with _live_turn(mcp_agent, "turn_repro"):
            first, _ = await _dispatch(mcp_agent, "mcp__files__read", iteration=0)
            await _dispatch(
                mcp_agent, "memory_feature", {"task": "recall"}, iteration=1
            )
            assert "mcp__files__read" in mcp_agent._known_tool_names()
            second, _ = await _dispatch(mcp_agent, "mcp__files__read", iteration=2)

        assert first == {"success": True, "tool": "mcp__files__read"}
        assert second == {"success": True, "tool": "mcp__files__read"}
        assert mcp_agent.mcp_read.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_eviction_still_runs_for_a_tool_an_earlier_turn_used(
        self, mcp_agent,
    ):
        """The protection is scoped to the live turn: LRU still sheds an owner
        the previous turn used, and the refusal then says so."""
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_one"):
            await _dispatch(mcp_agent, "mcp__files__read")
        async with _live_turn(mcp_agent, "turn_two"):
            await _dispatch(mcp_agent, "memory_feature", {"task": "recall"})
            assert "mcp__files__read" not in mcp_agent._direct_tools
            await _dispatch(mcp_agent, "mcp__files__read", iteration=1)

        assert mcp_agent.mcp_read.execute.await_count == 1
        assert len(store.rows) == 1
        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["in_set"] == "no"
        assert fields["turn"] == "turn_two"
        assert fields["accepted_earlier_this_turn"] == "no"

    @pytest.mark.asyncio
    async def test_a_feature_owner_the_turn_used_is_still_evictable(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        """Evicting a feature's tools leaves them resolvable through the
        feature, so only mount-only tools hold their owner. Holding feature
        owners too would let a busy turn grow the schema past the cap."""
        subject = dispatch_agent
        subject.MAX_DIRECT_TOOLS = 2
        github_issue = _make_tool("get_github_issue")
        github = _make_feature("github", [github_issue])
        memory = _make_feature(
            "memory_feature", [_make_tool("recall_a"), _make_tool("recall_b")]
        )
        subject.features = {"GitHubFeature": github, "MemoryFeature": memory}

        async with _live_turn(subject, "turn_feature"):
            await _dispatch(subject, "github", {"task": "read"})
            await _dispatch(subject, "get_github_issue", iteration=1)
            await _dispatch(subject, "memory_feature", {"task": "recall"}, iteration=2)
            assert "get_github_issue" not in subject._direct_tools  # evicted
            result, _ = await _dispatch(subject, "get_github_issue", iteration=3)

        assert result == {"success": True, "tool": "get_github_issue"}
        assert github_issue.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_eviction_from_outside_the_turn_still_holds_its_tools(
        self, mcp_agent,
    ):
        """An MCP server remounting from a background task runs eviction with
        no turn id of its own; the agent's live turn still holds its tools."""
        async with _live_turn(mcp_agent, "turn_background"):
            await _dispatch(mcp_agent, "mcp__files__read")
            with bind_current_turn_id(None):
                assert mcp_agent._get_current_turn_id() is None
                mcp_agent.register_dynamic_tools(
                    "mcp:other", [_make_tool("other_a"), _make_tool("other_b")]
                )
            assert "mcp__files__read" in mcp_agent._direct_tools
            result, _ = await _dispatch(mcp_agent, "mcp__files__read", iteration=1)

        assert result == {"success": True, "tool": "mcp__files__read"}

    @pytest.mark.asyncio
    async def test_a_finished_turn_holds_nothing(self, mcp_agent):
        """A task detached from a finished turn still carries that turn's id.
        The turn is over, so its tools are evictable again."""
        async with _live_turn(mcp_agent, "turn_done"):
            await _dispatch(mcp_agent, "mcp__files__read")
        with bind_current_turn_id("turn_done"):
            mcp_agent.register_dynamic_tools(
                "mcp:other", [_make_tool("other_a"), _make_tool("other_b")]
            )

        assert "mcp__files__read" not in mcp_agent._direct_tools

    @pytest.mark.asyncio
    async def test_when_nothing_is_evictable_the_cap_waits_for_the_turn(
        self, mcp_agent, caplog,
    ):
        """Every unpinned owner held by the live turn: eviction bails with a
        warning naming the held owner instead of unmounting it."""
        mcp_agent._pinned_features.add("memory_feature")
        async with _live_turn(mcp_agent, "turn_held"):
            with caplog.at_level(logging.WARNING):
                await _dispatch(mcp_agent, "mcp__files__read")
                await _dispatch(mcp_agent, "memory_feature", {"task": "recall"})

        assert "mcp__files__read" in mcp_agent._direct_tools
        assert "Cannot evict" in caplog.text
        assert "mcp:files" in caplog.text


# =============================================================================
# Step 1, codex route: execute_named_tool feeds the same ledger
# =============================================================================

async def _on_reader_task(executor, name, arguments):
    """Run one inline call the way the codex app-server does: on a task born
    with none of the turn's context, so only the executor's re-presented turn
    scope tells it which turn it belongs to."""
    return await asyncio.create_task(
        executor(name, arguments), context=contextvars.Context()
    )


def _evidence_logs(caplog, level):
    """Evidence fields of each ``not registered`` refusal logged at ``level``."""
    return [
        dict(
            token.split("=", 1)
            for token in record.getMessage().split(" | ", 1)[1].split()[1:]
        )
        for record in caplog.records
        if record.levelno == level and "not registered" in record.getMessage()
    ]


class TestNamedToolDispatch:
    @pytest.mark.asyncio
    async def test_inline_executor_does_not_lose_an_mcp_tool_to_exploration(
        self, mcp_agent,
    ):
        """The reproduction on the codex route: an inline MCP call, an inline
        feature exploration that overflows MAX_DIRECT_TOOLS, the same inline
        MCP call again — one turn, no configuration change.

        Before the ledger was fed from ``execute_named_tool``, the exploration
        evicted ``mcp:files`` and the second call raised
        ToolNotRegisteredError."""
        async with _live_turn(mcp_agent, "turn_inline"):
            executor = mcp_agent._make_inline_tool_executor("chat-inline")
            _, first = await _on_reader_task(
                executor, "mcp__files__read", {"query": "x"}
            )
            _, explored = await _on_reader_task(
                executor, "memory_feature", {"task": "recall"}
            )
            # Exploration ran and the cap was enforced: the newest owner was
            # the only evictable one, so the turn's MCP tool is what is left.
            assert set(mcp_agent._direct_tools) == {"mcp__files__read"}
            _, second = await _on_reader_task(
                executor, "mcp__files__read", {"query": "x"}
            )

        assert explored == {"success": True, "result": "memory_feature ran"}
        assert first == second == {"success": True, "tool": "mcp__files__read"}
        assert mcp_agent.mcp_read.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_inline_and_chat_calls_share_one_count(self, mcp_agent):
        """Both routes count into the same turn, once per call: an inline call
        between two chat-path calls is call 2, and the refusal after them is
        call 4 and knows the inline acceptance."""
        store = _attach_security_feature(mcp_agent)
        mcp_agent.MAX_DIRECT_TOOLS = 10
        async with _live_turn(mcp_agent, "turn_shared"):
            executor = mcp_agent._make_inline_tool_executor("chat-shared")
            await _dispatch(mcp_agent, "memory_feature", {"task": "recall"})
            await _on_reader_task(executor, "mcp__files__read", {"query": "x"})
            await _dispatch(mcp_agent, "ghost_tool", iteration=1)
            mcp_agent.unregister_dynamic_tools("mcp:files")
            await _dispatch(mcp_agent, "mcp__files__read", iteration=2)

        calls = [_evidence_fields(row["args_summary"]) for row in store.rows]
        assert [fields["call"] for fields in calls] == ["3", "4"]
        assert calls[1]["accepted_earlier_this_turn"] == "yes"
        assert calls[1]["accepted_at_call"] == "2"

    @pytest.mark.asyncio
    async def test_inline_bounce_of_a_tool_this_turn_used_is_a_warning(
        self, mcp_agent, caplog,
    ):
        """The same-turn bounce on the codex route (an MCP server removed
        mid-turn) is logged at WARNING with what the registry held."""
        async with _live_turn(mcp_agent, "turn_inline_bounce"):
            executor = mcp_agent._make_inline_tool_executor("chat-bounce")
            accepted_known = mcp_agent._registered_tool_names()
            await _on_reader_task(executor, "mcp__files__read", {"query": "x"})
            mcp_agent.unregister_dynamic_tools("mcp:files")
            with caplog.at_level(logging.INFO):
                with pytest.raises(ToolNotRegisteredError):
                    await _on_reader_task(
                        executor, "mcp__files__read", {"query": "x"}
                    )

        [fields] = _evidence_logs(caplog, logging.WARNING)
        assert fields["in_set"] == "no"
        assert fields["turn"] == "turn_inline_bounce"
        assert fields["call"] == "2"
        assert fields["accepted_earlier_this_turn"] == "yes"
        assert fields["accepted_at_call"] == "1"
        assert fields["accepted_sha256_16"] == accepted_known.digest
        assert int(fields["registry_generation"]) > int(
            fields["accepted_registry_generation"]
        )

    @pytest.mark.asyncio
    async def test_inline_subagent_call_is_recorded(self, mcp_agent, caplog):
        """The subagent branch records its acceptance too: a dispatcher the
        turn called inline, then disabled mid-turn, is refused as a name this
        turn already used."""
        memory = mcp_agent.features["MemoryFeature"]
        async with _live_turn(mcp_agent, "turn_inline_subagent"):
            executor = mcp_agent._make_inline_tool_executor("chat-subagent")
            await _on_reader_task(executor, "memory_feature", {"task": "recall"})
            memory.enabled = False
            with caplog.at_level(logging.INFO):
                with pytest.raises(ToolNotRegisteredError):
                    await _on_reader_task(
                        executor, "memory_feature", {"task": "recall"}
                    )

        [fields] = _evidence_logs(caplog, logging.WARNING)
        assert fields["call"] == "2"
        assert fields["accepted_at_call"] == "1"
        assert fields["features_enabled"] == "0/1"

    @pytest.mark.asyncio
    async def test_inline_argument_refusal_carries_evidence(self, mcp_agent):
        """The inline route's argument check writes the same evidence as the
        chat path's, and a call it refuses is not an acceptance."""
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_inline_args"):
            executor = mcp_agent._make_inline_tool_executor("chat-args")
            await _on_reader_task(executor, "mcp__files__read", {"query": "x"})
            for _ in range(2):
                await _on_reader_task(
                    executor, "mcp__files__read", {"querry": "x"}
                )

        assert len(store.rows) == 2
        first, second = (_evidence_fields(r["args_summary"]) for r in store.rows)
        assert first["in_set"] == "yes"
        assert first["turn"] == "turn_inline_args"
        assert first["call"] == "2"
        assert first["accepted_at_call"] == "1"
        assert second["call"] == "3"
        assert second["accepted_at_call"] == "1"

    @pytest.mark.asyncio
    async def test_inline_probe_for_a_tool_never_used_is_not_a_warning(
        self, mcp_agent, caplog,
    ):
        """Callers probe for an optional tool by name and handle the error;
        that refusal carries the evidence at INFO, not WARNING."""
        async with _live_turn(mcp_agent, "turn_probe"):
            executor = mcp_agent._make_inline_tool_executor("chat-probe")
            with caplog.at_level(logging.INFO):
                with pytest.raises(ToolNotRegisteredError):
                    await _on_reader_task(executor, "workflow_run", {})

        assert _evidence_logs(caplog, logging.WARNING) == []
        [fields] = _evidence_logs(caplog, logging.INFO)
        assert fields["call"] == "1"
        assert fields["accepted_earlier_this_turn"] == "no"

    @pytest.mark.asyncio
    async def test_a_call_a_running_tool_makes_is_not_counted_again(
        self, mcp_agent,
    ):
        """A tool that dispatches another by name (a workflow runner) does so
        inside its own call: the nested call takes that call's position, so
        the count stays the turn's own calls, on either route."""
        store = _attach_security_feature(mcp_agent)

        async def _run_nested(**_kwargs):
            return await mcp_agent.execute_named_tool(
                "mcp__files__read", {"query": "x"},
                session_id="nested", source="workflow_runner",
            )

        runner = _make_tool("run_steps")
        runner.execute.side_effect = _run_nested
        mcp_agent.register_dynamic_tools("runner", [runner])
        mcp_agent.MAX_DIRECT_TOOLS = 10

        async with _live_turn(mcp_agent, "turn_nested"):
            executor = mcp_agent._make_inline_tool_executor("chat-nested")
            await _dispatch(mcp_agent, "run_steps")
            await _on_reader_task(executor, "run_steps", {"query": "x"})
            await _dispatch(mcp_agent, "ghost_tool", iteration=1)

        assert mcp_agent.mcp_read.execute.await_count == 2
        [fields] = [_evidence_fields(row["args_summary"]) for row in store.rows]
        assert fields["call"] == "3"

    @pytest.mark.asyncio
    async def test_a_nested_call_still_holds_its_tool_against_eviction(
        self, mcp_agent,
    ):
        """The nested call is recorded at its enclosing call's position, so an
        MCP tool only a workflow runner called is held for the turn too."""
        async def _run_nested(**_kwargs):
            return await mcp_agent.execute_named_tool(
                "mcp__files__read", {"query": "x"},
                session_id="nested", source="workflow_runner",
            )

        runner = _make_feature(
            "runner_feature", [], name="RunnerFeature",
        )
        runner.execute_as_subagent.side_effect = _run_nested
        mcp_agent.features["RunnerFeature"] = runner

        async with _live_turn(mcp_agent, "turn_nested_held"):
            await _dispatch(mcp_agent, "runner_feature", {"task": "run"})
            await _dispatch(
                mcp_agent, "memory_feature", {"task": "recall"}, iteration=1
            )
            assert "mcp__files__read" in mcp_agent._direct_tools

    @pytest.mark.asyncio
    async def test_a_task_outliving_its_call_is_not_nested(self, mcp_agent):
        """A task created inside a call keeps the call's marker after the call
        returns (a lazily started reader loop can be such a task). Its later
        calls are the turn's own, not the finished call's."""
        store = _attach_security_feature(mcp_agent)
        spawned = []
        release = asyncio.Event()

        async def _later_call():
            await release.wait()
            return await mcp_agent.execute_named_tool(
                "mcp__files__read", {"query": "x"},
                session_id="later", source="reader",
            )

        async def _spawn(**_kwargs):
            spawned.append(asyncio.create_task(_later_call()))
            return {"success": True}

        spawner = _make_tool("spawn_reader")
        spawner.execute.side_effect = _spawn
        mcp_agent.register_dynamic_tools("spawner", [spawner])
        mcp_agent.MAX_DIRECT_TOOLS = 10

        async with _live_turn(mcp_agent, "turn_outlived"):
            await _dispatch(mcp_agent, "spawn_reader")
            release.set()
            await spawned[0]
            await _dispatch(mcp_agent, "ghost_tool", iteration=1)

        [fields] = [_evidence_fields(row["args_summary"]) for row in store.rows]
        assert fields["call"] == "3"


# =============================================================================
# Step 2: every refusal carries the allowlist evidence
# =============================================================================

class TestRejectionEvidence:
    @pytest.mark.asyncio
    async def test_rejection_row_carries_every_field(self, mcp_agent):
        """A tool accepted at call 1, then unmounted mid-turn (an MCP server
        removed) and refused at call 2: the row records the requested name,
        the set's size and digest, the registry generation that produced it,
        the turn and call index, and the earlier acceptance."""
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_fields"):
            accepted_known = mcp_agent._known_tool_names()
            await _dispatch(mcp_agent, "mcp__files__read", known=accepted_known)
            mcp_agent.unregister_dynamic_tools("mcp:files")
            refused_known = mcp_agent._known_tool_names()
            await _dispatch(
                mcp_agent, "mcp__files__read", iteration=1, known=refused_known
            )

        assert len(store.rows) == 1
        row = store.rows[0]
        assert row["tool_name"] == "mcp__files__read"
        assert row["action"] == ACTION_TOOL_VALIDATION
        fields = _evidence_fields(row["args_summary"])
        assert fields == {
            "in_set": "no",
            "size": str(len(refused_known)),
            "sha256_16": tool_name_set_digest(refused_known),
            "registry_generation": str(refused_known.registry_generation),
            "registry_generation_now": str(registry_generation(mcp_agent)),
            "features_enabled": "2/2",
            "unreadable_features": "none",
            "turn": "turn_fields",
            "call": "2",
            "accepted_earlier_this_turn": "yes",
            "accepted_at_call": "1",
            "accepted_size": str(len(accepted_known)),
            "accepted_sha256_16": tool_name_set_digest(accepted_known),
            "accepted_registry_generation": str(accepted_known.registry_generation),
        }
        # The transition the evidence exists to show: the set shrank and was
        # produced by a later registry than the one that accepted the name.
        assert len(refused_known) < len(accepted_known)
        assert fields["sha256_16"] != fields["accepted_sha256_16"]
        assert refused_known.registry_generation > accepted_known.registry_generation

    @pytest.mark.asyncio
    async def test_dispatcher_accepted_then_disabled_mid_turn(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        """A feature disabled mid-turn is a real capability boundary, so its
        dispatcher IS refused; the refusal says it answered earlier in the
        turn and that one fewer feature was enabled."""
        store = _attach_security_feature(dispatch_agent)
        github = _make_feature("github", [_make_tool("get_github_issue")])
        dispatch_agent.features["GitHubFeature"] = github
        async with _live_turn(dispatch_agent, "turn_disabled"):
            accepted_known = dispatch_agent._known_tool_names()
            first, _ = await _dispatch(
                dispatch_agent, "github", {"task": "read"}, known=accepted_known
            )
            github.enabled = False
            refused_known = dispatch_agent._known_tool_names()
            await _dispatch(
                dispatch_agent, "github", {"task": "read"}, iteration=1,
                known=refused_known,
            )

        assert first == {"success": True, "result": "github ran"}
        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["in_set"] == "no"
        assert fields["accepted_earlier_this_turn"] == "yes"
        assert fields["accepted_at_call"] == "1"
        assert fields["features_enabled"] == "1/2"
        assert accepted_known.features_enabled == 2

    @pytest.mark.asyncio
    async def test_the_most_recent_acceptance_is_reported(self, mcp_agent):
        """The acceptance closest to the refusal is the one that brackets the
        transition, so it is the one reported."""
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_recent"):
            await _dispatch(mcp_agent, "mcp__files__read")
            await _dispatch(mcp_agent, "mcp__files__read", iteration=1)
            mcp_agent.unregister_dynamic_tools("mcp:files")
            await _dispatch(mcp_agent, "mcp__files__read", iteration=2)

        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["call"] == "3"
        assert fields["accepted_at_call"] == "2"

    def test_a_call_is_never_earlier_than_itself(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        """"Earlier" is strict: an acceptance recorded for the refused call
        itself is not evidence that the name answered before it."""
        from kestrel_sovereign.agent.tool_allowlist import (
            allowlist_evidence,
            begin_tool_call,
            record_tool_acceptance,
        )

        with turn_span_scope("turn_self"):
            context = begin_tool_call(agent, {"tool"})
            record_tool_acceptance(agent, context, "tool")
            evidence = allowlist_evidence(agent, context, "tool")

        assert evidence.call_index == 1
        assert evidence.accepted_earlier_this_turn is False

    def test_a_later_acceptance_does_not_hide_an_earlier_one(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        """A parallel batch reaches acceptance in completion order, so call 3
        can be recorded before call 2 is refused. Call 1 still came first."""
        from kestrel_sovereign.agent.tool_allowlist import (
            allowlist_evidence,
            begin_tool_call,
            record_tool_acceptance,
        )

        with turn_span_scope("turn_parallel"):
            first, second, third = [
                begin_tool_call(agent, {"tool"}) for _ in range(3)
            ]
            record_tool_acceptance(agent, first, "tool")
            record_tool_acceptance(agent, third, "tool")
            evidence = allowlist_evidence(agent, second, "tool")

        assert evidence.call_index == 2
        assert evidence.accepted_earlier.call_index == 1

    def test_turn_id_is_reduced_to_plain_characters(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        """The audit read path withholds bracketed structure in front of
        ``args=``; a turn id must not be able to open one."""
        from kestrel_sovereign.agent.tool_allowlist import (
            allowlist_evidence,
            begin_tool_call,
        )

        with turn_span_scope('turn_[{"a":1}]'):
            context = begin_tool_call(agent, {"tool"})
        text = allowlist_evidence(agent, context, "ghost").describe()

        assert not set('[]{}"') & set(text)
        row = f"reason | {text} | args={{\"query\": \"x\"}}"
        assert text in remask_summary(row)

    @pytest.mark.asyncio
    async def test_accepted_earlier_is_false_for_a_name_never_accepted(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        store = _attach_security_feature(dispatch_agent)
        dispatch_agent.features["GitHubFeature"] = _make_feature(
            "github", [_make_tool("get_github_issue")]
        )
        async with _live_turn(dispatch_agent, "turn_never"):
            await _dispatch(dispatch_agent, "get_github_issue")
            await _dispatch(dispatch_agent, "ghost_tool", iteration=1)

        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["call"] == "2"
        assert fields["accepted_earlier_this_turn"] == "no"
        assert "accepted_at_call" not in fields

    @pytest.mark.asyncio
    async def test_accepted_earlier_is_false_when_another_turn_accepted_it(
        self, mcp_agent,
    ):
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_a"):
            await _dispatch(mcp_agent, "mcp__files__read")
        mcp_agent.unregister_dynamic_tools("mcp:files")
        async with _live_turn(mcp_agent, "turn_b"):
            await _dispatch(mcp_agent, "mcp__files__read")

        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["turn"] == "turn_b"
        assert fields["call"] == "1"
        assert fields["accepted_earlier_this_turn"] == "no"

    @pytest.mark.asyncio
    async def test_outside_a_turn_nothing_counts_as_earlier(self, mcp_agent):
        store = _attach_security_feature(mcp_agent)
        await _dispatch(mcp_agent, "mcp__files__read")
        mcp_agent.unregister_dynamic_tools("mcp:files")
        await _dispatch(mcp_agent, "mcp__files__read")

        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["turn"] == "none"
        assert fields["call"] == "unknown"
        assert fields["accepted_earlier_this_turn"] == "no"

    @pytest.mark.asyncio
    async def test_call_index_counts_across_iterations(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        store = _attach_security_feature(dispatch_agent)
        async with _live_turn(dispatch_agent, "turn_count"):
            for iteration in range(3):
                await _dispatch(dispatch_agent, "ghost_tool", iteration=iteration)

        assert [
            _evidence_fields(row["args_summary"])["call"] for row in store.rows
        ] == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_resolution_rejection_carries_evidence(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        """Allowlisted but resolvable by no loaded feature."""
        store = _attach_security_feature(dispatch_agent)
        async with _live_turn(dispatch_agent, "turn_resolution"):
            await _dispatch(
                dispatch_agent, "ghost_tool", known={"ghost_tool", "other"}
            )

        row = store.rows[0]
        assert row["action"] == ACTION_TOOL_RESOLUTION
        fields = _evidence_fields(row["args_summary"])
        assert fields["in_set"] == "yes"
        assert fields["size"] == "2"
        assert fields["sha256_16"] == tool_name_set_digest({"ghost_tool", "other"})
        # A plain set carries no provenance; the evidence says so rather than
        # inventing a generation.
        assert fields["registry_generation"] == "unknown"
        assert fields["features_enabled"] == "unknown/unknown"

    @pytest.mark.asyncio
    async def test_direct_tool_argument_refusal_carries_evidence(
        self, mcp_agent,
    ):
        """The argument check is the last guardrail: a call it refuses is not
        an acceptance, and its row carries the same evidence."""
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_args"):
            await _dispatch(mcp_agent, "mcp__files__read")
            await _dispatch(
                mcp_agent, "mcp__files__read", {"querry": "x"}, iteration=1
            )
            await _dispatch(
                mcp_agent, "mcp__files__read", {"querry": "x"}, iteration=2
            )

        assert len(store.rows) == 2
        first, second = (_evidence_fields(r["args_summary"]) for r in store.rows)
        assert first["in_set"] == "yes"
        assert first["call"] == "2"
        assert first["accepted_at_call"] == "1"
        # The refused call 2 did not replace call 1 as the last acceptance.
        assert second["accepted_at_call"] == "1"

    @pytest.mark.asyncio
    async def test_registry_generation_is_the_one_that_produced_the_set(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        """A batch derives the set once; a change later in the batch moves the
        live registry on, and the evidence keeps the two apart."""
        store = _attach_security_feature(dispatch_agent)
        known = dispatch_agent._known_tool_names()
        dispatch_agent.register_dynamic_tools("mcp:late", [_make_tool("late_tool")])

        async with _live_turn(dispatch_agent, "turn_generation"):
            await _dispatch(dispatch_agent, "late_tool", known=known)

        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["registry_generation"] == str(known.registry_generation)
        assert int(fields["registry_generation_now"]) > known.registry_generation

    @pytest.mark.asyncio
    async def test_unreadable_feature_is_named(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        """A feature whose ``get_tools()`` raises loses its names from the set
        without any registry change; the evidence names it."""
        store = _attach_security_feature(dispatch_agent)
        broken = _make_feature("github", [])
        broken.get_tools.side_effect = RuntimeError("client not ready")
        dispatch_agent.features["GitHubFeature"] = broken

        async with _live_turn(dispatch_agent, "turn_unreadable"):
            await _dispatch(dispatch_agent, "create_github_issue")

        fields = _evidence_fields(store.rows[0]["args_summary"])
        assert fields["unreadable_features"] == "GitHubFeature"

    @pytest.mark.asyncio
    async def test_evidence_holds_no_arguments(
        self, dispatch_agent,  # noqa: F811 - the imported fixture
    ):
        """Names, counts and digests only. The arguments stay in ``args=``,
        masked exactly as before."""
        store = _attach_security_feature(dispatch_agent)
        async with _live_turn(dispatch_agent, "turn_secret"):
            await _dispatch(
                dispatch_agent, "ghost_tool",
                {"api_key": "sk-live-secret", "repo": "owner/private-repo"},
            )

        summary = store.rows[0]["args_summary"]
        evidence = next(p for p in summary.split(" | ") if p.startswith("allowlist "))
        assert "owner/private-repo" not in evidence
        assert "sk-live-secret" not in summary
        assert "owner/private-repo" in summary.split(" | args=", 1)[1]

    @pytest.mark.asyncio
    async def test_audit_read_path_shows_the_evidence_verbatim(
        self, mcp_agent,
    ):
        """``remask_summary`` withholds bracketed structure it cannot parse in
        front of ``args=``. The evidence must survive it whole."""
        store = _attach_security_feature(mcp_agent)
        async with _live_turn(mcp_agent, "turn_read"):
            await _dispatch(mcp_agent, "mcp__files__read")
            mcp_agent.unregister_dynamic_tools("mcp:files")
            await _dispatch(mcp_agent, "mcp__files__read", {"query": "q"})

        stored = store.rows[0]["args_summary"]
        evidence = next(p for p in stored.split(" | ") if p.startswith("allowlist "))
        assert evidence in remask_summary(stored)

    @pytest.mark.asyncio
    async def test_rejection_is_logged_at_warning_with_the_evidence(
        self, mcp_agent, caplog,
    ):
        async with _live_turn(mcp_agent, "turn_log"):
            with caplog.at_level(logging.WARNING):
                await _dispatch(mcp_agent, "mcp__files__read")
                mcp_agent.unregister_dynamic_tools("mcp:files")
                await _dispatch(mcp_agent, "mcp__files__read", iteration=1)

        refusals = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "refused tool" in r.getMessage()
        ]
        assert len(refusals) == 1
        message = refusals[0].getMessage()
        assert "'mcp__files__read'" in message
        assert "accepted_earlier_this_turn=yes" in message
        assert "accepted_at_call=1" in message
        assert "turn=turn_log" in message


# =============================================================================
# The digest and the generation
# =============================================================================

class TestNameSetDigest:
    def test_digest_is_stable_for_the_same_set(self):
        names = ["github", "create_github_issue", "mcp__files__read"]
        assert tool_name_set_digest(names) == tool_name_set_digest(reversed(names))
        assert tool_name_set_digest(names) == tool_name_set_digest(set(names))
        assert tool_name_set_digest(names) == tool_name_set_digest(names + names)
        # Not Python's per-process salted hash(): a fixed value.
        assert tool_name_set_digest(["a", "b"]) == tool_name_set_digest(["b", "a"])
        assert tool_name_set_digest([]) == tool_name_set_digest(set())

    def test_digest_changes_when_the_set_changes(self):
        base = {"github", "create_github_issue"}
        assert tool_name_set_digest(base) != tool_name_set_digest(base - {"github"})
        assert tool_name_set_digest(base) != tool_name_set_digest(base | {"x"})
        # Joining cannot make two different sets collide.
        assert tool_name_set_digest({"a,b"}) != tool_name_set_digest({"a", "b"})

    def test_known_tool_names_survive_copy_and_pickle(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        """A frozenset copies and pickles; a subclass of one must too, with
        its provenance."""
        agent.register_dynamic_tools("mcp:files", [_make_tool("mcp__files__read")])
        known = agent._known_tool_names()

        for clone in (
            copy.copy(known),
            copy.deepcopy(known),
            pickle.loads(pickle.dumps(known)),
        ):
            assert isinstance(clone, KnownToolNames)
            assert clone == known
            assert clone.registry_generation == known.registry_generation
            assert clone.features_loaded == known.features_loaded
            assert clone.features_enabled == known.features_enabled
            assert clone.unreadable_features == known.unreadable_features
            assert clone.digest == known.digest

    def test_known_tool_names_digest_matches(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        agent.features = {"GitHubFeature": _make_feature("github", [_make_tool("t")])}
        known = agent._known_tool_names()
        assert isinstance(known, KnownToolNames)
        assert known.digest == tool_name_set_digest(set(known))


class TestRegistryGeneration:
    def test_known_tool_names_carries_the_current_generation(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        before = agent._known_tool_names().registry_generation
        agent.register_dynamic_tools("mcp:files", [_make_tool("mcp__files__read")])
        after = agent._known_tool_names()
        assert after.registry_generation > before
        assert after.registry_generation == registry_generation(agent)

    def test_mount_unmount_and_eviction_each_advance_it(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        agent.MAX_DIRECT_TOOLS = 1
        start = registry_generation(agent)
        agent.register_dynamic_tools("mcp:a", [_make_tool("a")])
        mounted = registry_generation(agent)
        assert mounted > start
        # Mounting a second owner over the cap evicts the first.
        agent.register_dynamic_tools("mcp:b", [_make_tool("b")])
        assert "a" not in agent._direct_tools
        assert registry_generation(agent) >= mounted + 2
        evicted = registry_generation(agent)
        agent.unregister_dynamic_tools("mcp:b")
        assert registry_generation(agent) == evicted + 1

    def test_a_no_op_unmount_leaves_it(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        start = registry_generation(agent)
        assert agent.unregister_dynamic_tools("mcp:never-mounted") == 0
        assert registry_generation(agent) == start

    def test_constitution_receipt_tool_advances_it(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        start = registry_generation(agent)
        agent.register_constitution_receipt_tool(canary="c", signal_id="s")
        registered = registry_generation(agent)
        agent.clear_constitution_receipt_tool()
        assert start < registered < registry_generation(agent)

    @pytest.mark.asyncio
    async def test_feature_load_disable_and_enable_each_advance_it(self, tmp_path):
        """The agent's own feature-map rails, driven for real."""
        lifecycle_agent = _agent(tmp_path)
        feature = _DispatchFeature(lifecycle_agent)

        start = registry_generation(lifecycle_agent)
        await lifecycle_agent._register_feature(feature)
        loaded = lifecycle_agent._known_tool_names()
        assert loaded.registry_generation > start
        assert {"dispatch_feature", "dispatch_do"} <= loaded

        await lifecycle_agent._unregister_feature_runtime(feature, unload=False)
        disabled = lifecycle_agent._known_tool_names()
        assert disabled.registry_generation > loaded.registry_generation
        assert "dispatch_do" not in disabled
        assert disabled.features_enabled == loaded.features_enabled - 1

        await lifecycle_agent._activate_feature_runtime(feature)
        enabled = lifecycle_agent._known_tool_names()
        assert enabled.registry_generation > disabled.registry_generation
        assert {"dispatch_feature", "dispatch_do"} <= enabled

    def test_ledgers_are_bounded(
        self, agent,  # noqa: F811 - the imported fixture
    ):
        from kestrel_sovereign.agent.tool_allowlist import begin_tool_call

        for index in range(MAX_TRACKED_TURNS + 5):
            with turn_span_scope(f"turn_{index}"):
                begin_tool_call(agent, set())
        assert len(agent._tool_turn_ledgers) == MAX_TRACKED_TURNS
        assert f"turn_{MAX_TRACKED_TURNS + 4}" in agent._tool_turn_ledgers
