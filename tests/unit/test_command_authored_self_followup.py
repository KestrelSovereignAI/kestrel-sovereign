"""The chat command surface must refuse to schedule `self_followup` (#3112 P1).

Why this file exists, since the scheduler already has guards:

`_prepare_self_followup_args` gates on session-truthiness, then on
turn-ownership. Both are PROXIES for the real question, which is who composed
the text. `!schedule deadline` runs inside `_turn_lifecycle`, so a human typing
a command owns the live turn exactly as the model does, and every downstream
guard sees an indistinguishable call.

`TaskManager.execute_command` is the one place in core where authorship is
a FACT rather than an inference: its only two callers are chat command
surfaces. So the refusal lives there, and this file tests it there.

The tests drive the real `execute_command` rather than calling the helper
directly. A test that called `_refuse_command_authored_self_followup` in
isolation would pass even if nothing wired it into the command path — which is
the wiring failure mode that has bitten this PR twice already.
"""

import pytest

from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.signals.sources.self_followup import (
    TASK_NAME as SELF_FOLLOWUP_TASK_NAME,
)


class _StubTaskStore:
    async def save(self, task):  # pragma: no cover - not reached in refusals
        return task


def _manager(monkeypatch, *, skill_id: str):
    """Build a manager whose command router resolves to `skill_id`.

    `execute_skill` is replaced with a recorder that raises: if the refusal
    fails to fire, the test fails LOUDLY at the point of the unwanted write
    rather than by a missing string in a message.
    """
    mgr = TaskManager.__new__(TaskManager)
    mgr._agents = {}
    mgr.task_store = _StubTaskStore()
    mgr.hooks_manager = None

    monkeypatch.setattr(
        TaskManager,
        "get_agent_for_command",
        lambda self, user_input: ("scheduler", skill_id),
        raising=False,
    )

    async def _explode(*args, **kwargs):
        raise AssertionError(
            "execute_skill was reached: the command-surface refusal did not "
            "fire, so a chat-typed self_followup row would have been written"
        )

    monkeypatch.setattr(TaskManager, "execute_skill", _explode, raising=False)
    return mgr


@pytest.mark.asyncio
async def test_command_surface_refuses_self_followup(monkeypatch):
    """The exploit path: a human types the command, and it must be refused."""
    mgr = _manager(monkeypatch, skill_id="schedule_add_deadline")

    result = await mgr.execute_command(
        f'!schedule deadline 2026-08-26T09:00:00Z {SELF_FOLLOWUP_TASK_NAME} '
        '{"intent": "merge PR 3112"}'
    )

    assert result is not None
    assert result["success"] is False
    assert result["refused"] == "command_authored_self_followup"


@pytest.mark.asyncio
async def test_refusal_survives_positional_binder_shredding(monkeypatch):
    """The refusal must not depend on `parse_command_args` (#3118).

    That binder is strictly positional, so a spaces-containing `args_json`
    smears across later parameters and `task_name` need not land in
    `task_name`. A guard that read the parsed args would miss this; scanning
    the raw text does not.
    """
    mgr = _manager(monkeypatch, skill_id="schedule_add")

    result = await mgr.execute_command(
        f'!schedule add "0 9 * * *" {SELF_FOLLOWUP_TASK_NAME} '
        '{"intent": "check whether CI went green, then merge"}'
    )

    assert result is not None
    assert result["refused"] == "command_authored_self_followup"


@pytest.mark.asyncio
async def test_ordinary_scheduler_commands_still_run(monkeypatch):
    """The opposite direction: the refusal must not swallow normal commands.

    A guard that refused every `schedule_*` command would 'pass' the two tests
    above while breaking the feature. This asserts the reached-execute_skill
    case, so over-broad matching fails here.
    """
    mgr = TaskManager.__new__(TaskManager)
    mgr._agents = {}
    mgr.task_store = _StubTaskStore()
    mgr.hooks_manager = None

    monkeypatch.setattr(
        TaskManager,
        "get_agent_for_command",
        lambda self, user_input: ("scheduler", "schedule_add"),
        raising=False,
    )

    reached = {}

    async def _record(self, agent_id, skill_id, args, sync=True, session_id=None):
        reached["skill_id"] = skill_id
        raise RuntimeError("stop here: routing was allowed through, which is the assertion")

    monkeypatch.setattr(TaskManager, "execute_skill", _record, raising=False)

    await mgr.execute_command('!schedule add "0 9 * * *" memory_consolidation {}')

    assert reached["skill_id"] == "schedule_add"


@pytest.mark.asyncio
async def test_non_scheduler_command_mentioning_task_name_is_untouched(monkeypatch):
    """Scope check: the guard keys on scheduler skills, not on the word alone.

    Someone running a memory search for the string 'self_followup' is not
    scheduling anything, and refusing that would be the over-match this
    narrow guard is meant to avoid.
    """
    mgr = TaskManager.__new__(TaskManager)
    mgr._agents = {}
    mgr.task_store = _StubTaskStore()
    mgr.hooks_manager = None

    monkeypatch.setattr(
        TaskManager,
        "get_agent_for_command",
        lambda self, user_input: ("memory", "memory_search"),
        raising=False,
    )

    reached = {}

    async def _record(self, agent_id, skill_id, args, sync=True, session_id=None):
        reached["skill_id"] = skill_id
        raise RuntimeError("stop here: routing was allowed through, which is the assertion")

    monkeypatch.setattr(TaskManager, "execute_skill", _record, raising=False)

    await mgr.execute_command(f"!memory search {SELF_FOLLOWUP_TASK_NAME}")

    assert reached["skill_id"] == "memory_search"


@pytest.mark.asyncio
async def test_task_name_inside_another_arguments_value_is_not_refused(
    monkeypatch,
):
    """Exact token, not substring (#3112 gate-2 P2).

    A SCHEDULER command that merely mentions the task name inside another
    argument's value was refused by the original substring scan. It cannot
    create a ``self_followup`` row -- to do that the name must bind to the
    ``task_name`` parameter, which requires its own whitespace-delimited
    token -- so refusing it protected nothing and blocked a legitimate
    schedule.

    Distinct from ``test_non_scheduler_command_mentioning_task_name...``:
    that one escapes via the ``schedule_`` skill-prefix check, so it passes
    with the substring bug still present. This one is a ``schedule_add``
    and can only pass on the token fix.
    """
    mgr = TaskManager.__new__(TaskManager)
    mgr._agents = {}
    mgr.task_store = _StubTaskStore()
    mgr.hooks_manager = None

    monkeypatch.setattr(
        TaskManager,
        "get_agent_for_command",
        lambda self, user_input: ("scheduler", "schedule_add"),
        raising=False,
    )

    reached = {}

    async def _record(self, agent_id, skill_id, args, sync=True, session_id=None):
        reached["skill_id"] = skill_id
        raise RuntimeError("stop here: routing was allowed through, which is the assertion")

    monkeypatch.setattr(TaskManager, "execute_skill", _record, raising=False)

    await mgr.execute_command(
        '!schedule add @hourly github_pr_watch '
        f'{{"repo":"owner/{SELF_FOLLOWUP_TASK_NAME}","pr":1}}'
    )

    assert reached["skill_id"] == "schedule_add", (
        "a scheduler command that only mentions the task name inside another "
        "value cannot create the row, so it must not be refused"
    )
