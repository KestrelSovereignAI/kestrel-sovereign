"""Irrevocable native authority across copied contexts (#3569)."""

import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.turn_scope import capture_turn_scope


class Authority:
    backend_type = "postgres"

    def require_work(self):
        pass

    async def lock_and_validate(self, connection):
        raise AssertionError("the pure context tests must not issue database work")


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["normal", "revoke"])
async def test_copied_child_cannot_regain_authority_after_scope_ends(retirement):
    proceed = asyncio.Event()

    async def child():
        await proceed.wait()
        with pytest.raises(ExecutionAuthorityError):
            require_execution_work()
        # A fresh child scope must not discard its stale ancestor.
        with pytest.raises(ExecutionAuthorityError):
            with bind_execution_custody(Authority()):
                pass

    with bind_execution_custody(Authority()) as scope:
        task = asyncio.create_task(child())
        if retirement == "revoke":
            scope.revoke("lost exact authority session")
    proceed.set()
    await task
    require_execution_work()  # No scope was leaked into the owner context.


def test_nested_ancestor_revocation_is_not_masked_by_live_child():
    with bind_execution_custody(Authority()) as parent:
        with bind_execution_custody(Authority()):
            parent.revoke("parent session lost")
            with pytest.raises(ExecutionAuthorityError, match="parent session lost"):
                require_execution_work()


@pytest.mark.asyncio
async def test_foreign_turn_executor_rebinds_same_mutable_authority():
    # This task exists before the turn publishes, like the inline transport.
    ready = asyncio.Event()
    captured = None

    async def foreign_reader():
        await ready.wait()
        with pytest.raises(ExecutionAuthorityError):
            with captured.bind():
                require_execution_work()

    task = asyncio.create_task(foreign_reader())
    with bind_execution_custody(Authority()) as scope:
        captured = capture_turn_scope(object())
        scope.revoke("turn lost authority")
        ready.set()
        await task


def test_repeated_revoke_never_changes_original_reason_or_revives():
    with bind_execution_custody(Authority()) as scope:
        scope.revoke("first loss")
        scope.revoke("second loss")
        with pytest.raises(ExecutionAuthorityError, match="first loss"):
            require_execution_work()

