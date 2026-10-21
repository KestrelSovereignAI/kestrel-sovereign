"""Direct-node classification shares the existing explicit-graph control contract."""

import asyncio

import httpx
import pytest

from kestrel_sovereign.execution_custody import (
    AcknowledgedExecutionCommit,
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
    retain_acknowledged_execution_commit,
    retain_native_execution_stream_close,
)


def _error_case(kind):
    if kind == "none":
        return None, False, False
    ordinary = ValueError("original ordinary carrier")
    authority = ExecutionAuthorityError("original denied admission")
    if kind == "ordinary":
        return ordinary, False, False
    if kind == "cancellation":
        return asyncio.CancelledError("original withdrawal"), False, False
    if kind == "authority":
        return authority, True, True
    if kind in {"committed", "unknown"}:
        return ExecutionCommitOutcomeError(kind), True, True
    if kind == "explicit-cause":
        ordinary.__cause__ = authority
        return ordinary, True, False
    if kind == "ordinary-group":
        return ExceptionGroup("ordinary members", [ordinary]), False, False
    if kind == "control-group":
        return ExceptionGroup("original members", [ordinary, authority]), True, False
    if kind == "nested-group":
        return (
            ExceptionGroup("outer", [ordinary, ExceptionGroup("inner", [authority])]),
            True,
            False,
        )
    if kind == "group-cause":
        group = ExceptionGroup("ordinary members", [ordinary])
        group.__cause__ = authority
        return group, True, False
    if kind == "cancellation-group":
        return (
            BaseExceptionGroup(
                "original lifecycle", [asyncio.CancelledError(), authority]
            ),
            True,
            False,
        )
    if kind == "control-cycle":
        ordinary.__cause__ = authority
        authority.__cause__ = ordinary
        return ordinary, True, False
    if kind == "ordinary-cycle":
        cause = RuntimeError("second ordinary carrier")
        ordinary.__cause__ = cause
        cause.__cause__ = ordinary
        return ordinary, False, False
    if kind == "implicit-context":
        ordinary.__context__ = authority
        return ordinary, False, False
    if kind == "authority-with-cause":
        authority.__cause__ = ordinary
        return authority, True, True
    raise AssertionError(f"unknown test case: {kind}")


CASES = (
    "none",
    "ordinary",
    "cancellation",
    "authority",
    "committed",
    "unknown",
    "explicit-cause",
    "ordinary-group",
    "control-group",
    "nested-group",
    "group-cause",
    "cancellation-group",
    "control-cycle",
    "ordinary-cycle",
    "implicit-context",
    "authority-with-cause",
)


def _snapshot(error):
    return tuple(
        (
            node,
            node.__cause__,
            node.__context__,
            node.__traceback__,
            node.__suppress_context__,
            tuple(
                (key, tuple(value) if isinstance(value, list) else value)
                for key, value in vars(node).items()
            ),
        )
        for node in _explicit_control_chain(error)
    )


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize("mode", ["default", "recursive", "direct"])
def test_same_original_node_and_graph_classification(kind, mode):
    error, recursive_expected, direct_expected = _error_case(kind)
    before = _snapshot(error)
    for _ in range(2):
        if mode == "default":
            assert is_execution_control_error(error) is recursive_expected
        else:
            recursive = mode == "recursive"
            expected = recursive_expected if recursive else direct_expected
            assert is_execution_control_error(error, recursive=recursive) is expected
        assert _snapshot(error) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
async def test_direct_acknowledged_commit_retains_exact_original_receipt(group):
    # A canonical settled owner is sufficient for this classifier unit test;
    # it is not evidence that a database transaction occurred in this case.
    owner = asyncio.get_running_loop().create_future()
    owner.set_result(None)
    receipt = AcknowledgedExecutionCommit(owner)
    ordinary = ValueError("original native acknowledgement carrier")
    error = (
        ExceptionGroup("carrier owns its own receipt", [ordinary])
        if group
        else ordinary
    )
    assert retain_acknowledged_execution_commit(error, receipt) is error
    before = _snapshot(error)
    assert is_execution_control_error(error)
    assert is_execution_control_error(error, recursive=False)
    assert is_execution_control_error(error, recursive=True)
    assert _snapshot(error) == before
    assert receipt.commit_owner is owner and owner.done()
    assert receipt.retirement_owner is None


@pytest.mark.asyncio
async def test_direct_native_retirement_keeps_actual_response_and_failed_close_owner():
    original = OSError("original native HTTP body retirement failure")

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"owned, provider-free response"

        async def aclose(self):
            raise original

    response = httpx.Response(200, stream=Body())
    retain_native_execution_stream_close(response)
    with pytest.raises(BaseException) as caught:
        await response.aclose()
    error = caught.value
    debts = execution_cleanup_debts(error)
    assert len(debts) == 1
    resource, closer = debts[0]
    assert resource is response and closer.done()
    assert closer.exception() is original
    assert any(node is original for node in _explicit_control_chain(error))
    before = _snapshot(error)
    assert is_execution_control_error(error)
    assert is_execution_control_error(error, recursive=False)
    assert is_execution_control_error(error, recursive=True)
    assert execution_cleanup_debts(error) == debts
    assert _snapshot(error) == before


def test_recursive_option_remains_keyword_only():
    with pytest.raises(TypeError):
        is_execution_control_error(ExecutionAuthorityError("original"), False)
