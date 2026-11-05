"""Malformed cursor refusal cannot erase actual original native control."""

import base64
import binascii
import sys
from types import SimpleNamespace

import pytest
from kestrel_sdk.native_a2a.utils.errors import InvalidParamsError

from kestrel_sovereign.a2a import request_handler
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ProcessRuntimeExecutionFence, bind_execution_custody,
)


def test_recursive_push_cursor_is_invalid_parameters():
    depth = sys.getrecursionlimit() + 50
    token = base64.urlsafe_b64encode(b"[" * depth + b"]" * depth).decode()
    with pytest.raises(InvalidParamsError):
        request_handler._push_after(token, SimpleNamespace(recipient_id="r", tenant="t"), "p", "task")


@pytest.mark.parametrize("error_type", [ValueError, UnicodeError, binascii.Error, RecursionError])
@pytest.mark.parametrize("outcome", ["denied", "unknown", "group"])
def test_push_cursor_does_not_normalize_original_control(monkeypatch, error_type, outcome):
    # Actual deliberately denied runtime provenance, not positive tenant or
    # physical authority. A codec failure may carry this original unchanged.
    with bind_execution_custody(ProcessRuntimeExecutionFence("owned-cursor-proof")) as scope:
        if outcome == "unknown":
            scope.preserve_commit_uncertainty("unknown")
        else:
            scope.revoke("original invocation retired")
        with pytest.raises(ExecutionAuthorityError) as observed:
            scope.require_work()
    original = observed.value
    if outcome == "group":
        original = ExceptionGroup("original outcomes", [LookupError("parse"), original])
    failure = error_type("declared cursor decoder failure")
    failure.__cause__ = original

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(request_handler.json, "loads", fail)
    with pytest.raises(error_type) as result:
        request_handler._push_after("W10=", SimpleNamespace(recipient_id="r", tenant="t"), "p", "task")
    assert result.value is failure and result.value.__cause__ is original
