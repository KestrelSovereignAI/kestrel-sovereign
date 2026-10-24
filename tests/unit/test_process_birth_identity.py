"""Kernel process lifetime identity is independent of epoch creation time."""

import errno
import os
import sys

import pytest

from kestrel_sovereign.multi_agent import liveness
from kestrel_sovereign.multi_agent.process_manager import (
    PidStatus,
    ProcessManager,
    recorded_birth_identity,
)


@pytest.mark.parametrize("value", [None, 0, True, "", "unknown:1", "linux:-1", "darwin:+1", "linux:01", "darwin:²", "linux:18446744073709551616"])
def test_invalid_birth_identity_never_becomes_a_lifetime_witness(value):
    assert recorded_birth_identity(value) is None


@pytest.mark.parametrize("value", ["linux:0", "linux:18446744073709551615", "darwin:1"])
def test_canonical_kernel_birth_values_round_trip(value):
    assert recorded_birth_identity(value) == value


@pytest.mark.parametrize("pid", [None, True, "1", 0, -1])
def test_invalid_pid_cannot_reach_kernel_birth_observation(pid):
    with pytest.raises(ValueError, match="positive PID"):
        ProcessManager.process_birth_identity(pid)


def test_actual_process_birth_remains_live_without_an_epoch_match():
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        pytest.skip("Native kernel process lifetime identity")
    birth = ProcessManager.process_birth_identity(os.getpid())
    assert recorded_birth_identity(birth) == birth
    status, _ = ProcessManager.identify_recorded_process(
        os.getpid(), 1.0, "original-owner", recorded_birth=birth, require_birth=True
    )
    assert status is PidStatus.LIVE


def test_opaque_birth_identity_cannot_use_an_epoch_or_hidden_pid_as_retirement(monkeypatch, tmp_path):
    identity = liveness.process_identity(require_kernel_session=True)
    original = PermissionError("original kernel birth observer refused access")

    def opaque(pid):
        raise original

    original_probe = ProcessManager._probe_process
    monkeypatch.setattr(ProcessManager, "process_birth_identity", staticmethod(opaque))
    monkeypatch.setattr(ProcessManager, "_probe_process", staticmethod(lambda pid: (False, None)))
    assert not liveness.recorded_process_exited(tmp_path / "live-owner", identity, require_kernel_session=True)
    monkeypatch.setattr(ProcessManager, "_probe_process", staticmethod(original_probe))
    with pytest.raises(liveness.ProcessIdentityUnavailable) as caught:
        liveness.process_identity(require_kernel_session=True)
    assert caught.value.__cause__ is original


def test_unknown_pid_presence_does_not_convert_opaque_birth_into_absence(monkeypatch):
    birth = ProcessManager.process_birth_identity(os.getpid())

    def opaque(pid):
        raise PermissionError("birth identity opaque")

    def unknown(pid, signal):
        raise PermissionError("kernel process presence opaque")

    monkeypatch.setattr(ProcessManager, "process_birth_identity", staticmethod(opaque))
    monkeypatch.setattr(os, "kill", unknown)
    status, _ = ProcessManager.identify_recorded_process(
        os.getpid(), 1.0, "original-owner", recorded_birth=birth, require_birth=True
    )
    assert status is PidStatus.UNDECIDABLE


def test_missing_birth_never_uses_an_older_epoch_record_as_strict_proof(tmp_path):
    identity = liveness.process_identity(require_kernel_session=True)
    identity.pop("birth_id")
    assert not liveness.recorded_process_exited(tmp_path / "old-owner", identity, require_kernel_session=True)


def test_native_kernel_capability_refuses_unsupported_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "unsupported-test-platform")
    with pytest.raises(OSError) as caught:
        liveness._kernel_session_id()
    assert caught.value.errno == errno.ENOTSUP
    with pytest.raises(OSError) as caught:
        ProcessManager.process_birth_identity(os.getpid())
    assert caught.value.errno == errno.ENOTSUP
