"""Regression coverage for the Windows private-file custody branch."""

import os
from pathlib import Path

from kestrel_sovereign import private_storage


class _WindowsOsProxy:
    """Exercise platform selection without changing process-global os.name."""

    name = "nt"

    def __init__(self) -> None:
        self.fchmod_calls = 0

    def __getattr__(self, attribute):
        return getattr(os, attribute)

    def fchmod(self, descriptor, mode):
        self.fchmod_calls += 1
        raise PermissionError("Windows rejects fchmod on read-only witnesses")


def test_read_only_witness_does_not_chmod_on_windows(monkeypatch, tmp_path: Path):
    witness = tmp_path / "hold-backend-binding"
    witness.write_bytes(b"sqlite\n")
    windows_os = _WindowsOsProxy()
    monkeypatch.setattr(private_storage, "os", windows_os)

    descriptor = private_storage.open_private_file(
        witness,
        os.O_RDONLY,
        label="Hold backend custody binding",
    )
    try:
        assert os.read(descriptor, 32) == b"sqlite\n"
    finally:
        os.close(descriptor)

    assert windows_os.fchmod_calls == 0
    assert witness.read_bytes() == b"sqlite\n"
