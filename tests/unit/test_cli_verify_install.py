"""
``kestrel verify-install`` CLI tests — sub-PR 2.2 of epic #1050
(bash-to-Python port of ``scripts/verify_clean_install.sh``).

Covers:
- argparse wires up under both the local subparser and the real
  ``kestrel`` parser
- ``cmd_verify_install`` exit codes:
  * 0 on all-pass
  * 1 on any-fail
  * 2 on bad selector / missing uv
- Test-selector narrowing: ``["1", "3", "5"]`` runs only those, no
  duplicates, preserving user order
- Bad selector ("nan", "9") errors clearly with exit code 2

Subprocess invocations are mocked via ``monkeypatch.setattr`` against
the module-private helpers so unit tests stay hermetic — the real
venv/pip/uvicorn matrix is exercised end-to-end via the integration
tier (and CI's real-pip sandbox).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign.cli_verify_install import (
    VerifyResult,
    _TEST_RUNNERS,
    _VALID_TEST_NUMBERS,
    add_verify_install_subcommand,
    cmd_verify_install,
)


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    add_verify_install_subcommand(sub)
    return p


def test_argparse_no_tests_is_empty_list():
    parser = _build_parser()
    args = parser.parse_args(["verify-install"])
    assert args.command == "verify-install"
    assert args.tests == []


def test_argparse_single_test():
    parser = _build_parser()
    args = parser.parse_args(["verify-install", "1"])
    assert args.tests == ["1"]


def test_argparse_multiple_tests_preserve_order():
    parser = _build_parser()
    args = parser.parse_args(["verify-install", "5", "1", "3"])
    assert args.tests == ["5", "1", "3"]


def test_kestrel_cli_registers_verify_install():
    """The full ``kestrel`` parser registers verify-install. Guards
    against a future cli.py refactor accidentally dropping the
    wiring."""
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["verify-install", "1", "3"])
    assert args.command == "verify-install"
    assert args.tests == ["1", "3"]


# ---------------------------------------------------------------------------
# cmd_verify_install — exit codes
# ---------------------------------------------------------------------------

class _Args:
    """Minimal argparse-style namespace for direct cmd_ calls."""

    def __init__(self, tests=None):
        self.tests = tests if tests is not None else []


def _patch_runners(passes: dict[int, bool] = None):
    """Build a context-manager that swaps every entry in
    ``_TEST_RUNNERS`` for a stub returning a single VerifyResult.

    ``passes`` maps test number → expected passed flag. Defaults to
    every test passing.
    """
    passes = passes if passes is not None else {n: True for n in _VALID_TEST_NUMBERS}
    called: list[int] = []

    def make_stub(n: int):
        def stub(work_dir: Path):
            called.append(n)
            return VerifyResult(f"Test {n}: stub", passes.get(n, True), "stubbed")
        return stub

    return called, {n: make_stub(n) for n in _VALID_TEST_NUMBERS}


def test_cmd_all_pass_exits_zero(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    called, stubs = _patch_runners()
    with patch.dict(_TEST_RUNNERS, stubs, clear=True):
        rc = cmd_verify_install(_Args())
    assert rc == 0
    # All 5 tests ran in canonical order
    assert called == [1, 2, 3, 4, 5]


def test_cmd_any_fail_exits_one(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    called, stubs = _patch_runners({1: True, 2: True, 3: False, 4: True, 5: True})
    with patch.dict(_TEST_RUNNERS, stubs, clear=True):
        rc = cmd_verify_install(_Args())
    assert rc == 1


def test_cmd_selection_narrows_runs(monkeypatch):
    """Only the requested tests should execute when the operator
    narrows with positional args."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    called, stubs = _patch_runners()
    with patch.dict(_TEST_RUNNERS, stubs, clear=True):
        rc = cmd_verify_install(_Args(tests=["1", "3", "5"]))
    assert rc == 0
    assert called == [1, 3, 5]


def test_cmd_selection_preserves_user_order(monkeypatch):
    """``kestrel verify-install 5 1 3`` runs in user order (5, 1, 3)
    — handy when an operator wants the slow test first to fail-fast."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    called, stubs = _patch_runners()
    with patch.dict(_TEST_RUNNERS, stubs, clear=True):
        rc = cmd_verify_install(_Args(tests=["5", "1", "3"]))
    assert rc == 0
    assert called == [5, 1, 3]


def test_cmd_selection_dedupes(monkeypatch):
    """Repeated test number runs once (wastes time otherwise)."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    called, stubs = _patch_runners()
    with patch.dict(_TEST_RUNNERS, stubs, clear=True):
        rc = cmd_verify_install(_Args(tests=["3", "3", "1"]))
    assert rc == 0
    assert called == [3, 1]


# ---------------------------------------------------------------------------
# Bad selectors / environment
# ---------------------------------------------------------------------------

def test_cmd_bad_selector_non_int(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    rc = cmd_verify_install(_Args(tests=["nan"]))
    assert rc == 2
    captured = capsys.readouterr()
    assert "not an integer" in captured.err


def test_cmd_bad_selector_out_of_range(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")
    rc = cmd_verify_install(_Args(tests=["9"]))
    assert rc == 2
    captured = capsys.readouterr()
    assert "out of range" in captured.err


def test_cmd_uv_missing_exits_two(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    rc = cmd_verify_install(_Args())
    assert rc == 2
    captured = capsys.readouterr()
    assert "uv" in captured.err and "not found" in captured.err


# ---------------------------------------------------------------------------
# Sub-test list-result handling
# ---------------------------------------------------------------------------

def test_cmd_handles_list_results(monkeypatch):
    """Tests 2 and 5 each return a *list* of VerifyResult (sub-results
    for the import + endpoint/entry-point checks). The dispatcher
    must flatten lists into the summary, and a single sub-failure
    must flip the overall exit code."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/uv")

    def list_stub_pass(_work_dir):
        return [
            VerifyResult("Test 2: stub (a)", True, "ok-a"),
            VerifyResult("Test 2: stub (b)", True, "ok-b"),
        ]

    def list_stub_partial(_work_dir):
        return [
            VerifyResult("Test 5: stub (a)", True, "ok-a"),
            VerifyResult("Test 5: stub (b)", False, "fail-b"),
        ]

    def single_stub_pass(_work_dir):
        return VerifyResult("stub", True, "ok")

    stubs = {
        1: single_stub_pass,
        2: list_stub_pass,
        3: single_stub_pass,
        4: single_stub_pass,
        5: list_stub_partial,
    }
    with patch.dict(_TEST_RUNNERS, stubs, clear=True):
        rc = cmd_verify_install(_Args())
    # one sub-result failed → exit 1
    assert rc == 1


# ---------------------------------------------------------------------------
# Cross-platform venv-path helper
# ---------------------------------------------------------------------------

def test_venv_exec_picks_per_platform(monkeypatch, tmp_path):
    """The venv resolver must pick ``Scripts\\python.exe`` on Windows
    and ``bin/python`` everywhere else."""
    from kestrel_sovereign import cli_verify_install as mod

    venv = tmp_path / "v"
    venv.mkdir()

    monkeypatch.setattr(mod.sys, "platform", "linux")
    assert mod._venv_exec(venv, "python") == venv / "bin" / "python"

    monkeypatch.setattr(mod.sys, "platform", "win32")
    assert mod._venv_exec(venv, "python") == venv / "Scripts" / "python.exe"


# ---------------------------------------------------------------------------
# Streaming subprocess — no capture_output
# ---------------------------------------------------------------------------

def test_install_targets_use_pypi_packages_not_dead_local_paths(monkeypatch):
    """Codex review on PR #1067: post-OSS-split (epic #462) the
    ``sdk/`` and feature directories no longer exist locally — the SDK
    and feature packages live on PyPI. The verifier must install from
    PyPI, not point at dead repo subpaths.

    Captures every ``_pip_install`` call argv across all 5 tests with
    a mocked subprocess and asserts none of the dead local paths
    appear.
    """
    from kestrel_sovereign import cli_verify_install as mod

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    # Skip uvicorn / urllib so test 2's /health probe doesn't try to
    # exercise a real server in this dry-run.
    monkeypatch.setattr(mod, "_start_uvicorn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "_stop_process", lambda proc: None)
    monkeypatch.setattr(mod, "_wait_for_health", lambda port, timeout=8.0: True)
    # Pretend every venv-exec resolves so _python_check / _pip_install
    # fall through to subprocess.run. Path.exists is class-wide so we
    # patch the cli_verify_install Path attribute.
    monkeypatch.setattr(
        mod.Path, "exists", lambda self: True, raising=False
    )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (
            mod._test_1_sdk_only,
            mod._test_3_feature_package,
            mod._test_4_sdk_feature_dev,
            mod._test_5_full_stack,
        ):
            fn(mod.Path(tmp))

    # Walk every ``pip install`` argv and confirm the install target is
    # a PyPI package name, not a local repo subpath. We can't just grep
    # the flattened command line — the Python ``import`` statements
    # legitimately contain ``kestrel_feature_wallet`` (the module name),
    # which collides with the dead-path string. So look at each pip-install
    # invocation specifically and check the LAST positional (the target).
    pip_install_targets: list[str] = []
    for argv in captured:
        # First arg is the pip exec path (ends with ``/pip`` or
        # ``\\pip.exe``); next must be ``install``; rest are flags +
        # the target.
        if not argv:
            continue
        first = argv[0]
        if not (first.endswith("/pip") or first.endswith("\\pip.exe")):
            continue
        if "install" not in argv:
            continue
        positionals = [
            a for a in argv[1:]
            if not a.startswith("-") and a != "install"
        ]
        if positionals:
            pip_install_targets.append(positionals[-1])

    for target in pip_install_targets:
        assert not target.endswith("/sdk"), (
            f"Test 1/4 must install kestrel-sovereign-sdk from PyPI, not "
            f"the dead $REPO/sdk path. pip install target: {target!r}"
        )
        assert not target.endswith("/kestrel_feature_wallet"), (
            f"Tests 3/4/5 must install kestrel-feature-wallet from PyPI, "
            f"not the dead $REPO/kestrel_feature_wallet path. "
            f"pip install target: {target!r}"
        )
        assert not target.endswith("/kestrel-feature-intelligence"), (
            f"Test 5 must install kestrel-feature-intelligence from PyPI, "
            f"not the dead $REPO/... path. pip install target: {target!r}"
        )

    # And the PyPI names must each appear at least once.
    assert "kestrel-sovereign-sdk" in pip_install_targets
    assert "kestrel-feature-wallet" in pip_install_targets
    assert "kestrel-feature-intelligence" in pip_install_targets


def test_run_streaming_does_not_capture(monkeypatch):
    """Codex's Tier 1.3 lesson: subprocess output must stream live,
    not be buffered with ``capture_output=True``. Asserts the helper
    delegates to ``subprocess.run`` without that flag."""
    from kestrel_sovereign import cli_verify_install as mod

    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rc = mod._run_streaming(["echo", "hi"])
    assert rc == 0
    assert "capture_output" not in captured_kwargs
    assert captured_kwargs.get("check") is False
