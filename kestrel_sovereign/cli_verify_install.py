"""
``kestrel verify-install`` CLI command — sub-PR 2.2 of epic #1050
(bash-to-Python port).

Direct port of ``scripts/verify_clean_install.sh``. Runs a 5-test
clean-install matrix, each in its own throwaway ``uv venv``, against
the local repository checkout:

    Test 1  SDK only                 — kestrel_sdk.features.base.Feature import
    Test 2  Core sovereign           — kestrel_sovereign.features.base.Feature
                                       import + uvicorn /health probe on
                                       127.0.0.1:18548
    Test 3  Feature package          — kestrel_feature_wallet import
    Test 4  SDK + feature dev mode   — wallet --no-deps -e + Feature
                                       subclass assertion
    Test 5  Full stack               — sovereign + wallet + intelligence
                                       + entry_point discovery

Usage::

    kestrel verify-install               # All 5 tests
    kestrel verify-install 1             # Only test 1
    kestrel verify-install 1 3 5         # Selected

Exit codes mirror the bash original: 0 on all-pass, 1 on any-fail.

Cross-platform: the venv ``python`` / ``pip`` / ``uvicorn``
executables live under ``Scripts\\`` on Windows and ``bin/`` elsewhere;
:func:`_venv_exec` picks the right one. Subprocess output is **streamed**
to stdout/stderr (not captured) — matches the Tier 1.3 lesson that
``capture_output=True`` makes long-running installs look hung.

The CI workflow ``.github/workflows/clean-install.yml`` uses a
SEPARATE, narrower verifier in ``scripts/ci/clean_install_verify.py``
(post-wizard health checks). The two scripts cover different things
and live independently — do not consolidate.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    """Outcome of a single test slot.

    ``passed`` is False when any assertion the test makes fails.
    ``message`` is one human-readable line; tests can append several
    sub-results by emitting multiple :class:`VerifyResult` objects.
    """

    name: str
    passed: bool
    message: str


# ---------------------------------------------------------------------------
# Cross-platform venv helpers
# ---------------------------------------------------------------------------

def _is_windows() -> bool:
    return sys.platform == "win32"


def _venv_exec(venv_dir: Path, name: str) -> Path:
    """Resolve a binary inside a venv, picking the right subdir per
    platform. ``name`` is the unsuffixed exe name (``python``, ``pip``,
    ``uvicorn``); on Windows we append ``.exe``.

    The rest of the repo follows this same idiom — see
    :mod:`kestrel_sovereign.multi_agent.process_manager`.
    """
    if _is_windows():
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _repo_root() -> Path:
    """Path to the repository root (one level up from this module's
    package). Resolves to the same location ``$REPO_ROOT`` did in the
    bash script."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Subprocess helper — streaming, no capture
# ---------------------------------------------------------------------------

def _run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> int:
    """Run ``cmd`` and stream its stdout/stderr live to the parent
    console. Returns the exit code. Codex's Tier 1.3 review caught
    that ``capture_output=True`` buffers everything until exit, which
    makes long-running installs/builds look hung — same lesson here:
    ``uv venv`` + ``pip install`` of the full sovereign tree can
    take 30-60s and operators want to see progress.
    """
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
    )
    return completed.returncode


# ---------------------------------------------------------------------------
# Shared per-test setup
# ---------------------------------------------------------------------------

def _make_venv(venv_dir: Path) -> bool:
    """Create a uv venv at ``venv_dir``. Returns True on success."""
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    rc = _run_streaming(["uv", "venv", str(venv_dir)])
    return rc == 0


def _pip_install(
    venv_dir: Path,
    *args: str,
) -> bool:
    """Run ``<venv>/bin/pip install <args>`` with VIRTUAL_ENV set so
    pip resolves the right interpreter on platforms (Linux) where it
    matters. Returns True on success.
    """
    pip = _venv_exec(venv_dir, "pip")
    if not pip.exists():
        return False
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    rc = _run_streaming([str(pip), "install", *args], env=env)
    return rc == 0


def _python_check(venv_dir: Path, snippet: str) -> bool:
    """Run a Python ``-c`` snippet in the venv. Returns True on
    exit code 0. The snippet is executed verbatim — assertions inside
    raise + propagate non-zero on failure.
    """
    py = _venv_exec(venv_dir, "python")
    if not py.exists():
        return False
    rc = _run_streaming([str(py), "-c", snippet])
    return rc == 0


# ---------------------------------------------------------------------------
# Test 1 — SDK only
# ---------------------------------------------------------------------------

def _test_1_sdk_only(work_dir: Path) -> VerifyResult:
    """``pip install $REPO/sdk`` then assert
    ``from kestrel_sdk.features.base import Feature`` works.
    """
    name = "Test 1: SDK only"
    repo = _repo_root()
    venv_dir = work_dir / "test1" / ".venv"
    if not _make_venv(venv_dir):
        return VerifyResult(name, False, "uv venv creation failed")
    if not _pip_install(venv_dir, str(repo / "sdk")):
        return VerifyResult(name, False, "pip install of sdk failed")
    ok = _python_check(
        venv_dir,
        "from kestrel_sdk.features.base import Feature; print('SDK OK')",
    )
    if not ok:
        return VerifyResult(
            name, False,
            "import kestrel_sdk.features.base.Feature failed",
        )
    return VerifyResult(
        name, True,
        "import kestrel_sdk.features.base.Feature",
    )


# ---------------------------------------------------------------------------
# Test 2 — Core sovereign + /health probe
# ---------------------------------------------------------------------------

def _start_uvicorn(
    venv_dir: Path,
    repo: Path,
    port: int,
    agent_dir: Path,
) -> "subprocess.Popen[bytes]":
    """Spawn ``<venv>/bin/uvicorn server:app`` as a background process.

    On POSIX we ``start_new_session=True`` so SIGTERM hits the whole
    group (uvicorn's worker children); on Windows we use
    ``CREATE_NEW_PROCESS_GROUP`` so we can later send
    ``CTRL_BREAK_EVENT``. Same idiom the multi_agent ProcessManager
    uses (kestrel_sovereign/multi_agent/process_manager.py).
    """
    uvicorn = _venv_exec(venv_dir, "uvicorn")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["KESTREL_DB_PATH"] = str(agent_dir)
    cmd = [
        str(uvicorn), "server:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--app-dir", str(repo),
    ]
    kwargs: dict = {"cwd": str(repo), "env": env}
    if _is_windows():
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]


def _stop_process(proc: "subprocess.Popen[bytes]") -> None:
    """Terminate a uvicorn background process best-effort.

    On POSIX we ``os.killpg(SIGTERM)`` so workers don't outlive the
    parent. On Windows we use ``taskkill /F /T`` to walk the process
    tree. Failures are swallowed — this is cleanup, not assertion.
    """
    if proc.poll() is not None:
        return
    try:
        if _is_windows():
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass


def _wait_for_health(port: int, timeout: float = 8.0) -> bool:
    """Poll ``http://127.0.0.1:<port>/health`` until it returns 200 or
    we time out. The bash original ``sleep 3``ed unconditionally; we
    poll because cold uvicorn boot can be slower under heavy load and
    a fixed sleep is racy."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_err = repr(e)
        time.sleep(0.5)
    if last_err:
        # Surface the last failure so debugging a CI flake doesn't
        # require re-running locally.
        print(f"  /health probe failed: {last_err}", file=sys.stderr)
    return False


def _test_2_core_sovereign(work_dir: Path) -> List[VerifyResult]:
    """``pip install $REPO`` then check the import AND boot uvicorn
    long enough to verify ``/health`` answers 200."""
    name = "Test 2: Core sovereign"
    repo = _repo_root()
    venv_dir = work_dir / "test2" / ".venv"
    results: List[VerifyResult] = []

    if not _make_venv(venv_dir):
        return [VerifyResult(name, False, "uv venv creation failed")]
    if not _pip_install(venv_dir, str(repo)):
        return [VerifyResult(name, False, "pip install of sovereign failed")]

    # Sub-result A: import smoke
    ok = _python_check(
        venv_dir,
        "from kestrel_sovereign.features.base import Feature; print('Sovereign OK')",
    )
    results.append(VerifyResult(
        f"{name} (import)",
        ok,
        "import kestrel_sovereign.features.base.Feature"
        if ok else
        "import kestrel_sovereign.features.base.Feature failed",
    ))

    # Sub-result B: /health probe
    agent_dir = work_dir / "test2" / "agent_data"
    agent_dir.mkdir(parents=True, exist_ok=True)
    port = 18548
    proc = _start_uvicorn(venv_dir, repo, port, agent_dir)
    try:
        ok = _wait_for_health(port, timeout=8.0)
        results.append(VerifyResult(
            f"{name} (/health)",
            ok,
            "/health endpoint responds 200" if ok
            else "/health endpoint did not respond within 8s",
        ))
    finally:
        _stop_process(proc)
    return results


# ---------------------------------------------------------------------------
# Test 3 — Feature package install (wallet)
# ---------------------------------------------------------------------------

def _test_3_feature_package(work_dir: Path) -> VerifyResult:
    """``pip install $REPO`` + ``pip install $REPO/kestrel_feature_wallet``
    + ``import WalletFeature``."""
    name = "Test 3: Feature package"
    repo = _repo_root()
    venv_dir = work_dir / "test3" / ".venv"
    if not _make_venv(venv_dir):
        return VerifyResult(name, False, "uv venv creation failed")
    if not _pip_install(venv_dir, str(repo)):
        return VerifyResult(name, False, "pip install of sovereign failed")
    if not _pip_install(venv_dir, str(repo / "kestrel_feature_wallet")):
        return VerifyResult(name, False, "pip install of wallet feature failed")
    ok = _python_check(
        venv_dir,
        "from kestrel_feature_wallet import WalletFeature; print('Wallet OK')",
    )
    if not ok:
        return VerifyResult(name, False, "import WalletFeature failed")
    return VerifyResult(name, True, "import WalletFeature")


# ---------------------------------------------------------------------------
# Test 4 — SDK + feature dev mode (--no-deps -e wallet)
# ---------------------------------------------------------------------------

def _test_4_sdk_feature_dev(work_dir: Path) -> VerifyResult:
    """``pip install $REPO/sdk`` + ``pip install --no-deps -e
    $REPO/kestrel_feature_wallet`` + assert ``WalletFeature`` is a
    ``kestrel_sdk.features.base.Feature`` subclass.

    Proves that a feature package author can develop against the SDK
    alone, without pulling in the full sovereign tree.
    """
    name = "Test 4: SDK + feature dev mode"
    repo = _repo_root()
    venv_dir = work_dir / "test4" / ".venv"
    if not _make_venv(venv_dir):
        return VerifyResult(name, False, "uv venv creation failed")
    if not _pip_install(venv_dir, str(repo / "sdk")):
        return VerifyResult(name, False, "pip install of sdk failed")
    if not _pip_install(
        venv_dir,
        "--no-deps",
        "-e",
        str(repo / "kestrel_feature_wallet"),
    ):
        return VerifyResult(name, False, "editable wallet install failed")
    ok = _python_check(
        venv_dir,
        (
            "from kestrel_sdk.features.base import Feature\n"
            "from kestrel_feature_wallet.wallet_feature import WalletFeature\n"
            "assert issubclass(WalletFeature, Feature), "
            "'WalletFeature must be a Feature subclass'\n"
            "print('Dev mode OK')\n"
        ),
    )
    if not ok:
        return VerifyResult(
            name, False,
            "WalletFeature is not a kestrel_sdk Feature subclass",
        )
    return VerifyResult(
        name, True,
        "WalletFeature is a kestrel_sdk.features.base.Feature subclass",
    )


# ---------------------------------------------------------------------------
# Test 5 — Full stack
# ---------------------------------------------------------------------------

def _test_5_full_stack(work_dir: Path) -> List[VerifyResult]:
    """``pip install $REPO + wallet + intelligence`` then assert all
    imports work AND that entry-point discovery finds every feature.
    """
    name = "Test 5: Full stack"
    repo = _repo_root()
    venv_dir = work_dir / "test5" / ".venv"
    results: List[VerifyResult] = []
    if not _make_venv(venv_dir):
        return [VerifyResult(name, False, "uv venv creation failed")]
    if not _pip_install(venv_dir, str(repo)):
        return [VerifyResult(name, False, "pip install of sovereign failed")]
    if not _pip_install(venv_dir, str(repo / "kestrel_feature_wallet")):
        return [VerifyResult(name, False, "pip install of wallet failed")]
    if not _pip_install(venv_dir, str(repo / "kestrel-feature-intelligence")):
        return [VerifyResult(name, False, "pip install of intelligence failed")]

    ok_imports = _python_check(
        venv_dir,
        (
            "from kestrel_sovereign.features.base import Feature\n"
            "from kestrel_feature_wallet import WalletFeature\n"
            "from kestrel_feature_intelligence import "
            "ReflectionFeature, CouncilFeature\n"
            "print('Full stack OK')\n"
        ),
    )
    results.append(VerifyResult(
        f"{name} (imports)",
        ok_imports,
        "all packages importable" if ok_imports
        else "one or more imports failed",
    ))

    ok_eps = _python_check(
        venv_dir,
        (
            "import importlib.metadata\n"
            "eps = importlib.metadata.entry_points()\n"
            "group = eps.select(group='kestrel_sovereign.features')\n"
            "names = {ep.name for ep in group}\n"
            "for needed in ('WalletFeature', 'ReflectionFeature', 'CouncilFeature'):\n"
            "    assert needed in names, f'{needed} not in {names}'\n"
            "print(f'Entry points discovered: {sorted(names)}')\n"
        ),
    )
    results.append(VerifyResult(
        f"{name} (entry_points)",
        ok_eps,
        "entry_point discovery finds all features" if ok_eps
        else "entry_point discovery missing one or more features",
    ))
    return results


# ---------------------------------------------------------------------------
# Test runner registry
# ---------------------------------------------------------------------------

# Each entry returns either a single VerifyResult or a list of them.
# Module-private so callers + tests share the canonical mapping.
_TEST_RUNNERS: dict[int, Callable[[Path], object]] = {
    1: _test_1_sdk_only,
    2: _test_2_core_sovereign,
    3: _test_3_feature_package,
    4: _test_4_sdk_feature_dev,
    5: _test_5_full_stack,
}

_VALID_TEST_NUMBERS = frozenset(_TEST_RUNNERS.keys())


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_verify_install_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel verify-install [TESTS...]`` under the parent
    subparsers. Called from :func:`kestrel_sovereign.cli.build_parser`.
    Mirrors the cli_release / cli_deploy locality pattern: keeps the
    venv/uvicorn machinery out of the hot path for operators who never
    run install verification.
    """
    p = subparsers.add_parser(
        "verify-install",
        help="Verify clean-install matrix (5 isolated venvs) — port of "
             "scripts/verify_clean_install.sh (epic #1050 tier 2)",
    )
    p.add_argument(
        "tests",
        nargs="*",
        metavar="TEST",
        help="Test numbers to run (1-5). Omit to run all.",
    )


# ---------------------------------------------------------------------------
# Main command handler
# ---------------------------------------------------------------------------

def cmd_verify_install(args) -> int:
    """Dispatch ``kestrel verify-install [TESTS...]``.

    Exit codes:
        0 — every requested test passed
        1 — one or more tests failed
        2 — bad test selector (non-int or out of range)

    The bash original returned 0/1 only; we reserve 2 for argument
    errors so CI can distinguish "verifier had a bug" from "install
    is broken".
    """
    raw_selectors: List[str] = list(getattr(args, "tests", []) or [])
    if not raw_selectors:
        selected = sorted(_VALID_TEST_NUMBERS)
    else:
        selected = []
        for s in raw_selectors:
            try:
                n = int(s)
            except ValueError:
                print(
                    f"error: test selector {s!r} is not an integer",
                    file=sys.stderr,
                )
                return 2
            if n not in _VALID_TEST_NUMBERS:
                print(
                    f"error: test {n} is out of range; valid: "
                    f"{sorted(_VALID_TEST_NUMBERS)}",
                    file=sys.stderr,
                )
                return 2
            selected.append(n)
        # Preserve user order but de-dupe (running test 3 twice wastes time).
        seen = set()
        deduped = []
        for n in selected:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        selected = deduped

    # ``uv`` is required to create venvs. Fail loud if it's missing,
    # matching the bash ``set -euo pipefail``-style behaviour.
    if shutil.which("uv") is None:
        print(
            "error: `uv` not found on PATH; install from "
            "https://docs.astral.sh/uv/",
            file=sys.stderr,
        )
        return 2

    # Use a tempdir for venvs. Cleanup on exit (success or fail) — same
    # as the bash trap. Each test gets its own subdir so a failed test
    # leaves the others alone.
    with tempfile.TemporaryDirectory(prefix="kestrel-clean-install-") as td:
        work_dir = Path(td)
        all_results: List[VerifyResult] = []
        for n in selected:
            print(f"[verify] running test {n}", file=sys.stderr)
            runner = _TEST_RUNNERS[n]
            out = runner(work_dir)
            if isinstance(out, list):
                all_results.extend(out)
            else:
                all_results.append(out)

    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)

    print("")
    print("=" * 67)
    print(" Clean Install Verification Summary")
    print("=" * 67)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}")
    print("-" * 67)
    print(f"  Passed: {passed}  |  Failed: {failed}")
    print("=" * 67)

    return 0 if failed == 0 else 1


__all__ = [
    "VerifyResult",
    "add_verify_install_subcommand",
    "cmd_verify_install",
]
