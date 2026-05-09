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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


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


def _python_check(
    venv_dir: Path,
    snippet: str,
    *,
    env_extra: Optional[dict] = None,
) -> bool:
    """Run a Python ``-c`` snippet in the venv. Returns True on
    exit code 0. The snippet is executed verbatim — assertions inside
    raise + propagate non-zero on failure.

    ``env_extra`` overlays variables on top of the parent env (used by
    test 2's identity bootstrap to set ``KESTREL_DB_PATH`` so the
    Sovereign DB initializes in the test agent dir, not the operator's
    real one).
    """
    py = _venv_exec(venv_dir, "python")
    if not py.exists():
        return False
    env = None
    if env_extra:
        env = os.environ.copy()
        env.update(env_extra)
    rc = _run_streaming([str(py), "-c", snippet], env=env)
    return rc == 0


# ---------------------------------------------------------------------------
# Test 1 — SDK only
# ---------------------------------------------------------------------------

def _test_1_sdk_only(work_dir: Path) -> VerifyResult:
    """``pip install kestrel-sovereign-sdk`` (from PyPI) then assert
    ``from kestrel_sdk.features.base import Feature`` works.

    The SDK lives at https://github.com/KestrelSovereignAI/kestrel-sovereign-sdk
    and is published as ``kestrel-sovereign-sdk`` on PyPI (see
    pyproject.toml ``dependencies``). The bash predecessor pointed at a
    local ``$REPO/sdk`` path that no longer exists post-OSS-split;
    codex review on PR #1067 caught the dead path.
    """
    name = "Test 1: SDK only"
    venv_dir = work_dir / "test1" / ".venv"
    if not _make_venv(venv_dir):
        return VerifyResult(name, False, "uv venv creation failed")
    if not _pip_install(venv_dir, "kestrel-sovereign-sdk"):
        return VerifyResult(name, False, "pip install of kestrel-sovereign-sdk failed")
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
    """Spawn ``<venv>/bin/uvicorn kestrel_sovereign.server:app`` as a background process.

    On POSIX we ``start_new_session=True`` so SIGTERM hits the whole
    group (uvicorn's worker children); on Windows we use
    ``CREATE_NEW_PROCESS_GROUP`` so we can later send
    ``CTRL_BREAK_EVENT``. Same idiom the multi_agent ProcessManager
    uses (kestrel_sovereign/multi_agent/process_manager.py).

    The module reference is the in-package path because verify-install
    is exercising a pip-installed wheel; the ``--app-dir`` shim that
    used to make the root-level ``server:app`` resolvable is no longer
    needed.
    """
    uvicorn = _venv_exec(venv_dir, "uvicorn")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["KESTREL_DB_PATH"] = str(agent_dir)
    cmd = [
        str(uvicorn), "kestrel_sovereign.server:app",
        "--host", "127.0.0.1",
        "--port", str(port),
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

    # Sub-result B: /health probe.
    #
    # Codex review on PR #1067 caught that the bash predecessor seeded an
    # identity via ``create_kestrel_identity(...)`` before launching
    # uvicorn — without that the /health route's
    # ``get_agent_did_async()`` raises and the endpoint returns 503.
    # Bootstrap the agent in-venv so the test reflects a real fresh
    # install + first run, not a half-configured server.
    agent_dir = work_dir / "test2" / "agent_data"
    agent_dir.mkdir(parents=True, exist_ok=True)
    constitution_path = repo / "docs" / "principles" / "KESTREL_CONSTITUTION.md"
    if not _python_check(
        venv_dir,
        (
            "from kestrel_sovereign.inception_service "
            "import create_kestrel_identity\n"
            f"create_kestrel_identity({str(agent_dir)!r}, "
            f"{str(constitution_path)!r})\n"
            "print('identity bootstrapped')\n"
        ),
        env_extra={"KESTREL_DB_PATH": str(agent_dir)},
    ):
        results.append(VerifyResult(
            f"{name} (/health)",
            False,
            "agent identity bootstrap (create_kestrel_identity) failed",
        ))
        return results

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
# Extracted-feature discovery (used by Tests 3/4/5)
# ---------------------------------------------------------------------------

def _extracted_feature_packages() -> List[Tuple[str, List[str]]]:
    """Return (package_name, [Feature class names]) for every Feature
    package registered as truly extracted (``core = false``) in the
    feature registry.

    Tests 3/4/5 must not hardcode specific feature package names —
    those would couple sovereign to specific extensions, which the
    open-core split (epic #462) explicitly forbids. Driving off the
    registry means new extracted features are auto-verified the moment
    they're registered with ``core = false``, and a feature that's
    claimed extracted but can't be ``pip install``ed surfaces as a
    real verification failure.

    Multiple registry entries can point at the same package (e.g.
    ``[reflection]`` and ``[council]`` both ship inside
    ``kestrel-feature-intelligence``). Dedupe by package name and
    union the Feature class lists.

    **Filtered to ``kestrel-feature-*`` packages only.** The registry
    also lists provider plugins (``kestrel-voice-elevenlabs``,
    ``kestrel-voice-deepgram``, ``kestrel-voice-openai``) under
    ``core = false``, but those register under a different entry-point
    group (``kestrel_sovereign.voice_providers``) and implement
    provider interfaces, not ``kestrel_sdk.features.base.Feature``.
    Running Tests 4 and 5 against them would fail spuriously
    (Feature-subclass and ``kestrel_sovereign.features`` entry-point
    assertions both miss). The naming convention is the discriminator
    here; if a future Feature package adopts a non-default name it
    must register under the conventional prefix to be picked up.
    """
    from kestrel_sovereign.feature_registry import get_registry

    by_package: Dict[str, Set[str]] = {}
    for info in get_registry().values():
        if info.core:
            continue
        if not info.package or info.package == "kestrel-sovereign":
            continue
        if not info.package.startswith("kestrel-feature-"):
            continue
        by_package.setdefault(info.package, set()).update(info.features)
    return [(pkg, sorted(classes)) for pkg, classes in sorted(by_package.items())]


# ---------------------------------------------------------------------------
# Test 3 — Feature package install (each `core = false` package)
# ---------------------------------------------------------------------------

def _test_3_feature_package(work_dir: Path) -> List[VerifyResult]:
    """For every package registered as ``core = false`` in
    ``feature_registry.toml``: ``pip install $REPO`` +
    ``pip install <package>`` (from PyPI) + verify each declared
    Feature class imports.

    Post-OSS-split (epic #462), feature packages publish independently.
    Sovereign no longer ships any one of them inline, so the verifier
    can't single out a particular one — it iterates whatever the
    registry currently claims is extracted.
    """
    base_name = "Test 3: Feature package"
    extracted = _extracted_feature_packages()
    if not extracted:
        return [VerifyResult(
            base_name, True,
            "no extracted feature packages registered (core = false)",
        )]

    repo = _repo_root()
    results: List[VerifyResult] = []
    for idx, (package, feature_classes) in enumerate(extracted):
        sub_name = f"{base_name} ({package})"
        venv_dir = work_dir / f"test3-{idx}" / ".venv"
        if not _make_venv(venv_dir):
            results.append(VerifyResult(sub_name, False, "uv venv creation failed"))
            continue
        if not _pip_install(venv_dir, str(repo)):
            results.append(VerifyResult(sub_name, False, "pip install of sovereign failed"))
            continue
        if not _pip_install(venv_dir, package):
            results.append(VerifyResult(
                sub_name, False, f"pip install of {package} failed",
            ))
            continue
        # Resolve via the entry-point group — package authors are only
        # required to declare the class under
        # ``kestrel_sovereign.features``, not to re-export it from
        # ``__init__.py``. Hitting ``ep.load()`` exercises the
        # declared `module.path:ClassName` and proves the class is
        # actually importable along the documented contract.
        ok = _python_check(
            venv_dir,
            (
                "import importlib.metadata as md\n"
                "group = md.entry_points(group='kestrel_sovereign.features')\n"
                "by_name = {ep.name: ep for ep in group}\n"
                f"for cls_name in {sorted(feature_classes)!r}:\n"
                "    assert cls_name in by_name, "
                "f'entry point {cls_name!r} not registered'\n"
                "    obj = by_name[cls_name].load()\n"
                "    assert obj.__name__ == cls_name, "
                "f'entry point loaded {obj.__name__!r}, expected {cls_name!r}'\n"
                f"print('{package} OK')\n"
            ),
        )
        if not ok:
            results.append(VerifyResult(
                sub_name, False,
                f"entry-point load of {feature_classes} failed",
            ))
            continue
        results.append(VerifyResult(
            sub_name, True,
            f"entry-points loaded: {', '.join(feature_classes)}",
        ))
    return results


# ---------------------------------------------------------------------------
# Test 4 — SDK + feature dev mode (--no-deps per extracted package)
# ---------------------------------------------------------------------------

def _test_4_sdk_feature_dev(work_dir: Path) -> List[VerifyResult]:
    """For every ``core = false`` package: ``pip install
    kestrel-sovereign-sdk`` + ``pip install --no-deps <package>`` +
    assert each declared Feature class is a
    ``kestrel_sdk.features.base.Feature`` subclass.

    Proves that a feature package author can develop against the SDK
    alone, without pulling in the full sovereign tree. ``--no-deps``
    keeps the install minimal — if the SDK interface is sufficient for
    the package to import, the dev-mode contract holds.
    """
    base_name = "Test 4: SDK + feature dev mode"
    extracted = _extracted_feature_packages()
    if not extracted:
        return [VerifyResult(
            base_name, True,
            "no extracted feature packages registered (core = false)",
        )]

    results: List[VerifyResult] = []
    for idx, (package, feature_classes) in enumerate(extracted):
        sub_name = f"{base_name} ({package})"
        venv_dir = work_dir / f"test4-{idx}" / ".venv"
        if not _make_venv(venv_dir):
            results.append(VerifyResult(sub_name, False, "uv venv creation failed"))
            continue
        if not _pip_install(venv_dir, "kestrel-sovereign-sdk"):
            results.append(VerifyResult(
                sub_name, False, "pip install of kestrel-sovereign-sdk failed",
            ))
            continue
        if not _pip_install(venv_dir, "--no-deps", package):
            results.append(VerifyResult(
                sub_name, False, f"--no-deps install of {package} failed",
            ))
            continue
        # Resolve through the entry-point group: load each declared
        # Feature class via the registered `module.path:ClassName` and
        # assert it subclasses the SDK's Feature base. Avoids the
        # assumption that packages re-export their Feature classes
        # from the package root.
        ok = _python_check(
            venv_dir,
            (
                "import importlib.metadata as md\n"
                "from kestrel_sdk.features.base import Feature\n"
                "group = md.entry_points(group='kestrel_sovereign.features')\n"
                "by_name = {ep.name: ep for ep in group}\n"
                f"for cls_name in {sorted(feature_classes)!r}:\n"
                "    assert cls_name in by_name, "
                "f'entry point {cls_name!r} not registered'\n"
                "    obj = by_name[cls_name].load()\n"
                "    assert issubclass(obj, Feature), "
                "f'{cls_name!r} must be a kestrel_sdk Feature subclass'\n"
                f"print('{package} dev mode OK')\n"
            ),
        )
        if not ok:
            results.append(VerifyResult(
                sub_name, False,
                f"one or more of {feature_classes} are not kestrel_sdk Feature subclasses",
            ))
            continue
        results.append(VerifyResult(
            sub_name, True,
            f"{', '.join(feature_classes)} are kestrel_sdk.features.base.Feature subclasses",
        ))
    return results


# ---------------------------------------------------------------------------
# Test 5 — Full stack (install $REPO + every extracted package)
# ---------------------------------------------------------------------------

def _test_5_full_stack(work_dir: Path) -> List[VerifyResult]:
    """``pip install $REPO`` + every ``core = false`` package, then
    assert each declared Feature class imports AND that
    ``kestrel_sovereign.features`` entry-point discovery finds every
    Feature class.
    """
    name = "Test 5: Full stack"
    extracted = _extracted_feature_packages()
    if not extracted:
        return [VerifyResult(name, True, "no extracted feature packages registered")]

    repo = _repo_root()
    venv_dir = work_dir / "test5" / ".venv"
    results: List[VerifyResult] = []
    if not _make_venv(venv_dir):
        return [VerifyResult(name, False, "uv venv creation failed")]
    if not _pip_install(venv_dir, str(repo)):
        return [VerifyResult(name, False, "pip install of sovereign failed")]
    for package, _classes in extracted:
        if not _pip_install(venv_dir, package):
            return [VerifyResult(name, False, f"pip install of {package} failed")]

    expected_classes: List[str] = sorted({c for _, classes in extracted for c in classes})
    expected_repr = repr(tuple(expected_classes))

    # Imports: load every expected Feature class via its entry point
    # rather than `from <module> import <Class>` — package authors
    # aren't required to re-export from `__init__.py`. ``ep.load()``
    # exercises the declared `module.path:ClassName` so the import
    # path the SDK contract guarantees is the one tested.
    ok_imports = _python_check(
        venv_dir,
        (
            "import importlib.metadata as md\n"
            "from kestrel_sovereign.features.base import Feature\n"
            "group = md.entry_points(group='kestrel_sovereign.features')\n"
            "by_name = {ep.name: ep for ep in group}\n"
            f"for cls_name in {expected_repr}:\n"
            "    assert cls_name in by_name, "
            "f'entry point {cls_name!r} not registered'\n"
            "    by_name[cls_name].load()\n"
            "print('Full stack OK')\n"
        ),
    )
    results.append(VerifyResult(
        f"{name} (imports)",
        ok_imports,
        "all packages importable via entry-point resolution" if ok_imports
        else "one or more entry-point loads failed",
    ))

    # Discovery: enumerate the entry-point group and assert every
    # expected class name is registered. Catches the case where the
    # package was installed but the entry point itself was malformed
    # / missing — `ok_imports` above already covered actual load
    # failures, this asserts the group surface specifically.
    ok_eps = _python_check(
        venv_dir,
        (
            "import importlib.metadata\n"
            "eps = importlib.metadata.entry_points()\n"
            "group = eps.select(group='kestrel_sovereign.features')\n"
            "names = {ep.name for ep in group}\n"
            f"for needed in {expected_repr}:\n"
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
