"""Local clean-install validation: the macOS/Windows axis on your own machine.

The clean-install GitHub workflow only runs the billed macOS/Windows runners
as a pre-publish gate (and a weekly sweep). To avoid the gate failing AFTER
you cut a release tag, run this on your dev machine BEFORE tagging. It
reproduces the same readiness assertions the workflow runs, for free, on
whatever OS you are sitting at, Windows included.

Every assertion is delegated to ``scripts/ci/clean_install_verify.py``, the
same pure-Python checks the workflow uses, so local and CI stay in lockstep.

Usage (any Python 3.11+, from a fresh checkout)::

    python scripts/ci/clean_install_local.py          # sync install method (default)
    python scripts/ci/clean_install_local.py wheel    # wheel install method

It builds into the project venv (sync) or a throwaway venv (wheel), so it
does not disturb an editable dev install. Stdlib only.

Exit status: 0 pass, 1 an assertion failed or cleanup was inconclusive,
2 the checkout or environment is not safe to rehearse in.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_NAME = "Kestrel"
DEFAULT_PYTHON_VERSION = "3.13"

# Inception writes durable identity/config files. An existing local install
# would make this a dirty-state test and could silently select cloud routes.
STATE_PATHS = (".env", "kestrel.toml", "multi_agent.toml", "agent_data")

# The only KESTREL_* variable the rehearsal accepts from the caller: it
# selects the test runtime, not a database, Hold, identity or deployment.
ALLOWED_KESTREL_ENV = frozenset({"KESTREL_PY_VERSION"})

# Never inherited by the rehearsal's children, so it cannot discover a paid
# route from credentials exported in the invoking shell or reuse an
# operator's data directories.
STRIPPED_ENV = (
    "KESTREL_HOME", "KESTREL_DB_PATH", "AGENT_DATA_DIR", "KESTREL_DATA_DIR",
    "KESTREL_MULTI_AGENT_CONFIG",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
    "OPENROUTER_MANAGEMENT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY", "XAI_API_KEY", "MOONSHOT_API_KEY",
)

# Mirror the workflow env: the wizard takes its non-interactive, test-instance
# path and never blocks on a prompt or pollutes a real agent DB. The paired
# test-instance and audit-mode settings defer genesis as *pending* (never
# passed) even when this host runs Ollama; the harness calls no model.
REHEARSAL_ENV = {
    "KESTREL_DB_BACKEND": "sqlite",
    "KESTREL_AUDIT_MODE": "skip",
    "KESTREL_SKIP_REACHABILITY_PROBE": "1",
    "KESTREL_NONINTERACTIVE": "1",
    "KESTREL_TEST_INSTANCE": "1",
    "PYTHONSAFEPATH": "1",
}

VERIFY_STEPS: tuple[tuple[str, ...], ...] = (
    ("wizard-artifacts",),
    ("doctor",),  # the kestrel CLI itself, not a verifier subcommand
    ("identity", "--agent-name", AGENT_NAME),
    ("constitution", "--agent-name", AGENT_NAME),
    ("memory", "--agent-name", AGENT_NAME),
    ("test-instance", "--agent-name", AGENT_NAME),
    ("start-and-health", "--agent-name", AGENT_NAME),
    ("host-and-chat-503",),
    ("did-persists", "--agent-name", AGENT_NAME),
)


class RehearsalRefused(RuntimeError):
    """The checkout or environment is not safe to rehearse in."""


class StepFailed(RuntimeError):
    """A command in the rehearsal exited non-zero."""


def refuse_unsafe(root: Path, environ: Mapping[str, str]) -> None:
    for name in sorted(environ):
        if name == "DATABASE_URL" or (
            name.startswith("KESTREL_") and name not in ALLOWED_KESTREL_ENV
        ):
            raise RehearsalRefused(
                f"unset {name} before running the isolated clean-install rehearsal"
            )
    for state in STATE_PATHS:
        path = root / state
        if path.exists() or path.is_symlink():
            raise RehearsalRefused(
                f"clean-install rehearsal requires no existing {state}; use a fresh checkout"
            )


def child_env(environ: Mapping[str, str], harness_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if k not in STRIPPED_ENV}
    env.update(REHEARSAL_ENV)
    env["KESTREL_HOST_DB_PATH"] = str(harness_dir / "host-features.db")
    return env


def venv_executable(venv: Path, name: str) -> Path:
    """``name`` inside ``venv``: ``Scripts\\name.exe`` on Windows, ``bin/name`` elsewhere."""
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


class Rehearsal:
    def __init__(self, root: Path, method: str, python_version: str, env: dict[str, str],
                 harness_dir: Path):
        self.root = root
        self.method = method
        self.python_version = python_version
        self.env = env
        self.harness_dir = harness_dir
        self.python: list[str] = []
        self.kestrel: list[str] = []
        self.ports_ready = False

    def run(self, argv: Sequence[str]) -> None:
        returncode = run_child(argv, cwd=self.root, env=self.env)
        if returncode != 0:
            raise StepFailed(f"{' '.join(argv)} exited {returncode}")

    def install(self) -> None:
        self.run(["uv", "python", "install", self.python_version])
        if self.method == "wheel":
            # Mirror `pip install kestrel-sovereign`: build one wheel and
            # install it into a disposable venv OUTSIDE the checkout. A bare
            # `uv venv` would reuse ./.venv and overwrite the developer's
            # editable install; an out-of-tree path keeps it untouched.
            dist = self.harness_dir / "dist"
            venv = self.harness_dir / "venv"
            self.run(["uv", "build", "--wheel", "--out-dir", str(dist)])
            wheels = sorted(dist.glob("*.whl"))
            if len(wheels) != 1:
                raise StepFailed(f"expected one wheel in {dist}, found {len(wheels)}")
            self.run(["uv", "venv", "--python", self.python_version, str(venv)])
            self.run(["uv", "pip", "install", "--python", str(venv), str(wheels[0])])
            self.python = [str(venv_executable(venv, "python"))]
            self.kestrel = [str(venv_executable(venv, "kestrel"))]
        else:
            # Source-clone path: `uv sync` resolves ./.venv to the lock, the
            # dev env's normal state, and `uv run` invokes through it.
            self.run(["uv", "sync"])
            self.python = ["uv", "run", "python"]
            self.kestrel = ["uv", "run", "kestrel"]

    def validate(self) -> None:
        print("==> Setup wizard (kestrel setup --quickstart)", flush=True)
        self.run([*self.kestrel, "setup", "--quickstart"])
        # A developer may already run a real host on 8888 or an agent on
        # 8801. Rebind only this newly created test config to free ports.
        self.run([*self.python, "scripts/ci/clean_install_local_ports.py"])
        self.ports_ready = True
        print("==> Readiness assertions", flush=True)
        verifier = "scripts/ci/clean_install_verify.py"
        for step in VERIFY_STEPS:
            if step == ("doctor",):
                self.run([*self.kestrel, "doctor"])
            else:
                self.run([*self.python, verifier, *step])

    def cleanup(self) -> bool:
        """Remove the harness unless a process may still be using it."""
        if self.ports_ready:
            try:
                returncode = run_child(
                    [*self.python, "scripts/ci/clean_install_local_cleanup.py",
                     "--root", str(self.root), "--check-ports"],
                    cwd=self.root, env=self.env,
                )
            except (KeyboardInterrupt, SystemExit):
                self._warn_preserved()
                raise
            if returncode != 0:
                self._warn_preserved()
                return False
        shutil.rmtree(self.harness_dir)
        return True

    def _warn_preserved(self) -> None:
        print(
            f"warning: preserving clean-install harness {self.harness_dir}; "
            "verify and stop its processes before removing it",
            file=sys.stderr,
        )


def run_child(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
    """Run ``argv`` to completion and return its exit status.

    An interrupt (Ctrl-C) or termination of this runner does not cut the
    running step short, as bash does not cut its foreground command short:
    the step finishes its own teardown (the verifier stops the host it
    started) before the runner's cleanup begins. ``subprocess.run`` instead
    kills the child 0.25 s into a KeyboardInterrupt. The interrupt is
    re-raised once the child has exited.
    """
    proc = subprocess.Popen(list(argv), cwd=cwd, env=dict(env))
    try:
        return proc.wait()
    except (KeyboardInterrupt, SystemExit):
        while True:
            try:
                proc.wait()
                break
            except KeyboardInterrupt:
                continue
        raise


def _raise_exit(signum: int, _frame: object) -> None:
    """Turn a termination signal into SystemExit, so ``finally`` cleanup runs.

    Python's default SIGTERM/SIGHUP handling ends the process without running
    ``finally``; bash's ``trap ... EXIT`` ran on both. Exit status 128+signum
    is the shell convention for death by signal.
    """
    raise SystemExit(128 + signum)


def install_termination_handlers() -> None:
    """Handle SIGTERM, SIGHUP and (Windows) Ctrl-Break, unless ignored at start.

    A signal ignored when the runner starts stays ignored, as bash leaves it:
    under ``nohup`` SIGHUP is ignored, and replacing that with a handler would
    reset it to "die" in every step the runner starts.
    """
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):  # per-platform; getattr skips absent
        signum = getattr(signal, name, None)
        if signum is not None and signal.getsignal(signum) is signal.SIG_DFL:
            signal.signal(signum, _raise_exit)


def python_version_from(environ: Mapping[str, str]) -> str:
    """``KESTREL_PY_VERSION``, treating empty as unset (bash ``${VAR:-default}``)."""
    return environ.get("KESTREL_PY_VERSION") or DEFAULT_PYTHON_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("method", nargs="?", default="sync", choices=("sync", "wheel"),
                        help="install method (default: sync)")
    args = parser.parse_args(argv)

    try:
        refuse_unsafe(REPO_ROOT, os.environ)
    except RehearsalRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    python_version = python_version_from(os.environ)
    install_termination_handlers()
    harness_dir = Path(tempfile.mkdtemp(prefix="kestrel-clean-install-"))
    rehearsal = Rehearsal(
        REPO_ROOT, args.method, python_version, child_env(os.environ, harness_dir), harness_dir
    )
    print(
        f"==> Clean-install local validation: {args.method} install / "
        f"Python {python_version} / {platform.system()}",
        flush=True,
    )
    status = 0
    try:
        rehearsal.install()
        rehearsal.validate()
    except StepFailed as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        status = 1
    finally:
        if not rehearsal.cleanup():
            status = 1
    if status == 0:
        print(f"==> PASS: clean install OK on {platform.system()} ({args.method}). Safe to tag.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
