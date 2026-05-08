"""``kestrel agent docker {create,chat,retire}`` CLI command —
sub-PR 3.2 of epic #1050 (bash-to-Python port of
``scripts/sovereign-agent.sh``).

Runs Kestrel agents in Docker containers with cryptographic isolation.
The agent receives ``KESTREL_DATA_KEY`` via ``-e`` but cannot read
anything else off the host filesystem — only ``-v <data_dir>:/data``
is mounted, so ``~/.zshrc``, ``~/.ssh/``, etc. are unreachable.

Subverbs:

- ``kestrel agent docker create <name> <data_dir>`` — runs the
  inception flow (``inception_service.py --name --output /data``)
  inside the container against the volume-mounted DB.
- ``kestrel agent docker chat <data_dir>`` — runs ``main.py`` against
  the volume-mounted DB with ``-it`` for interactive chat.
- ``kestrel agent docker retire <data_dir>`` — runs the retirement
  flow (graceful shutdown + final state export).

Coexists with the in-process ``kestrel create`` / ``kestrel shell``
commands; this is the Docker-isolated variant.

Cross-platform: docker works on Windows; the volume mount uses POSIX
path separators (docker accepts forward-slashes on Windows hosts too).
``Path(data_dir).expanduser()`` handles ``~``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from kestrel_sovereign._subprocess_helpers import run_streaming


_IMAGE_NAME = "kestrel-sovereign"
_DOCKERFILE_REL = Path("docker") / "Dockerfile.sovereign"

_KEY_GEN_HINT = (
    "  python -c \"import secrets; print(secrets.token_urlsafe(32))\""
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Repo root — used as the docker build context."""
    return Path(__file__).resolve().parent.parent


def _check_data_key(env: Optional[dict] = None) -> Optional[str]:
    """Return the value of ``KESTREL_DATA_KEY`` if set, else None.

    The bash predecessor errored out with a clear message + the
    ``secrets.token_urlsafe(32)`` hint; the caller composes that
    message based on this return.
    """
    e = env if env is not None else os.environ
    val = e.get("KESTREL_DATA_KEY")
    if not val:
        return None
    return val


def _print_missing_key_error() -> None:
    """Match the bash predecessor's friendly hint verbatim."""
    print(
        "error: KESTREL_DATA_KEY is not set!\n"
        "\n"
        "Generate a key with:\n"
        f"{_KEY_GEN_HINT}\n"
        "\n"
        "Then set it:\n"
        "  export KESTREL_DATA_KEY=\"your-key-here\"\n"
        "\n"
        "Store it safely (password manager, ~/.zshrc, etc.)",
        file=sys.stderr,
    )


def _resolve_data_dir(raw: str) -> Path:
    """Expand ``~`` against the operator's home and create the dir if
    missing — matches the bash predecessor's
    ``data_dir=${data_dir/#\\~/$HOME}`` + ``mkdir -p``.
    """
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_image(
    repo: Path,
    *,
    force_rebuild: bool = False,
    no_cache: bool = False,
) -> int:
    """Build the kestrel-sovereign image. Returns exit code: 0 on success.

    - Implicit path (``force_rebuild=False``): used by create/chat/run.
      Calls ``docker image inspect``; if the image exists, return
      immediately. Otherwise build with cache (fast first build).
    - Explicit path (``force_rebuild=True``): used by ``kestrel agent
      docker build``. Always rebuilds. ``no_cache`` chooses cache vs
      no-cache.

    Codex review v4 on PR #1071 split these two semantics: previously
    ``force_rebuild=True`` implied ``--no-cache``, so a plain
    ``kestrel agent docker build`` (without ``--no-cache``) couldn't
    rebuild a stale-but-cached image — the ``inspect`` short-circuit
    fired first.
    """
    if not force_rebuild:
        rc = run_streaming(
            ["docker", "image", "inspect", _IMAGE_NAME],
            # Suppress stdout because ``inspect`` dumps the manifest;
            # we only care about the exit code.
        )
        if rc == 0:
            print(f"[INFO] Using existing {_IMAGE_NAME} image")
            return 0
        print(f"[INFO] Building {_IMAGE_NAME} image ...")
    elif no_cache:
        print(f"[INFO] Rebuilding {_IMAGE_NAME} image (--no-cache) ...")
    else:
        print(f"[INFO] Rebuilding {_IMAGE_NAME} image ...")

    cmd = ["docker", "build"]
    if no_cache:
        cmd.append("--no-cache")
    cmd += [
        "-f", str(repo / _DOCKERFILE_REL),
        "-t", _IMAGE_NAME,
        str(repo),
    ]
    return run_streaming(cmd)


def _docker_run(
    *,
    data_dir: Path,
    data_key: str,
    container_argv: List[str],
    interactive: bool = False,
) -> int:
    """``docker run --rm [-it] -e KESTREL_DATA_KEY=... -v dir:/data
    <image> <container_argv>``.

    POSIX path separators in the volume mount (docker accepts
    forward-slashes on Windows). ``-it`` only when ``interactive`` to
    keep ``create`` / ``retire`` non-interactive.
    """
    cmd = ["docker", "run", "--rm"]
    if interactive:
        cmd += ["-it"]
    cmd += [
        "-e", f"KESTREL_DATA_KEY={data_key}",
        "-v", f"{data_dir.as_posix()}:/data",
        _IMAGE_NAME,
    ]
    cmd += container_argv
    return run_streaming(cmd)


# ---------------------------------------------------------------------------
# Subverb handlers
# ---------------------------------------------------------------------------

def _cmd_create(args) -> int:
    """``kestrel agent docker create <name> <data_dir>``."""
    name: str = args.name
    raw_dir: str = args.data_dir

    data_key = _check_data_key()
    if data_key is None:
        _print_missing_key_error()
        return 1

    repo = _repo_root()
    rc = _ensure_image(repo)
    if rc != 0:
        return rc

    data_dir = _resolve_data_dir(raw_dir)

    print(f"[INFO] Creating agent {name!r} in {data_dir} ...")
    # NOTE: --output is the explicit alias for --output-dir in
    # inception_service.py. Don't rely on argparse prefix-abbreviation
    # behaviour (matches the bash predecessor's same comment).
    rc = _docker_run(
        data_dir=data_dir,
        data_key=data_key,
        container_argv=[
            "inception_service.py",
            "--name", name,
            "--output", "/data",
        ],
    )
    if rc != 0:
        return rc

    print(f"[INFO] Agent {name!r} created successfully!")
    print("")
    print("Start chatting with:")
    print(f"  kestrel agent docker chat {raw_dir}")
    return 0


def _cmd_chat(args) -> int:
    """``kestrel agent docker chat <data_dir>``."""
    raw_dir: str = args.data_dir

    data_key = _check_data_key()
    if data_key is None:
        _print_missing_key_error()
        return 1

    repo = _repo_root()
    rc = _ensure_image(repo)
    if rc != 0:
        return rc

    data_dir = _resolve_data_dir(raw_dir)
    if not (data_dir / "kestrel_prime.db").is_file():
        print(
            f"error: no agent found in {data_dir}\n"
            f"       Create one first with: "
            f"kestrel agent docker create <name> {raw_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"[INFO] Starting chat with agent in {data_dir} ...")
    print("")
    return _docker_run(
        data_dir=data_dir,
        data_key=data_key,
        container_argv=[
            "main.py",
            "/data/kestrel_prime.db",
        ],
        interactive=True,
    )


def _cmd_run(args) -> int:
    """``kestrel agent docker run <data_dir> <command...>``.

    Generic escape hatch — runs a custom command inside the isolated
    agent container, with the host ``data_dir`` volume-mounted at
    ``/data`` and ``KESTREL_DATA_KEY`` passed through. Mirrors the
    bash predecessor's ``sovereign-agent.sh run <dir> <cmd...>`` for
    ad-hoc inspection / scripting against the encrypted volume.

    Codex review on PR #1071 caught that the deletion dropped this
    surface; this restores parity with the bash.
    """
    raw_dir: str = args.data_dir
    command: List[str] = list(getattr(args, "container_command", None) or [])
    if not command:
        print(
            "error: `kestrel agent docker run` requires a command.\n"
            "Example: kestrel agent docker run ~/emma_data "
            "python -c 'print(\"hello\")'",
            file=sys.stderr,
        )
        return 1

    data_key = _check_data_key()
    if data_key is None:
        _print_missing_key_error()
        return 1

    repo = _repo_root()
    rc = _ensure_image(repo)
    if rc != 0:
        return rc

    data_dir = _resolve_data_dir(raw_dir)
    return _docker_run(
        data_dir=data_dir,
        data_key=data_key,
        container_argv=command,
        interactive=False,
    )


def _cmd_retire(args) -> int:
    """``kestrel agent docker retire <data_dir>``."""
    raw_dir: str = args.data_dir
    assume_yes: bool = bool(getattr(args, "yes", False))

    data_key = _check_data_key()
    if data_key is None:
        _print_missing_key_error()
        return 1

    repo = _repo_root()
    rc = _ensure_image(repo)
    if rc != 0:
        return rc

    data_dir = _resolve_data_dir(raw_dir)
    if not (data_dir / "kestrel_prime.db").is_file():
        print(
            f"error: no agent found in {data_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"[WARN] This will retire the agent in {data_dir}")
    if not assume_yes:
        try:
            confirm = input("Are you sure? (yes/no) ")
        except EOFError:
            confirm = ""
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            return 0

    return _docker_run(
        data_dir=data_dir,
        data_key=data_key,
        container_argv=[
            "retirement_service.py",
            "/data/kestrel_prime.db",
        ],
    )


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def add_agent_docker_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel agent docker {create,chat,retire}`` under
    the parent subparsers.

    Called from :func:`kestrel_sovereign.cli.build_parser`. Top-level
    is ``agent`` because future agent-management subverbs (e.g.
    ``agent local`` for the in-process flow) might land here too —
    keeping ``docker`` as a sub-namespace leaves room for the layered
    surface to grow without breaking flags.
    """
    agent_p = subparsers.add_parser(
        "agent",
        help="Manage Kestrel agents — Docker-isolated lifecycle "
             "(create/chat/retire). Port of scripts/sovereign-agent.sh "
             "(epic #1050 tier 3).",
    )
    agent_sub = agent_p.add_subparsers(dest="agent_command")

    docker_p = agent_sub.add_parser(
        "docker",
        help="Run agent in a Docker container with KESTREL_DATA_KEY "
             "isolation",
    )
    docker_sub = docker_p.add_subparsers(dest="agent_docker_command")

    create_p = docker_sub.add_parser(
        "create",
        help="Create a new agent (runs inception in a Docker "
             "container)",
    )
    create_p.add_argument(
        "name",
        help="Agent name (e.g. Emma)",
    )
    create_p.add_argument(
        "data_dir",
        help="Host directory for agent data — created if missing; "
             "``~`` is expanded (e.g. ~/emma_data)",
    )

    chat_p = docker_sub.add_parser(
        "chat",
        help="Chat with an existing agent (interactive Docker run)",
    )
    chat_p.add_argument(
        "data_dir",
        help="Host directory containing the agent's "
             "``kestrel_prime.db`` (``~`` is expanded)",
    )

    retire_p = docker_sub.add_parser(
        "retire",
        help="Retire an agent — graceful shutdown + final state "
             "export",
    )
    retire_p.add_argument(
        "data_dir",
        help="Host directory containing the agent's "
             "``kestrel_prime.db`` (``~`` is expanded)",
    )
    retire_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive y/n confirmation prompt — useful "
             "for scripted retirement flows",
    )

    # ``kestrel agent docker run <data_dir> <command...>`` — generic
    # escape hatch that mirrors the bash predecessor's ``run`` subverb
    # for ad-hoc commands inside the isolated agent container. Codex
    # review v3 on PR #1071 caught the missing surface.
    run_p = docker_sub.add_parser(
        "run",
        help="Run a custom command inside the isolated agent container "
             "(escape hatch for ad-hoc inspection / scripting).",
    )
    run_p.add_argument(
        "data_dir",
        help="Host directory containing the agent's data (``~`` is expanded).",
    )
    # ``dest="container_command"`` to avoid colliding with the
    # top-level subparser's ``dest="command"`` — argparse merges all
    # positionals into the same Namespace, so a bare ``"command"``
    # here would clobber the top-level dispatch field with a list and
    # crash ``cli.main()`` (codex review v9 on PR #1079 caught it).
    run_p.add_argument(
        "container_command",
        nargs=argparse.REMAINDER,
        metavar="command",
        help="Command (and args) to run inside the container — e.g. "
             "``python -c 'print(\"hello\")'``.",
    )

    # ``kestrel agent docker build [--no-cache]`` — explicit rebuild
    # path. Codex review on PR #1071 caught that the helper had a
    # ``force_rebuild`` branch but no argparse path could reach it, so
    # operators with stale images had no way to force a refresh after
    # Dockerfile or code changes short of running ``docker rmi`` first.
    build_p = docker_sub.add_parser(
        "build",
        help="Build the kestrel-sovereign Docker image (run this after "
             "Dockerfile.sovereign or kestrel_sovereign/ changes; "
             "create/chat/retire skip the build when the image already "
             "exists).",
    )
    build_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Pass ``--no-cache`` to ``docker build`` so layers are "
             "rebuilt from scratch — slower, but the right escape hatch "
             "if a base-image refresh is needed.",
    )


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_agent(args) -> int:
    """Dispatch ``kestrel agent ...``.

    Exit codes:
        0 — success
        1 — runtime error (missing KESTREL_DATA_KEY, missing
            ``kestrel_prime.db``, docker build/run non-zero)
    """
    agent_sub = getattr(args, "agent_command", None)
    if agent_sub != "docker":
        print(
            "Usage: kestrel agent docker {create|chat|retire} ...",
            file=sys.stderr,
        )
        return 1

    docker_sub = getattr(args, "agent_docker_command", None)
    if docker_sub == "create":
        return _cmd_create(args)
    if docker_sub == "chat":
        return _cmd_chat(args)
    if docker_sub == "retire":
        return _cmd_retire(args)
    if docker_sub == "run":
        return _cmd_run(args)
    if docker_sub == "build":
        return _cmd_build(args)

    print(
        "Usage:\n"
        "  kestrel agent docker create <name> <data_dir>\n"
        "  kestrel agent docker chat   <data_dir>\n"
        "  kestrel agent docker retire <data_dir> [--yes]\n"
        "  kestrel agent docker run    <data_dir> <command...>\n"
        "  kestrel agent docker build  [--no-cache]\n"
        "\n"
        "Examples:\n"
        "  KESTREL_DATA_KEY=... kestrel agent docker create Emma ~/emma_data\n"
        "  KESTREL_DATA_KEY=... kestrel agent docker chat ~/emma_data\n"
        "  KESTREL_DATA_KEY=... kestrel agent docker retire ~/test_agent\n"
        "  KESTREL_DATA_KEY=... kestrel agent docker run ~/emma_data python -c 'print(1)'\n"
        "  kestrel agent docker build --no-cache",
        file=sys.stderr,
    )
    return 1


def _cmd_build(args) -> int:
    """``kestrel agent docker build [--no-cache]`` — explicit rebuild
    path. The bash predecessor's image lifecycle was implicit
    (``docker image inspect`` → build if missing); this gives operators
    a way to force a rebuild after Dockerfile or kestrel_sovereign/
    changes without remembering the underlying ``docker build`` argv.
    Codex review on PR #1071 caught the missing surface.
    """
    # ``build`` is the explicit rebuild surface — when an operator runs
    # it, they want the image rebuilt, period. Without ``force_rebuild``
    # the ``docker image inspect`` short-circuits and the command is a
    # no-op for the common stale-image case (codex review v4 on PR #1071).
    # ``--no-cache`` then chooses *how* the rebuild runs: with or
    # without docker layer cache. Both rebuild.
    repo = _repo_root()
    return _ensure_image(repo, force_rebuild=True, no_cache=bool(getattr(args, "no_cache", False)))


__all__ = [
    "add_agent_docker_subcommand",
    "cmd_agent",
]
