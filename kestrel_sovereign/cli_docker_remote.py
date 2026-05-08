"""``kestrel docker remote {build,run}`` CLI command — sub-PR 3.3 of
epic #1050 (bash-to-Python port of ``scripts/build_docker_remote.sh``
and ``scripts/run_docker_remote.sh``).

Lightweight Docker workflow for running Kestrel against remote LLM
providers (OpenAI, Anthropic, etc.) — the ``Dockerfile.agent.remote``
image skips torch / spacy / chromadb so the image is ~500MB instead
of ~32GB.

Subverbs:

- ``kestrel docker remote build [--tag latest] [--platform linux/amd64]``
  builds ``kestrel-remote:<tag>`` from ``Dockerfile.agent.remote``.
- ``kestrel docker remote run [--port 8888] [--env-file .env]`` parses
  ``.env``, validates ``OPENAI_API_KEY`` is set, stops/removes any
  existing container, picks the right Ollama host for the platform
  (``host.docker.internal`` on macOS/Windows, ``172.17.0.1`` on
  Linux), runs the container detached, and prints the access URL.

Coexists with ``kestrel deploy build`` (Cloud Run / GCR) — that's the
production multi-arch buildx path; this is the local single-host
remote-LLM path.

Cross-platform: docker works on Windows; ``host.docker.internal``
resolves on Docker Desktop (mac + Windows). On Linux the gateway IP
is ``172.17.0.1`` (the bash predecessor's behaviour).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from kestrel_sovereign._subprocess_helpers import run_streaming


_IMAGE_NAME = "kestrel-remote"
_DEFAULT_TAG = "latest"
_DEFAULT_PLATFORM = "linux/amd64"
_DEFAULT_HOST_PORT = 8888
_CONTAINER_PORT = 8888
_CONTAINER_NAME = "kestrel-remote"
_DOCKERFILE_REMOTE = "Dockerfile.agent.remote"

# Optional env-vars forwarded into the container on ``run``. The bash
# predecessor passed each as ``-e KEY="${KEY:-}"`` — empty values are
# silently dropped here so the ``-e`` array stays clean.
#
# Note that ``OPENAI_API_KEY`` is required (we error if missing) and is
# forwarded explicitly above this list.
_FORWARDED_ENV = (
    "KESTREL_API_KEY",
    "KESTREL_DATA_KEY",
    "REPLICATE_API_TOKEN",
    "TAVILY_API_KEY",
    "RUNPOD_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Repo root — the docker build context."""
    return Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> Dict[str, str]:
    """Read a ``.env`` file via ``dotenv.dotenv_values`` with
    ``interpolate=False``.

    Same idiom as :func:`kestrel_sovereign.features.deploy.secrets.load_env_file`
    — keep ``${PLACEHOLDER}`` values literal so any password/key that
    happens to contain ``${...}`` is forwarded verbatim, not silently
    expanded. Imported lazily so the module has no eager dotenv import.
    """
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")
    from dotenv import dotenv_values
    raw = dotenv_values(str(path), interpolate=False)
    return {k: v for k, v in raw.items() if v is not None}


def _detect_ollama_host() -> str:
    """Return the docker-bridge URL the container should use to reach
    a host-side Ollama.

    Bash predecessor used ``uname == Darwin`` → ``host.docker.internal``,
    else ``172.17.0.1``. Docker Desktop on Windows also resolves
    ``host.docker.internal``, so we treat ``win32`` like ``darwin``.
    """
    if sys.platform in ("darwin", "win32"):
        return "http://host.docker.internal:11434"
    return "http://172.17.0.1:11434"


# ---------------------------------------------------------------------------
# Subverb handlers
# ---------------------------------------------------------------------------

def _cmd_build(args) -> int:
    """``kestrel docker remote build``.

    ``docker build -f Dockerfile.agent.remote -t kestrel-remote:<tag>
    --platform <platform> .``
    """
    tag: str = args.tag or _DEFAULT_TAG
    platform: str = args.platform or _DEFAULT_PLATFORM

    repo = _repo_root()
    image = f"{_IMAGE_NAME}:{tag}"

    print(f"Building {image} for {platform} ...")
    print("Using lightweight dependencies (no torch, spacy, chromadb)")
    rc = run_streaming(
        [
            "docker", "build",
            "-f", _DOCKERFILE_REMOTE,
            "-t", image,
            "--platform", platform,
            ".",
        ],
        cwd=repo,
    )
    if rc != 0:
        return rc

    # Bash predecessor printed the image size with
    # ``docker images $IMAGE --format "{{.Size}}"``. We omit it because
    # ``docker build`` already streams that line into its own output.
    print(f"Built {image}")
    print(f"To run: kestrel docker remote run")
    return 0


def _cmd_run(args) -> int:
    """``kestrel docker remote run``.

    1. Load .env (validates ``OPENAI_API_KEY`` is present).
    2. Stop + remove any existing ``kestrel-remote`` container.
    3. ``docker run -d`` with port mapping, env vars, host gateway.
    """
    env_file_arg: Optional[str] = getattr(args, "env_file", None)
    # Port resolution mirrors the bash predecessor's ``${KESTREL_PORT:-8888}``:
    # explicit ``--port`` wins; else ``KESTREL_PORT`` env var; else
    # default 8888. Codex review v4 on PR #1071 caught that dropping
    # the env-var path silently broke wrappers that exported
    # KESTREL_PORT to avoid colliding with a busy/live 8888.
    if args.port is not None:
        host_port = int(args.port)
    elif os.environ.get("KESTREL_PORT"):
        try:
            host_port = int(os.environ["KESTREL_PORT"])
        except ValueError:
            print(
                f"error: KESTREL_PORT={os.environ['KESTREL_PORT']!r} is "
                f"not an integer.",
                file=sys.stderr,
            )
            return 1
    else:
        host_port = _DEFAULT_HOST_PORT
    tag: str = args.tag or _DEFAULT_TAG

    repo = _repo_root()
    env_file = (
        Path(env_file_arg) if env_file_arg else (repo / ".env")
    ).expanduser()

    try:
        env_vars = _load_env_file(env_file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not env_vars.get("OPENAI_API_KEY"):
        print(
            f"error: OPENAI_API_KEY not set in {env_file}",
            file=sys.stderr,
        )
        return 1

    image = f"{_IMAGE_NAME}:{tag}"

    # Stop and remove existing container — ``|| true`` in bash; we
    # silence stderr by passing ``check=False`` (run_streaming default)
    # and ignore non-zero exit codes.
    run_streaming(["docker", "stop", _CONTAINER_NAME])
    run_streaming(["docker", "rm", _CONTAINER_NAME])

    print(f"Starting {_CONTAINER_NAME} on port {host_port} ...")

    ollama_host = _detect_ollama_host()

    cmd: List[str] = [
        "docker", "run", "-d",
        "--name", _CONTAINER_NAME,
        "-p", f"{host_port}:{_CONTAINER_PORT}",
        "--add-host=host.docker.internal:host-gateway",
        # Required (validated above).
        "-e", f"OPENAI_API_KEY={env_vars['OPENAI_API_KEY']}",
        "-e", f"OLLAMA_HOST={ollama_host}",
    ]
    for key in _FORWARDED_ENV:
        # The bash predecessor passed empty defaults (``${KEY:-}``);
        # we drop empties to keep the ``-e`` flag list focused.
        val = env_vars.get(key)
        if val:
            cmd += ["-e", f"{key}={val}"]
    cmd.append(image)

    rc = run_streaming(cmd)
    if rc != 0:
        print("error: docker run failed", file=sys.stderr)
        return rc

    # Wait for /health — if Kestrel inside the container crashes
    # immediately, ``docker run -d`` still returns 0. Codex review on
    # PR #1071 caught the false-success path: the bash predecessor
    # waited and curled /health, surfacing logs on failure.
    health_url = f"http://localhost:{host_port}/health"
    print(f"   Polling {health_url} (up to 30s) ...")
    if not _wait_for_container_health(health_url, timeout=30.0):
        print(
            "error: container started but /health did not respond within 30s",
            file=sys.stderr,
        )
        print("   Last 50 log lines:", file=sys.stderr)
        # Stream the last 50 log lines so the operator sees why startup
        # failed without having to re-run ``docker logs`` themselves.
        run_streaming(["docker", "logs", "--tail", "50", _CONTAINER_NAME])
        return 1

    print(f"Kestrel Agent running at http://localhost:{host_port}")
    print(f"   Health: {health_url}")
    print(f"   API Docs: http://localhost:{host_port}/docs")
    if not env_vars.get("KESTREL_API_KEY"):
        print(
            "  no KESTREL_API_KEY set — server will generate one. "
            f"Check logs: docker logs {_CONTAINER_NAME} | grep -i key"
        )
    return 0


def _wait_for_container_health(url: str, *, timeout: float) -> bool:
    """Poll ``<url>`` until 200 or timeout. Returns False on timeout
    or persistent error. Used to verify the container actually serves
    Kestrel after ``docker run -d`` rather than just trusting the
    container-start exit code."""
    import time
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Argparse subcommand wiring
# ---------------------------------------------------------------------------

def get_or_create_docker_subparsers(
    subparsers: "argparse._SubParsersAction",
) -> "argparse._SubParsersAction":
    """Create (or fetch) the shared ``kestrel docker`` parent subparsers.

    The ``kestrel docker`` parent is shared by multiple modules:

    - :mod:`cli_docker_remote` owns ``remote`` (this file).
    - :mod:`cli_docker_build` owns ``build`` (Cloud Build / GCR
      specialty images, epic #1050 tier 4).

    Whichever module is wired up first creates the parser; the second
    one fetches the existing parser via the private
    ``_name_parser_map`` of the parent ``_SubParsersAction``. That is
    a documented attribute of argparse's subparsers action and is the
    standard way to compose subverb registrations across modules.
    """
    existing = subparsers.choices.get("docker")
    if existing is not None:
        # Find the docker subparsers action attached to the existing parser.
        for action in existing._actions:  # type: ignore[attr-defined]
            if isinstance(action, argparse._SubParsersAction):
                return action
        # Defensive: existing parser without subparsers — recreate.
        return existing.add_subparsers(dest="docker_command")
    docker_p = subparsers.add_parser(
        "docker",
        help="Docker workflows for Kestrel — local remote-LLM container "
             "(remote) and Cloud Build / GCR specialty images (build).",
    )
    return docker_p.add_subparsers(dest="docker_command")


def add_docker_subcommand(
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Register ``kestrel docker remote {build,run}`` under the parent
    subparsers.

    ``kestrel docker`` is namespaced separately from ``kestrel deploy``
    (which targets Cloud Run / GCR). The ``remote`` infix leaves room
    for future docker subverbs (e.g. ``kestrel docker local``) without
    breaking flags.
    """
    docker_sub = get_or_create_docker_subparsers(subparsers)

    remote_p = docker_sub.add_parser(
        "remote",
        help="Remote-LLM mode (lightweight image, ~500MB, no "
             "torch/spacy/chromadb)",
    )
    remote_sub = remote_p.add_subparsers(dest="docker_remote_command")

    build_p = remote_sub.add_parser(
        "build",
        help="Build the kestrel-remote Docker image",
    )
    build_p.add_argument(
        "--tag",
        type=str,
        default=_DEFAULT_TAG,
        help=f"Image tag (default: {_DEFAULT_TAG})",
    )
    build_p.add_argument(
        "--platform",
        type=str,
        default=_DEFAULT_PLATFORM,
        help=f"Build platform (default: {_DEFAULT_PLATFORM})",
    )

    run_p = remote_sub.add_parser(
        "run",
        help="Run the kestrel-remote container against the remote-LLM "
             "config in .env",
    )
    run_p.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Host port to bind (default: {_DEFAULT_HOST_PORT})",
    )
    run_p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to the .env file (default: .env in the repo root)",
    )
    run_p.add_argument(
        "--tag",
        type=str,
        default=_DEFAULT_TAG,
        help=f"Image tag to run (default: {_DEFAULT_TAG})",
    )


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------

def cmd_docker(args) -> int:
    """Dispatch ``kestrel docker ...``.

    Routes the ``build`` subverb into :mod:`cli_docker_build` and the
    ``remote`` subverb into the local handlers.

    Exit codes:
        0 — success
        1 — runtime error (missing .env, missing OPENAI_API_KEY,
            docker build/run non-zero, missing GCP_PROJECT_ID for
            ``build``, etc.)
    """
    docker_sub = getattr(args, "docker_command", None)
    if docker_sub == "build":
        # Local import — keeps cli_docker_build off the hot path for
        # operators who only run ``kestrel docker remote``.
        from kestrel_sovereign.cli_docker_build import cmd_docker_build
        return cmd_docker_build(args)

    if docker_sub == "remote":
        remote_sub = getattr(args, "docker_remote_command", None)
        if remote_sub == "build":
            return _cmd_build(args)
        if remote_sub == "run":
            return _cmd_run(args)
        print(
            "Usage:\n"
            "  kestrel docker remote build [--tag latest] [--platform linux/amd64]\n"
            "  kestrel docker remote run   [--port 8888] [--env-file .env] [--tag latest]",
            file=sys.stderr,
        )
        return 1

    print(
        "Usage:\n"
        "  kestrel docker build  <preset> [--tag latest] [--no-cache]\n"
        "  kestrel docker build  --list\n"
        "  kestrel docker remote build [--tag latest] [--platform linux/amd64]\n"
        "  kestrel docker remote run   [--port 8888] [--env-file .env] [--tag latest]",
        file=sys.stderr,
    )
    return 1


__all__ = [
    "add_docker_subcommand",
    "cmd_docker",
]
