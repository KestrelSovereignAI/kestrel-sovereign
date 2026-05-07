"""Docker image build/push port — replaces ``scripts/cloudrun/build.sh``.

Sub-PR 1.3 of epic #1050 (host-side bash → Python). The original bash
script ran two ``docker buildx build`` invocations — one for the
single-agent ``kestrel`` image and one for the ``kestrel-multi_agent``
host image — both tagged with both ``:$TAG`` and ``:latest`` and pushed
to ``gcr.io/<project>/...``. ``scripts/cloudrun/build_multi_agent.sh``
covered the same multi_agent image with a single-arch ``docker build``
fallback for operators without buildx; we subsume that flow via the
``multi_arch=False`` flag on :func:`build_target`.

Public surface
--------------

* :data:`DEFAULT_TARGETS` — the two canonical Cloud Run images.
* :class:`BuildTarget` — declarative description of one image build.
* :class:`BuildResult` — outcome of one build.
* :class:`BuildError` — raised on docker non-zero exit; carries the
  failed image ref + the docker invocation that failed.
* :func:`resolve_github_token` — env-first / ``gh auth token`` fallback.
* :func:`build_target` — build (and optionally push) one target.
* :func:`build_all` — iterate targets, accumulate results, raise on
  the first error (with the partial result list attached).

Independence from :class:`DeployManager`
----------------------------------------

The agent-tool surface (``DeployFeature.deploy_agent``) intentionally
does NOT expose builds — agents shouldn't be running ``docker buildx``.
This module is standalone, mirroring the pattern :mod:`secrets` set in
sub-PR 1.2.

Subprocess handling
-------------------

``subprocess`` is imported lazily inside the functions so loading the
module is cheap (matches the :mod:`secrets` idiom). All subprocess
invocations use the list form — no shell, no quoting, no
platform-specific escaping. The ``runner`` parameter is injectable so
unit tests can mock without touching real ``docker``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BuildTarget:
    """One image we know how to build for the cloudrun deploy.

    Attributes:
        image_name: Short name for the image (without registry prefix or
            tag), e.g. ``"kestrel"`` or ``"kestrel-multi_agent"``. The
            full ref is ``gcr.io/<project>/<image_name>:<tag>``.
        dockerfile: Path to the Dockerfile, relative to the project
            root, e.g. ``Path("docker/Dockerfile.cloudrun")``.
        description: Human-readable label used in log lines.
    """

    image_name: str
    dockerfile: Path
    description: str


@dataclass
class BuildResult:
    """Outcome of one :func:`build_target` invocation.

    ``image_refs`` is the list of full registry refs that were tagged
    (e.g. ``["gcr.io/p/kestrel:v1.2.3", "gcr.io/p/kestrel:latest"]``).
    ``pushed`` is True iff ``--push`` was effectively part of the
    invocation. ``skipped_reason`` is set when a target was filtered
    out (e.g. via ``--target`` mismatch in :func:`build_all`); for a
    skipped target neither ``image_refs`` nor ``pushed`` carry meaning.
    """

    target: BuildTarget
    image_refs: List[str] = field(default_factory=list)
    pushed: bool = False
    skipped_reason: Optional[str] = None
    duration_seconds: Optional[float] = None


class BuildError(Exception):
    """Raised when ``docker buildx build`` (or ``docker push``) returns
    a non-zero exit code.

    Holds the failed image ref + the exact docker invocation that
    failed so the operator can reproduce/debug without re-running the
    whole pipeline. ``stderr`` is captured when available — the
    injected runner may or may not capture it; defaults pass through.
    """

    def __init__(
        self,
        image_ref: str,
        command: List[str],
        stderr: str = "",
        partial_results: Optional[List[BuildResult]] = None,
    ) -> None:
        self.image_ref = image_ref
        self.command = command
        self.stderr = stderr
        # ``partial_results`` is populated by :func:`build_all` so callers
        # can show ``built kestrel; failed kestrel-multi_agent`` rather
        # than throwing away the successful first build's diagnostics.
        self.partial_results: List[BuildResult] = partial_results or []
        cmd_str = " ".join(command)
        suffix = f"\nstderr: {stderr}" if stderr else ""
        super().__init__(
            f"docker build failed for {image_ref}: exit non-zero from "
            f"`{cmd_str}`{suffix}"
        )


# ---------------------------------------------------------------------------
# Default targets — the two canonical cloudrun images
# ---------------------------------------------------------------------------

# Mirrors ``scripts/cloudrun/build.sh`` exactly: same image names, same
# Dockerfile paths. The legacy ``build_multi_agent.sh`` used a different
# image name (``kestrel-multi-agent`` with hyphens) but that script was
# the redundant fallback and the canonical name is the underscore form
# pushed by ``build.sh`` (which is what deploy_config.toml profiles
# actually reference).
DEFAULT_TARGETS: List[BuildTarget] = [
    BuildTarget(
        image_name="kestrel",
        dockerfile=Path("docker/Dockerfile.cloudrun"),
        description="single-agent",
    ),
    # Image name matches scripts/cloudrun/build.sh (the canonical
    # multi-arch builder) and .github/workflows/deploy.yml (the CI
    # path) — both use ``kestrel-multi_agent`` (underscore). The legacy
    # scripts/cloudrun/{build_multi_agent.sh,deploy_dev.sh,deploy_multi_agent_dev.sh}
    # diverged on ``kestrel-multi-agent`` (hyphen); that bash-vs-bash
    # inconsistency is a pre-existing config drift the epic's Tier 1.4
    # reconciliation will resolve. Picking the underscore here keeps
    # ``kestrel deploy build`` aligned with CI's image name and the
    # canonical buildx flow. Codex review on PR #1060 flagged the
    # divergence; the answer is to fix the divergent path in 1.4, not
    # to perpetuate two image names for the same build.
    BuildTarget(
        image_name="kestrel-multi_agent",
        dockerfile=Path("docker/Dockerfile.multi_agent"),
        description="multi_agent host",
    ),
]


# ---------------------------------------------------------------------------
# GitHub token resolution
# ---------------------------------------------------------------------------

def resolve_github_token() -> Optional[str]:
    """Return a GitHub token using the same precedence as ``build.sh``.

    Order:

    1. ``GITHUB_TOKEN`` environment variable (wins if non-empty).
    2. ``gh auth token`` shell invocation (whatever the operator's gh
       CLI is logged in as).

    Returns the token string, or None if neither source has one. The
    caller is responsible for warning the operator — this function is
    side-effect-free besides the optional ``gh`` subprocess.

    The ``gh auth token`` step is best-effort: any failure (gh missing,
    not logged in, error exit) is swallowed and treated as "no token".
    The bash script used ``gh auth token 2>/dev/null || true`` — same
    intent here.
    """
    env_value = os.getenv("GITHUB_TOKEN")
    if env_value:
        return env_value

    # Lazy import — see module docstring rationale.
    import subprocess

    try:
        completed = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # gh CLI not installed.
        return None
    except OSError:
        # Other exec failures (permissions, etc.).
        return None

    if completed.returncode != 0:
        return None

    token = (completed.stdout or "").strip()
    return token or None


# ---------------------------------------------------------------------------
# Single-target build
# ---------------------------------------------------------------------------

def _default_runner(cmd: List[str], env: Optional[dict] = None) -> Any:
    """Default subprocess runner — ``subprocess.run`` with check=False.

    Tests inject their own runner; production code uses this. We
    intentionally don't pass ``check=True`` — we want to inspect the
    return code ourselves and wrap a non-zero exit in :class:`BuildError`
    rather than raising :class:`subprocess.CalledProcessError` which
    leaks the stdlib exception class across our public boundary.
    """
    import subprocess

    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _build_buildx_command(
    target: BuildTarget,
    *,
    project_id: str,
    tag: str,
    platforms: Sequence[str],
    push: bool,
    has_github_token: bool,
    project_root: Path,
) -> List[str]:
    """Construct the ``docker buildx build`` argv for the multi-arch
    build path. Mirrors ``scripts/cloudrun/build.sh`` exactly:

        docker buildx build \\
            --platform <comma-joined platforms> \\
            -f <project_root>/<dockerfile> \\
            [--secret id=github_token,env=GITHUB_TOKEN] \\
            -t gcr.io/<project>/<image>:<tag> \\
            -t gcr.io/<project>/<image>:latest \\
            [--push] \\
            <project_root>

    The ``--secret`` flag is omitted when no token is available — adding
    it without a token would break the build on environments that don't
    set ``GITHUB_TOKEN`` and can't satisfy the BuildKit secret read.
    Builds whose Dockerfile doesn't reference the secret simply work
    either way.
    """
    dockerfile_path = project_root / target.dockerfile
    image_base = f"gcr.io/{project_id}/{target.image_name}"

    cmd: List[str] = [
        "docker", "buildx", "build",
        "--platform", ",".join(platforms),
        "-f", str(dockerfile_path),
    ]
    if has_github_token:
        cmd.extend(["--secret", "id=github_token,env=GITHUB_TOKEN"])
    cmd.extend(["-t", f"{image_base}:{tag}"])
    # The bash script always writes the ``:latest`` tag in addition to
    # the named tag; deploys that pin ``:latest`` rely on this. We only
    # skip it when the named tag itself is ``latest`` (avoids a
    # redundant duplicate ``-t`` arg, though docker would tolerate it).
    if tag != "latest":
        cmd.extend(["-t", f"{image_base}:latest"])
    if push:
        cmd.append("--push")
    else:
        # Without --push or --load (or another --output), buildx leaves
        # the result only in the build cache — the CLI promised
        # "local-only images" with --no-push but the operator would get
        # nothing usable. ``--load`` writes the result into the local
        # docker daemon, which is exactly what local smoke tests want.
        # Codex review on PR #1060 caught this. ``--load`` only supports
        # single-platform builds; the validator above forces that
        # constraint when push=False.
        cmd.append("--load")
    cmd.append(str(project_root))
    return cmd


def _build_plain_command(
    target: BuildTarget,
    *,
    project_id: str,
    tag: str,
    project_root: Path,
    has_github_token: bool = False,
) -> List[str]:
    """Construct the legacy single-arch ``docker build`` argv used when
    ``multi_arch=False`` — mirrors ``scripts/cloudrun/build_multi_agent.sh``.

    The cloudrun + multi_agent Dockerfiles already use BuildKit's
    ``RUN --mount=type=secret,id=github_token`` to install private
    deps. If the operator has a GITHUB_TOKEN, pass ``--secret`` here
    too — modern ``docker build`` enables BuildKit by default and
    accepts the same flag (we also set ``DOCKER_BUILDKIT=1`` in the
    subprocess env at the call site for older docker installs). Codex
    review on PR #1060: dropping the secret here silently broke
    Dockerfiles that depend on it.
    """
    dockerfile_path = project_root / target.dockerfile
    image_base = f"gcr.io/{project_id}/{target.image_name}"

    cmd: List[str] = [
        "docker", "build",
        "-f", str(dockerfile_path),
    ]
    if has_github_token:
        cmd.extend(["--secret", "id=github_token,env=GITHUB_TOKEN"])
    cmd.extend(["-t", f"{image_base}:{tag}"])
    if tag != "latest":
        cmd.extend(["-t", f"{image_base}:latest"])
    cmd.append(str(project_root))
    return cmd


def _push_command(image_ref: str) -> List[str]:
    """``docker push <ref>`` argv for the legacy fallback flow."""
    return ["docker", "push", image_ref]


def _image_refs(target: BuildTarget, project_id: str, tag: str) -> List[str]:
    """The full registry refs that a build of this target writes.

    Always returns at least the ``:tag`` ref; adds ``:latest`` when the
    tag isn't already ``latest`` (matches :func:`_build_buildx_command`
    and :func:`_build_plain_command`).
    """
    image_base = f"gcr.io/{project_id}/{target.image_name}"
    refs = [f"{image_base}:{tag}"]
    if tag != "latest":
        refs.append(f"{image_base}:latest")
    return refs


def build_target(
    target: BuildTarget,
    *,
    project_id: str,
    tag: str,
    platforms: Sequence[str] = ("linux/amd64", "linux/arm64"),
    push: bool = True,
    multi_arch: bool = True,
    github_token: Optional[str] = None,
    project_root: Optional[Path] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> BuildResult:
    """Build (and optionally push) one image target.

    Args:
        target: The :class:`BuildTarget` describing what to build.
        project_id: GCP project ID (registry path is
            ``gcr.io/<project_id>/<image>``).
        tag: Image tag for the named build (``:latest`` is always also
            written unless ``tag`` itself is ``"latest"``).
        platforms: Buildx target platforms, only used in multi-arch
            mode. Defaults to ``("linux/amd64", "linux/arm64")``,
            matching the bash script.
        push: If True, ``--push`` is added in multi-arch mode and a
            separate ``docker push`` is run in plain mode. If False,
            the build stays local — useful for smoke tests.
        multi_arch: If True (default), use ``docker buildx build``. If
            False, fall back to plain ``docker build`` + ``docker push``
            (the legacy ``build_multi_agent.sh`` flow, for operators
            without buildx).
        github_token: When provided, set in the subprocess env as
            ``GITHUB_TOKEN`` AND, in multi-arch mode, the
            ``--secret id=github_token,env=GITHUB_TOKEN`` flag is added
            so the Dockerfile can consume it via BuildKit secrets.
            When None, the secret flag is omitted.
        project_root: Defaults to ``Path.cwd()`` (matches the
            ``deploy_config.toml`` / secrets.py CWD-relative
            convention; codex review on PR #1057 settled the rule).
            The build context is the project root.
        runner: Injectable replacement for :func:`subprocess.run`.
            Tests pass a ``MagicMock``; production code defaults to
            :func:`_default_runner`. The runner is called as
            ``runner(cmd, env=env)`` and must return an object with
            ``.returncode`` (and, if available, ``.stderr``).

    Returns:
        :class:`BuildResult` with timing and tagged refs populated.

    Raises:
        BuildError: If any docker invocation returns a non-zero exit
            code. The exception carries the failed image ref + the
            exact argv for debugging.
    """
    if project_root is None:
        project_root = Path.cwd()
    if runner is None:
        runner = _default_runner

    # ``--load`` (used in multi_arch + push=False mode below) only
    # supports a single platform — it writes the image into the local
    # docker daemon, which can't store a manifest list. Reject early
    # with a clear message rather than emitting a buildx command that
    # silently fails. Codex review on PR #1060 caught this.
    if multi_arch and not push and len(platforms) > 1:
        raise ValueError(
            "kestrel deploy build --no-push requires a single --platforms "
            f"value (got {list(platforms)}). buildx --load only supports "
            "single-platform builds. Pass e.g. --platforms linux/amd64 "
            "or drop --no-push to keep the multi-arch push path."
        )

    # Build the subprocess environment — we want the docker invocation
    # to inherit the parent env (PATH, DOCKER_HOST, etc.) but with
    # ``GITHUB_TOKEN`` overlaid when we have one. Using ``os.environ.copy()``
    # keeps Docker's auth helpers (``DOCKER_CONFIG``, etc.) available.
    env: Optional[dict]
    if github_token is not None:
        env = os.environ.copy()
        env["GITHUB_TOKEN"] = github_token
    else:
        env = None  # inherit parent env unmodified

    refs = _image_refs(target, project_id, tag)
    has_token = github_token is not None

    started = time.monotonic()

    if multi_arch:
        cmd = _build_buildx_command(
            target,
            project_id=project_id,
            tag=tag,
            platforms=platforms,
            push=push,
            has_github_token=has_token,
            project_root=project_root,
        )
        completed = runner(cmd, env=env)
        rc = getattr(completed, "returncode", 0)
        if rc != 0:
            stderr = getattr(completed, "stderr", "") or ""
            raise BuildError(refs[0], cmd, stderr=stderr)
    else:
        # Plain ``docker build`` — single-arch local build. Pass the
        # GitHub token along: modern ``docker build`` honors BuildKit
        # secrets, and DOCKER_BUILDKIT=1 forces older installs onto
        # the BuildKit path so ``RUN --mount=type=secret,id=github_token``
        # in Dockerfile.cloudrun / Dockerfile.multi_agent can read it.
        # Codex review on PR #1060: dropping the secret in fallback
        # mode silently broke private-dep installs.
        if env is None:
            env = os.environ.copy()
        env.setdefault("DOCKER_BUILDKIT", "1")
        cmd = _build_plain_command(
            target,
            project_id=project_id,
            tag=tag,
            project_root=project_root,
            has_github_token=has_token,
        )
        completed = runner(cmd, env=env)
        rc = getattr(completed, "returncode", 0)
        if rc != 0:
            stderr = getattr(completed, "stderr", "") or ""
            raise BuildError(refs[0], cmd, stderr=stderr)

        # In plain mode ``--push`` doesn't exist; we run ``docker push``
        # for each tagged ref iff the caller asked for a push.
        if push:
            for ref in refs:
                push_cmd = _push_command(ref)
                pushed = runner(push_cmd, env=env)
                push_rc = getattr(pushed, "returncode", 0)
                if push_rc != 0:
                    push_stderr = getattr(pushed, "stderr", "") or ""
                    raise BuildError(ref, push_cmd, stderr=push_stderr)

    duration = time.monotonic() - started

    return BuildResult(
        target=target,
        image_refs=refs,
        pushed=push,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Multi-target orchestrator
# ---------------------------------------------------------------------------

def build_all(
    *,
    project_id: str,
    tag: str = "latest",
    targets: Sequence[BuildTarget] = (),
    platforms: Sequence[str] = ("linux/amd64", "linux/arm64"),
    push: bool = True,
    multi_arch: bool = True,
    github_token: Optional[str] = None,
    project_root: Optional[Path] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> List[BuildResult]:
    """Iterate ``targets`` calling :func:`build_target` for each.

    Args:
        targets: Sequence of :class:`BuildTarget` to build. Empty
            defaults to :data:`DEFAULT_TARGETS` — the two canonical
            cloudrun images.
        Other args are passed through unchanged to :func:`build_target`.

    Behaviour on error
    ------------------
    The first failing target raises :class:`BuildError`. We use
    ``stop_on_error=True`` semantics implicitly because the bash
    script also halts on the first failure (``set -e``). To preserve
    diagnostics about earlier successful builds, the partial result
    list is attached to the exception's ``partial_results`` attribute,
    so the CLI can print "built kestrel in 92.4s; failed
    kestrel-multi_agent" instead of swallowing the success.

    Returns:
        One :class:`BuildResult` per target (in the order they were
        passed in / in the order of :data:`DEFAULT_TARGETS`).
    """
    if not targets:
        targets = DEFAULT_TARGETS

    results: List[BuildResult] = []
    for tgt in targets:
        try:
            result = build_target(
                tgt,
                project_id=project_id,
                tag=tag,
                platforms=platforms,
                push=push,
                multi_arch=multi_arch,
                github_token=github_token,
                project_root=project_root,
                runner=runner,
            )
        except BuildError as e:
            # Re-raise with the partial results attached so the CLI
            # can render context. We don't swallow — operators want
            # exit code 1 from a failed build, not "2 of 2 attempted".
            e.partial_results = list(results)
            raise
        results.append(result)

    return results


__all__ = [
    "BuildTarget",
    "BuildResult",
    "BuildError",
    "DEFAULT_TARGETS",
    "resolve_github_token",
    "build_target",
    "build_all",
]
