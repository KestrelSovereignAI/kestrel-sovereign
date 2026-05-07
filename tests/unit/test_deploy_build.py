"""Tests for ``kestrel_sovereign.features.deploy.build``.

Sub-PR 1.3 of epic #1050 (bash-to-Python port of
``scripts/cloudrun/build.sh``).

Docker is mocked via the injectable ``runner`` parameter on
:func:`build_target` / :func:`build_all` so the unit tests don't touch
real docker. We assert the constructed argv matches the bash script's
shape exactly (``--platform``, ``-f``, ``--secret``, ``-t`` x2, ``--push``,
build context).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.features.deploy.build import (
    DEFAULT_TARGETS,
    BuildError,
    BuildResult,
    BuildTarget,
    build_all,
    build_target,
    resolve_github_token,
)


# ---------------------------------------------------------------------------
# Smoke: dataclasses
# ---------------------------------------------------------------------------

def test_build_target_dataclass_fields():
    """BuildTarget carries image_name / dockerfile / description."""
    t = BuildTarget(
        image_name="kestrel",
        dockerfile=Path("docker/Dockerfile.cloudrun"),
        description="single-agent",
    )
    assert t.image_name == "kestrel"
    assert t.dockerfile == Path("docker/Dockerfile.cloudrun")
    assert t.description == "single-agent"


def test_build_result_dataclass_defaults():
    """BuildResult image_refs / pushed / skipped_reason / duration default
    cleanly (so tests don't have to construct the full thing)."""
    t = BuildTarget("k", Path("docker/Dockerfile.cloudrun"), "x")
    r = BuildResult(target=t)
    assert r.image_refs == []
    assert r.pushed is False
    assert r.skipped_reason is None
    assert r.duration_seconds is None


def test_build_error_carries_image_ref_and_command():
    """BuildError exposes image_ref + command + stderr for the CLI to
    render diagnostics without re-running the build."""
    cmd = ["docker", "buildx", "build", "-f", "Dockerfile.cloudrun", "."]
    err = BuildError(
        image_ref="gcr.io/test/kestrel:latest",
        command=cmd,
        stderr="boom",
    )
    assert err.image_ref == "gcr.io/test/kestrel:latest"
    assert err.command == cmd
    assert err.stderr == "boom"
    # Message should be useful as-is in the operator's terminal.
    assert "gcr.io/test/kestrel:latest" in str(err)
    assert "docker buildx build" in str(err)
    assert "boom" in str(err)


def test_default_targets_match_bash_script():
    """DEFAULT_TARGETS encodes the exact two images the bash build.sh
    pushed: kestrel + kestrel-multi_agent (underscore form, matching
    deploy_config.toml profiles, NOT the legacy hyphen build_multi_agent.sh
    name)."""
    assert len(DEFAULT_TARGETS) == 2
    names = [t.image_name for t in DEFAULT_TARGETS]
    assert names == ["kestrel", "kestrel-multi_agent"]
    dfs = [t.dockerfile for t in DEFAULT_TARGETS]
    assert dfs == [
        Path("docker/Dockerfile.cloudrun"),
        Path("docker/Dockerfile.multi_agent"),
    ]


# ---------------------------------------------------------------------------
# resolve_github_token
# ---------------------------------------------------------------------------

def test_resolve_github_token_env_wins(monkeypatch):
    """GITHUB_TOKEN env value beats `gh auth token`."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fromenv")
    # gh shouldn't even be called — patch it to fail loudly if it is.
    with patch("subprocess.run") as run:
        token = resolve_github_token()

    assert token == "ghp_fromenv"
    run.assert_not_called()


def test_resolve_github_token_falls_back_to_gh(monkeypatch):
    """No env var → invoke ``gh auth token`` and use its stdout."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    fake_completed = MagicMock(returncode=0, stdout="ghp_fromgh\n", stderr="")
    with patch("subprocess.run", return_value=fake_completed) as run:
        token = resolve_github_token()

    assert token == "ghp_fromgh"
    # gh auth token, not gh auth status / login / etc.
    args, kwargs = run.call_args
    assert args[0] == ["gh", "auth", "token"]


def test_resolve_github_token_returns_none_when_neither_source(monkeypatch):
    """No env var + gh fails → None (caller logs the warning)."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    fake_completed = MagicMock(returncode=1, stdout="", stderr="not logged in")
    with patch("subprocess.run", return_value=fake_completed):
        token = resolve_github_token()

    assert token is None


def test_resolve_github_token_returns_none_when_gh_missing(monkeypatch):
    """gh CLI not installed (FileNotFoundError) → None, no traceback."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with patch("subprocess.run", side_effect=FileNotFoundError("no gh")):
        token = resolve_github_token()

    assert token is None


def test_resolve_github_token_returns_none_for_empty_gh_output(monkeypatch):
    """gh exits 0 but emits empty/whitespace → None.

    Without this guard we'd hand an empty string downstream, which
    triggers the BuildKit ``--secret`` flag without a real value and
    breaks builds that try to consume it.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    fake_completed = MagicMock(returncode=0, stdout="\n", stderr="")
    with patch("subprocess.run", return_value=fake_completed):
        token = resolve_github_token()

    assert token is None


# ---------------------------------------------------------------------------
# build_target — argv construction
# ---------------------------------------------------------------------------

@pytest.fixture
def kestrel_target() -> BuildTarget:
    return BuildTarget(
        image_name="kestrel",
        dockerfile=Path("docker/Dockerfile.cloudrun"),
        description="single-agent",
    )


def test_build_target_multi_arch_push_with_token(kestrel_target, tmp_path):
    """Multi-arch + push + token: matches the canonical build.sh argv."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    result = build_target(
        kestrel_target,
        project_id="test-project",
        tag="v1.2.3",
        platforms=("linux/amd64", "linux/arm64"),
        push=True,
        multi_arch=True,
        github_token="ghp_test",
        project_root=tmp_path,
        runner=runner,
    )

    runner.assert_called_once()
    cmd = runner.call_args.args[0]

    # Shape of the argv mirrors scripts/cloudrun/build.sh exactly.
    assert cmd[0:3] == ["docker", "buildx", "build"]
    assert "--platform" in cmd
    plat_idx = cmd.index("--platform")
    assert cmd[plat_idx + 1] == "linux/amd64,linux/arm64"

    assert "-f" in cmd
    f_idx = cmd.index("-f")
    assert cmd[f_idx + 1] == str(tmp_path / "docker" / "Dockerfile.cloudrun")

    # --secret flag present because we have a token.
    assert "--secret" in cmd
    secret_idx = cmd.index("--secret")
    assert cmd[secret_idx + 1] == "id=github_token,env=GITHUB_TOKEN"

    # Two -t flags: :v1.2.3 and :latest.
    tags = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-t"]
    assert tags == [
        "gcr.io/test-project/kestrel:v1.2.3",
        "gcr.io/test-project/kestrel:latest",
    ]

    assert "--push" in cmd
    # Last arg is the build context (project_root).
    assert cmd[-1] == str(tmp_path)

    # GITHUB_TOKEN propagated into subprocess env.
    env = runner.call_args.kwargs["env"]
    assert env is not None
    assert env["GITHUB_TOKEN"] == "ghp_test"

    # Result wiring.
    assert result.target is kestrel_target
    assert result.image_refs == [
        "gcr.io/test-project/kestrel:v1.2.3",
        "gcr.io/test-project/kestrel:latest",
    ]
    assert result.pushed is True
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0.0


def test_build_target_multi_arch_no_push(kestrel_target, tmp_path):
    """--no-push uses ``--load`` (writes into local docker daemon) and
    requires a single platform. Without ``--load`` a buildx build with
    no output target leaves the result only in the build cache, which
    contradicts the CLI's "local-only images" promise (codex review
    on PR #1060)."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    result = build_target(
        kestrel_target,
        project_id="test-project",
        tag="latest",
        platforms=("linux/amd64",),  # buildx --load only supports single-arch
        push=False,
        multi_arch=True,
        github_token=None,
        project_root=tmp_path,
        runner=runner,
    )

    cmd = runner.call_args.args[0]
    assert "--push" not in cmd
    assert "--load" in cmd, "--load is required for buildx no-push to produce a usable local image"
    # tag=latest → only one -t flag (don't write :latest twice).
    tags = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-t"]
    assert tags == ["gcr.io/test-project/kestrel:latest"]
    assert result.pushed is False


def test_build_target_multi_arch_no_push_rejects_multiple_platforms(
    kestrel_target, tmp_path
):
    """--no-push + multi-platform is invalid (buildx --load only works
    single-arch). Codex review on PR #1060: silently emitting a buildx
    cmd with no output produced no usable image."""
    runner = MagicMock()  # never called

    with pytest.raises(ValueError, match="single --platforms"):
        build_target(
            kestrel_target,
            project_id="test-project",
            tag="latest",
            platforms=("linux/amd64", "linux/arm64"),
            push=False,
            multi_arch=True,
            project_root=tmp_path,
            runner=runner,
        )

    runner.assert_not_called()


def test_build_target_no_multi_arch_legacy_flow(kestrel_target, tmp_path):
    """--no-multi-arch uses ``docker build`` + ``docker push`` per ref —
    matches build_multi_agent.sh."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    result = build_target(
        kestrel_target,
        project_id="test-project",
        tag="v1.2.3",
        push=True,
        multi_arch=False,
        github_token=None,
        project_root=tmp_path,
        runner=runner,
    )

    # Three runner calls: one build, two pushes (one per tag).
    assert runner.call_count == 3
    build_cmd = runner.call_args_list[0].args[0]
    assert build_cmd[0:2] == ["docker", "build"]
    assert "buildx" not in build_cmd
    # No --secret in plain mode.
    assert "--secret" not in build_cmd
    # No --platform in plain mode.
    assert "--platform" not in build_cmd

    push_cmds = [c.args[0] for c in runner.call_args_list[1:]]
    assert push_cmds == [
        ["docker", "push", "gcr.io/test-project/kestrel:v1.2.3"],
        ["docker", "push", "gcr.io/test-project/kestrel:latest"],
    ]
    assert result.pushed is True


def test_build_target_no_multi_arch_with_token_passes_secret(kestrel_target, tmp_path):
    """Codex review on PR #1060: plain (``--no-multi-arch``) builds
    with a resolved GITHUB_TOKEN must still pass ``--secret`` to
    docker — the cloudrun + multi_agent Dockerfiles use BuildKit
    secrets and would otherwise see an empty token. ``DOCKER_BUILDKIT=1``
    is also set in the subprocess env so older docker installs use
    BuildKit.
    """
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    build_target(
        kestrel_target,
        project_id="test-project",
        tag="latest",
        push=False,
        multi_arch=False,
        github_token="token-from-test",
        project_root=tmp_path,
        runner=runner,
    )

    cmd = runner.call_args.args[0]
    assert cmd[0:2] == ["docker", "build"]
    assert "--secret" in cmd
    secret_idx = cmd.index("--secret")
    assert cmd[secret_idx + 1] == "id=github_token,env=GITHUB_TOKEN"
    # GITHUB_TOKEN must be in the subprocess env.
    env = runner.call_args.kwargs.get("env") or {}
    assert env.get("GITHUB_TOKEN") == "token-from-test"
    assert env.get("DOCKER_BUILDKIT") == "1"


def test_build_target_no_multi_arch_no_push(kestrel_target, tmp_path):
    """Plain mode + --no-push: only the build call, no push calls."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    build_target(
        kestrel_target,
        project_id="test-project",
        tag="latest",
        push=False,
        multi_arch=False,
        project_root=tmp_path,
        runner=runner,
    )

    assert runner.call_count == 1
    cmd = runner.call_args_list[0].args[0]
    assert cmd[0:2] == ["docker", "build"]


def test_build_target_no_token_omits_secret_flag(kestrel_target, tmp_path):
    """github_token=None: --secret flag is NOT added (would break builds
    that don't set GITHUB_TOKEN in their environment)."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    build_target(
        kestrel_target,
        project_id="test-project",
        tag="latest",
        github_token=None,
        project_root=tmp_path,
        runner=runner,
    )

    cmd = runner.call_args.args[0]
    assert "--secret" not in cmd
    # And the env override is not applied either.
    env = runner.call_args.kwargs["env"]
    assert env is None


def test_build_target_default_project_root_is_cwd(kestrel_target, monkeypatch, tmp_path):
    """project_root=None → Path.cwd() (matches deploy_config.toml /
    secrets.py CWD-relative convention)."""
    monkeypatch.chdir(tmp_path)
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    build_target(
        kestrel_target,
        project_id="test-project",
        tag="latest",
        runner=runner,
    )

    cmd = runner.call_args.args[0]
    # Build context arg = cwd.
    assert cmd[-1] == str(tmp_path)


def test_build_target_raises_build_error_on_buildx_failure(kestrel_target, tmp_path):
    """docker buildx exit non-zero → BuildError with command + stderr."""
    runner = MagicMock(
        return_value=MagicMock(returncode=1, stderr="missing docker daemon")
    )

    with pytest.raises(BuildError) as exc_info:
        build_target(
            kestrel_target,
            project_id="test-project",
            tag="v1.0.0",
            project_root=tmp_path,
            runner=runner,
        )

    err = exc_info.value
    assert err.image_ref == "gcr.io/test-project/kestrel:v1.0.0"
    assert err.command[0:3] == ["docker", "buildx", "build"]
    assert "missing docker daemon" in err.stderr


def test_build_target_raises_build_error_on_push_failure(kestrel_target, tmp_path):
    """In legacy mode, a failed ``docker push`` after a successful build
    surfaces as BuildError with the push command and ref attached."""
    # First call (build) succeeds; second call (first push) fails.
    runner = MagicMock(side_effect=[
        MagicMock(returncode=0, stderr=""),
        MagicMock(returncode=1, stderr="auth required"),
    ])

    with pytest.raises(BuildError) as exc_info:
        build_target(
            kestrel_target,
            project_id="test-project",
            tag="v1.0.0",
            push=True,
            multi_arch=False,
            project_root=tmp_path,
            runner=runner,
        )

    err = exc_info.value
    assert err.image_ref == "gcr.io/test-project/kestrel:v1.0.0"
    assert err.command[0:2] == ["docker", "push"]
    assert "auth required" in err.stderr


# ---------------------------------------------------------------------------
# build_all — multi-target orchestration
# ---------------------------------------------------------------------------

def test_build_all_iterates_default_targets(tmp_path):
    """No explicit targets → DEFAULT_TARGETS → two builds, in order."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    results = build_all(
        project_id="test-project",
        tag="latest",
        project_root=tmp_path,
        runner=runner,
    )

    assert len(results) == 2
    assert results[0].target.image_name == "kestrel"
    assert results[1].target.image_name == "kestrel-multi_agent"

    # Two buildx calls (one per target), since multi_arch defaults True.
    assert runner.call_count == 2


def test_build_all_respects_explicit_targets(tmp_path):
    """Explicit targets list → only those built."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    only_main = [DEFAULT_TARGETS[0]]
    results = build_all(
        project_id="test-project",
        tag="latest",
        targets=only_main,
        project_root=tmp_path,
        runner=runner,
    )

    assert len(results) == 1
    assert results[0].target.image_name == "kestrel"
    assert runner.call_count == 1


def test_build_all_propagates_buildx_args(tmp_path):
    """Platforms / push / multi_arch / token / tag flow into build_target."""
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    build_all(
        project_id="test-project",
        tag="v1.2.3",
        targets=[DEFAULT_TARGETS[0]],
        platforms=("linux/amd64",),
        push=False,
        multi_arch=True,
        github_token="ghp_test",
        project_root=tmp_path,
        runner=runner,
    )

    cmd = runner.call_args.args[0]
    plat_idx = cmd.index("--platform")
    assert cmd[plat_idx + 1] == "linux/amd64"
    assert "--push" not in cmd
    assert "--secret" in cmd
    tags = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-t"]
    assert tags == [
        "gcr.io/test-project/kestrel:v1.2.3",
        "gcr.io/test-project/kestrel:latest",
    ]


def test_build_all_first_failing_target_raises_with_partial_results(tmp_path):
    """First target succeeds, second fails: BuildError carries the
    partial success in ``partial_results`` so the CLI can render it."""
    # First target's build call → ok; second target's build call → fail.
    runner = MagicMock(side_effect=[
        MagicMock(returncode=0, stderr=""),    # kestrel buildx ok
        MagicMock(returncode=1, stderr="oops"),  # kestrel-multi_agent buildx fails
    ])

    with pytest.raises(BuildError) as exc_info:
        build_all(
            project_id="test-project",
            tag="latest",
            project_root=tmp_path,
            runner=runner,
        )

    err = exc_info.value
    assert err.image_ref == "gcr.io/test-project/kestrel-multi_agent:latest"
    assert len(err.partial_results) == 1
    assert err.partial_results[0].target.image_name == "kestrel"
    # The successful build's refs are preserved.
    assert err.partial_results[0].image_refs == [
        "gcr.io/test-project/kestrel:latest",
    ]


def test_build_all_never_invokes_real_docker(tmp_path):
    """Sanity: with the runner mocked, the test process never shells out.

    This is the explicit invariant the test plan calls for. Verified by
    ensuring ``subprocess.run`` is never called — only the injected
    runner is.
    """
    runner = MagicMock(return_value=MagicMock(returncode=0, stderr=""))

    with patch("subprocess.run") as real_run:
        results = build_all(
            project_id="test-project",
            tag="latest",
            project_root=tmp_path,
            runner=runner,
        )

    assert len(results) == 2
    real_run.assert_not_called()
