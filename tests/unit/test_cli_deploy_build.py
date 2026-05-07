"""CLI tests for ``kestrel deploy build``.

Sub-PR 1.3 of epic #1050. Exercises argparse wiring and the dispatch
into :mod:`kestrel_sovereign.features.deploy.build`. The build module
itself is mocked here — its own behaviour is covered in
``tests/unit/test_deploy_build.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.cli_deploy import (
    add_deploy_subcommands,
    cmd_deploy,
)
from kestrel_sovereign.features.deploy.build import (
    BuildError,
    BuildResult,
    BuildTarget,
    DEFAULT_TARGETS,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    add_deploy_subcommands(sub)
    return p


def _make_args(**overrides):
    """Argparse-like namespace with all build-relevant defaults."""
    base = {
        "target": None,
        "profile": None,
        "tag": "latest",
        "lines": 100,
        "json": False,
        # secrets sync flags (inert for build, but argparse populates them)
        "secrets_profile": None,
        "env_file": None,
        "dry_run": False,
        # build flags
        "build_target": None,
        "no_push": False,
        "no_multi_arch": False,
        "build_platforms": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def fake_project_root(tmp_path, monkeypatch):
    """``chdir`` into ``tmp_path`` so deploy_config.toml resolution lands
    somewhere harmless. CLI helpers resolve relative to CWD (matches
    ``kestrel_sovereign.config.load_config`` and the agent !deploy path).
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_minimal_deploy_config(project_root: Path) -> None:
    """Minimal deploy_config.toml that satisfies the build path's
    project-ID resolution."""
    (project_root / "deploy_config.toml").write_text(
        '[manager]\n'
        'gcp_project_id = "test-project"\n'
    )


def _ok_build_result(image_name: str, *, pushed: bool = True) -> BuildResult:
    target = next(
        (t for t in DEFAULT_TARGETS if t.image_name == image_name),
        BuildTarget(image_name, Path("docker/Dockerfile.cloudrun"), "test"),
    )
    return BuildResult(
        target=target,
        image_refs=[f"gcr.io/test-project/{image_name}:latest"],
        pushed=pushed,
        duration_seconds=12.5,
    )


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def test_argparse_build_basic():
    """``kestrel deploy build`` parses with all defaults."""
    parser = _build_parser()
    args = parser.parse_args(["deploy", "build"])

    assert args.command == "deploy"
    assert args.target == "build"
    assert args.tag == "latest"
    assert args.build_target is None
    assert args.no_push is False
    assert args.no_multi_arch is False
    assert args.build_platforms is None


def test_argparse_build_all_flags():
    """All documented build flags parse together."""
    parser = _build_parser()
    args = parser.parse_args([
        "deploy", "build",
        "--tag", "v1.2.3",
        "--target", "kestrel",
        "--no-push",
        "--no-multi-arch",
        "--platforms", "linux/amd64",
        "--json",
    ])

    assert args.target == "build"
    assert args.tag == "v1.2.3"
    assert args.build_target == "kestrel"
    assert args.no_push is True
    assert args.no_multi_arch is True
    assert args.build_platforms == "linux/amd64"
    assert args.json is True


def test_argparse_build_help_runs(capsys):
    """``kestrel deploy --help`` mentions the build flags (smoke).

    argparse exits with SystemExit(0) on --help, so we catch it.
    """
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["deploy", "--help"])

    captured = capsys.readouterr()
    # Long flag names should be present in the help.
    assert "--no-push" in captured.out
    assert "--no-multi-arch" in captured.out
    assert "--platforms" in captured.out


# ---------------------------------------------------------------------------
# Dispatcher: prerequisites
# ---------------------------------------------------------------------------

def test_cmd_deploy_build_missing_project_id(fake_project_root, monkeypatch, capsys):
    """No GCP_PROJECT_ID env + no deploy_config.toml → friendly error."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    rc = cmd_deploy(_make_args(target="build"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "deploy_config.toml not found" in captured.err


def test_cmd_deploy_build_uses_env_project_id(fake_project_root, monkeypatch):
    """GCP_PROJECT_ID env wins outright — no deploy_config.toml needed.

    Matches build.sh exactly (it never reads any config). The
    Python-side fallback to deploy_config.toml is additive.
    """
    monkeypatch.setenv("GCP_PROJECT_ID", "env-project")
    # No deploy_config.toml in the fake_project_root — should still work.

    fake_results = [_ok_build_result("kestrel"), _ok_build_result("kestrel-multi_agent")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ) as mock_build:
        rc = cmd_deploy(_make_args(target="build"))

    assert rc == 0
    assert mock_build.call_args.kwargs["project_id"] == "env-project"


def test_cmd_deploy_build_falls_back_to_config_project_id(fake_project_root, monkeypatch):
    """No env var → deploy_config.toml's [manager].gcp_project_id."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    _write_minimal_deploy_config(fake_project_root)

    fake_results = [_ok_build_result("kestrel"), _ok_build_result("kestrel-multi_agent")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ) as mock_build:
        rc = cmd_deploy(_make_args(target="build"))

    assert rc == 0
    assert mock_build.call_args.kwargs["project_id"] == "test-project"


# ---------------------------------------------------------------------------
# Dispatcher: full build paths (mocked build_all)
# ---------------------------------------------------------------------------

def test_cmd_deploy_build_happy_path_pretty_output(fake_project_root, monkeypatch, capsys):
    """Happy path: per-target line + summary footer, exit 0."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel"), _ok_build_result("kestrel-multi_agent")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ):
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value="ghp_test",
        ):
            rc = cmd_deploy(_make_args(target="build", tag="v1.2.3"))

    captured = capsys.readouterr()
    assert rc == 0
    # Pre-build banner lines.
    assert "building kestrel:" in captured.out
    assert "building kestrel-multi_agent:" in captured.out
    # Per-target completion lines.
    assert "built kestrel" in captured.out
    assert "built kestrel-multi_agent" in captured.out
    assert "(pushed)" in captured.out
    # Summary footer.
    assert "2 built, 0 errors" in captured.out


def test_cmd_deploy_build_target_filter(fake_project_root, monkeypatch):
    """``--target kestrel`` narrows DEFAULT_TARGETS to one image."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ) as mock_build:
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            rc = cmd_deploy(_make_args(target="build", build_target="kestrel"))

    assert rc == 0
    targets = mock_build.call_args.kwargs["targets"]
    assert len(targets) == 1
    assert targets[0].image_name == "kestrel"


def test_cmd_deploy_build_target_filter_unknown_errors(fake_project_root, monkeypatch, capsys):
    """``--target bogus`` → exit 1 with available-targets hint."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    rc = cmd_deploy(_make_args(target="build", build_target="bogus"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown build target 'bogus'" in captured.err
    # Both default target names listed for the operator.
    assert "kestrel" in captured.err
    assert "kestrel-multi_agent" in captured.err


def test_cmd_deploy_build_no_push_propagates(fake_project_root, monkeypatch, capsys):
    """``--no-push`` flows into build_all(push=False) and the rendered
    line says ``(local only)``.

    Codex review on PR #1060 v2: ``--no-push`` must default to a single
    platform (buildx --load doesn't support multi-arch); without
    ``--platforms`` the CLI now picks ``linux/amd64`` so the advertised
    ``--no-push`` workflow works without extra flags.
    """
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel", pushed=False)]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ) as mock_build:
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            rc = cmd_deploy(_make_args(
                target="build",
                build_target="kestrel",
                no_push=True,
            ))

    captured = capsys.readouterr()
    assert rc == 0
    assert mock_build.call_args.kwargs["push"] is False
    # Single-platform default for --no-push (multi-arch buildx --load
    # only supports single-platform).
    assert mock_build.call_args.kwargs["platforms"] == ("linux/amd64",)
    assert "(local only)" in captured.out


def test_cmd_deploy_build_no_multi_arch_propagates(fake_project_root, monkeypatch):
    """``--no-multi-arch`` flows into build_all(multi_arch=False)."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ) as mock_build:
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            cmd_deploy(_make_args(
                target="build",
                build_target="kestrel",
                no_multi_arch=True,
            ))

    assert mock_build.call_args.kwargs["multi_arch"] is False


def test_cmd_deploy_build_platforms_propagates(fake_project_root, monkeypatch):
    """``--platforms`` parses to a tuple; default falls back to amd64+arm64."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ) as mock_build:
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            cmd_deploy(_make_args(
                target="build",
                build_target="kestrel",
                build_platforms="linux/amd64,linux/arm64,linux/arm/v7",
            ))

    platforms = mock_build.call_args.kwargs["platforms"]
    assert platforms == ("linux/amd64", "linux/arm64", "linux/arm/v7")


def test_cmd_deploy_build_warns_when_no_token(fake_project_root, monkeypatch, capsys):
    """No GITHUB_TOKEN → warning printed but build still runs (matches
    bash script)."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ):
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            rc = cmd_deploy(_make_args(
                target="build", build_target="kestrel",
            ))

    captured = capsys.readouterr()
    assert rc == 0
    assert "WARNING: No GITHUB_TOKEN" in captured.err


def test_cmd_deploy_build_error_surfaces_partial_results(fake_project_root, monkeypatch, capsys):
    """BuildError from build_all → exit 1, stderr printed, partial
    successes rendered so the operator sees what got built."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    partial = _ok_build_result("kestrel")
    err = BuildError(
        image_ref="gcr.io/test-project/kestrel-multi_agent:latest",
        command=["docker", "buildx", "build", "..."],
        stderr="failed to fetch base image: 503",
        partial_results=[partial],
    )

    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        side_effect=err,
    ):
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value="ghp_test",
        ):
            rc = cmd_deploy(_make_args(target="build"))

    captured = capsys.readouterr()
    assert rc == 1
    # Partial success rendered.
    assert "built kestrel" in captured.out
    # Error message + stderr both visible.
    assert "docker build failed for gcr.io/test-project/kestrel-multi_agent:latest" in captured.err
    assert "failed to fetch base image" in captured.err


def test_cmd_deploy_build_docker_not_installed_friendly_error(
    fake_project_root, monkeypatch, capsys
):
    """If docker isn't on PATH, ``_default_runner`` raises FileNotFoundError
    inside build_all. The CLI must surface this as a clean message, not
    a Python traceback."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        side_effect=FileNotFoundError("[Errno 2] No such file or directory: 'docker'"),
    ):
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            rc = cmd_deploy(_make_args(target="build"))

    captured = capsys.readouterr()
    assert rc == 1
    assert "build failed" in captured.err
    assert "docker" in captured.err


def test_cmd_deploy_build_json_output_happy_path(fake_project_root, monkeypatch, capsys):
    """``--json`` emits a structured payload (success/results array)."""
    import json as _json
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    fake_results = [_ok_build_result("kestrel"), _ok_build_result("kestrel-multi_agent")]
    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        return_value=fake_results,
    ):
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value="ghp_test",
        ):
            rc = cmd_deploy(_make_args(target="build", tag="v1.2.3", json=True))

    captured = capsys.readouterr()
    assert rc == 0
    parsed = _json.loads(captured.out)
    assert parsed["success"] is True
    assert parsed["project_id"] == "test-project"
    assert parsed["tag"] == "v1.2.3"
    assert len(parsed["results"]) == 2
    names = [r["image_name"] for r in parsed["results"]]
    assert names == ["kestrel", "kestrel-multi_agent"]
    assert parsed["results"][0]["pushed"] is True


def test_cmd_deploy_build_json_output_error_path(fake_project_root, monkeypatch, capsys):
    """``--json`` on BuildError emits ``success=False`` + error block +
    partial results."""
    import json as _json
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    partial = _ok_build_result("kestrel")
    err = BuildError(
        image_ref="gcr.io/test-project/kestrel-multi_agent:latest",
        command=["docker", "buildx", "build", "..."],
        stderr="boom",
        partial_results=[partial],
    )

    with patch(
        "kestrel_sovereign.features.deploy.build.build_all",
        side_effect=err,
    ):
        with patch(
            "kestrel_sovereign.features.deploy.build.resolve_github_token",
            return_value=None,
        ):
            rc = cmd_deploy(_make_args(target="build", json=True))

    captured = capsys.readouterr()
    assert rc == 1
    parsed = _json.loads(captured.out)
    assert parsed["success"] is False
    assert parsed["error"]["image_ref"] == "gcr.io/test-project/kestrel-multi_agent:latest"
    assert parsed["error"]["stderr"] == "boom"
    # Partial successful result preserved in the array.
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["image_name"] == "kestrel"


def test_cmd_deploy_build_empty_platforms_errors(fake_project_root, monkeypatch, capsys):
    """``--platforms ,,,`` (all-empty) → friendly error rather than
    invoking buildx with an empty platform list."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    rc = cmd_deploy(_make_args(
        target="build",
        build_platforms=",,,",
    ))

    captured = capsys.readouterr()
    assert rc == 1
    assert "non-empty" in captured.err
