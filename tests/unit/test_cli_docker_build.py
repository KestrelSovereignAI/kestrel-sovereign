"""``kestrel docker build <preset>`` CLI tests — sub-PR 4 of epic
#1050 (bash-to-Python port of ``scripts/docker/build_*.sh``).

Covers:
- argparse wiring under the local subparser and the real ``kestrel``
  parser (where ``build`` shares the ``kestrel docker`` parent with
  ``remote``)
- ``--list`` prints the preset table without invoking gcloud
- ``GCP_PROJECT_ID`` validation
- Unknown preset → exit code 1 listing valid presets
- Each known preset shells out to ``gcloud builds submit`` with the
  right Cloud Build config
- ``--no-cache`` is propagated into the synthesized cloudbuild yaml
- ``simpletuner`` uses the vendored ``docker/cloudbuild-simpletuner.yaml``
- Missing Dockerfile preflight blocks the build cleanly
- Exit code is the ``gcloud builds submit`` exit code on failure
- Streams subprocess output (no ``capture_output=True``)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli_docker_build


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    cli_docker_build.add_docker_build_subcommand(sub)
    return p


def test_argparse_build_no_preset():
    parser = _build_parser()
    # no preset is fine when --list is used; bare invocation is
    # caught at runtime.
    args = parser.parse_args(["docker", "build"])
    assert args.command == "docker"
    assert args.docker_command == "build"
    assert args.preset is None
    assert args.tag == "latest"
    assert args.no_cache is False
    assert args.list is False


def test_argparse_build_preset():
    parser = _build_parser()
    args = parser.parse_args(["docker", "build", "gpu-cloud"])
    assert args.preset == "gpu-cloud"


def test_argparse_build_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "docker", "build", "lora-trainer",
        "--tag", "v9", "--no-cache",
    ])
    assert args.preset == "lora-trainer"
    assert args.tag == "v9"
    assert args.no_cache is True


def test_argparse_build_list_flag():
    parser = _build_parser()
    args = parser.parse_args(["docker", "build", "--list"])
    assert args.list is True
    assert args.preset is None


def test_argparse_build_invalid_preset_rejected():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["docker", "build", "totally-fake"])


def test_kestrel_cli_registers_docker_build():
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["docker", "build", "ollama-server"])
    assert args.command == "docker"
    assert args.docker_command == "build"
    assert args.preset == "ollama-server"


def test_kestrel_cli_docker_remote_still_works():
    """Sanity: introducing ``kestrel docker build`` must not break
    the existing ``kestrel docker remote {build,run}`` surface."""
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["docker", "remote", "build"])
    assert args.docker_command == "remote"
    assert args.docker_remote_command == "build"


# ---------------------------------------------------------------------------
# --list flag
# ---------------------------------------------------------------------------

def test_list_prints_table_without_calling_gcloud(monkeypatch, capsys):
    called = {"n": 0}
    monkeypatch.setattr(
        cli_docker_build, "run_streaming",
        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or 0,
    )
    rc = cli_docker_build.cmd_docker_build(_Args(
        list=True, preset=None, tag="latest", no_cache=False,
    ))
    assert rc == 0
    assert called["n"] == 0
    out = capsys.readouterr().out
    # Every preset is mentioned.
    for name in ("gpu-cloud", "lora-trainer", "ollama-server", "simpletuner"):
        assert name in out
    # The shared-image config-drift note.
    assert "kestrel-lora" in out


# ---------------------------------------------------------------------------
# Missing GCP_PROJECT_ID
# ---------------------------------------------------------------------------

def test_missing_gcp_project_errors(monkeypatch, capsys):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset="gpu-cloud", tag="latest", no_cache=False,
    ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "GCP_PROJECT_ID" in err


# ---------------------------------------------------------------------------
# Unknown preset
# ---------------------------------------------------------------------------

def test_no_preset_no_list_errors(monkeypatch, capsys):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset=None, tag="latest", no_cache=False,
    ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing preset" in err


# ---------------------------------------------------------------------------
# Each known preset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset_name,image,dockerfile", [
    ("gpu-cloud", "kestrel-gpu", "docker/Dockerfile.gpu"),
    ("lora-trainer", "kestrel-lora", "docker/Dockerfile.lora-trainer"),
    ("ollama-server", "kestrel-ollama", "docker/Dockerfile.ollama-server"),
])
def test_each_preset_invokes_gcloud_builds_submit(
    monkeypatch, tmp_path, preset_name, image, dockerfile,
):
    """For every synthesized-yaml preset, ``gcloud builds submit
    --config=<yaml> --project=<id> .`` is invoked and the synthesized
    yaml references the right image + dockerfile."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []
    yaml_seen: dict = {}

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        captured.append(list(cmd))
        # The first arg after gcloud builds submit is --config=<path>;
        # peek at the yaml contents before the caller deletes it.
        for a in cmd:
            if a.startswith("--config="):
                p = Path(a.split("=", 1)[1])
                if p.is_file():
                    yaml_seen["text"] = p.read_text(encoding="utf-8")
        return 0

    monkeypatch.setattr(cli_docker_build, "run_streaming", fake_run)
    # Ensure the dockerfile preflight passes — point repo root at a
    # tmp dir we populate.
    (tmp_path / "docker").mkdir()
    (tmp_path / dockerfile).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / dockerfile).write_text("FROM alpine\n")
    monkeypatch.setattr(cli_docker_build, "_repo_root", lambda: tmp_path)

    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset=preset_name, tag="v2", no_cache=False,
    ))
    assert rc == 0
    assert len(captured) == 1
    argv = captured[0]
    assert argv[0:3] == ["gcloud", "builds", "submit"]
    assert any(a.startswith("--config=") for a in argv)
    assert "--project=test-proj" in argv
    assert argv[-1] == "."

    yaml = yaml_seen.get("text", "")
    assert image in yaml
    assert dockerfile in yaml
    assert "v2" in yaml


def test_no_cache_propagates_into_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    yaml_seen: dict = {}

    def fake_run(cmd, *, cwd=None, env=None, check=False):
        for a in cmd:
            if a.startswith("--config="):
                p = Path(a.split("=", 1)[1])
                if p.is_file():
                    yaml_seen["text"] = p.read_text(encoding="utf-8")
        return 0

    monkeypatch.setattr(cli_docker_build, "run_streaming", fake_run)
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/Dockerfile.gpu").write_text("FROM alpine\n")
    monkeypatch.setattr(cli_docker_build, "_repo_root", lambda: tmp_path)

    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset="gpu-cloud", tag="latest", no_cache=True,
    ))
    assert rc == 0
    assert "--no-cache" in yaml_seen.get("text", "")


def test_simpletuner_uses_vendored_yaml(monkeypatch, tmp_path):
    """``simpletuner`` is the one preset that points at a vendored
    cloudbuild yaml on disk (``docker/cloudbuild-simpletuner.yaml``).
    The CLI must use it as-is rather than synthesizing one."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    captured: list = []
    monkeypatch.setattr(
        cli_docker_build, "run_streaming",
        lambda cmd, **kw: captured.append(list(cmd)) or 0,
    )
    # Provide the vendored yaml + the dockerfile preflight.
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/Dockerfile.simpletuner").write_text("FROM alpine\n")
    (tmp_path / "docker/cloudbuild-simpletuner.yaml").write_text(
        "steps: []\n"
    )
    monkeypatch.setattr(cli_docker_build, "_repo_root", lambda: tmp_path)

    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset="simpletuner", tag="latest", no_cache=False,
    ))
    assert rc == 0
    argv = captured[0]
    config = next(a for a in argv if a.startswith("--config="))
    config_path = config.split("=", 1)[1]
    # Vendored yaml lives at <repo>/docker/cloudbuild-simpletuner.yaml.
    assert config_path.endswith("cloudbuild-simpletuner.yaml")
    assert str(tmp_path) in config_path


def test_simpletuner_missing_vendored_yaml_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    monkeypatch.setattr(
        cli_docker_build, "run_streaming",
        lambda cmd, **kw: 0,
    )
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/Dockerfile.simpletuner").write_text("FROM alpine\n")
    # No cloudbuild-simpletuner.yaml on disk.
    monkeypatch.setattr(cli_docker_build, "_repo_root", lambda: tmp_path)

    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset="simpletuner", tag="latest", no_cache=False,
    ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "cloudbuild-simpletuner.yaml" in err


def test_missing_dockerfile_preflight(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    monkeypatch.setattr(
        cli_docker_build, "run_streaming",
        lambda cmd, **kw: 0,
    )
    # Empty repo root — no Dockerfile present.
    (tmp_path / "docker").mkdir()
    monkeypatch.setattr(cli_docker_build, "_repo_root", lambda: tmp_path)

    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset="gpu-cloud", tag="latest", no_cache=False,
    ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Dockerfile.gpu" in err


def test_gcloud_failure_returns_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-proj")
    monkeypatch.setattr(
        cli_docker_build, "run_streaming",
        lambda cmd, **kw: 7,
    )
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker/Dockerfile.gpu").write_text("FROM alpine\n")
    monkeypatch.setattr(cli_docker_build, "_repo_root", lambda: tmp_path)

    rc = cli_docker_build.cmd_docker_build(_Args(
        list=False, preset="gpu-cloud", tag="latest", no_cache=False,
    ))
    assert rc == 7


# ---------------------------------------------------------------------------
# Preset registry surface
# ---------------------------------------------------------------------------

def test_preset_registry_has_four_entries():
    assert set(cli_docker_build._PRESETS.keys()) == {
        "gpu-cloud", "lora-trainer", "ollama-server", "simpletuner",
    }


def test_simpletuner_and_lora_share_image():
    """Pre-existing config drift carry-over from the bash predecessors.
    Documented in the module docstring; this test pins the behavior so
    a future cleanup is a deliberate decision, not an accidental rename.
    """
    assert (
        cli_docker_build._PRESETS["simpletuner"].image
        == cli_docker_build._PRESETS["lora-trainer"].image
        == "kestrel-lora"
    )


# ---------------------------------------------------------------------------
# Streaming subprocess — no capture
# ---------------------------------------------------------------------------

def test_module_uses_shared_streaming_helper():
    from kestrel_sovereign import _subprocess_helpers as sh
    assert cli_docker_build.run_streaming is sh.run_streaming
