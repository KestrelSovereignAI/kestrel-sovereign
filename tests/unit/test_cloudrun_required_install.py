"""Required image installation failures must reach Docker's RUN exit status."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Executes the Linux image's POSIX installation shell"
)


def _probe_wheel(directory: Path, module: str) -> Path:
    """Make a valid, dependency-free wheel installed by the real uv binary."""
    dist_info = f"{module}-0.0.0.dist-info"
    files = {
        f"{module}/__init__.py": b"",
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.3\nName: {module.replace('_', '-')}\n"
            "Version: 0.0.0\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ).encode(),
    }
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow((name, "sha256=" + digest.decode(), len(data)))
    writer.writerow((f"{dist_info}/RECORD", "", ""))
    files[f"{dist_info}/RECORD"] = record.getvalue().encode()
    wheel = directory / f"{module}-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


def _installation_command() -> str:
    """Use the actual RUN program; substitute only owned dependency targets."""
    dockerfile = Path(__file__).resolve().parents[2] / "docker/Dockerfile.multi_agent"
    text = dockerfile.read_text()
    command = text.split("RUN --mount=type=secret,id=github_token \\\n", 1)[1]
    command = command.split("\n\n", 1)[0].replace("\\\n", " ")
    command = command.replace("/app/.venv/bin/python", '"$INSTALL_TEST_PYTHON"')
    command = command.replace("/app/.venv", '"$INSTALL_TEST_VENV"')
    command = command.replace("/run/secrets/github_token", '"$INSTALL_TEST_TOKEN_FILE"')
    core = 'uv pip install --python "$INSTALL_TEST_PYTHON" .'
    feature = (
        '"kestrel-feature-github-app @ '
        'git+https://github.com/KestrelSovereignAI/kestrel-feature-github-app.git"'
    )
    assert command.count(core) == command.count(feature) == 1
    return command.replace(core, core[:-1] + '"$INSTALL_TEST_CORE"').replace(
        feature, '"$INSTALL_TEST_FEATURE"'
    )


def _run_install(tmp_path: Path, *, failure: str | None, authenticated: bool):
    assert shutil.which("uv") is not None, "The image installation contract requires uv"
    shell = shutil.which("sh")
    assert shell is not None, "The Linux image installation contract requires a POSIX shell"
    venv = tmp_path / "image venv"
    core = _probe_wheel(tmp_path, "docker_core_probe")
    feature = _probe_wheel(tmp_path, "docker_feature_probe")
    token_file = tmp_path / "build-secret"
    token_file.write_text("synthetic-build-token" if authenticated else "")
    gitconfig = tmp_path / "gitconfig"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "GIT_ASKPASS"}
        and not key.startswith("GIT_CONFIG_")
    }
    env.update(
        INSTALL_TEST_VENV=str(venv),
        INSTALL_TEST_PYTHON=str(venv / "bin/python"),
        INSTALL_TEST_CORE=str(core),
        INSTALL_TEST_FEATURE=str(feature),
        INSTALL_TEST_TOKEN_FILE=str(token_file),
        UV_PYTHON=sys.executable,
        UV_PYTHON_DOWNLOADS="never",
        UV_CACHE_DIR=str(tmp_path / "uv-cache"),
        GIT_CONFIG_GLOBAL=str(gitconfig),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
    )
    if failure == "venv":
        env["UV_PYTHON"] = str(tmp_path / "missing-python")
    elif failure == "core":
        env["INSTALL_TEST_CORE"] = str(tmp_path / "missing_core-0.0.0-py3-none-any.whl")
    elif failure == "feature":
        env["INSTALL_TEST_FEATURE"] = "git+" + (tmp_path / "missing-feature-repository").as_uri()
    result = subprocess.run(
        [shell, "-c", _installation_command()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, venv, gitconfig


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("failure", ["venv", "core", "feature"])
def test_required_installation_failure_fails_the_image(tmp_path, failure, authenticated):
    result, _, _ = _run_install(tmp_path, failure=failure, authenticated=authenticated)
    assert result.returncode != 0, (
        f"Required {failure} installation failed but Docker RUN reported success:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    expected_error = {"venv": "missing-python", "core": "missing_core", "feature": "missing-feature-repository"}
    assert expected_error[failure] in result.stderr


@pytest.mark.parametrize("authenticated", [False, True])
def test_successful_image_installs_both_packages_and_clears_build_auth(tmp_path, authenticated):
    result, venv, gitconfig = _run_install(tmp_path, failure=None, authenticated=authenticated)
    assert result.returncode == 0, result.stdout + result.stderr
    installed = subprocess.run(
        [str(venv / "bin/python"), "-I", "-c", "import docker_core_probe, docker_feature_probe"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert installed.returncode == 0, installed.stderr
    assert not gitconfig.exists() or "synthetic-build-token" not in gitconfig.read_text()
