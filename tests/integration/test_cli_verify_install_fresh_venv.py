"""Live subprocess regression for the verify-install package installer (#3020)."""

from __future__ import annotations

import textwrap

import pytest

from kestrel_sovereign.cli_verify_install import (
    _make_venv,
    _pip_install,
    _python_check,
    _venv_exec,
)


pytestmark = pytest.mark.integration


def test_pip_install_succeeds_in_fresh_uv_venv_without_seeded_pip(tmp_path):
    """The verifier reaches an installer in the exact venv shape it creates."""

    project = tmp_path / "local-package"
    package = project / "src" / "verify_install_probe"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [project]
            name = "kestrel-verify-install-probe"
            version = "0.0.0"

            [tool.hatch.build.targets.wheel]
            packages = ["src/verify_install_probe"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "INSTALLED_BY_VERIFY = True\n",
        encoding="utf-8",
    )

    venv = tmp_path / "fresh" / ".venv"
    assert _make_venv(venv)
    assert not _venv_exec(venv, "pip").exists(), (
        "the regression fixture must retain uv's unseeded venv shape"
    )
    assert _pip_install(venv, "--no-deps", str(project))
    assert _python_check(
        venv,
        "import verify_install_probe; "
        "assert verify_install_probe.INSTALLED_BY_VERIFY is True",
    )
