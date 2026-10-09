"""The lock's lines against the real ``uv`` resolver, offline (#3502).

``test_core_lock_install.py`` drives every install surface against a fake venv
whose resolver models what uv does with the guard's constraint lines. These
check that model against uv itself: hand-made wheels on a ``--find-links``
directory, ``--offline --no-index``, an editable checkout whose build backend
requires nothing, and a throwaway venv. Nothing is fetched. Each install goes
through ``CoreInstallGuard.run``, so the constraints file uv reads is the one
the guard writes.

They need a ``uv`` that runs here, and :func:`uv_runs` checks for one by
creating a venv the way every test does. A host without one skips them with
the reason in the skip message AND a warning in the run's summary, so a
green run that never reached uv cannot pass for one that did. Under CI (``CI``
set, as GitHub Actions sets it) the same finding FAILS instead: these are the
only tests of the guard's lines against the resolver they are written for.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from typing import Optional

import pytest

from kestrel_sovereign import cli, paths
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.cli_features import CoreInstallGuard
from tests.utils.fixture_wheels import write_checkout, write_wheel


_LOCK = """\
version = 1
revision = 3

[[package]]
name = "kestrel-sovereign"
version = "0.52.0"
source = {{ editable = "." }}

[[package]]
name = "anthropic"
version = "0.117.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "depx"
version = "{depx}"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "localy"
version = "1.0"
source = {{ registry = "https://pypi.org/simple" }}
"""

_STATE = """\
import importlib.metadata as md, json
state = {}
for dist in md.distributions():
    url = dist.read_text("direct_url.json")
    state[dist.metadata["Name"].lower()] = [dist.version, json.loads(url) if url else None]
print(json.dumps(state))
"""


class Host:
    """A venv, an offline index, and a core checkout whose uv.lock holds both."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.wheels = tmp_path / "wheels"
        self.wheels.mkdir()
        write_wheel(self.wheels, "anthropic", "0.117.0")
        write_wheel(self.wheels, "anthropic", "1.11.0")
        write_wheel(self.wheels, "depx", "1.1")
        # A locked version, and a local build of it the lock never named.
        write_wheel(self.wheels, "localy", "1.0")
        write_wheel(self.wheels, "localy", "1.0+cu1")
        # Unrelated features, each pressing on a locked package.
        write_wheel(self.wheels, "needs-anthropic-1", "1.0", requires=["anthropic>=1"])
        write_wheel(self.wheels, "needs-depx-1-1", "1.0", requires=["depx>=1.1"])
        write_wheel(self.wheels, "takes-any-depx", "1.0", requires=["depx>=1.0"])
        write_wheel(self.wheels, "takes-any-localy", "1.0", requires=["localy>=1.0"])
        self.depx_checkout = write_checkout(tmp_path / "depx-checkout", "depx", "1.0")

        self.core = tmp_path / "core"
        self.core.mkdir()
        # The lock pins depx at 0.9, which nothing publishes: a declared
        # checkout is held on what it builds, not at the locked version.
        (self.core / "uv.lock").write_text(_LOCK.format(depx="0.9"), encoding="utf-8")

        venv = tmp_path / "venv"
        subprocess.run(
            ["uv", "venv", "-q", "--offline", "--python", sys.executable, str(venv)],
            check=True, capture_output=True, text=True,
        )
        self.python = next(
            path for path in (venv / "bin" / "python", venv / "Scripts" / "python.exe")
            if path.exists()
        )
        # The guard installs into the interpreter it runs under, and reads the
        # lock beside the checkout core is installed from.
        monkeypatch.setattr(sys, "executable", str(self.python))
        monkeypatch.setattr(
            cli, "_core_install_shape",
            lambda: fr.CoreInstallShape(
                version="0.52.0",
                provenance=fr.Provenance.direct(str(self.core), editable=True),
            ),
        )

    def offline(self, *pip_args) -> list:
        return ["--offline", "--no-index", "--find-links", str(self.wheels), *pip_args]

    def setup(self, *pip_args) -> None:
        """Put the venv in a starting state, outside any guard."""
        result = CoreInstallGuard.unguarded().run(self.offline(*pip_args))
        assert result.returncode == 0, result.stderr

    def state(self) -> dict:
        out = subprocess.run(
            [str(self.python), "-c", _STATE], check=True, capture_output=True, text=True,
        )
        return json.loads(out.stdout)


def _uv_unavailable(directory) -> Optional[str]:
    """Why ``uv`` cannot run these tests here, or None when it can.

    Runs the one uv command every test starts with, so "uv is on PATH" is not
    mistaken for "uv works": a sandbox that refuses uv's system calls makes it
    panic (exit 101) on that very command.
    """
    uv = shutil.which("uv")
    if uv is None:
        return "uv is not on PATH"
    try:
        probe = subprocess.run(
            [uv, "venv", "-q", "--offline", "--python", sys.executable,
             str(directory / "venv")],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`uv venv` could not run: {exc}"
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip()[-400:]
        return f"`uv venv` exited {probe.returncode}: {detail or 'no output'}"
    return None


@pytest.fixture(scope="module")
def uv_runs(tmp_path_factory):
    """Skip loudly, or fail under CI, when uv cannot run these tests here."""
    reason = _uv_unavailable(tmp_path_factory.mktemp("uv-probe"))
    if reason is None:
        return
    message = f"offline uv resolver tests did not run: {reason}"
    if os.environ.get("CI"):
        pytest.fail(
            f"{message} (CI must run them: they are the only check of the "
            "install guard's constraint lines against uv itself, #3502)",
            pytrace=False,
        )
    warnings.warn(message, stacklevel=1)
    pytest.skip(message)


@pytest.fixture
def host(uv_runs, tmp_path, monkeypatch):
    # A guard given no manifest reads the project directory's host manifest,
    # which on a dev host is that host's real one.
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    return Host(tmp_path, monkeypatch)


def _editable(entry) -> bool:
    return bool(entry[1] and entry[1].get("dir_info", {}).get("editable"))


def test_a_url_installed_locked_package_is_not_moved_by_a_feature(host):
    """The ruling's case on the real resolver: no ``--upgrade``, no manifest.

    Anthropic is installed from a URL at the locked 0.117.0, and a feature
    needs ``anthropic>=1``. The guard pins it whatever its source, so uv
    refuses; unheld, uv replaces it with 1.11.0.
    """
    url = (host.wheels / "anthropic-0.117.0-py3-none-any.whl").as_uri()
    host.setup(f"anthropic @ {url}")

    result = CoreInstallGuard.snapshot().run(host.offline("needs-anthropic-1"))

    assert result.returncode != 0
    assert "No solution found" in result.stderr  # a refusal, not a bad argv
    after = host.state()
    assert after["anthropic"][0] == "0.117.0"
    assert after["anthropic"][1]["url"] == url
    assert "needs-anthropic-1" not in after

    control = CoreInstallGuard.unguarded().run(host.offline("needs-anthropic-1"))

    assert control.returncode == 0
    assert host.state()["anthropic"] == ["1.11.0", None]


def _declared_depx(host):
    return CoreInstallGuard.snapshot(
        {"depx": fr.SourceEntry(package="depx", editable=str(host.depx_checkout))}
    )


def test_a_declared_checkout_is_never_replaced_from_the_index(host):
    """A feature needs a depx its declared checkout does not build: refused."""
    host.setup("-e", str(host.depx_checkout))

    result = _declared_depx(host).run(host.offline("needs-depx-1-1"))

    assert result.returncode != 0
    assert "No solution found" in result.stderr  # a refusal, not a bad argv
    assert host.state()["depx"][0] == "1.0"
    assert _editable(host.state()["depx"])

    control = CoreInstallGuard.unguarded().run(host.offline("needs-depx-1-1"))

    assert control.returncode == 0
    assert host.state()["depx"] == ["1.1", None]


def test_an_upgrade_keeps_a_declared_checkout_linked(host):
    """uv's eager ``--upgrade`` takes the newest depx unless the hold keeps the link."""
    host.setup("-e", str(host.depx_checkout))

    result = _declared_depx(host).run(host.offline("--upgrade", "takes-any-depx"))

    assert result.returncode == 0, result.stderr
    after = host.state()
    assert after["takes-any-depx"][0] == "1.0"
    assert after["depx"][0] == "1.0"
    assert _editable(after["depx"])

    control = CoreInstallGuard.unguarded().run(
        host.offline("--upgrade", "takes-any-depx"),
    )

    assert control.returncode == 0
    assert host.state()["depx"] == ["1.1", None]


def test_a_local_build_cannot_stand_in_for_the_locked_version(host):
    """``==1.0`` would admit ``1.0+cu1``, and uv's ``--upgrade`` takes it.

    The lock pins ``localy`` 1.0 and the index also offers the local build
    ``1.0+cu1``. The guard's ``===`` pin keeps 1.0; unheld, the upgrade moves
    to the local build, a version the lock never named.
    """
    host.setup("localy===1.0")

    result = CoreInstallGuard.snapshot().run(host.offline("--upgrade", "takes-any-localy"))

    assert result.returncode == 0, result.stderr
    assert host.state()["localy"][0] == "1.0"

    control = CoreInstallGuard.unguarded().run(
        host.offline("--upgrade", "takes-any-localy"),
    )

    assert control.returncode == 0
    assert host.state()["localy"][0] == "1.0+cu1"


def test_prefer_pypi_moves_a_declared_checkout_to_its_locked_version(host):
    """``kestrel update --prefer-pypi`` on a package the manifest declares editable.

    The lock pins depx at 1.1, and its declared checkout builds 1.0. Reconcile
    plans an index install of depx, reinstalling only depx. Held to the
    manifest's own declaration, uv keeps depx on the checkout the plan was
    replacing (#3502). Held to the declarations the preference produces, the
    switch lands the locked version from the index.
    """
    (host.core / "uv.lock").write_text(_LOCK.format(depx="1.1"), encoding="utf-8")
    host.setup("-e", str(host.depx_checkout))
    manifest = {"depx": fr.SourceEntry(package="depx", editable=str(host.depx_checkout))}
    switch = host.offline("--upgrade", "depx")

    contradicted = CoreInstallGuard.snapshot(manifest).run(switch, reinstall="depx")

    assert contradicted.returncode == 0, contradicted.stderr
    assert host.state()["depx"][0] == "1.0"
    assert _editable(host.state()["depx"])

    preferred = fr.preferred_source_index(
        manifest, ["depx"], {"depx": str(host.depx_checkout)}, "pypi",
    )
    result = CoreInstallGuard.snapshot(preferred).run(switch, reinstall="depx")

    assert result.returncode == 0, result.stderr
    assert host.state()["depx"] == ["1.1", None]


def test_the_project_directorys_manifest_holds_a_checkout_from_any_cwd(
    host, tmp_path, monkeypatch,
):
    """``feature install`` takes no manifest, so the guard reads the host's.

    That is the project directory's (``KESTREL_HOME`` here), whatever the cwd,
    and its relative ``editable`` names the checkout beside it. So uv keeps
    depx linked through an eager ``--upgrade``. With no manifest there, depx
    is pinned at its locked 0.9, which nothing publishes, and uv refuses: the
    declaration, found where the host keeps it, is what holds the link.
    """
    host.setup("-e", str(host.depx_checkout))
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    upgrade = host.offline("--upgrade", "takes-any-depx")
    manifest = tmp_path / ".kestrel-host-features.toml"

    unheld = CoreInstallGuard.snapshot().run(upgrade)

    assert unheld.returncode != 0
    assert "No solution found" in unheld.stderr
    assert _editable(host.state()["depx"])

    manifest.write_text(
        f'[[feature]]\nname = "depx"\neditable = "{host.depx_checkout.name}"\n',
        encoding="utf-8",
    )
    result = CoreInstallGuard.snapshot().run(upgrade)

    assert result.returncode == 0, result.stderr
    after = host.state()
    assert after["takes-any-depx"][0] == "1.0"
    assert after["depx"][0] == "1.0"
    assert _editable(after["depx"])
