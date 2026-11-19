"""Worker build pins must follow the actual Core version on this branch."""
from pathlib import Path
import re
import tomllib

import pytest
from packaging.markers import Marker


@pytest.mark.parametrize("suffix", [".in", ".txt"])
def test_worker_build_pin_matches_core_project_version(suffix):
    root = Path(__file__).parents[3]
    version = tomllib.loads((root/"pyproject.toml").read_text())["project"]["version"]
    requirements = root/"docker/visual-browser"/("requirements-two-domain"+suffix)
    pins = re.findall(r"^kestrel-sovereign==([^\s;\\]+)", requirements.read_text(), re.MULTILINE)
    assert pins == [version], "worker build input and generated lock must use this Core project version"


@pytest.mark.parametrize("reverse_order", [False, True])
def test_worker_shared_dependencies_follow_core_tested_lock(reverse_order):
    root = Path(__file__).parents[3]
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = list(reversed(lock["package"])) if reverse_order else lock["package"]
    versions = _locked_versions(packages, python_full_version="3.12.3")
    worker = dict(re.findall(r"^([\w.-]+)==([^\s;\\]+)",
        (root / "docker/visual-browser/requirements-two-domain.txt").read_text(), re.MULTILINE))
    # The pinned browser image owns its Playwright/pyee pair. Other shared
    # dependencies must match the versions tested by Core CI.
    drift = {name: (version, versions[name]) for name, version in worker.items()
        if name in versions and name not in {"playwright", "pyee"} and version != versions[name]}
    assert not drift, f"Worker dependency versions differ from Core's tested lock: {drift}"


def _locked_versions(packages, *, python_full_version):
    environment = {
        "python_full_version": python_full_version,
        "python_version": ".".join(python_full_version.split(".")[:2]),
        "implementation_name": "cpython", "implementation_version": python_full_version,
        "os_name": "posix", "sys_platform": "linux", "platform_machine": "aarch64",
        "platform_system": "Linux", "platform_python_implementation": "CPython",
        "platform_release": "", "platform_version": "",
    }
    versions = {}
    for package in packages:
        markers = package.get("resolution-markers")
        if "version" not in package or markers and not any(
            Marker(marker).evaluate(environment=environment) for marker in markers
        ):
            continue
        name, version = package["name"], package["version"]
        assert name not in versions or versions[name] == version, f"Ambiguous worker lock entry: {name}"
        versions[name] = version
    return versions

