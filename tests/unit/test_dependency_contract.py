"""Security contracts for declared dependencies and the resolved lock.

A dependency that ships in a published extra is attack surface even when
nothing imports it: ``chromadb==1.5.9`` sat unused in the ``local`` extra
(propagated into ``all-features``/``full``) while carrying an unpatched
pre-authentication code-injection CVE (CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c).

These tests pin two contracts: known-vulnerable, unused packages stay out of
the declared and resolved graphs, and patched dependency floors cannot be
silently lowered while refreshing ``uv.lock``.
"""

from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages that must never appear in the declared or resolved dependency
#: graph. Add an entry (with the issue reference) when a package is removed
#: for being unused-and-vulnerable; remove it only with an explicit,
#: reviewed decision to reintroduce the dependency.
BANNED_PACKAGES = {
    "chromadb": "#2547 — unused; unpatched critical CVE-2026-45829",
    "langchain": "#2546 — unused; family carries open high-severity advisories",
    "langchain-community": "#2546 — unused; pulls langsmith/langchain-classic alerts",
    "langsmith": "#2546 — transitive of unused langchain-community; open advisories",
    "torchaudio": "#2546 — unused; latest release pins a vulnerable Torch line",
}

#: Lowest patched versions for direct dependencies that closed the final
#: Dependabot release block in #2546. Every declaration and lock entry for
#: these packages must remain at or above the corresponding floor.
SECURITY_FLOORS = {
    "aiohttp": "3.14.1",
    "diffusers": "0.38.0",
    "httplib2": "0.32.0",
    "pypdf": "6.13.3",
    "pytest": "9.0.3",
    "requests": "2.33.0",
    "setuptools": "83.0.0",
    "torch": "2.13.0",
    "transformers": "5.5.0",
    "web3": "7.15.0",
}


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _declared_dependency_lines(pyproject: dict):
    project = pyproject["project"]
    yield from project.get("dependencies", [])
    for extra, deps in project.get("optional-dependencies", {}).items():
        for dep in deps:
            yield f"[{extra}] {dep}"
    for group, deps in pyproject.get("dependency-groups", {}).items():
        for dep in deps:
            if isinstance(dep, str):
                yield f"[dependency-group:{group}] {dep}"


def _declared_requirements(pyproject: dict):
    project = pyproject["project"]
    yield from project.get("dependencies", [])
    for deps in project.get("optional-dependencies", {}).values():
        yield from deps
    for deps in pyproject.get("dependency-groups", {}).values():
        yield from (dep for dep in deps if isinstance(dep, str))


def test_banned_packages_not_declared():
    pyproject = _pyproject()
    offenders = [
        line
        for line in _declared_dependency_lines(pyproject)
        for banned in BANNED_PACKAGES
        if banned in line.lower()
    ]
    assert not offenders, (
        f"Banned package declared in pyproject.toml: {offenders}. "
        f"Reasons: {BANNED_PACKAGES}"
    )


def test_banned_packages_not_locked():
    with open(REPO_ROOT / "uv.lock", "rb") as f:
        lock = tomllib.load(f)
    locked = {pkg["name"].lower() for pkg in lock.get("package", [])}
    offenders = sorted(locked & {b.lower() for b in BANNED_PACKAGES})
    assert not offenders, (
        f"Banned package present in uv.lock: {offenders}. "
        f"Reasons: {BANNED_PACKAGES}"
    )


def test_security_floors_are_declared():
    requirements = [Requirement(raw) for raw in _declared_requirements(_pyproject())]

    for package, floor_text in SECURITY_FLOORS.items():
        floor = Version(floor_text)
        declarations = [
            requirement
            for requirement in requirements
            if canonicalize_name(requirement.name) == canonicalize_name(package)
        ]
        assert declarations, f"Missing security-floored dependency: {package}"

        for requirement in declarations:
            lower_bounds = [
                Version(specifier.version)
                for specifier in requirement.specifier
                if specifier.operator in {">=", "==", "==="}
                and "*" not in specifier.version
            ]
            assert lower_bounds and max(lower_bounds) >= floor, (
                f"{requirement} permits a version below the security floor "
                f"{package}>={floor}"
            )


def test_security_floors_are_locked():
    with open(REPO_ROOT / "uv.lock", "rb") as f:
        lock = tomllib.load(f)

    locked = {}
    for package in lock.get("package", []):
        locked.setdefault(canonicalize_name(package["name"]), []).append(
            Version(package["version"])
        )

    for package, floor_text in SECURITY_FLOORS.items():
        versions = locked.get(canonicalize_name(package), [])
        floor = Version(floor_text)
        assert versions, f"Security-floored dependency missing from uv.lock: {package}"
        assert all(version >= floor for version in versions), (
            f"uv.lock contains {package} below {floor}: {versions}"
        )
