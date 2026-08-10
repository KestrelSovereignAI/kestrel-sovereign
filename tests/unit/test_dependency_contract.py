"""Security contracts for declared dependencies and the resolved lock.

A dependency that ships in a published extra is attack surface even when
nothing imports it: ``chromadb==1.5.9`` sat unused in the ``local`` extra
(propagated into ``all-features``/``full``) while carrying an unpatched
pre-authentication code-injection CVE (CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c).

These tests pin two contracts: known-vulnerable, unused packages stay out of
the declared and resolved graphs, and patched dependency floors cannot be
silently lowered while refreshing ``uv.lock``.
"""

import tomllib
from pathlib import Path

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

# Private inference leases require the 0.35 owner-scoped idle-renewal and
# absolute lifetime-bound contracts, and #2755 adds typed, capability-negotiated
# private host ingress. This is intentionally a Core-only release gate: sibling
# packages are released from their own repositories, but Core must never
# silently lower its declared/locked line to accommodate an older Frinz or
# observability constraint. Their compatible releases remain a documented
# release-cascade prerequisite in README.md.
SDK_RELEASE_CASCADE_SPECIFIERS = frozenset({(">=", "0.35.1"), ("<", "0.36")})
SDK_RELEASE_CASCADE_CONTRACTS = {
    "base": frozenset({"tracing"}),
    "observability": frozenset({"metrics", "tracing"}),
}
SDK_RELEASE_CASCADE_DOWNSTREAM_REQUIREMENTS = {
    # These are release prerequisites, not declarations about sibling repos'
    # current branches. Each downstream must publish/test this line before a
    # Core release can be cut.
    "frinz": ">=0.35.1,<0.36",
    "observability fleet": ">=0.35.1,<0.36",
}


def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _lock() -> dict:
    with open(REPO_ROOT / "uv.lock", "rb") as f:
        return tomllib.load(f)


def _locked_root_package(lock: dict) -> dict:
    return next(
        package
        for package in lock["package"]
        if package["name"] == "kestrel-sovereign"
        and package.get("source") == {"editable": "."}
    )


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
    lock = _lock()
    locked = {pkg["name"].lower() for pkg in lock.get("package", [])}
    offenders = sorted(locked & {b.lower() for b in BANNED_PACKAGES})
    assert not offenders, (
        f"Banned package present in uv.lock: {offenders}. Reasons: {BANNED_PACKAGES}"
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
    lock = _lock()

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


def test_windows_tzdata_is_a_direct_base_dependency_and_is_locked():
    """Windows IANA scheduling cannot depend on optional Pandas/Phoenix trees."""

    direct = [
        Requirement(raw)
        for raw in _pyproject()["project"]["dependencies"]
        if canonicalize_name(Requirement(raw).name) == "tzdata"
    ]
    assert len(direct) == 1
    marker = direct[0].marker
    assert marker is not None
    assert marker.evaluate({"sys_platform": "win32"})
    assert not marker.evaluate({"sys_platform": "linux"})

    root = _locked_root_package(_lock())
    locked_direct = [
        dependency
        for dependency in root["dependencies"]
        if dependency["name"] == "tzdata"
    ]
    assert locked_direct == [{"name": "tzdata", "marker": "sys_platform == 'win32'"}]

    locked_metadata = [
        requirement
        for requirement in root["metadata"]["requires-dist"]
        if requirement["name"] == "tzdata"
    ]
    assert locked_metadata == locked_direct
    assert any(package["name"] == "tzdata" for package in _lock()["package"])


def _sdk_contract_requirement(raw_requirements, *, extras):
    requirements = [
        Requirement(raw)
        for raw in raw_requirements
        if canonicalize_name(Requirement(raw).name)
        == canonicalize_name("kestrel-sovereign-sdk")
    ]
    assert len(requirements) == 1
    requirement = requirements[0]
    assert requirement.extras == extras
    assert {
        (specifier.operator, specifier.version) for specifier in requirement.specifier
    } == SDK_RELEASE_CASCADE_SPECIFIERS
    return requirement


def test_sdk_035_release_cascade_contract_is_declared():
    """Core and its observability extra must declare the v0.35 SDK line.

    This deliberately does not inspect sibling worktrees: their compatible
    Frinz/observability releases are an external release prerequisite, while
    this repository can reliably guard only its own published constraints.
    """

    pyproject = _pyproject()
    _sdk_contract_requirement(
        pyproject["project"]["dependencies"],
        extras=SDK_RELEASE_CASCADE_CONTRACTS["base"],
    )
    _sdk_contract_requirement(
        pyproject["project"]["optional-dependencies"]["observability"],
        extras=SDK_RELEASE_CASCADE_CONTRACTS["observability"],
    )

    # The human release contract identifies downstream gates without probing
    # their (possibly dirty or unavailable) repositories from Core CI.
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8").casefold()
    for downstream, specifier in SDK_RELEASE_CASCADE_DOWNSTREAM_REQUIREMENTS.items():
        assert downstream in readme
        assert f"kestrel-sovereign-sdk{specifier}" in readme


def test_sdk_035_release_cascade_contract_is_locked():
    """The resolved lock must carry the same v0.35 line before Core ships."""

    root = _locked_root_package(_lock())
    locked_contracts = {
        (
            frozenset(requirement.get("extras", [])),
            requirement.get("marker"),
            requirement.get("specifier"),
        )
        for requirement in root["metadata"]["requires-dist"]
        if requirement["name"] == "kestrel-sovereign-sdk"
    }
    assert locked_contracts == {
        (SDK_RELEASE_CASCADE_CONTRACTS["base"], None, ">=0.35.1,<0.36"),
        (
            SDK_RELEASE_CASCADE_CONTRACTS["observability"],
            "extra == 'observability'",
            ">=0.35.1,<0.36",
        ),
    }

    sdk_versions = [
        Version(package["version"])
        for package in _lock()["package"]
        if package["name"] == "kestrel-sovereign-sdk"
    ]
    assert sdk_versions
    assert all(
        Version("0.35.1") <= version < Version("0.36.0")
        for version in sdk_versions
    )
