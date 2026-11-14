"""Security contracts for declared dependencies and the resolved lock.

A dependency that ships in a published extra is attack surface even when
nothing imports it: ``chromadb==1.5.9`` sat unused in the ``local`` extra
(propagated into ``all-features``/``full``) while carrying an unpatched
pre-authentication code-injection CVE (CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c).

These tests pin three contracts: known-vulnerable, unused packages stay out
of the declared and resolved graphs; patched dependency floors cannot be
silently lowered while refreshing ``uv.lock`` or ``package-lock.json``; and
the browser libraries the console loads from a CDN ship the locked, patched
version rather than an older pin the npm lock never touches.
"""

import json
import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
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
#: Dependabot release block in #2546 (aiohttp and pypdf raised in #3771).
#: Every declaration and lock entry for these packages must remain at or
#: above the corresponding floor.
SECURITY_FLOORS = {
    "aiohttp": "3.14.4",
    "diffusers": "0.38.0",
    "httplib2": "0.32.0",
    "pypdf": "6.21.4",
    "pytest": "9.0.3",
    "requests": "2.33.0",
    # GHSA-jhr6-gm9c-rqjv (#3718). The advisory names 5.6.0, which only warns
    # before importing a local model directory's custom code; 6.0 refuses it.
    "sentence-transformers": "6.0.0",
    "setuptools": "83.0.0",
    "torch": "2.13.0",
    "transformers": "5.5.0",
    "web3": "7.15.0",
}

#: Lowest patched versions for undeclared transitive dependencies (#3771).
#: Only ``uv.lock`` carries these fixes, so a lock refresh must not resolve
#: back below them. Remove an entry only when the package leaves the graph.
TRANSITIVE_SECURITY_FLOORS = {
    # huggingface-hub and torch (ml/local extras)
    "fsspec": "2026.6.0",
    # requests-oauthlib via msrest (cloud extra)
    "oauthlib": "4.0.0",
    # google-auth (base) and ldap3 (observability extra)
    "pyasn1": "0.6.5",
}

#: Lowest patched versions in ``package-lock.json`` (#3771). The npm lock
#: feeds only the frontend tests; ``CONSOLE_CDN_PACKAGES`` covers what ships.
NPM_SECURITY_FLOORS = {
    "dompurify": "3.4.17",
    # jsdom (frontend test DOM)
    "undici": "7.30.1",
    # css-tree via jsdom (frontend test DOM)
    "source-map-js": "1.2.2",
}

#: npm packages the shipped console loads from a CDN URL in ``index.html``
#: rather than from ``package-lock.json``. A lock bump alone never reaches a
#: browser: #3771 raised the lock to DOMPurify 3.4.17 while the console still
#: loaded 3.4.9. Each must be pinned to the exact locked version, so the
#: frontend tests exercise what ships and its npm floor covers the console.
CONSOLE_CDN_PACKAGES = frozenset({"dompurify"})

CONSOLE_HTML = REPO_ROOT / "kestrel_sovereign" / "static" / "index.html"
_JSDELIVR_NPM_SCRIPT = re.compile(
    r"^https://cdn\.jsdelivr\.net/npm/"
    r"(?P<name>(?:@[^/@]+/)?[^/@]+)@(?P<version>[^/]+)/"
)

# Private inference leases require the owner-scoped idle-renewal and absolute
# lifetime-bound contracts, #2755 adds typed private host ingress, 0.36 adds
# the public operator contribution contracts, 0.37.1 adds lifecycle-owned
# context-clause contributions, 0.38.1 adds awaited preparation before
# synchronous context rendering, and 0.39.0 adds the decisions contract (LLM
# contract v7, #3424) that ``LLMService.decide`` is built on. Version 0.39.1
# adds the shared outbound policy and native pinned transports; 0.39.2 fixes
# their canonical host identity before DNS and native requests (#3679).
# Version 0.40 replaces draft A2A types with native AgentSkill contracts.
# The coordinated 0.41 floor includes the shared computer-use, artifact,
# resource-scope and workspace-transfer contracts (#3679).
# Version 0.42 adds the single vendored native A2A implementation shared with
# Feature.get_agent_skills; Core must consume its a2a extra.
# Version 0.42.1 enforces Major.Minor version negotiation and exposes the
# canonical version parser used by Core's signed Agent Card validation.
# Version 0.42.2 enforces all standard HTTP header and query carriers.
# Version 0.42.3 corrects native HTTP+JSON protocol error media types.
# Version 0.42.4 owns the certifi floor for HTTPX/requests; OS roots are unaffected.
# Version 0.43 adds model exposure and governed registration bindings.
# Version 0.45 adds the authenticated human-control and acquisition-custody
# contracts. Version 0.46 adds the host-selected deferred-publication context.
# Version 0.48 adds public argument coercion and the host control-error predicate.
# Version 0.49 adds native completion evidence and the streaming EOF contract.
# Version 0.51 requires explicit intent IDs on governed bindings and preserves
# original uncertain tool exceptions. Core must require that exact contract.
# A private development wheel does not satisfy the normal lock gate.
# This is a Core-only release gate: sibling packages release independently.
# Core must never lower its declared/locked line to accommodate an older
# consumer. Host feature acceptance remains a release prerequisite in README.md.
# Version 0.53.1 includes explicit source lifetime and protocol/HTTP terminal
# refusals consumed by Core's native request and authentication adapter.
SDK_RELEASE_CASCADE_SPECIFIERS = frozenset({(">=", "0.53.1"), ("<", "1")})
SDK_RELEASE_CASCADE_CONTRACTS = {
    "base": frozenset({"a2a", "httpx", "tracing"}),
    "observability": frozenset({"metrics"}),
}

def _pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _lock() -> dict:
    with open(REPO_ROOT / "uv.lock", "rb") as f:
        return tomllib.load(f)


def _assert_requirement_floor(requirement: Requirement, floor: Version) -> None:
    lower_bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">=", "==", "==="}
        and "*" not in specifier.version
    ]
    assert lower_bounds and max(lower_bounds) >= floor, (
        f"{requirement} permits a version below the required floor "
        f"{requirement.name}>={floor}"
    )


def test_installed_remote_mcp_campaign_requires_the_public_feature_contract():
    """Public MCP must retain native intent, refusal and unknown-outcome receipts."""
    dependencies = _pyproject()["project"]["optional-dependencies"]["mcp-integration"]
    (requirement,) = [
        Requirement(raw)
        for raw in dependencies
        if Requirement(raw).name == "kestrel-feature-mcp"
    ]
    floor = Version("0.17.0")
    _assert_requirement_floor(requirement, floor)
    assert not requirement.specifier.contains("0.11.2")
    assert not requirement.specifier.contains("0.12.0")
    assert not requirement.specifier.contains("0.13.0")
    assert not requirement.specifier.contains("0.14.1")
    assert not requirement.specifier.contains("0.15.0")
    assert not requirement.specifier.contains("0.16.0")
    assert requirement.specifier.contains(floor)
    (package,) = [
        item for item in _lock()["package"] if item["name"] == "kestrel-feature-mcp"
    ]
    assert Version(package["version"]) >= floor
    assert package["source"] == {"registry": "https://pypi.org/simple"}


def test_native_a2a_streams_require_server_scoped_shutdown():
    """MUST then full TCK boots must not share a latched SSE shutdown flag."""
    (requirement,) = [
        Requirement(raw)
        for raw in _pyproject()["project"]["dependencies"]
        if canonicalize_name(Requirement(raw).name) == "sse-starlette"
    ]
    _assert_requirement_floor(requirement, Version("3.5.0"))
    assert Version("3.5.0") in requirement.specifier
    assert Version("3.6.0") in requirement.specifier
    (package,) = [
        item for item in _lock()["package"] if item["name"] == "sse-starlette"
    ]
    assert Version(package["version"]) >= Version("3.5.0")
    assert package["source"] == {"registry": "https://pypi.org/simple"}


def test_shared_sdk_owns_the_current_https_trust_bundle_policy():
    """Core must not pin an obsolete CA bundle against its shared HTTPS SDK."""
    declared = [Requirement(raw) for raw in _pyproject()["project"]["dependencies"]]
    assert not any(canonicalize_name(item.name) == "certifi" for item in declared)
    packages = {item["name"]: item for item in _lock()["package"]}
    sdk = packages["kestrel-sovereign-sdk"]
    assert any(item["name"] == "certifi" for item in sdk["optional-dependencies"]["a2a"])
    assert Version(packages["certifi"]["version"]) >= Version("2026.10.23")
    assert packages["certifi"]["source"] == {"registry": "https://pypi.org/simple"}


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
            _assert_requirement_floor(requirement, floor)


def _assert_locked_floors(floors: dict[str, str]) -> None:
    locked = {}
    for package in _lock().get("package", []):
        locked.setdefault(canonicalize_name(package["name"]), []).append(
            Version(package["version"])
        )

    for package, floor_text in floors.items():
        versions = locked.get(canonicalize_name(package), [])
        floor = Version(floor_text)
        assert versions, f"Security-floored dependency missing from uv.lock: {package}"
        assert all(version >= floor for version in versions), (
            f"uv.lock contains {package} below {floor}: {versions}"
        )


def test_security_floors_are_locked():
    _assert_locked_floors(SECURITY_FLOORS)


def test_transitive_security_floors_are_locked():
    _assert_locked_floors(TRANSITIVE_SECURITY_FLOORS)


def _npm_locked_versions(names) -> dict[str, list[str]]:
    with open(REPO_ROOT / "package-lock.json", encoding="utf-8") as f:
        packages = json.load(f)["packages"]
    locked = {}
    for path, entry in packages.items():
        name = path.rsplit("node_modules/", 1)[-1]
        if path and name in names:
            locked.setdefault(name, []).append(entry["version"])
    return locked


class _ScriptSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag, attrs):
        src = dict(attrs).get("src")
        if tag == "script" and src:
            self.sources.append(src)


def _console_script_sources() -> list[str]:
    parser = _ScriptSources()
    parser.feed(CONSOLE_HTML.read_text(encoding="utf-8"))
    return parser.sources


def test_npm_security_floors_are_locked():
    locked = _npm_locked_versions(NPM_SECURITY_FLOORS)

    for package, floor_text in NPM_SECURITY_FLOORS.items():
        versions = locked.get(package, [])
        floor = Version(floor_text)
        assert versions, (
            f"Security-floored npm package missing from package-lock.json: {package}"
        )
        assert all(Version(version) >= floor for version in versions), (
            f"package-lock.json contains {package} below {floor}: {versions}"
        )


def test_console_cdn_scripts_ship_the_locked_patched_version():
    """The console's CDN pin is the version the lock and its floor vouch for."""
    sources = _console_script_sources()
    locked = _npm_locked_versions(CONSOLE_CDN_PACKAGES)

    for package in sorted(CONSOLE_CDN_PACKAGES):
        loads = [src for src in sources if package in src.lower()]
        assert len(loads) == 1, (
            f"index.html must load {package} from exactly one script: {loads}"
        )
        match = _JSDELIVR_NPM_SCRIPT.match(loads[0])
        assert match and match["name"] == package, (
            f"index.html loads {package} from an unrecognised URL: {loads[0]}"
        )
        assert locked.get(package) == [match["version"]], (
            f"index.html pins {package}@{match['version']} but package-lock.json "
            f"resolves {locked.get(package)}; the frontend tests run the locked "
            f"version, so the console must ship exactly that version"
        )
        floor = Version(NPM_SECURITY_FLOORS[package])
        assert Version(match["version"]) >= floor, (
            f"index.html ships {package}@{match['version']}, below {floor}"
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


def test_anyio_requires_native_cancellation_preserving_connection_completion():
    """Every consumer must receive the real shutdown fix, including installs."""
    direct = [
        Requirement(raw)
        for raw in _pyproject()["project"]["dependencies"]
        if canonicalize_name(Requirement(raw).name) == "anyio"
    ]
    assert len(direct) == 1
    _assert_requirement_floor(direct[0], Version("4.16.0"))
    assert Version("4.15.1") not in direct[0].specifier
    assert Version("4.16.0") in direct[0].specifier
    assert Version("4.17.0") in direct[0].specifier
    root = _locked_root_package(_lock())
    assert [
        requirement
        for requirement in root["metadata"]["requires-dist"]
        if requirement["name"] == "anyio"
    ] == [{"name": "anyio", "specifier": ">=4.16.0"}]
    _assert_locked_floors({"anyio": "4.16.0"})


def test_asyncpg_runtime_probe_supports_current_minor_releases():
    """Require native retirement fixes without capping future minor releases."""
    direct = [
        Requirement(raw)
        for raw in _pyproject()["project"]["dependencies"]
        if canonicalize_name(Requirement(raw).name) == "asyncpg"
    ]
    assert len(direct) == 1
    assert Version("0.31") not in direct[0].specifier
    assert Version("0.32") in direct[0].specifier
    assert Version("0.33") in direct[0].specifier

    root = _locked_root_package(_lock())
    locked_metadata = [
        requirement
        for requirement in root["metadata"]["requires-dist"]
        if requirement["name"] == "asyncpg"
    ]
    assert locked_metadata == [{"name": "asyncpg", "specifier": ">=0.32.0"}]


def _sdk_contract_requirement(
    raw_requirements, *, extras, specifiers=SDK_RELEASE_CASCADE_SPECIFIERS,
):
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
    } == specifiers
    return requirement


def test_sdk_native_release_cascade_contract_is_declared():
    """Core owns the transport floor; observability adds metrics to that floor.

    This deliberately does not inspect sibling worktrees. Installed host
    feature acceptance remains a separate release prerequisite; this
    repository guards its own published constraints.
    """

    pyproject = _pyproject()
    base = _sdk_contract_requirement(
        pyproject["project"]["dependencies"],
        extras=SDK_RELEASE_CASCADE_CONTRACTS["base"],
    )
    observability = _sdk_contract_requirement(
        pyproject["project"]["optional-dependencies"]["observability"],
        extras=SDK_RELEASE_CASCADE_CONTRACTS["observability"],
        specifiers=frozenset(),
    )
    for specifier in (base.specifier, base.specifier & observability.specifier):
        assert Version("0.38.2") not in specifier
        assert Version("0.39.0") not in specifier
        assert Version("0.39.1") not in specifier
        assert Version("0.39.2") not in specifier
        assert Version("0.40.0") not in specifier
        assert Version("0.41.0") not in specifier
        assert Version("0.42.0") not in specifier
        assert Version("0.42.1") not in specifier
        assert Version("0.42.2") not in specifier
        assert Version("0.42.3") not in specifier
        assert Version("0.42.4") not in specifier
        assert Version("0.43.0") not in specifier
        assert Version("0.44.0") not in specifier
        assert Version("0.44.1") not in specifier
        assert Version("0.45.0") not in specifier
        assert Version("0.46.0") not in specifier
        assert Version("0.47.0") not in specifier
        assert Version("0.48.0") not in specifier
        assert Version("0.49.0") not in specifier
        assert Version("0.50.0") not in specifier
        assert Version("0.51.0") not in specifier
        assert Version("0.51.1") not in specifier
        assert Version("0.51.2") not in specifier
        assert Version("0.52.0") not in specifier
        assert Version("0.52.1") not in specifier
        assert Version("0.52.2") not in specifier
        assert Version("0.53.0") not in specifier
        assert Version("0.53.1") in specifier
        assert Version("1.0.0") not in specifier


def test_sdk_native_release_cascade_contract_is_locked():
    """The normal lock must exercise the declared publication-context SDK floor."""

    root = _locked_root_package(_lock())
    locked_contracts = {
        (
            frozenset(requirement.get("extras", [])),
            requirement.get("marker"),
            frozenset(
                (specifier.operator, specifier.version)
                for specifier in SpecifierSet(requirement.get("specifier", ""))
            ),
        )
        for requirement in root["metadata"]["requires-dist"]
        if requirement["name"] == "kestrel-sovereign-sdk"
    }
    assert locked_contracts == {
        (SDK_RELEASE_CASCADE_CONTRACTS["base"], None, SDK_RELEASE_CASCADE_SPECIFIERS),
        (
            SDK_RELEASE_CASCADE_CONTRACTS["observability"],
            "extra == 'observability'",
            frozenset(),
        ),
    }

    sdk_versions = [
        Version(package["version"])
        for package in _lock()["package"]
        if package["name"] == "kestrel-sovereign-sdk"
    ]
    assert sdk_versions
    assert all(
        Version("0.52.2") <= version < Version("1.0.0") for version in sdk_versions
    )


def test_native_provider_completion_release_is_required():
    dependencies = {
        canonicalize_name(requirement.name): requirement
        for requirement in map(Requirement, _pyproject()["project"]["dependencies"])
    }
    providers = dependencies["kestrel-llms"]
    assert providers.extras == {"all"}
    _assert_requirement_floor(providers, Version("0.2.0"))
    assert Version("0.1.8") not in providers.specifier
    assert Version("0.2.0") in providers.specifier
    assert Version("0.2.99") in providers.specifier
    assert Version("0.3.0") not in providers.specifier
    for name, floor in (("openai", "3.46.0"), ("google-genai", "2.34.0")):
        _assert_requirement_floor(dependencies[name], Version(floor))
