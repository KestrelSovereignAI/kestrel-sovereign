"""Dependency-contract assertions for published extras (#2547).

A dependency that ships in a published extra is attack surface even when
nothing imports it: ``chromadb==1.5.9`` sat unused in the ``local`` extra
(propagated into ``all-features``/``full``) while carrying an unpatched
pre-authentication code-injection CVE (CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c).

These tests pin the contract that known-vulnerable, unused packages stay out
of the declared dependency graph AND the resolved lockfile, so one cannot
silently re-enter via a copy-pasted extras line or a transitive resolution.
"""

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Packages that must never appear in the declared or resolved dependency
#: graph. Add an entry (with the issue reference) when a package is removed
#: for being unused-and-vulnerable; remove it only with an explicit,
#: reviewed decision to reintroduce the dependency.
BANNED_PACKAGES = {
    "chromadb": "#2547 — unused; unpatched critical CVE-2026-45829",
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
