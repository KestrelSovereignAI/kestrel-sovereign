"""Core's ``uv.lock`` as the declaration of what the host venv holds (#3502).

CI installs core's dependencies from ``uv.lock``, so the lock is the only
statement of which versions have been tested together. ``uv sync`` honours it,
and nothing else on the host does: every feature install goes through
``uv pip`` (see ``cli_features._extension_install_run``), which never reads the
lock. Left unconstrained, a feature install resolves a locked package to
whatever the index has. ``kestrel update``'s reconcile passes ``--upgrade``,
which lets uv move every package in the resolution, so a host ran
``anthropic`` 1.11.0 against a lock pinning 0.117.0. The update re-applied it
each time, right after its own ``uv sync`` had put the locked version back.

One reading of the lock serves two purposes:

* :meth:`CoreLock.constraint_lines` gives one ``name==version`` line per locked
  package. Every feature install carries them, so a feature can neither
  upgrade nor downgrade a package the lock pins. A conflict fails the install
  instead of moving the package.
* :func:`lock_drift` lists every installed package whose version differs from
  the lock. It reports and never repairs.

The lock read is core's own: the ``uv.lock`` in the checkout core is installed
from (:func:`core_lock_checkout`). A core installed from an index wheel has no
lock beside it, and then there is nothing to hold or compare.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.feature_reconcile import (
    CORE_DISTRIBUTION,
    canonical_package,
    version_is_valid,
)

LOCK_FILENAME = "uv.lock"


class CoreLockError(ValueError):
    """``uv.lock`` exists and cannot be read as a lock.

    Distinct from "there is no lock". A missing lock declares nothing, so there
    is nothing to hold. An unreadable one declares something nobody can
    recover, and an install that went ahead anyway could not be shown to keep
    it.
    """


@dataclass(frozen=True)
class LockedPackage:
    """One package the lock pins for this environment."""

    name: str
    version: str


@dataclass(frozen=True)
class CoreLock:
    """What core's ``uv.lock`` pins for THIS interpreter and platform.

    ``packages`` excludes core itself, which has its own source policy
    (``feature_reconcile.core_install_constraints``), and any entry whose
    resolution markers do not apply here. ``undetermined`` names packages for
    which more than one locked version applies, or whose version or markers
    cannot be evaluated. They are not constrained, and the drift report names
    them, because a guess either way would be a statement the lock never made.
    """

    path: Path
    packages: Tuple[LockedPackage, ...] = ()
    undetermined: Tuple[str, ...] = ()

    def versions(self) -> Dict[str, str]:
        return {package.name: package.version for package in self.packages}

    def constraint_lines(self, exclude: Iterable[str] = ()) -> List[str]:
        """``name==version`` for every locked package not in *exclude*.

        *exclude* holds canonical names the caller has a deliberate reason to
        leave free. The host manifest's editable entries are the one such
        reason: an operator who links a checkout of a locked package has
        declared its source, and a version pin would make that entry
        uninstallable. :func:`lock_drift` still reports it.
        """
        skip = {canonical_package(name) for name in exclude}
        return [
            f"{package.name}=={package.version}"
            for package in self.packages
            if package.name not in skip
        ]


@dataclass(frozen=True)
class LockDrift:
    """An installed package whose version is not the locked one."""

    name: str
    locked: str
    installed: str

    def describe(self) -> str:
        return f"{self.name} {self.installed} installed, uv.lock pins {self.locked}"


@dataclass(frozen=True)
class LockCheck:
    """The venv compared against core's lock.

    ``lock`` is None when core is installed from somewhere with no lock beside
    it. ``error`` is set when a lock exists and could not be read. Neither case
    is drift, and neither is a clean bill: no comparison was made.
    """

    lock: Optional[CoreLock] = None
    error: Optional[str] = None
    drift: Tuple[LockDrift, ...] = ()
    checkout: Optional[Path] = None

    @property
    def compared(self) -> bool:
        return self.lock is not None and self.error is None

    @property
    def matches(self) -> bool:
        return self.compared and not self.drift and not self.lock.undetermined

    def report_lines(self) -> List[str]:
        """Operator-readable lines for every mismatch, plus the one remedy.

        Shared by ``kestrel doctor`` and ``kestrel update`` so the two cannot
        describe one venv differently. Empty when there is nothing to say.
        """
        if self.error is not None:
            return [f"core lock could not be read: {self.error}"]
        if self.lock is None:
            return []
        lines = [drift.describe() for drift in self.drift]
        if self.lock.undetermined:
            lines.append(
                "uv.lock names no single version for this environment of: "
                + ", ".join(self.lock.undetermined)
            )
        if self.drift:
            lines.append(
                "This venv is not running the versions CI tested. "
                "`kestrel update --no-pull` re-applies the lock (`uv sync`, "
                "then feature installs held to it); to run other versions, "
                "move the lock deliberately with `uv lock` and a green CI run."
            )
        return lines


def core_lock_checkout(provenance) -> Optional[Path]:
    """The checkout whose ``uv.lock`` describes this core, or None.

    *provenance* is core's PEP 610 :class:`~kestrel_sovereign.feature_reconcile.Provenance`.
    The lock lives beside the project it locks, so only a core installed from a
    local directory has one: an editable link (the ``uv sync`` workflow), or a
    non-editable install from the same directory. An index wheel, a VCS ref and
    an archive carry no lock. Unknown provenance names no directory either.
    """
    if provenance is None or not provenance.known or not provenance.url:
        return None
    if provenance.vcs or provenance.archive_hash:
        return None
    checkout = Path(provenance.url).expanduser()
    return checkout if checkout.is_dir() else None


def core_lock_path(checkout: Optional[Path]) -> Optional[Path]:
    """``<checkout>/uv.lock`` when it exists, else None."""
    if checkout is None:
        return None
    path = Path(checkout) / LOCK_FILENAME
    return path if path.is_file() else None


def _applies(entry: dict, environment: Optional[Dict[str, str]]) -> Optional[bool]:
    """Whether a lock entry applies here. None when its markers will not evaluate.

    An entry with no ``resolution-markers`` belongs to every fork of the
    resolution. One with markers belongs to the forks they name, and uv makes
    those forks disjoint, so at most one entry per package should apply.
    """
    from packaging.markers import InvalidMarker, Marker, UndefinedComparison
    from packaging.markers import UndefinedEnvironmentName

    markers = entry.get("resolution-markers")
    if not markers:
        return True
    if not isinstance(markers, list):
        return None
    try:
        return any(Marker(str(m)).evaluate(environment) for m in markers)
    except (InvalidMarker, UndefinedComparison, UndefinedEnvironmentName, TypeError):
        return None


def read_core_lock(
    path: Path, *, environment: Optional[Dict[str, str]] = None,
) -> CoreLock:
    """Read *path* as a ``uv.lock``, keeping what applies to this environment.

    *environment* overrides PEP 508 marker values for evaluating resolution
    markers (default: the running interpreter's). Raises :class:`CoreLockError`
    when the file cannot be read or is not a lock.
    """
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CoreLockError(f"{path}: {exc}") from exc
    entries = data.get("package")
    if not isinstance(entries, list):
        raise CoreLockError(f"{path}: no [[package]] entries")

    applicable: Dict[str, List[str]] = {}
    undetermined = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise CoreLockError(f"{path}: a [[package]] entry has no name")
        name = canonical_package(entry["name"])
        if name == CORE_DISTRIBUTION:
            continue
        source = entry.get("source")
        if isinstance(source, dict) and "virtual" in source:
            # A virtual package is never installed, so it has no version to pin.
            continue
        applies = _applies(entry, environment)
        if applies is None:
            undetermined.add(name)
            continue
        if not applies:
            continue
        version = entry.get("version")
        if not isinstance(version, str) or not version_is_valid(version):
            undetermined.add(name)
            continue
        applicable.setdefault(name, []).append(version)

    packages = []
    for name in sorted(applicable):
        versions = sorted(set(applicable[name]))
        if len(versions) != 1 or name in undetermined:
            undetermined.add(name)
            continue
        packages.append(LockedPackage(name=name, version=versions[0]))
    return CoreLock(
        path=Path(path),
        packages=tuple(packages),
        undetermined=tuple(sorted(undetermined)),
    )


def installed_distributions() -> Dict[str, str]:
    """``{canonical name: version}`` for every distribution this interpreter sees.

    The first distribution found on ``sys.path`` wins, which is the one
    ``importlib.metadata.version`` and an ``import`` would resolve.
    ``invalidate_caches`` because the installs this follows ran in subprocesses
    after the import system cached its directory listings.
    """
    import importlib
    import importlib.metadata as md

    importlib.invalidate_caches()
    found: Dict[str, str] = {}
    for dist in md.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # noqa: BLE001 - one damaged dist is not a verdict
            continue
        if not name or not version:
            continue
        found.setdefault(canonical_package(name), version)
    return found


def _same_version(installed: str, locked: str) -> bool:
    """PEP 440 equality when both parse (``1.0`` is ``1.0.0``), else the raw text."""
    if version_is_valid(installed) and version_is_valid(locked):
        from packaging.version import Version

        return Version(installed) == Version(locked)
    return installed == locked


def lock_drift(lock: CoreLock, installed: Mapping[str, str]) -> Tuple[LockDrift, ...]:
    """Every locked package installed at a version other than the locked one.

    A locked package that is not installed is not drift: the lock covers every
    extra and dependency group, and a host installs the ones it uses.
    """
    drift = []
    for package in lock.packages:
        have = installed.get(package.name)
        if have is None or _same_version(have, package.version):
            continue
        drift.append(LockDrift(name=package.name, locked=package.version, installed=have))
    return tuple(drift)


def load_core_lock(checkout: Optional[Path]) -> Optional[CoreLock]:
    """``<checkout>/uv.lock`` read for this environment, or None when there is none.

    Raises :class:`CoreLockError` when the lock exists and cannot be read. The
    one entry point both the install guard and the drift check use, so the
    lock an install is held to is the lock the venv is compared against.
    """
    path = core_lock_path(checkout)
    return read_core_lock(path) if path is not None else None


def check_venv_against_lock(
    checkout: Optional[Path],
    *,
    installed: Callable[[], Mapping[str, str]] = installed_distributions,
) -> LockCheck:
    """Compare this venv against ``<checkout>/uv.lock``. Reads only."""
    try:
        lock = load_core_lock(checkout)
    except CoreLockError as exc:
        return LockCheck(error=str(exc), checkout=checkout)
    if lock is None:
        return LockCheck(checkout=checkout)
    return LockCheck(lock=lock, drift=lock_drift(lock, installed()), checkout=checkout)
