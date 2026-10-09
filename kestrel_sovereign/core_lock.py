"""Core's ``uv.lock`` as the declaration of what the host venv holds (#3502).

CI installs core's dependencies from ``uv.lock``, so the lock is the only
statement of which versions have been tested together. ``uv sync`` honours it,
and nothing else on the host does: every feature install goes through
``uv pip`` (see ``cli_features._extension_install_run``), which never reads the
lock. Left unconstrained, a feature install resolves a locked package to
whatever the index has, and ``uv pip install --upgrade`` (the update action of
``kestrel update``'s reconcile) is eager: it may move every package in the
resolution, not only the one named. A host ran ``anthropic`` 1.11.0 against a
lock pinning 0.117.0, re-installed by the update's feature steps right after
its own ``uv sync`` had put the locked version back.

One reading of the lock serves two purposes:

* :meth:`CoreLock.constraint_lines` gives one line per locked package: its
  ``name===version``, or, for a package the host manifest declares editable,
  a hold on that checkout. Every feature install carries them, so a feature
  can neither upgrade nor downgrade a package the lock pins, nor resolve a
  declared checkout's package from the index. A conflict fails the install
  instead of moving the package, and a locked package with no single version
  for this environment refuses it.
* :func:`lock_drift` lists every installed package whose version differs from
  the lock. It reports and never repairs.

The lock read is core's own: the ``uv.lock`` in the checkout core is installed
from (:func:`core_lock_checkout`). A core installed from an index wheel has no
lock beside it, and then there is nothing to hold or compare.
"""

from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from kestrel_sovereign.feature_reconcile import (
    CORE_DISTRIBUTION,
    canonical_package,
    version_is_valid,
)

LOCK_FILENAME = "uv.lock"


class CoreLockError(ValueError):
    """Core's ``uv.lock`` cannot be used: it will not read, cannot be found, or
    names no single version here of a package it covers.

    Distinct from "there is no lock". A missing lock declares nothing, so there
    is nothing to hold. An unreadable one declares something nobody can
    recover, and so does one hidden behind core install metadata that will not
    read. One that leaves a package's version undetermined here cannot hold
    that package (:meth:`CoreLock.constraint_lines`). An install that went
    ahead anyway could not be shown to keep it.
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
    which more than one locked version applies, none applies, or whose version
    or markers cannot be evaluated. They have no version to pin, so an install
    is refused rather than held to every package but them
    (:meth:`constraint_lines`), and the drift report names them, because a
    guess either way would be a statement the lock never made.
    """

    path: Path
    packages: Tuple[LockedPackage, ...] = ()
    undetermined: Tuple[str, ...] = ()

    def versions(self) -> Dict[str, str]:
        return {package.name: package.version for package in self.packages}

    def names(self) -> Tuple[str, ...]:
        """Every package this lock covers here: the pinned and the undetermined."""
        return tuple(package.name for package in self.packages) + self.undetermined

    def constraint_lines(
        self, declared_checkouts: Mapping[str, str] = MappingProxyType({}),
    ) -> List[str]:
        """The line every install carries for each package this lock covers.

        Decided from two inputs and nothing else: this lock, for the versions,
        and *declared_checkouts*, the packages the host manifest declares
        ``editable``, mapped to the checkout each names. Not from the
        arguments of the install the lines are for, and not from how a package
        is installed now: either let one install, or one damaged
        ``direct_url.json``, take a locked package out from under the lock
        (#3502).

        * A package with no declared checkout gets ``name===<locked version>``,
          however it is installed. A linked copy at another version is put
          back to the locked one or the install fails, and so is one installed
          from a URL. ``===``, not ``==``: PEP 440 lets ``==1.0`` match any
          local build ``1.0+<label>``, and uv prefers one when an index offers
          it, so ``==`` would let an install move a locked package to a build
          the lock never named (and :func:`lock_drift` would then report it).
          Measured on uv 0.9.22 and pip 25.0: ``===`` refuses the local build,
          and still matches the locked version however the artifact spells it
          (``2024.01.01`` for a locked ``2024.1.1``, ``v1.2``,
          ``1.0.0.POST1``), because both compare the parsed version.
        * A package with a declared checkout gets ``-e name @ <file URL>`` (see
          :func:`checkout_hold_line`). It stays on that checkout: the resolver
          may neither replace the link from the index nor move it to another
          version, and an install whose resolution needs another version fails.
          Its locked version is not pinned, since the operator chose to run
          what that checkout builds.

        Raises :class:`CoreLockError` when an :attr:`undetermined` package has
        no declared checkout. It has no version to carry, so the lines would
        hold every package but that one, and an install held to them would be
        free to move it: the lines are a whole hold of the lock or none. The
        drift check never asks for lines, so it still reports what it can
        compare.
        """
        checkouts = {
            canonical_package(name): checkout
            for name, checkout in declared_checkouts.items()
            if checkout
        }
        unpinnable = [name for name in self.undetermined if name not in checkouts]
        if unpinnable:
            raise CoreLockError(
                f"{self.path} names no single version for this environment of: "
                + ", ".join(unpinnable)
                + ". An install could not hold them at a locked version. Either "
                "this interpreter or platform is outside the environments the "
                "lock was resolved for, or its markers or versions do not "
                "evaluate here"
            )
        versions = self.versions()
        lines = []
        for name in sorted(self.names()):
            if name in checkouts:
                lines.append(checkout_hold_line(name, checkouts[name]))
            else:
                lines.append(f"{name}==={versions[name]}")
        return lines

    def checkout_held(self, declared_checkouts: Mapping[str, str]) -> Tuple[str, ...]:
        """The packages this lock covers that :meth:`constraint_lines` holds by checkout."""
        held = {canonical_package(name) for name, path in declared_checkouts.items() if path}
        return tuple(sorted(name for name in self.names() if name in held))


def checkout_hold_line(name: str, checkout: str) -> str:
    """The constraint that keeps *name* on its editable *checkout*.

    ``-e name @ file:///<checkout>``. Measured on uv 0.9.22 against an
    editable dependency: the link is kept by a plain install and by an eager
    ``--upgrade``; an install whose resolution needs another version fails
    instead of taking an index wheel; a package not linked from the checkout
    is linked from it when the install resolves it; a constraint on a package
    the install never resolves is ignored. A ``name @ file:///...`` constraint
    without ``-e`` is not a substitute: uv replaces the editable link with a
    non-editable build of the same directory.

    pip refuses an editable constraint outright, so only an install that runs
    on uv can carry this line (``CoreInstallGuard`` refuses the rest).

    The path is made absolute without touching the filesystem, as the
    ``-e <checkout>`` an install of the entry itself names. uv compares the two
    after resolving links (``/tmp`` against ``/private/tmp`` on macOS).
    """
    absolute = os.path.abspath(os.path.expanduser(checkout))
    return f"-e {canonical_package(name)} @ {Path(absolute).as_uri()}"


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
    it. ``error`` is set when the lock could not be read or located
    (:class:`CoreLockError`). Neither case is drift, and neither is a clean
    bill: no comparison was made.
    """

    lock: Optional[CoreLock] = None
    error: Optional[str] = None
    drift: Tuple[LockDrift, ...] = ()
    #: Locked packages that are installed but whose installed metadata names
    #: no version, so they could not be compared. Named, never counted as a
    #: match: a damaged dist-info still imports whatever code is beside it.
    unreadable: Tuple[str, ...] = ()

    @property
    def compared(self) -> bool:
        return self.lock is not None and self.error is None

    @property
    def matches(self) -> bool:
        return (
            self.compared
            and not self.drift
            and not self.lock.undetermined
            and not self.unreadable
        )

    @property
    def needs_attention(self) -> bool:
        """A mismatch, or a lock that exists and could not be compared.

        Not "anything short of a match": a core with no lock beside it has
        nothing to compare, and that is not something an operator must act on.
        """
        return not self.matches and (self.compared or self.error is not None)

    @property
    def headline(self) -> str:
        """One line naming which outcome this is."""
        if self.matches:
            return (
                f"venv matches core's uv.lock ({len(self.lock.packages)} "
                "locked packages)"
            )
        if self.drift:
            return "venv differs from core's uv.lock"
        if self.compared:
            return "venv was only partly compared against core's uv.lock"
        if self.error is not None:
            return "venv was not compared against core's uv.lock"
        return "no uv.lock beside the core install, nothing to compare"

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
        if self.unreadable:
            lines.append(
                "installed metadata names no version for: "
                + ", ".join(self.unreadable)
            )
        if self.drift:
            lines.append(
                f"This venv is not running the versions CI tested ({self.lock.path}). "
                "`kestrel update --no-pull` re-applies the lock (`uv sync`, "
                "then feature installs held to it) and restarts onto it. To "
                "run other versions, move the lock deliberately with `uv lock` "
                "and a green CI run."
            )
        return lines


def core_lock_checkout(provenance) -> Optional[Path]:
    """The checkout whose ``uv.lock`` describes this core, or None.

    *provenance* is core's PEP 610 :class:`~kestrel_sovereign.feature_reconcile.Provenance`.
    The lock lives beside the project it locks, so only a core installed from a
    local directory has one: an editable link (the ``uv sync`` workflow), or a
    non-editable install from the same directory. An index wheel, a VCS ref and
    an archive carry no lock.

    Raises :class:`CoreLockError` when core's provenance is unknown. Damaged
    install metadata may hide a checkout with a lock beside it, and answering
    "no lock" would let every install run unheld on exactly the host whose
    install is already broken. Whether that directory still holds a lock is
    :func:`core_lock_path`'s question, not this one's.
    """
    if provenance is None:
        return None
    if not provenance.known:
        raise CoreLockError(
            "kestrel-sovereign's install metadata (direct_url.json) would not "
            "read, so the checkout it was installed from, and the uv.lock "
            "beside it, cannot be located. Reinstall core so that metadata is "
            "rewritten: `uv sync --reinstall-package kestrel-sovereign` in "
            "its checkout"
        )
    if not provenance.url or provenance.vcs or provenance.archive_hash:
        return None
    checkout = Path(provenance.url).expanduser()
    # A local directory is always an absolute path once its file: URL is
    # decoded. Anything else is a remote address, with no directory beside it.
    return checkout if checkout.is_absolute() else None


def core_lock_path(checkout: Optional[Path]) -> Optional[Path]:
    """``<checkout>/uv.lock``, or None when nothing is there.

    None ONLY for an absent path: nothing exists at it, or *checkout* is not a
    directory and so holds nothing. Whatever else exists there is a lock the
    host declares, and :func:`read_core_lock` decides whether it reads. A
    directory, a socket or a dangling link named ``uv.lock`` is not "no lock",
    and neither is a path whose status the OS refuses
    (:class:`CoreLockError`): reading any of them as absent let every install
    run unheld and the drift check report nothing to compare.
    """
    if checkout is None:
        return None
    path = Path(checkout) / LOCK_FILENAME
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise CoreLockError(f"{path}: {exc}") from exc
    return path


def _open_lock(path: Path):
    """*path* opened for reading, refused unless it is a regular file.

    Checked on the open descriptor, so what is read is what was checked.
    ``O_NONBLOCK`` because a FIFO named ``uv.lock`` would otherwise block the
    open until something wrote to it; on a regular file the flag is inert.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CoreLockError(f"{path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CoreLockError(f"{path}: not a regular file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


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
        with _open_lock(path) as handle:
            data = tomllib.load(handle)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        # tomllib decodes the bytes itself, so a lock that is not UTF-8 raises
        # UnicodeDecodeError rather than TOMLDecodeError.
        raise CoreLockError(f"{path}: {exc}") from exc
    entries = data.get("package")
    if not isinstance(entries, list):
        raise CoreLockError(f"{path}: no [[package]] entries")

    applicable: Dict[str, List[str]] = {}
    undetermined = set()
    forked = set()
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
            forked.add(name)
            continue
        version = entry.get("version")
        if not isinstance(version, str) or not version_is_valid(version):
            undetermined.add(name)
            continue
        applicable.setdefault(name, []).append(version)

    # uv's forks partition every environment the lock supports, so a package
    # none of whose forks applies here is one whose markers this reader cannot
    # evaluate faithfully (a uv conflict fork keyed on an ``extra``, say).
    # Saying so beats silently neither pinning nor comparing it.
    undetermined.update(forked - set(applicable))
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


def installed_versions(
    names: Iterable[str], path: Optional[List[str]] = None,
) -> Dict[str, Optional[str]]:
    """``{name: installed version}`` for each of *names* installed on *path*.

    A name that is not installed is absent from the result. One that is
    installed but whose metadata names no version (``METADATA`` missing,
    unreadable, or without a ``Version``) maps to None: it cannot be compared,
    and leaving it out would read as "not installed".

    *path* defaults to ``sys.path``: the distributions this interpreter
    imports. The first distribution found wins, which is the one
    ``importlib.metadata.version`` and an ``import`` would resolve. Looked up
    by name, which ``importlib.metadata`` matches against the dist-info
    directory, so a damaged entry is still found. ``invalidate_caches``
    because the installs this follows ran in subprocesses after the import
    system cached its directory listings.
    """
    import importlib
    import importlib.metadata as md

    importlib.invalidate_caches()
    found: Dict[str, Optional[str]] = {}
    for name in names:
        context = {"name": name} if path is None else {"name": name, "path": path}
        dist = next(iter(md.distributions(**context)), None)
        if dist is None:
            continue
        try:
            version = dist.version
        except (OSError, ValueError, TypeError, KeyError):
            version = None
        found[canonical_package(name)] = version or None
    return found


def _same_version(installed: str, locked: str) -> bool:
    """PEP 440 equality when both parse (``1.0`` is ``1.0.0``), else the raw text."""
    if version_is_valid(installed) and version_is_valid(locked):
        from packaging.version import Version

        return Version(installed) == Version(locked)
    return installed == locked


def lock_drift(
    lock: CoreLock, installed: Mapping[str, Optional[str]],
) -> Tuple[LockDrift, ...]:
    """Every locked package installed at a version other than the locked one.

    A locked package that is not installed is not drift: the lock covers every
    extra and dependency group, and a host installs the ones it uses. Nor is
    one installed with no readable version (None in *installed*): that is
    :attr:`LockCheck.unreadable`, named apart because nothing was compared.
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
    one reader the install guard and the drift check share, so the lock an
    install is held to is the lock the venv is compared against.
    """
    path = core_lock_path(checkout)
    return read_core_lock(path) if path is not None else None


def check_venv_against_lock(
    read_lock: Callable[[], Optional[CoreLock]],
    *,
    installed: Optional[
        Callable[[Iterable[str]], Mapping[str, Optional[str]]]
    ] = None,
) -> LockCheck:
    """Compare the installed versions against the lock *read_lock* returns.

    Reads only, and never repairs. *read_lock* is called here so a lock that
    exists and cannot be read becomes :attr:`LockCheck.error` rather than an
    exception at the caller: a health report must still be produced.
    *installed* is asked for the locked names and defaults to
    :func:`installed_versions`, this interpreter's own ``sys.path``.
    """
    try:
        lock = read_lock()
    except CoreLockError as exc:
        return LockCheck(error=str(exc))
    if lock is None:
        return LockCheck()
    versions = (installed or installed_versions)(
        [package.name for package in lock.packages]
    )
    unreadable = tuple(
        package.name for package in lock.packages
        if package.name in versions and versions[package.name] is None
    )
    return LockCheck(
        lock=lock, drift=lock_drift(lock, versions), unreadable=unreadable,
    )
