"""Keep every test away from the operator's real host-runtime state (#3286).

Two halves, both wired as autouse fixtures in ``tests/conftest.py``:

**Isolation.** Host runtime state -- the host-feature database, its Hold
custody evidence, the Phoenix trace store -- lives below ``host_data_dir()``,
which is ``~/.kestrel/host-data`` unless the environment says otherwise. Before
#3286 only the unit tier redirected it. An integration test that set
``KESTREL_DB_PATH`` alone and entered the real server lifespan therefore made
``prepare_host_database`` consult the real
``~/.kestrel/host-data/host-features.db`` as the "previous default" host
database: with the production host running that refused on its live sidecars
(and the failed lifespan's ``stop_receipt_store_error`` stayed on the shared
``server.app``, turning later unit tests' requests into 503s); with the host
stopped, the same call would ``os.replace`` production's database into a
pytest temp directory. :func:`redirect_host_data_root` moves the root into a
per-test directory for the whole suite through ``KESTREL_HOST_DATA_DIR``.

That seam redirects the host-data root and nothing else. ``KESTREL_HOME`` would
also move ``project_dir()``, and the integration tiers read the checkout's
``kestrel.toml`` through it; ``HOME`` would hide the operator's global git
configuration from tests that shell out to ``git``. The unit tier redirects
both anyway, in ``tests/unit/conftest.py``, inside the same temporary root.

**Enforcement.** A convention is not a guarantee: the original leak was never
visible from inside the suite. :class:`HostDataTripwire` watches the real
host-data root for the whole session in three ways and fails the session if
any sees this run touch it:

* an in-process audit hook (:pep:`578`) records every audited filesystem or
  SQLite access under the root, attributed to the running test. It is exact,
  and it is the only way to see a *modification* while a live host is also
  writing the same files;
* a filesystem snapshot taken at session start and compared at session end
  catches what the hook cannot see, such as a subprocess a test launched. It
  compares every entry strictly (created, removed, replaced, size, mtime)
  except the files a live writer owns: those another process held open at
  session start (``lsof``) and the members of each SQLite family that had
  sidecars then (the same inference the production migration makes). For
  those, only creation and removal of the main file count;
* at session end, any process descended from this session that still holds a
  file under the root open is reported, live-writer files included.

The remaining gap, stated plainly: a subprocess a test launched that changes
the *contents* of a file a live writer held open at session start (the running
host's database, WAL, or Phoenix log) and exits before the session ends is
indistinguishable from the live writer's own changes. Nothing outside the
kernel can attribute that write; the audit hook covers it only in-process.

Tests that deliberately exercise path resolution opt out of the isolation with
``@pytest.mark.owns_host_paths`` and redirect ``HOME``/``KESTREL_HOME``
themselves; the tripwire still applies to them.
"""

from __future__ import annotations

import functools
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import psutil
import pytest

from kestrel_sovereign import paths
from kestrel_sovereign.host_features.storage import SQLITE_AUXILIARY_SUFFIXES

#: Marker name for tests that own host/home path resolution themselves.
OWNS_HOST_PATHS_MARKER = "owns_host_paths"

#: Name of the fixture's own temporary root. It is a sibling of each test's
#: ``tmp_path``, not a child: the unit tier writes a host manifest here, and a
#: directory the test owns is the wrong place for the fixture's state — one
#: test asserts its ``tmp_path`` is empty, and every test is entitled to.
ISOLATION_DIRNAME = "_kestrel_host_runtime_isolation"


def redirect_host_data_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``host_data_dir()`` and every resolver that mirrors it below ``root``.

    Nothing is created: each storage owner creates and validates its own
    directory, the same custody path production takes. The redirect also
    reaches managed child processes through :data:`SPAWNED_ENV_PIN`.
    """
    host_data = root / "host-data"
    monkeypatch.setenv(paths.HOST_DATA_DIR_ENV, str(host_data))
    SPAWNED_ENV_PIN.pin(str(host_data), monkeypatch)
    return host_data


class SpawnedEnvPin:
    """Keep the test's host-data root in every managed child's environment.

    ``paths.spawned_agent_env`` is the launch environment for every managed
    agent process (``ProcessManager``, ``kestrel shell``, ``doctor``, setup).
    It copies ``os.environ`` and then lets the project ``.env`` overwrite it:
    the production precedence, deliberately. So a checkout whose ``.env``
    names ``KESTREL_HOST_DATA_DIR`` or ``KESTREL_HOME`` would hand a child
    the operator's host-data root, and ``ProcessManager`` pins that choice as
    the child's previous-default host database. Setting the variable in
    ``os.environ`` cannot win against the file.

    Installed once per session, this wraps ``spawned_agent_env`` at every
    binding (``paths`` and each module that imported the name) and re-applies
    the isolated root *after* the ``.env`` merge. ``KESTREL_HOST_DATA_DIR``
    outranks ``KESTREL_HOME`` and ``HOME`` in both the default and the
    previous-default resolver, so it alone decides the child's host-data
    root. Outside an isolated test (the ``owns_host_paths`` opt-out) the
    wrapper is a pass-through, so production precedence stays testable.
    """

    def __init__(self) -> None:
        self._original: Optional[Callable[[Path], dict]] = None
        self._wrapper: Optional[Callable[[Path], dict]] = None
        self._root: Optional[str] = None

    @property
    def root(self) -> Optional[str]:
        return self._root

    def install(self) -> None:
        if self._wrapper is not None:
            return
        original = paths.spawned_agent_env
        self._original = original

        @functools.wraps(original)
        def spawned_agent_env(project_dir: Path) -> dict:
            env = original(project_dir)
            if self._root is not None:
                env[paths.HOST_DATA_DIR_ENV] = self._root
            return env

        self._wrapper = spawned_agent_env
        self._rebind(original, spawned_agent_env)

    def uninstall(self) -> None:
        if self._wrapper is None or self._original is None:
            return
        self._rebind(self._wrapper, self._original)
        self._wrapper = None
        self._root = None

    def pin(self, root: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin ``root`` for the current test; ``monkeypatch`` undoes it."""
        if self._wrapper is None:
            raise RuntimeError("SpawnedEnvPin.pin() before install()")
        monkeypatch.setattr(self, "_root", root)

    @staticmethod
    def _rebind(old: Callable, new: Callable) -> None:
        # A module that imported the name before installation holds its own
        # binding, so every binding in every loaded module is replaced.
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if not isinstance(namespace, dict):
                continue
            for name, value in list(namespace.items()):
                if value is old:
                    namespace[name] = new


#: The one pin the suite installs (see ``tests/conftest.py``).
SPAWNED_ENV_PIN = SpawnedEnvPin()


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------

#: Audit events whose arguments name a filesystem path. Reads count as well as
#: writes: consulting the real host database was the original defect.
_PATH_AUDIT_EVENTS = frozenset(
    {
        "open",
        "sqlite3.connect",
        "os.chflags",
        "os.chmod",
        "os.chown",
        "os.fwalk",
        "os.link",
        "os.listdir",
        "os.lchflags",
        "os.lchmod",
        "os.lchown",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.scandir",
        "os.symlink",
        "os.truncate",
        "os.utime",
        "os.walk",
        "shutil.copyfile",
        "shutil.copymode",
        "shutil.copystat",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
    }
)


def real_host_data_roots() -> tuple[Path, ...]:
    """The host-data roots an unisolated test would resolve in this process.

    Called at import, before any fixture edits the environment, so it names the
    operator's roots: ``host_data_dir()`` honours an exported
    ``KESTREL_HOST_DATA_DIR`` or ``KESTREL_HOME``, and ``~/.kestrel/host-data``
    is the fallback behind them.
    """
    candidates = (
        paths.host_data_dir(),
        Path(os.path.abspath(Path.home() / ".kestrel" / "host-data")),
    )
    roots: list[Path] = []
    for candidate in candidates:
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


@dataclass(frozen=True)
class _Entry:
    kind: str
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _Baseline:
    """One root as the session found it."""

    entries: dict[str, _Entry]
    #: Sidecars of live SQLite families: their writer creates and removes them.
    live_sidecars: frozenset[str]
    #: Files a live writer owns, whose contents and inode are its business.
    live_files: frozenset[str]


def _kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def _live_sqlite_families(names: set[str]) -> set[str]:
    """Main file and sidecars of every SQLite family in one directory with a
    live writer, inferred from sidecars being present."""
    members: set[str] = set()
    for name in names:
        sidecars = {f"{name}{suffix}" for suffix in SQLITE_AUXILIARY_SUFFIXES}
        if sidecars & names:
            members |= sidecars | {name}
    return members


def _relative_key(root: Path, path: str) -> Optional[str]:
    """``path`` relative to ``root`` as a snapshot key, or ``None`` if outside."""
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if real_path == real_root:
        return "."
    if not real_path.startswith(real_root + os.sep):
        return None
    return Path(os.path.relpath(real_path, real_root)).as_posix()


def _held_open_by_other_processes(root: Path) -> Optional[frozenset[str]]:
    """Files under ``root`` another process holds open now, via ``lsof``.

    ``None`` when ``lsof`` is unavailable or fails: the caller then knows only
    the SQLite inference, and every other file is compared strictly. A
    running host's non-SQLite files (its Phoenix log) then report as modified:
    a false alarm the operator can read, rather than a silent pass.
    """
    lsof = shutil.which("lsof")
    if lsof is None:
        return None
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pn", "+D", str(root)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # lsof exits 1 when nothing is open, which is an answer, not a failure.
    if result.returncode not in (0, 1):
        return None
    own = os.getpid()
    pid: Optional[int] = None
    held: set[str] = set()
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid != own:
            key = _relative_key(root, line[1:])
            if key is not None and key != ".":
                held.add(key)
    return frozenset(held)


def _descendants_holding(root: Path) -> list[str]:
    """Processes this session started that hold a file under ``root`` open."""
    findings: list[str] = []
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return findings
    for child in children:
        try:
            open_files = child.open_files()
            command = " ".join(child.cmdline()[:4])
        except psutil.Error:
            continue
        for open_file in open_files:
            key = _relative_key(root, open_file.path)
            if key is not None:
                findings.append(
                    f"{root}: pid {child.pid} ({command}) still holds {key} open"
                )
    return findings


def _snapshot(root: Path) -> tuple[dict[str, _Entry], frozenset[str]]:
    """Every entry below ``root`` plus the members of live SQLite families.

    Returns ``({relative path: entry}, live SQLite family members)``.
    """
    entries: dict[str, _Entry] = {}
    live_members: set[str] = set()

    def walk(directory: Path, relative: str) -> None:
        try:
            children = list(os.scandir(directory))
        except FileNotFoundError:
            return
        names = {child.name for child in children}
        live_members.update(
            f"{relative}{name}" for name in _live_sqlite_families(names)
        )
        for child in children:
            try:
                st = child.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            key = f"{relative}{child.name}"
            kind = _kind(st.st_mode)
            entries[key] = _Entry(
                kind=kind,
                inode=st.st_ino,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            )
            if kind == "directory":
                walk(Path(child.path), f"{key}/")

    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return entries, frozenset()
    entries["."] = _Entry(
        kind=_kind(root_stat.st_mode), inode=root_stat.st_ino, size=0, mtime_ns=0
    )
    if stat.S_ISDIR(root_stat.st_mode):
        walk(root, "")
    return entries, frozenset(live_members)


def _baseline(root: Path) -> _Baseline:
    entries, live_members = _snapshot(root)
    sidecars = frozenset(
        key
        for key in live_members
        if key.endswith(tuple(SQLITE_AUXILIARY_SUFFIXES))
    )
    held = _held_open_by_other_processes(root) if entries else frozenset()
    return _Baseline(
        entries=entries,
        live_sidecars=sidecars,
        live_files=live_members | (held or frozenset()),
    )


def _compare(baseline: _Baseline, after: dict[str, _Entry]) -> list[str]:
    before = baseline.entries
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key in baseline.live_sidecars:
            # A live writer creates and removes its own sidecars; only the
            # audit hook can attribute those to this run.
            continue
        old = before.get(key)
        new = after.get(key)
        if old is None:
            changes.append(f"created {key} ({new.kind})")
            continue
        if new is None:
            changes.append(f"removed {key} ({old.kind})")
            continue
        if old.kind != new.kind:
            changes.append(f"replaced {key} ({old.kind} -> {new.kind})")
            continue
        if key in baseline.live_files or new.kind == "directory":
            # A live writer's sizes, times, and atomically replaced inodes are
            # its own; a directory's times track its entry set, which is
            # compared entry by entry.
            continue
        if old.inode != new.inode:
            changes.append(f"replaced {key} (inode {old.inode} -> {new.inode})")
        elif (old.size, old.mtime_ns) != (new.size, new.mtime_ns):
            changes.append(
                f"modified {key} (size {old.size} -> {new.size}, "
                f"mtime_ns {old.mtime_ns} -> {new.mtime_ns})"
            )
    return changes


def _as_path_text(value: object) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return os.fsdecode(value)
    if isinstance(value, os.PathLike):
        path = os.fspath(value)
        return os.fsdecode(path) if isinstance(path, bytes) else path
    return None


class HostDataTripwire:
    """Fail the session if this run touched the real host-data roots."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = roots
        self._prefixes = tuple(
            dict.fromkeys(
                prefix
                for root in roots
                for prefix in (str(root), os.path.realpath(root))
            )
        )
        self._baselines: dict[Path, _Baseline] = {}
        self._accesses: list[tuple[str, str, str]] = []
        self._armed = False
        self._hook_installed = False

    def _names_a_root(self, text: str) -> bool:
        if text.startswith("file:"):
            # An SQLite URI filename, e.g. ``file:/path/db?mode=ro``.
            text = text[len("file:"):].split("?", 1)[0]
        if not os.path.isabs(text):
            text = os.path.abspath(text)
        for prefix in self._prefixes:
            if text == prefix or text.startswith(prefix + os.sep):
                return True
        return False

    def _audit(self, event: str, args: tuple) -> None:
        if not self._armed or event not in _PATH_AUDIT_EVENTS:
            return
        for arg in args:
            text = _as_path_text(arg)
            if text is not None and self._names_a_root(text):
                self._accesses.append(
                    (
                        os.environ.get("PYTEST_CURRENT_TEST", "<outside a test>"),
                        event,
                        text,
                    )
                )
                return

    def arm(self) -> None:
        """Snapshot the roots, then start recording in-process accesses."""
        self._accesses.clear()
        for root in self.roots:
            self._baselines[root] = _baseline(root)
        if not self._hook_installed:
            # Audit hooks cannot be removed; ``disarm`` makes this one inert.
            sys.addaudithook(self._audit)
            self._hook_installed = True
        self._armed = True

    def disarm(self) -> list[str]:
        """Stop recording and describe everything this run did to the roots."""
        self._armed = False
        findings: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for access in self._accesses:
            if access in seen:
                continue
            seen.add(access)
            test, event, path = access
            findings.append(f"{test}: {event} {path}")
        for root, baseline in self._baselines.items():
            after, _ = _snapshot(root)
            findings.extend(
                f"{root}: {change}" for change in _compare(baseline, after)
            )
            findings.extend(_descendants_holding(root))
        return findings

    @staticmethod
    def failure_message(findings: list[str]) -> str:
        return (
            "Host-data tripwire (#3286): this test run reached the operator's "
            "real host-runtime state. No test may resolve the real "
            "~/.kestrel/host-data; the autouse isolation in tests/conftest.py "
            "redirects it, and a test that opts out with "
            f"@pytest.mark.{OWNS_HOST_PATHS_MARKER} must redirect HOME and "
            "KESTREL_HOME itself.\n  "
            + "\n  ".join(findings)
        )


__all__ = [
    "HostDataTripwire",
    "ISOLATION_DIRNAME",
    "OWNS_HOST_PATHS_MARKER",
    "SPAWNED_ENV_PIN",
    "SpawnedEnvPin",
    "real_host_data_roots",
    "redirect_host_data_root",
]
