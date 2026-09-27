"""Keep every test away from the operator's real runtime state (#3286).

Two halves, both wired from ``tests/conftest.py``:

**Isolation.** Every variable in the guarded registry
``paths.RUNTIME_PATH_ENV_NAMES`` is pinned below a per-test temporary root or
removed, as :data:`ISOLATED_LAYOUT` says (:func:`pin_runtime_paths`). When
``tests/conftest.py`` is imported (:func:`install_session_guard`), before any
package module can freeze a path at import or a session fixture can resolve
one, **every** registered variable is pinned below a session root, the removed
ones included: a variable that is set cannot be refilled by the
``load_dotenv(override=False)`` that importing ``server`` runs against the
operator's ``.env``. Before #3286 only the unit tier isolated anything.
``tests/integration/test_feature_install_discover_e2e.py``'s
``TestFeatureStoreAPI`` client fixtures set ``KESTREL_DB_PATH`` alone and
entered the shared ``server.app`` lifespan, so ``prepare_host_database``
consulted the real ``~/.kestrel/host-data/host-features.db`` as the "previous
default" host database. With the production host running that refused on its
live sidecars; with the host stopped the same call would ``os.replace``
production's database into a pytest temporary directory.

**Enforcement.** A convention is not a guarantee, and a snapshot of the real
directory cannot tell a test's writes from a running host's. Instead
``KESTREL_TEST_STORAGE_ROOTS`` names the test's own temporary roots (the
session root, pytest's base temporary directory and the system temporary
directory), and every resolver that turns a path-shaped variable or
configuration into a storage path passes its answer through
``paths.guard_storage_path``, which refuses anything outside them with
``StoragePathOutsideTestRootsError``. An allow-list, not a list of the
operator's roots: environment, ``.env`` files and ``multi_agent.toml`` can
name any path, so the operator's roots cannot be enumerated, but anything
they name lies outside the test's temporary roots and is refused, whatever
production is writing at the time. ``tests/unit/test_runtime_path_census.py``
proves every such read goes through the guard. Children inherit the variable,
so a spawned process that picks up an operator path fails loudly too.
Refusals are also raised as an audit event; :class:`RefusalRecorder` turns one
a broad ``except`` swallowed into a test failure.

Tests that deliberately exercise path resolution opt out of the isolation with
``@pytest.mark.owns_host_paths``: every registered variable is removed, and
they must redirect ``HOME``/``KESTREL_HOME`` themselves. The guard still
applies to them.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pytest

from kestrel_sovereign import paths

#: Marker name for tests that own host/home path resolution themselves.
OWNS_HOST_PATHS_MARKER = "owns_host_paths"

#: Name of the fixture's own temporary root. It is a sibling of each test's
#: ``tmp_path``, not a child: the unit tier writes a host manifest here, and a
#: directory the test owns is the wrong place for the fixture's state — one
#: test asserts its ``tmp_path`` is empty, and every test is entitled to.
ISOLATION_DIRNAME = "_kestrel_host_runtime_isolation"

#: The isolated ``KESTREL_HOME``, below the isolation root.
PROJECT_HOME_DIRNAME = "kestrel-home"

#: Where each registered runtime-path variable points below a per-test
#: isolation root, or ``None`` to remove it for the test. Exactly the keys of
#: ``paths.RUNTIME_PATH_ENV_NAMES`` (``tests/unit/test_runtime_path_census.py``
#: enforces it). A removed variable is one whose *presence* changes behaviour
#: and whose default already follows a pinned one: the identity-export and
#: data-dir overrides fall back to ``KESTREL_DB_PATH``, the launcher pins are
#: written only by a launcher, and the project-root override falls back to
#: source discovery. At session level these too are pinned, below
#: :data:`SESSION_ONLY_DIRNAME`, to paths nothing creates.
ISOLATED_LAYOUT: dict[str, Optional[str]] = {
    paths.HOME_ENV: PROJECT_HOME_DIRNAME,
    paths.HOST_DB_PATH_ENV: "host-data/host-features.db",
    paths.DERIVED_HOST_DB_PATH_ENV: None,
    paths.HOST_DB_PREVIOUS_DEFAULT_ENV: None,
    paths.HOST_DB_LEGACY_PATH_ENV: None,
    paths.AGENT_DB_PATH_ENV: "agent-data",
    # The identity/key resolver's legacy root; unset it defaults to a
    # cwd-relative ``agent_data``, which names the checkout's own state.
    paths.LEGACY_AGENT_DATA_DIR_ENV: "agent-data",
    paths.DATA_DIR_ENV: None,
    paths.IDENTITY_EXPORT_DIR_ENV: None,
    paths.PHOENIX_WORKING_DIR_ENV: "host-data/phoenix",
    # Defaults: ``./storage_cache`` (the checkout), ``~/.kestrel/trash`` and
    # ``~/.kestrel`` (the operator's home).
    paths.CACHE_DIR_ENV: "storage-cache",
    paths.TRASH_DIR_ENV: "trash",
    paths.SERVE_STATE_DIR_ENV: "serve-state",
    paths.PROJECT_ROOT_ENV: None,
    # A registry nothing creates: unset, the host loads a cwd-relative
    # ``multi_agent.toml``, which in the operator's checkout names the
    # operator's agents.
    paths.MULTI_AGENT_CONFIG_ENV: "multi_agent.toml",
    # Unset, local MPS training defaults below the operator's real ``HOME``.
    paths.TRAINING_WORKING_DIR_ENV: "training",
}

#: Below the session root: where the variables a test removes are pinned for
#: the session. Inert values by construction: the launcher pins never match
#: the pinned ``KESTREL_HOST_DB_PATH``, so they are ignored, and a missing
#: ``KESTREL_PROJECT_ROOT`` falls back to source discovery.
SESSION_ONLY_DIRNAME = "session-only"

#: The session's own isolation root, exported so a child session (an xdist
#: worker, a nested pytest) knows the runtime-path values it inherited are
#: pins, not the operator's configuration.
SESSION_ROOT_ENV = "KESTREL_TEST_SESSION_ISOLATION_ROOT"


def allowed_storage_roots(basetemp: Optional[Path] = None) -> tuple[Path, ...]:
    """The roots this process lets a resolved runtime path lie in.

    The system temporary directory (every ``mkdtemp`` and the default pytest
    base temporary directory live below it) plus an explicit ``--basetemp``
    and ``PYTEST_DEBUG_TEMPROOT``, which may lie elsewhere. On POSIX, also
    ``/tmp``: it is ``gettempdir()`` on Linux CI, and many tests use it as a
    placeholder root, so without it a macOS run (whose ``gettempdir()`` is
    per-user) would refuse what CI allows.
    """
    roots = [Path(tempfile.gettempdir())]
    if os.name == "posix" and os.path.isdir("/tmp"):
        roots.append(Path("/tmp"))
    debug_temproot = os.environ.get("PYTEST_DEBUG_TEMPROOT")
    if debug_temproot:
        roots.append(Path(debug_temproot))
    if basetemp is not None:
        roots.append(basetemp)
    return tuple(Path(os.path.abspath(root)) for root in roots)


def allow_storage_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Add ``roots`` to :data:`paths.STORAGE_ROOTS_ENV`; return the full list.

    Merged with a value inherited from an enclosing session, whose temporary
    roots a nested session's own lie below anyway.
    """
    inherited = [
        Path(entry)
        for entry in os.environ.get(paths.STORAGE_ROOTS_ENV, "").split(os.pathsep)
        if entry
    ]
    merged = tuple(dict.fromkeys([*inherited, *roots]))
    os.environ[paths.STORAGE_ROOTS_ENV] = os.pathsep.join(
        str(entry) for entry in merged
    )
    return merged


def pin_runtime_paths(
    root: Path,
    setenv: Callable[[str, str], None],
    delenv: Optional[Callable[[str], None]],
) -> None:
    """Point every registered runtime-path variable as :data:`ISOLATED_LAYOUT` says.

    With no ``delenv`` (the session pre-pin) a removed variable is pinned
    below :data:`SESSION_ONLY_DIRNAME` instead, so nothing is left unset for
    a ``.env`` load to fill. Only the project home is created, because a
    configured ``KESTREL_HOME`` names an existing project. Each storage owner
    creates and validates its own directory, the same custody path production
    takes.
    """
    for name, relative in ISOLATED_LAYOUT.items():
        if relative is not None:
            setenv(name, str(root / relative))
        elif delenv is None:
            setenv(name, str(root / SESSION_ONLY_DIRNAME / name))
        else:
            delenv(name)
    (root / PROJECT_HOME_DIRNAME).mkdir(exist_ok=True)


def isolate_host_runtime_paths(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-test :func:`pin_runtime_paths`, undone by ``monkeypatch``."""
    pin_runtime_paths(
        root,
        monkeypatch.setenv,
        lambda name: monkeypatch.delenv(name, raising=False),
    )
    # ``project_dir`` memoizes on ``(KESTREL_HOME, cwd)``; the key changes with
    # the value, but per-test homes would otherwise evict real entries.
    paths.reset_cache()


def install_session_guard() -> Optional[Path]:
    """Allow only temporary roots and pin a session root, before any import.

    Called from the top of ``tests/conftest.py``, so it runs before any package
    module is imported: a path frozen at import (``DEFAULT_TRASH_DIR``,
    ``cli_serve.STATE_DIR``) or resolved by a session fixture lands in the
    session root, and a guarded resolver never meets the operator's. pytest's
    ``--basetemp`` is added in ``pytest_configure``, once options are parsed.

    A child session (an xdist worker, a nested pytest) inherits the parent's
    allow-list and pins; its runtime-path values are already temporary.
    Returns the session root this process created, or ``None``.
    """
    allow_storage_roots(allowed_storage_roots())
    if os.environ.get(SESSION_ROOT_ENV):
        return None
    session_root = Path(tempfile.mkdtemp(prefix="kestrel-test-session-"))
    os.environ[SESSION_ROOT_ENV] = str(session_root)
    pin_runtime_paths(session_root, os.environ.__setitem__, None)
    return session_root


@dataclass(frozen=True)
class Refusal:
    path: str
    source: str


class RefusalRecorder:
    """Record every ``StoragePathOutsideTestRootsError`` raised in this process.

    A refusal can be swallowed by a broad ``except`` on its way up; the audit
    event cannot. Installed once per process (audit hooks are permanent).
    """

    def __init__(self) -> None:
        self._refusals: list[Refusal] = []
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        sys.addaudithook(self._hook)
        self._installed = True

    def _hook(self, event: str, args: tuple) -> None:
        if event == paths.STORAGE_PATH_REFUSED_AUDIT_EVENT:
            self._refusals.append(Refusal(*args))

    def drain(self) -> list[Refusal]:
        refusals, self._refusals = self._refusals, []
        return refusals


#: The one recorder the suite installs (see ``tests/conftest.py``).
REFUSAL_RECORDER = RefusalRecorder()


def refusal_failure_message(refusals: list[Refusal]) -> str:
    lines = [
        "This test resolved a runtime path outside its temporary roots; the "
        "resolver refused (#3286). Isolate the test, or mark it "
        "owns_host_paths and redirect HOME/KESTREL_HOME yourself:"
    ]
    lines.extend(
        f"  - {refusal.source} {refusal.path}" for refusal in refusals
    )
    return "\n".join(lines)
