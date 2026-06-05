"""Allowlisted update/install profiles for ``update_then_restart`` (#1539).

The restart coordinator must never run arbitrary shell from request
fields. An ``update_then_restart`` request names one of a small,
curated set of profiles defined here; each profile expands to a fixed
sequence of argv-list steps (no shell, ``shell=False``). Only the repo
path and the target ref flow in as *data*, and both are validated
before they are ever handed to a subprocess.

Keeping the profiles here — rather than accepting a command string on
the request — is what makes update-and-restart auditable: a reviewer
can read the exact commands a profile will run, and a request can only
select a profile, never compose one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# A git ref/branch/tag/sha is data, never a command. Restrict it to the
# characters git refs legitimately use and forbid a leading dash so a
# crafted ref can never be parsed by git as an option (e.g. ``--upload-pack``).
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")


def is_valid_target_ref(ref: str) -> bool:
    """True if ``ref`` is a safe git ref/branch/tag/sha to pass as argv."""
    if not ref or len(ref) > 200:
        return False
    if ".." in ref or ref.endswith(".lock"):
        return False
    return bool(_REF_RE.match(ref))


def repo_is_git_checkout(path: str) -> bool:
    """True if ``path`` is an existing directory holding a ``.git`` entry."""
    if not path:
        return False
    try:
        p = Path(path)
        return p.is_dir() and (p / ".git").exists()
    except OSError:
        return False


def default_sovereign_repo_path() -> str:
    """Best-effort resolve of the local Sovereign checkout root.

    Walks up from this module's location looking for the ``.git`` that
    roots the working tree. Returns ``""`` if none is found (e.g. a
    pip-installed deployment with no checkout) — callers then require an
    explicit ``repo_path``.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return str(parent)
    return ""


@dataclass(frozen=True)
class UpdateStep:
    """One allowlisted command in a profile's update sequence."""

    name: str
    argv: List[str]
    cwd: Optional[str] = None
    # Read-only steps (e.g. resolving HEAD) observe state; they never
    # mutate the checkout and a non-zero exit is non-fatal to the update.
    read_only: bool = False


@dataclass(frozen=True)
class UpdateProfile:
    name: str
    description: str
    supports_migrations: bool
    _build: Callable[[str, str, bool], List[UpdateStep]] = field(repr=False)

    def build_steps(
        self, *, repo_path: str, target_ref: str, allow_migrations: bool,
    ) -> List[UpdateStep]:
        return self._build(repo_path, target_ref, allow_migrations)


def _sovereign_local_uv_sync(
    repo_path: str, target_ref: str, allow_migrations: bool,
) -> List[UpdateStep]:
    git = ["git", "-C", repo_path]
    return [
        # Fetch the *specific* requested ref so FETCH_HEAD points at the
        # commit we want to land on (``--tags`` keeps tag refs current for
        # tag targets, ``--prune`` drops deleted upstream refs).
        UpdateStep(
            "fetch",
            git + ["fetch", "--tags", "--prune", "origin", target_ref],
        ),
        # Detach straight onto the just-fetched commit. A named
        # ``git checkout <branch>`` would switch to the *local* branch at
        # its OLD commit — ``git fetch`` updates remote-tracking refs, not
        # the local branch — so the freshly-fetched commits would never
        # land and ``uv sync`` would re-install stale code (the headline
        # bug this feature exists to prevent). Checking out FETCH_HEAD
        # lands on the fetched commit regardless of whether the ref is a
        # branch, tag, or sha.
        UpdateStep("checkout", git + ["checkout", "--detach", "FETCH_HEAD"]),
        UpdateStep("install", ["uv", "sync"], cwd=repo_path),
        # Always capture the actual landed commit so the post-restart
        # completion signal can prove which ref we booted into.
        UpdateStep(
            "resolve_ref", git + ["rev-parse", "HEAD"], read_only=True,
        ),
    ]


SOVEREIGN_LOCAL_UV_SYNC = UpdateProfile(
    name="sovereign_local_uv_sync",
    description=(
        "Update a local Sovereign checkout: git fetch the target ref and "
        "detach onto the fetched commit (so branch targets actually "
        "advance), then `uv sync` to install. Schema migrates "
        "additively on the subsequent boot, so this profile defines no "
        "explicit migration step."
    ),
    supports_migrations=False,
    _build=_sovereign_local_uv_sync,
)


UPDATE_PROFILES = {p.name: p for p in (SOVEREIGN_LOCAL_UV_SYNC,)}

# Names an ``update_then_restart`` request may select. Anything else is
# rejected at request time and, defensively, by the coordinator.
KNOWN_UPDATE_PROFILES = frozenset(UPDATE_PROFILES)


def get_update_profile(name: str) -> Optional[UpdateProfile]:
    return UPDATE_PROFILES.get(name)
