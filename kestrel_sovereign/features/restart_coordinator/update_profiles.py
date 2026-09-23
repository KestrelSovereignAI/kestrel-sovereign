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
import subprocess
import sys
import time
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


# How long one resolved default branch (and its tag-collision answer) is
# reused. The executor re-verifies an agent-filed update row at several
# boundaries per coordinator tick, and the boot-time issuance adoption
# re-verifies every stored row; one ``git`` process per verification would
# block the event loop for ~10ms each. The check made immediately before an
# update profile runs passes ``fresh=True`` and never uses this cache, so a
# re-pointed ``origin/HEAD`` or a new tag cannot be missed where it matters.
_DEFAULT_BRANCH_TTL_SECONDS = 30.0
_default_branch_cache: dict[tuple[str, ...], tuple[float, object]] = {}


def _git(repo_path: str, *args: str, timeout: float = 5):
    try:
        return subprocess.run(
            ["git", "-C", repo_path, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _read_checkout_default_branch(repo_path: str) -> str:
    result = _git(repo_path, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD")
    if result is None or result.returncode != 0:
        return ""
    ref = (result.stdout or "").strip()
    # The full target (not ``--short``, which may keep a disambiguating
    # prefix) is ``refs/remotes/origin/<branch>``. The update profile fetches
    # from ``origin``, so only a branch of that remote is the default.
    prefix = "refs/remotes/origin/"
    if not ref.startswith(prefix):
        return ""
    branch = ref[len(prefix):]
    return branch if is_valid_target_ref(branch) else ""


def _read_local_tag_exists(repo_path: str, name: str) -> bool | None:
    result = _git(repo_path, "show-ref", "--verify", "--quiet", f"refs/tags/{name}")
    if result is None:
        return None
    if result.returncode == 0:
        return True
    # ``show-ref --verify`` exits 1 exactly when the ref is absent; anything
    # else is a failure to read the namespace, which answers nothing.
    return False if result.returncode == 1 else None


def _cached(key: tuple[str, ...], fresh: bool, read):
    now = time.monotonic()
    cached = _default_branch_cache.get(key)
    if (
        not fresh
        and cached is not None
        and now - cached[0] < _DEFAULT_BRANCH_TTL_SECONDS
    ):
        return cached[1]
    value = read()
    _default_branch_cache[key] = (now, value)
    return value


def checkout_default_branch(repo_path: str, *, fresh: bool = False) -> str:
    """The checkout's default branch as ``origin/HEAD`` names it, or ``""``.

    The same source ``kestrel update`` reattaches a detached checkout to
    (``cli_lifecycle._git_reattach_if_safely_detached``), but with no fallback:
    this answers an authority question (#3339), so an unconfigured
    ``origin/HEAD`` means "no default branch is known", never ``"main"``.
    ``fresh=True`` re-reads ``origin/HEAD`` (and refreshes the cache).
    """
    if not repo_path:
        return ""
    return _cached(
        ("branch", repo_path), fresh,
        lambda: _read_checkout_default_branch(repo_path),
    )


def checkout_has_tag(
    repo_path: str, name: str, *, fresh: bool = False,
) -> bool | None:
    """Whether the checkout's local tag namespace has ``refs/tags/<name>``.

    ``None`` when the namespace could not be read. The profile's
    ``git fetch --tags`` mirrors origin's tags into this namespace, so after
    any update has run it reflects the remote; a tag created on origin since
    the last fetch is seen only by :func:`origin_has_tag`.
    """
    if not repo_path or not name:
        return None
    return _cached(
        ("tag", repo_path, name), fresh,
        lambda: _read_local_tag_exists(repo_path, name),
    )


def origin_has_tag(repo_path: str, name: str) -> bool | None:
    """Whether ``origin`` itself advertises ``refs/tags/<name>`` right now.

    One read-only ``git ls-remote`` (network). ``None`` when origin could
    not be asked, which is not evidence of absence.
    """
    if not repo_path or not name:
        return None
    result = _git(
        repo_path, "ls-remote", "--tags", "origin", f"refs/tags/{name}",
        timeout=30,
    )
    if result is None or result.returncode != 0:
        return None
    wanted = {f"refs/tags/{name}", f"refs/tags/{name}^{{}}"}
    return any(
        line.split("\t", 1)[-1].strip() in wanted
        for line in (result.stdout or "").splitlines()
    )


def clear_checkout_default_branch_cache() -> None:
    """Forget every cached default-branch and tag answer."""
    _default_branch_cache.clear()


@dataclass(frozen=True)
class UpdateStep:
    """One allowlisted command in a profile's update sequence."""

    name: str
    argv: List[str]
    cwd: Optional[str] = None
    # Read-only steps (e.g. resolving HEAD) observe state; they never
    # mutate the checkout and a non-zero exit is non-fatal to the update.
    read_only: bool = False
    # Best-effort steps may mutate but are expected to fail in some
    # legitimate configurations (e.g. reattaching a branch when the target
    # ref is a tag/sha); a non-zero exit is non-fatal to the update.
    allow_failure: bool = False
    # Coordinator-native routine name. Most steps are a single argv exec;
    # a native step is a small fixed routine implemented in the coordinator
    # (still argv-exec subprocesses, never a shell) for logic that needs a
    # runtime ref comparison a single command can't express. ``argv`` then
    # documents the mutating command the routine may run; ``native_args``
    # carries its validated parameters.
    native: Optional[str] = None
    native_args: tuple = ()


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


def _host_feature_manifest() -> Optional[Path]:
    """The host's out-of-tree feature manifest, if present.

    ``kestrel feature sync`` reads ``.kestrel-host-features.toml`` from the host
    process cwd (the launch/data root). The profile is built in that same
    process, so resolve it here and pass an ABSOLUTE path to the step so manifest
    discovery does not depend on the step's own cwd.
    """
    manifest = (Path.cwd() / ".kestrel-host-features.toml").resolve()
    return manifest if manifest.exists() else None


def _sovereign_local_uv_sync(
    repo_path: str, target_ref: str, allow_migrations: bool,
) -> List[UpdateStep]:
    git = ["git", "-C", repo_path]
    steps = [
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
        # Reattach the local branch for branch targets. The detach above
        # guarantees we run the fetched commit, but left as-is it strands
        # the checkout on a detached HEAD after every update, breaking the
        # operator's `git status`/`kestrel update` pull step ("not
        # currently on a branch"). Reattaching needs a runtime ref
        # comparison no single git command expresses — a name can exist as
        # BOTH a tag and a branch, and `fetch <name>` lands on the TAG
        # commit (tags win), so blindly attaching to origin/<name> would
        # install a different commit than the one fetched. The
        # coordinator-native routine mirrors the fetch's own precedence:
        # skip when the name is a tag (stay detached on the tag commit),
        # skip when origin has no such branch (sha targets), verify the
        # branch tip equals FETCH_HEAD, then `checkout -B` — forcing the
        # local branch onto the fetched commit (a deploy checkout must
        # land on origin's tip; stray local commits stay recoverable via
        # the reflog).
        UpdateStep(
            "reattach_branch",
            git + ["checkout", "-B", target_ref, "FETCH_HEAD"],
            allow_failure=True,
            native="reattach_branch",
            native_args=(repo_path, target_ref),
        ),
        UpdateStep("install", ["uv", "sync"], cwd=repo_path),
    ]

    # Restore out-of-tree feature packages that bare ``uv sync`` prunes (F234).
    # ``kestrel update`` runs ``kestrel feature sync`` immediately after its
    # sync for exactly this reason; ``update_then_restart`` must not diverge, or
    # a host restarts with its isolated/entry-point feature packages missing.
    # Only add the step when a manifest actually exists (feature sync exits 1 on
    # a missing manifest); a host with no out-of-tree features has nothing to
    # prune and needs no restore.
    manifest = _host_feature_manifest()
    if manifest is not None:
        # Invoke via the RUNNING interpreter (``sys.executable -m ...``), never a
        # bare ``kestrel`` PATH lookup: a host launched from a venv by absolute
        # path (systemd/cron) may not have ``kestrel`` on PATH, and a machine
        # with multiple installs could resolve it to the wrong interpreter and
        # restore features into a venv the host isn't running (codex P1).
        steps.append(
            UpdateStep(
                "feature_sync",
                [
                    sys.executable, "-m", "kestrel_sovereign.cli",
                    "feature", "sync", "--manifest", str(manifest),
                ],
                cwd=repo_path,
            )
        )

    # Always capture the actual landed commit LAST so the post-restart
    # completion signal can prove which ref we booted into.
    steps.append(
        UpdateStep("resolve_ref", git + ["rev-parse", "HEAD"], read_only=True)
    )
    return steps


SOVEREIGN_LOCAL_UV_SYNC = UpdateProfile(
    name="sovereign_local_uv_sync",
    description=(
        "Update a local Sovereign checkout: git fetch the target ref, "
        "land on the fetched commit (reattaching the local branch for "
        "branch targets, detached for tags/shas), then `uv sync` to "
        "install. Schema migrates additively on the subsequent boot, so "
        "this profile defines no explicit migration step."
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
