"""Kestrel CLI — the ``kestrel feature ...`` command family.

Extracted from ``cli.py`` (#1678) to keep the monolithic entrypoint
manageable. Follows the established ``cli_<group>.py`` convention (see
``cli_embeddings``, ``cli_ipfs``, ``cli_runpod``, ...): this module owns the
feature commands and their argparse registration via
:func:`add_feature_subparser`.

Shared, test-patched helpers (``_get_project_dir``,
``_get_toml_disabled_features``, ``_set_toml_disabled_features``,
``_detect_running_agent_server``) stay defined in ``cli.py`` and are referenced
here through the ``cli`` module object at call time, so existing
``patch("kestrel_sovereign.cli.<helper>")`` test seams keep working unchanged.
"""
import functools
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Non-patched constants — import directly. (``MultiAgentConfig`` itself is
# referenced as ``cli.MultiAgentConfig`` because the test suite patches it.)
# ``feature_reconcile`` holds the pure planning types and imports nothing from
# the CLI, so naming the core distribution here is not a cycle.
from kestrel_sovereign.feature_reconcile import (
    CORE_DISTRIBUTION,
    canonical_package,
    package_for_label,
    resolve_registry_name,
)
from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME

# Return code for "core drifted and could not be repaired". Distinct from the
# generic ``1`` so callers can tell an unrepaired SAFETY failure from an
# ordinary package failure: `--continue-on-error` may wave the latter through,
# never this (issue #2949). See :meth:`CoreInstallGuard.verify`.
CORE_UNSAFE = 2

# Return code for "core is on its declared path but was NOT updated" — a
# declared editable checkout whose pull failed. Distinct from CORE_UNSAFE (core
# is the wrong install) and from a plain package failure, and like CORE_UNSAFE
# it is NOT continuable: `--continue-on-error` is documented for an optional
# package that would not install, and restarting the fleet onto stale core code
# while reporting success is not that (issue #2949).
CORE_STALE = 3

# Return code for "core is the declared install, at the declared version, and
# its dependencies do not resolve". A THIRD core state, because the operator
# instruction differs from both above: nothing needs restoring (CORE_UNSAFE)
# and nothing needs pulling (CORE_STALE) — a version conflict has to be fixed
# before this core can load. Reusing either code would print a sentence that is
# false in front of the operator. Not continuable, for the reason all three are
# not: `--continue-on-error` covers an optional package that would not install
# (issue #3047).
CORE_UNRESOLVED = 4

# Exit-code severity, worst first. `max()` on the raw ints is WRONG (CORE_STALE
# is 3 and CORE_UNSAFE is 2), which is exactly why ranking these by hand at each
# assignment site kept losing the stronger code. An undeclared core outranks an
# unloadable one, which outranks one merely running old code.
_RC_SEVERITY = (CORE_UNSAFE, CORE_UNRESOLVED, CORE_STALE, 1, 0)


#: Every core state no caller may continue past, and the sentence each one
#: means. ONE table, because the gates that ask are in three different
#: functions and a code added without all three learning it is silently
#: continuable — which is the failure `--continue-on-error` keeps producing.
#: The text is the operator's INSTRUCTION, and it differs per state: restore,
#: pull, or fix a conflict. That is why they are not one code.
NON_CONTINUABLE_CORE = {
    CORE_UNSAFE: (
        "core is not the declared install and could not be repaired. "
        "Refusing to continue onto an undeclared core"
    ),
    CORE_STALE: (
        "the declared core checkout could not be updated, so a restart would "
        "run stale code. Resolve the checkout (or pass --allow-dirty)"
    ),
    CORE_UNRESOLVED: (
        "core is the declared install, but the pass that validates its "
        "dependencies did not complete, so a restart could bring agents up on "
        "a core they cannot load. Resolve what failed and re-run"
    ),
}


def core_state_refusal(rc: int):
    """The operator sentence for a non-continuable core state, else ``None``.

    The one question a gate asks about an exit code, so "is this continuable"
    has exactly one answer everywhere it is asked.
    """
    return NON_CONTINUABLE_CORE.get(rc)


def _worst_rc(*codes: int) -> int:
    """Fold several exit codes into the most severe one.

    ``cmd_feature_sync`` tracks two INDEPENDENT facts: did some package fail to
    install, and is core in a state a restart must not proceed over. They shared
    a single ``rc``, so an optional feature failing *after* a failed core pull
    overwrote :data:`CORE_STALE` with ``1`` — a code `--continue-on-error`
    ignores by contract, which restarted the fleet onto stale core code and
    reported success. That is the very outcome CORE_STALE was added to prevent
    (issue #2949).

    Two facts get two variables, and the ranking lives in one place so a new
    assignment site cannot forget it.
    """
    for rank in _RC_SEVERITY:
        if rank in codes:
            return rank
    # An unrecognised non-zero code still outranks success.
    return next((code for code in codes if code), 0)

# A ``file:`` URL path that is really a Windows drive: ``/C:/src`` — or the
# legacy bar spelling ``/C|/src`` that older tools still emit.
_DRIVE_URL_PATH = re.compile(r"^/[A-Za-z][:|](/|$)")


def _resolve_feature_name(name: str, registry: dict) -> Optional[str]:
    """Resolve a user-provided name to a registry short name.

    The implementation lives in
    :func:`kestrel_sovereign.feature_reconcile.resolve_registry_name` so the
    reconcile planner resolves manifest names by exactly these rules; this is
    the CLI's long-standing name (and patch seam) for it.
    """
    return resolve_registry_name(name, registry)


def _status_icon(status) -> str:
    """Return a status icon for display."""
    from kestrel_sovereign.feature_registry import FeatureStatus
    return {
        FeatureStatus.ENABLED: "\u2713",
        FeatureStatus.INSTALLED: "\u2713",
        FeatureStatus.DISABLED: "\u2717",
        FeatureStatus.AVAILABLE: "\u25cb",
    }.get(status, "?")


def cmd_feature(args) -> int:
    """Dispatch feature subcommands."""
    feature_commands = {
        "list": cmd_feature_list,
        "inventory": cmd_feature_inventory,
        "install": cmd_feature_install,
        "upgrade": cmd_feature_upgrade,
        "sync": cmd_feature_sync,
        "status": cmd_feature_status,
        "enable": cmd_feature_enable,
        "disable": cmd_feature_disable,
        "info": cmd_feature_info,
        "scaffold": cmd_feature_scaffold,
        "skills": cmd_feature_skills,
    }

    handler = feature_commands.get(args.feature_command)
    if handler is None:
        print(
            "Usage: kestrel feature "
            "{list|inventory|install|upgrade|sync|status|enable|disable|info|scaffold|skills}"
        )
        return 1
    return handler(args)


def cmd_feature_inventory(args) -> int:
    """Generate the discovered feature/tool/endpoint/command inventory."""
    from kestrel_sovereign.feature_inventory import (
        build_inventory,
        render_inventory_json,
        render_inventory_markdown,
        write_canonical_inventory,
    )

    if args.write:
        write_canonical_inventory()
        print("Wrote KESTREL_FEATURES.md")
        return 0

    inventory = build_inventory()
    output = (
        render_inventory_json(inventory)
        if args.format == "json"
        else render_inventory_markdown(inventory)
    )
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(output, end="")
    return 0


def cmd_feature_list(args) -> int:
    """List all features with installed/enabled status."""
    from kestrel_sovereign.feature_registry import (
        get_registry,
        FeatureStatus,
    )

    project_dir = cli._get_project_dir()
    toml_disabled = set(cli._get_toml_disabled_features(project_dir))

    registry = get_registry()

    # Override status for features disabled via kestrel.toml
    for info in registry.values():
        feature_classes = set(info.features)
        if feature_classes & toml_disabled:
            info.status = FeatureStatus.DISABLED

    # Separate installed/core from available-to-install
    installed = {}
    available = {}
    for name, info in sorted(registry.items()):
        if info.status in (FeatureStatus.ENABLED, FeatureStatus.INSTALLED, FeatureStatus.DISABLED):
            installed[name] = info
        else:
            available[name] = info

    # Print installed features
    if installed:
        print()
        print(f"  {'INSTALLED':<40} {'STATUS'}")
        print(f"  {'─' * 50}")
        for name, info in installed.items():
            icon = _status_icon(info.status)
            status_str = info.status.value
            label = info.description if len(info.description) <= 36 else info.description[:33] + "..."
            core_tag = " (core)" if info.core else ""
            print(f"  {icon} {name:<38}{core_tag:>7} {status_str}")

    # Print available features
    if available:
        print()
        print(f"  {'AVAILABLE TO INSTALL':<40} {'PACKAGE'}")
        print(f"  {'─' * 50}")
        for name, info in available.items():
            print(f"  \u25cb {name:<38} {info.package}")

    if not installed and not available:
        print("  No features found in registry")

    print()
    return 0


def cmd_feature_install(args) -> int:
    """Install a feature package via pip."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        print("Run 'kestrel feature list' to see available features")
        return 1

    info = registry[pkg_name]

    if info.core:
        print(f"Feature '{pkg_name}' is a core feature (already included in kestrel-sovereign)")
        return 0

    package = info.package
    print(f"Installing {package}...")

    # This package depends on kestrel-sovereign, so the install can resolve core
    # from the index and replace the operator's core with a wheel copy. Same
    # guard as `feature sync` / `update`'s reconcile — a single command is not
    # a safer path than a batch (issue #2949).
    guard = CoreInstallGuard.snapshot()

    result = guard.run([package])

    if result.returncode != 0:
        # Try git URL as fallback
        if info.git:
            print(f"pip install failed, trying git: {info.git}")
            result = guard.run([f"git+{info.git}"])

    if result.returncode == 0:
        print(f"Installed {package}")
        return guard.verify()
    else:
        print(f"Failed to install {package}")
        if result.stderr:
            # Show last few lines of error
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        if guard.constraints:
            print(
                f"  note: core is pinned to {guard.constraints[0]} for this "
                "install so a feature cannot silently replace it. If this is a "
                "version conflict, move core to a version the feature accepts "
                "— do not remove the pin."
            )
        bound_note = guard.manifest_bound_note()
        if bound_note:
            print(f"  note: {bound_note}")
        # A failed install can be the very thing that broke the link. An
        # unrepaired core outranks the install failure: the caller must be able
        # to tell "this package did not install" from "the venv's core is not
        # the declared one", because only the second forbids a restart.
        return guard.verify() or 1


def _file_url_to_path(url: str) -> str:
    """Decode a PEP 610 ``file:`` URL into a filesystem path.

    ``direct_url.json`` records a URL, not a path: a checkout under
    ``/src/My Project`` arrives as ``file:///src/My%20Project``, and one on
    Windows as ``file:///C:/src/kestrel`` (``file://server/share/...`` for a UNC
    share). Chopping the scheme off textually leaves ``/src/My%20Project`` or
    ``/C:/src/kestrel`` — paths that do not exist, which here is worse than
    merely untidy: :class:`CoreInstallGuard` captures this value as the checkout
    core must be restored *from*, so a mangled path turns a recoverable swap
    into an unrecoverable one (issue #2949).

    The URL's own shape picks the conversion, not the host platform, so a record
    decodes to the path whoever wrote it meant, wherever it is read. A
    Windows-authored path read on POSIX then simply does not exist there, which
    is the honest answer — synthesising ``/C:/src/kestrel`` instead invites a
    match against something that was never the checkout.
    """
    from urllib.parse import unquote, urlsplit

    parts = urlsplit(url)
    path = unquote(parts.path)
    host = unquote(parts.netloc)
    if host and host.lower() != "localhost":
        # RFC 8089 non-local authority: a Windows UNC share.
        return "\\\\" + host + path.replace("/", "\\")
    if _DRIVE_URL_PATH.match(path):
        # urlsplit keeps the URL's leading slash on ``/C:/src`` (and on the
        # legacy ``/C|/src`` spelling); the drive letter is the real root.
        return path[1] + ":" + (path[3:].replace("/", "\\") or "\\")
    return path


#: Outcomes of re-reading ``direct_url.json`` that are not file contents.
_ABSENT = object()
_UNREADABLE = object()


def _reread_direct_url(dist):
    """Read ``direct_url.json`` ourselves, distinguishing absent from unreadable.

    ``Distribution.read_text`` returns None for both, by design — it suppresses
    ``FileNotFoundError`` and ``PermissionError`` together. The guard needs them
    apart: absent is the positive evidence of an index install, unreadable is
    the state where nothing may be inferred.

    Returns the file's text, :data:`_ABSENT`, or :data:`_UNREADABLE`. A
    distribution whose metadata directory we cannot identify reads as ABSENT:
    index installs (which have no such file) are the overwhelming majority, so
    treating an unlocatable file as unknown would hold every ordinary core on
    any layout we cannot follow.

    Anchors on ``_path`` — the metadata directory — because that is where
    ``read_text`` looks and therefore where the file is. ``locate_file`` is the
    tempting public alternative and it is the WRONG anchor: it resolves against
    ``_path.parent`` (site-packages) because it exists to find *package* files,
    so a re-read through it silently checks a path the file was never at, and
    every unreadable file comes back "absent" — the exact fail-open this
    function was added to close.
    """
    base = getattr(dist, "_path", None)
    if base is None:
        return _ABSENT
    try:
        return base.joinpath("direct_url.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return _ABSENT
    except (OSError, ValueError, AttributeError):
        # PermissionError, IsADirectoryError, a decode failure, or a path object
        # that does not support reading: the file's status was never established,
        # so nothing may be concluded from it.
        return _UNREADABLE


def _direct_url_provenance(dist_name: str):
    """Read *dist_name*'s PEP 610 provenance. Returns a :class:`Provenance`.

    ``direct_url.json`` is written by every install that named its source
    directly — a VCS ref, a local path, a remote archive, an editable checkout.
    An install resolved from a package index writes no such file, so its absence
    is the only positive evidence of an index install (issue #2949).

    Returns the value object rather than a tuple deliberately: the third state
    (metadata that exists but will not read) has to be impossible to drop, and
    a caller destructuring three positional values can drop it. Every "did this
    come from an index?" question goes through :attr:`Provenance.is_from_index`,
    which already fails closed on unknown.

    ``url`` is normalized to a filesystem path for ``file:`` URLs and kept
    verbatim otherwise.
    """
    import importlib.metadata as md
    import json

    from kestrel_sovereign.feature_reconcile import Provenance

    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError:
        # Not installed, so there is no provenance to have. Absent, and known.
        return Provenance.from_index_install()
    except Exception:
        # The distribution is there but unreadable — that is not "no file".
        return Provenance.unknown()
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return Provenance.unknown()
    if raw is None:
        # `read_text` cannot be trusted to mean "absent". The stdlib's
        # PathDistribution suppresses FileNotFoundError, PermissionError,
        # IsADirectoryError, NotADirectoryError and KeyError alike and returns
        # None for all of them — so a direct_url.json that EXISTS but cannot be
        # read arrives here identical to one that was never written, and would
        # be reported as positive evidence of an index install. That is the
        # fail-open this reader exists to close, reintroduced by trusting an
        # API that is documented to swallow the difference.
        #
        # So re-read it ourselves and classify by what actually goes wrong.
        raw = _reread_direct_url(dist)
        if raw is _ABSENT:
            return Provenance.from_index_install()  # genuinely not there
        if raw is _UNREADABLE:
            return Provenance.unknown()
    if not raw.strip():
        return Provenance.unknown()  # present but empty is damage, not an index
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return Provenance.unknown()
    if not isinstance(data, dict):
        return Provenance.unknown()
    # PEP 610 REQUIRES ``url``. A file that parses but omits it, leaves it
    # empty, or gives a non-string is damaged — and damaged is unknown, not an
    # index install. Deriving "no url, therefore no direct URL" reproduced the
    # fail-open one layer down, where the type could no longer help.
    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        return Provenance.unknown()
    dir_info = data.get("dir_info")
    editable = bool(dir_info.get("editable")) if isinstance(dir_info, dict) else False

    # The rest of the identity lives OUTSIDE ``url``. PEP 610 puts a VCS
    # revision in ``vcs_info``, an archive's hashes in ``archive_info``, and the
    # subdirectory alongside them — so two commits of one repository, or two
    # builds of one archive path, carry the same url. Comparing urls alone
    # reports a real replacement as no change, which is the version-pin mistake
    # this whole change exists to correct, one field further in.
    def _text(value):
        return value if isinstance(value, str) and value.strip() else None

    vcs_info = data.get("vcs_info")
    revision = None
    vcs_kind = None
    if isinstance(vcs_info, dict):
        # `vcs` is REQUIRED by PEP 610 and lives outside `url`, which carries
        # only the transport address. Dropping it made the same address served
        # by two different VCS at one revision string compare identical.
        vcs_kind = _text(vcs_info.get("vcs"))
        # commit_id is the resolved pin; requested_revision is what was asked
        # for (a branch or tag, which moves). Prefer the resolved one.
        revision = _text(vcs_info.get("commit_id")) or _text(
            vcs_info.get("requested_revision")
        )

    archive_info = data.get("archive_info")
    archive_hash = None
    if isinstance(archive_info, dict):
        hashes = archive_info.get("hashes")
        if isinstance(hashes, dict):
            # Sorted so an equal set of hashes always compares equal, whatever
            # order the writer happened to emit.
            archive_hash = ",".join(
                f"{name}={digest}"
                for name, digest in sorted(hashes.items())
                if _text(digest)
            ) or None
        if archive_hash is None:
            archive_hash = _text(archive_info.get("hash"))  # legacy single field

    if url.startswith("file:"):
        path = _file_url_to_path(url)
        if not path:
            return Provenance.unknown()  # a file: URL that will not resolve
        url = path
    return Provenance.direct(
        url,
        editable=editable,
        revision=revision,
        subdirectory=_text(data.get("subdirectory")),
        archive_hash=archive_hash,
        vcs=vcs_kind,
    )


def _from_index(dist_name: str) -> bool:
    """Did *dist_name* come from a package index?

    The one question a ``pypi`` source declaration actually asks. An index
    resolution records no PEP 610 ``direct_url.json``; every other way of naming
    a source — editable checkout, VCS ref, local path, remote archive — records
    one. So this is the whole predicate, and ``not is_editable`` is not a
    substitute for it: a git-installed package is non-editable too, and used to
    satisfy a declaration it violates (issue #2949).

    A distribution that is not installed has no direct URL, so it reads as
    "from an index" here; callers check presence separately.

    **Fails closed on unknown.** Unreadable or damaged provenance answers False,
    because this predicate is asked in order to decide whether a declared source
    is satisfied, and "we could not tell" is not a yes. The cost of failing
    closed is a reinstall that was not strictly needed; the cost of failing open
    is the wrong source silently passing — which is the whole defect.

    Read through ``cli.`` so provenance has exactly ONE patchable seam: a test
    that stubs where a package came from must not be able to leave this helper
    answering from the developer's real venv.
    """
    return cli._direct_url_provenance(dist_name).is_from_index


def _editable_install_path(dist_name: str) -> Optional[str]:
    """Return the local source path if *dist_name* is an editable install.

    PEP 660 editable installs record ``direct_url.json`` with
    ``dir_info.editable == true``. Editable installs track a local checkout, so
    pip cannot meaningfully "upgrade" them.

    Narrower than :func:`_direct_url_provenance` on purpose: callers asking
    "can I upgrade this?" want the editable checkout only. Callers asking
    "where did this come from?" must use the provenance reader, because this
    one returns None for a non-editable direct URL exactly as it does for an
    index wheel.
    """
    # Unknown provenance is not an editable checkout: this answers "is there a
    # checkout to pull/relink?", and there is no path to hand back.
    return _direct_url_provenance(dist_name).editable_path


def _installed_extension_distributions() -> list:
    """Enumerate distinct pip distributions backing installed extension entry-points.

    Kestrel already knows what is loaded — each feature/provider class is wired
    through an entry-point whose distribution we can resolve — so no curated
    requirements file is needed. Returns a list of dicts sorted by name:
    ``{dist, version, editable_path, entries: [group:name, ...]}``.
    """
    import importlib.metadata as md
    from kestrel_sovereign.feature_registry import iter_extension_entry_points

    by_dist: dict = {}
    for group, ep in iter_extension_entry_points():
        dist = getattr(ep, "dist", None)
        if dist is None:
            continue
        entry = by_dist.setdefault(
            dist.name,
            {
                "dist": dist.name,
                "version": None,
                "editable_path": cli._editable_install_path(dist.name),
                "entries": [],
            },
        )
        if entry["version"] is None:
            try:
                entry["version"] = md.version(dist.name)
            except Exception:
                entry["version"] = "?"
        short_group = group.rsplit(".", 1)[-1]
        entry["entries"].append(f"{short_group}:{ep.name}")
    return sorted(by_dist.values(), key=lambda e: e["dist"])


def _installed_requirements(dist_name: str) -> tuple:
    """The raw ``Requires-Dist`` lines of the copy of *dist_name* ON DISK.

    A seam of its own, like :func:`_installed_version`, because the question it
    answers is about the artifact in the venv and not about what any install
    command asked for. Returns ``()`` for anything not installed or whose
    metadata will not read — an absent answer is not an unmet requirement.
    """
    import importlib.metadata as md

    try:
        return tuple(md.metadata(canonical_package(dist_name)).get_all(
            "Requires-Dist"
        ) or ())
    except Exception:  # noqa: BLE001 - absent or unreadable metadata
        return ()


def _unsatisfied_requirements(dist_name: str, extras=()) -> tuple:
    """Every active requirement of *dist_name* this venv does not satisfy.

    The venv-reading half of :func:`feature_reconcile.unsatisfied_requirements`,
    wired to the same seams the rest of this module reads through so the double
    that models a venv models this too.
    """
    import kestrel_sovereign.feature_reconcile as fr

    return fr.unsatisfied_requirements(
        dist_name,
        tuple(extras or ()),
        cli._installed_requirements,
        cli._installed_version,
    )


def _installed_version(dist_name: str) -> Optional[str]:
    """The version of *dist_name* installed in THIS venv, right now.

    Asked after an install to report what that install actually did — and read
    from the venv, never from the installer's output. The backend is uv
    whenever uv is on PATH (:func:`_install_backend_argv`), and the two
    backends describe an install in prose that does not match: uv writes
    ``+ pkg==ver`` to stderr, pip writes ``Successfully installed pkg-ver`` to
    stdout. Parsing pip's line made every upgrade on the DEFAULT backend read
    as "up to date", which also swallowed the restart notice that only prints
    when something moved (issue #2949).

    Lookup is by :func:`canonical_package`, so the three spellings of one
    distribution (the operator's, the registry's, and the package's own
    ``Name:``) resolve to one answer — including the dotted form PEP 503 folds
    and a dash/underscore swap does not.

    ``invalidate_caches`` because the install ran in a subprocess after this
    process started: the import system's directory caches predate it, and a
    package installed since then is otherwise invisible without re-execing.
    """
    import importlib
    import importlib.metadata as md

    importlib.invalidate_caches()
    try:
        return md.version(canonical_package(dist_name))
    except md.PackageNotFoundError:
        return None


def cmd_feature_upgrade(args) -> int:
    """Upgrade installed feature/provider packages via pip.

    Discovers the packages from their live entry-points (no requirements file),
    skips editable/dev installs (they track a local checkout), and upgrades the
    rest in place.
    """
    dists = cli._installed_extension_distributions()

    from kestrel_sovereign.feature_registry import load_registry

    requested = getattr(args, "names", None) or []
    registry = load_registry()
    # Three spellings of one distribution meet here: what the operator typed,
    # what the registry catalogues, and what the package's own METADATA wrote
    # (``ep.dist.name`` — ``Kestrel_Feature_Voice`` is legal for the project the
    # registry calls ``kestrel-feature-voice``). PEP 503 says those are the same
    # distribution, so every comparison and lookup is keyed on the canonical
    # form; matching raw strings reported an installed package missing and lost
    # its git fallback (issue #2949).
    git_urls = {
        canonical_package(info.package): info.git
        for info in registry.values() if info.package
    }

    if requested:
        wanted = set()
        for name in requested:
            pkg = _resolve_feature_name(name, registry)
            wanted.add(canonical_package(registry[pkg].package if pkg else name))
        unmatched = wanted - {canonical_package(d["dist"]) for d in dists}
        dists = [d for d in dists if canonical_package(d["dist"]) in wanted]
        for name in sorted(unmatched):
            print(f"Skipping '{name}': not an installed extension package")
        if not dists:
            return 1

    if not dists:
        print("No installed feature/provider packages to upgrade.")
        print(
            "Core features ship with kestrel-sovereign; add extras with "
            "'kestrel feature install <name>'."
        )
        return 0

    dry_run = getattr(args, "dry_run", False)

    # `--upgrade` is the most likely command to drag core forward: pip is free
    # to satisfy the newer feature's core requirement from the index, replacing
    # the operator's core. Guard the whole batch (issue #2949).
    guard = CoreInstallGuard.snapshot()

    print()
    print(f"  {'PACKAGE':<34} {'CURRENT':<10} {'ACTION'}")
    print(f"  {'-' * 62}")

    rc = 0
    upgraded = 0
    for d in dists:
        name = d["dist"]
        current = d["version"] or "?"
        if d["editable_path"]:
            print(f"  {name:<34} {current:<10} skip (editable -> {d['editable_path']})")
            continue
        if dry_run:
            print(f"  {name:<34} {current:<10} would upgrade")
            continue

        git_url = git_urls.get(canonical_package(name))
        result = guard.run(["--upgrade", name])
        if result.returncode != 0 and git_url:
            result = guard.run(["--upgrade", f"git+{git_url}"])

        if result.returncode != 0:
            rc = 1
            print(f"  {name:<34} {current:<10} FAILED")
            for line in (result.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}")
            if guard.constraints:
                print(
                    f"      note: core is pinned to {guard.constraints[0]}; a "
                    "version conflict here is a real skew, not the pin's fault."
                )
            bound_note = guard.manifest_bound_note()
            if bound_note:
                print(f"      note: {bound_note}")
            continue

        new_version = _installed_version(name)
        if new_version and new_version != current:
            upgraded += 1
            print(f"  {name:<34} {current:<10} upgraded -> {new_version}")
        else:
            print(f"  {name:<34} {current:<10} up to date")

    print()
    if not dry_run:
        print(f"  {upgraded} package(s) upgraded.")
        if upgraded:
            print("  Restart the host/agents to load the upgraded code.")
        # Detection half: an upgrade that bypassed the pin can't leave the
        # command reporting success over a replaced core. A core state is
        # returned verbatim rather than folded into rc — see verify() — and the
        # set of them lives in `core_state_refusal`, not in a comparison here.
        core_rc = guard.verify()
        if core_state_refusal(core_rc):
            return core_rc
        if core_rc:
            rc = 1
    return rc


# ---------------------------------------------------------------------------
# `kestrel feature sync` — the restore counterpart to `feature upgrade`.
#
# `feature upgrade` discovers packages from their *live* entry-points, which is
# perfect for upgrading but useless for restoring: once `uv sync` (or any
# environment rebuild) prunes an out-of-tree feature package, its entry-point
# is gone and there is nothing left to discover. `feature sync` closes that gap
# by reading a host-local manifest of the packages this host should keep
# installed and reinstalling whatever is missing — without ever declaring
# features in pyproject.toml (which would invert the open-core dependency
# direction).
# ---------------------------------------------------------------------------

DEFAULT_HOST_MANIFEST = ".kestrel-host-features.toml"


def _host_manifest_path(args) -> Path:
    """Resolve the host manifest path (``--manifest`` override or default)."""
    override = getattr(args, "manifest", None)
    if override:
        return Path(override).expanduser()
    return Path.cwd() / DEFAULT_HOST_MANIFEST


def _load_host_manifest(path: Path) -> list:
    """Parse a host manifest into a list of feature entry dicts.

    Schema (TOML)::

        [[feature]]
        name = "voice"                 # registry name or raw dist/package name
        editable = "/path/to/checkout" # editable source: install -e + git pull
        # --- OR ---
        pypi = ">=0.3,<0.4"            # PyPI source: pip --upgrade (spec/floor)
        extras = ["local"]             # optional extras

    ``editable`` and ``pypi`` are the two source forms (#1788) and are mutually
    exclusive: an entry records *one* place a package comes from. ``pypi = ""``
    means "from PyPI, any version". An entry with neither is the legacy
    "present / install latest from PyPI" form, preserved for back-compat.

    Raises ``ValueError`` on a malformed entry so the caller can report it.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - 3.10 and earlier
        import tomli as tomllib  # type: ignore

    with open(path, "rb") as f:
        data = tomllib.load(f)

    entries = data.get("feature", [])
    if not isinstance(entries, list):
        raise ValueError("manifest '[[feature]]' must be an array of tables")

    cleaned = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"manifest feature #{i + 1} is missing a 'name'")
        label = entry["name"]
        editable = entry.get("editable")
        if editable is not None and not isinstance(editable, str):
            raise ValueError(f"manifest '{label}': 'editable' must be a string path")
        pypi = entry.get("pypi")
        if pypi is not None and not isinstance(pypi, str):
            raise ValueError(
                f"manifest '{label}': 'pypi' must be a string version spec "
                "(e.g. \">=0.3,<0.4\" or \"\" for any)"
            )
        if pypi:
            # Reject a malformed specifier HERE, where the operator can see
            # which line is wrong. Downstream it fails silently and expensively:
            # `version_satisfies` conservatively answers True when a spec will
            # not evaluate, so every installed version "satisfies" a garbage
            # window — and the constraint rendered from it is `<pkg><spec>`,
            # which for `banana` is the package NAME `kestrel-sovereignbanana`
            # and so constrains nothing at all. A guard that pins an unrelated
            # package and then reports conformity is worse than no guard,
            # because it reports success (issue #2949).
            # The GUARD's validator, not `SpecifierSet` directly — they are
            # deliberately not the same question. `===` parses but carries no
            # version operand, so it matches nothing: accepted here it becomes
            # an installer requirement the backend rejects, or a core policy
            # that can never conform. Validating with a laxer rule than the
            # consumer applies is how the two sides come to disagree, which is
            # the defect pattern this whole change exists to remove.
            from kestrel_sovereign.feature_reconcile import spec_is_valid

            if not spec_is_valid(pypi):
                raise ValueError(
                    f"manifest '{label}': 'pypi' is not a usable PEP 440 version "
                    f"spec: {pypi!r}. Use e.g. \">=0.3,<0.4\", or \"\" for any "
                    "version from the index."
                )
        if editable is not None and pypi is not None:
            raise ValueError(
                f"manifest '{label}': 'editable' and 'pypi' are mutually "
                "exclusive — a feature has one source"
            )
        extras = entry.get("extras", []) or []
        # A bare string ("local") would silently iterate into characters
        # (['l','o','c','a','l']); require an explicit array.
        if not isinstance(extras, list) or not all(isinstance(x, str) for x in extras):
            raise ValueError(f"manifest '{label}': 'extras' must be an array of strings")
        cleaned.append({
            "name": str(label),
            "editable": editable,
            "pypi": pypi,
            "extras": list(extras),
        })
    return cleaned


def _pip_spec(target: str, extras: list) -> str:
    """Render a pip requirement spec with optional extras (``pkg[a,b]``)."""
    return f"{target}[{','.join(extras)}]" if extras else target


def _toml_basic_string(value: str) -> str:
    """Quote *value* as a valid TOML basic string.

    Escapes backslashes (Windows checkout paths), quotes, and control chars so
    a captured manifest is always re-parseable without pulling in a TOML
    writer dependency.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _have_uv() -> bool:
    """Is uv the installer backend on this host?

    The two backends are not interchangeable for the core guard: uv can scope a
    reinstall to one package, pip cannot (see :func:`_extension_install_run`).
    One predicate so the argv builder and the reinstall-scoping logic can never
    disagree about which backend is about to run.
    """
    import shutil

    return shutil.which("uv") is not None


def _install_backend_argv(pip_args: list, extra_args=()) -> list:
    """The exact argv an extension install runs as.

    uv-created virtualenvs frequently ship without ``pip``, so a bare
    ``python -m pip`` would fail. Prefer ``uv pip install --python <interp>``
    (pinned to *this* interpreter so a worktree can't retarget the wrong env),
    and fall back to ``python -m pip install`` only when uv isn't on PATH.

    Split out of :func:`_extension_install_run` because the recovery command the
    guard hands an operator after a failed automatic repair must be the command
    that actually ran. Rendering ``uv pip install ...`` unconditionally
    advertises a uv this host may not have, and drops ``--python``, so running
    the advertised "fix" by hand could mutate a different environment — the
    retargeting this guard exists to prevent (issue #2949).
    """
    if _have_uv():
        return [
            "uv", "pip", "install", "--python", sys.executable,
            *extra_args, *pip_args,
        ]
    return [sys.executable, "-m", "pip", "install", *extra_args, *pip_args]


def _install_commands(pip_args: list, extra_args=(), reinstall=None) -> list:
    """Every argv one install runs as, in order.

    One function so the executor and the recovery command the guard prints can
    never describe different operations: a printed fix that drops the scoping
    the real install used is issue #2949, re-opened by hand.

    Without *reinstall* an install is one command. With it — a source switch,
    where a plain install is a no-op because the already-linked version
    satisfies the spec — the two backends diverge:

    * uv scopes a reinstall to one package (``--reinstall-package``), so the
      switch stays a single command and no dependency is ever a candidate.
    * pip has no such flag; its ``--force-reinstall`` is all-or-nothing and
      cascades to every resolved dependency, core included. So the switch
      splits in two, and the ORDER is the protection:

      1. ``--upgrade`` (non-forcing) resolves the requested version from the
         index *with* its dependencies. A requirement the core pin forbids has
         no solution and fails HERE — while the package on disk is still the
         one that was working. Non-forcing, so a satisfying editable core keeps
         its link.
      2. ``--force-reinstall --no-deps`` replaces the package itself and only
         it. This is the destructive pass, and it runs after the resolve it
         depends on has already succeeded. It is needed because pass 1 is a
         no-op when the index publishes the same version the checkout builds —
         the case that motivated all of this.
      3. A plain install — the very command a non-switch install runs — so the
         environment is resolved around the artifact that ACTUALLY landed.

      Running 1 and 2 the other way round is a bug, not a style choice: the
      wheel lands, the dependency resolve then fails, and the caller reports a
      failure over an environment it has already changed.

    Pass 3 is there because pass 1 does not resolve the artifact pass 2
    installs. pip builds its candidate for a *name* requirement from the
    installed distribution whenever that distribution already satisfies the
    spec, so in the no-op case pass 1 reads the metadata of the copy on disk —
    the checkout build — and pass 2 then writes the index artifact with
    ``--no-deps``. Two artifacts at one version are not obliged to declare the
    same dependencies (an extra, a raised floor, a dependency added after the
    tag), so without pass 3 the environment is left holding a distribution
    whose dependency set was validated by NEITHER pass (issue #3047).

    Pass 3 resolves because by then the index artifact IS the installed one, so
    the candidate pip builds carries its metadata. It can only run after pass 2,
    and that is the honest cost: a requirement only the new artifact declares is
    refused with the switch already made. Moving that check ahead of pass 2
    would mean resolving the index artifact without installing it — which needs
    the resolver to ignore what is installed, and then core has to come from the
    index too. An editable core builds a version the index may never have
    published, so that resolve would refuse every install on a dev host. A late
    honest failure beats an early false one; the earlier check is still there in
    pass 1 for every case where pip can make it.
    """
    extra = list(extra_args)
    if not reinstall:
        return [_install_backend_argv(pip_args, extra)]
    if _have_uv():
        return [
            _install_backend_argv(pip_args, [*extra, "--reinstall-package", reinstall]),
        ]
    return [
        _install_backend_argv(pip_args, [*extra, "--upgrade"]),
        _install_backend_argv(pip_args, [*extra, "--force-reinstall", "--no-deps"]),
        _install_backend_argv(pip_args, extra),
    ]


def _is_windows() -> bool:
    """Whether a command printed for the operator needs Windows quoting."""
    return os.name == "nt"


#: The Windows shell :func:`_render_command` quotes for. One string cannot serve
#: both of Windows' shells: PowerShell needs the ``&`` call operator before a
#: quoted program, and a leading ``&`` is a syntax error to ``cmd.exe``. So the
#: operator is told which shell the printed command is quoted for rather than
#: handed a coin flip. PowerShell is the choice because it is what Windows
#: presents by default (Windows Terminal, the Start-menu entry) and what this
#: project's own Windows guidance in ``README.md`` already assumes.
WINDOWS_SHELL = "PowerShell"

#: Tokens PowerShell's argument-mode parser passes through verbatim, so the
#: common ``-m`` / ``--python`` / ``C:\\venv\\python.exe`` stay readable.
#: Deliberately narrow: everything else — including ``,`` (array operator),
#: ``@`` (splat), ``$``, and the redirection pair — is quoted.
_PS_BARE_TOKEN = re.compile(r"^[A-Za-z0-9_.:=+/\\-]+$")


def _powershell_quote(arg: str) -> str:
    """Quote *arg* as a PowerShell literal string.

    Single quotes suppress every parse rule PowerShell would otherwise apply —
    ``$`` expansion, the backtick escape, and the argument-mode metacharacters
    ``< > | & , ; @ ( ) { }`` — so a requirement such as
    ``kestrel-sovereign>=0.52,<0.53`` survives as ONE argument. A literal
    single quote doubles, which is PowerShell's only escape inside them.
    """
    if _PS_BARE_TOKEN.match(arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def _render_command(argv: list) -> str:
    """Render *argv* as a command the operator's shell will run as this argv.

    A recovery command exists solely to be pasted after an automatic repair
    failed, so one the shell rejects is worth no more than printing nothing —
    and one the shell silently reads as something ELSE is worse than both.

    ``shlex.join`` speaks POSIX ``sh`` only, and :func:`subprocess.list2cmdline`
    is not its Windows counterpart: it quotes for the MS C runtime the *child*
    parses, not for the shell that reads the line first. It leaves a version
    window like ``kestrel-sovereign>=0.52,<0.53`` bare, and ``<`` / ``>`` are
    shell syntax before they are ever an argument — PowerShell refuses the line
    outright, ``cmd.exe`` redirects into a file instead of passing the pin. Both
    ends of that are the retargeting this guard exists to prevent (issue #2949),
    re-entered through the printed fix.

    Windows therefore gets :data:`WINDOWS_SHELL` quoting: literal single-quoted
    arguments, behind the ``&`` call operator so a quoted interpreter path is
    RUN rather than echoed back as a string.
    """
    if _is_windows():
        return "& " + " ".join(_powershell_quote(a) for a in argv)
    return shlex.join(argv)


def _render_commands(argvs: list) -> str:
    """Render a whole install SEQUENCE as one line the operator's shell runs.

    A single command renders exactly as :func:`_render_command` — the common
    case, and the only one on a uv host. A pip source switch is three passes
    (see :func:`_install_commands`) and every one has to reach the operator:
    printing a prefix of them advertises a repair that does part of the job.

    The ORDER is the protection (:func:`_install_commands`), so the rendered
    line has to carry it: pass 2 replaces the package with ``--no-deps`` and is
    only correct once the resolve in pass 1 has succeeded. POSIX gets ``&&``,
    which says exactly that.

    Windows PowerShell 5.1 — what ships with Windows — has no ``&&``, and a
    line the shell refuses is worth nothing to someone whose core is already
    off its declared source. Joining with ``;`` parses, but it runs pass 2
    whatever pass 1 did: the destructive pass lands after the resolve that
    forbade it, which is the ordering hazard :func:`_install_commands` exists
    to prevent, handed to the operator as the fix for it. So each following
    pass is nested inside ``if ($?) { ... }`` — PowerShell sets ``$?`` from a
    native command's exit code, and nesting rather than chaining keeps every
    pass conditional on every pass before it.
    """
    rendered = [_render_command(argv) for argv in argvs]
    if not _is_windows():
        return " && ".join(rendered)
    line = rendered[-1]
    for earlier in reversed(rendered[:-1]):
        line = f"{earlier}; if ($?) {{ {line} }}"
    return line


def _render_shell() -> Optional[str]:
    """Name the shell :func:`_render_command` just quoted for, when it matters.

    ``None`` on POSIX: ``sh`` quoting is what a POSIX operator's shell reads
    anyway, so naming it would be noise. On Windows the name is load-bearing —
    it is the difference between a command that runs and one that errors.
    """
    return WINDOWS_SHELL if _is_windows() else None


def _extension_install_run(pip_args: list, *, constraints, reinstall=None, timeout=None):
    """Install via uv when available, else the interpreter's own pip.

    Prefer :meth:`CoreInstallGuard.run` — it decides ``constraints`` for you and
    verifies the result. ``constraints`` is keyword-only and REQUIRED so a new
    caller has to make the core-guard decision explicitly instead of inheriting
    a silent default that reopens issue #2949; pass ``None`` only when core
    itself is the install target (it is never constrained against itself).

    ``reinstall`` names the ONE package a caller wants force-reinstalled (used
    when a feature switches source, e.g. an editable checkout → a PyPI wheel,
    where a plain install is a no-op because the linked version already
    satisfies the spec). Never append a bare ``--force-reinstall`` to
    *pip_args* instead: that flag cascades to every resolved dependency, and
    core is a dependency of every feature package, so it reinstalls core from
    the index as a wheel over the editable link — issue #2949 on the path most
    likely to hit it. A version constraint cannot stop that, because the index
    carries the same version the checkout builds and a same-version wheel
    satisfies the pin exactly.

    The backend (uv, else this interpreter's pip) and the argv sequence it runs
    as come from :func:`_install_commands` — one command on uv, two on pip,
    which the guard also renders its recovery command from. They run in order
    and stop at the first failure, so the result returned is always the one
    that decided the outcome. ``timeout`` bounds each installer subprocess (it
    is :func:`subprocess.run`'s), so a two-pass pip switch can take up to twice
    it; every path is still bounded, which is what a caller with nobody
    watching needs.

    ``uv pip`` vs ``uv sync`` — do NOT "fix" this by switching installers.
    ``uv pip`` is uv's *pip-compatible* layer: it operates on a bare virtualenv
    and deliberately does not read ``uv.lock`` or the project config. That is
    exactly what we want here, because feature packages are intentionally NOT
    declared in ``pyproject.toml`` (declaring them would invert the open-core
    dependency direction) — ``uv sync`` would prune every one of them.

    The cost of project-blindness is that uv never learns the lock declares the
    root as ``source = { editable = "." }``. It sees an ordinary dependency
    named ``kestrel-sovereign``, and when the installed version fails a
    feature's pin it resolves one from the index — silently replacing the
    editable checkout with a wheel copy (issue #2949). ``constraints`` closes
    that hole: a list of PEP 508 requirement strings written to a temporary
    constraints file and passed as ``-c``. Callers pin core to the version it is
    installed at, so a real version skew fails the resolve loudly instead of
    swapping the install underneath the operator. The pin bounds core's
    VERSION; ``reinstall`` is what keeps its SOURCE — the two holes are
    different and need both.

    ``--no-deps`` is NOT the fix for either: on its own it stops feature
    dependencies resolving at all and hides the version-skew signal rather than
    surfacing it. The pip source switch does use it — but only in a pass that
    follows one which resolved those dependencies, and fails if they do not
    resolve (:func:`_install_commands`).
    """
    import tempfile

    constraint_file = None
    extra_args: list = []
    if constraints:
        fd, constraint_file = tempfile.mkstemp(
            prefix="kestrel-constraints-", suffix=".txt",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(constraints) + "\n")
        extra_args = ["-c", constraint_file]

    try:
        result = None
        commands = _install_commands(pip_args, extra_args, reinstall)
        for index, cmd in enumerate(commands):
            try:
                completed = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired as expired:
                # A bound raises before the result exists, so the position
                # would be lost — and with it the only way to know that a kill
                # in the WRITE pass left an artifact nothing has validated.
                raise InstallSequenceTimeout(
                    expired, index=index, total=len(commands),
                ) from expired
            result = InstallSequenceResult(
                completed, index=index, total=len(commands),
            )
            if result.returncode != 0:
                break
        return result
    finally:
        if constraint_file:
            try:
                os.unlink(constraint_file)
            except OSError:
                pass


class InstallSequenceTimeout(subprocess.TimeoutExpired):
    """A timed-out install that still knows WHICH pass it was in.

    A subclass so every existing ``except subprocess.TimeoutExpired`` keeps
    catching it, and so :func:`_killed_process` can rebuild the position the
    raise would otherwise have thrown away.
    """

    def __init__(self, expired: subprocess.TimeoutExpired, *, index: int, total: int):
        super().__init__(
            expired.cmd, expired.timeout, output=expired.output, stderr=expired.stderr,
        )
        self.index = index
        self.total = total


class InstallSequenceResult(subprocess.CompletedProcess):
    """A finished install sequence, plus WHICH of its passes ended it.

    The exit code alone cannot say what a late failure means. A pip source
    switch is three passes and only the LAST resolves the dependencies of what
    landed (:func:`_install_commands`), so a caller has to tell "the resolve
    refused after the artifact was written" from "the destructive pass failed
    over an artifact already in place" — the second is the case re-reading core
    was always the right answer for.

    Position, not argv shape: the closing pass of a switch and the only pass of
    an ordinary install are the SAME command, and reading the argv called an
    ordinary failed install a refused resolve. What distinguishes them is that
    one ran after two passes had already changed the environment.
    """

    def __init__(self, completed, *, index: int, total: int):
        super().__init__(
            completed.args, completed.returncode, completed.stdout, completed.stderr,
        )
        self.index = index
        self.total = total

    @property
    def resolve_incomplete(self) -> bool:
        """Did a multi-pass sequence end without its closing resolve succeeding?

        Not "did the last pass refuse". The environment is in the SAME state
        whether the closing resolve refused or an earlier pass stopped the
        sequence before it ran: the ``--no-deps`` pass has written an artifact
        and nothing has validated what it declares. Asking only about the last
        pass missed the second case entirely.

        Pass 1 is exempt — it fails before anything is written, so that is an
        ordinary failed install over an environment still intact — and so is a
        single-command sequence, which is pass 1 and nothing else.

        Deliberately conservative where it cannot tell: a ``--no-deps`` pass
        that fails after a pass 1 which really did resolve leaves a fine
        environment, and this reports it anyway, because the exit code cannot
        distinguish that from a pass 1 that no-oped. The costs are not
        symmetric — a false positive costs a re-run, a false negative restarts
        a fleet onto a core it cannot load.

        A killed process never reaches this class at all — the timeout path
        builds a plain :class:`~subprocess.CompletedProcess`
        (:func:`_killed_process`) — which is how "a bound ended it" stays
        distinct from "the sequence did not finish".
        """
        return self.returncode != 0 and self.total > 1 and self.index > 0


def _killed_process(expired: subprocess.TimeoutExpired) -> subprocess.CompletedProcess:
    """A timed-out install, shaped like the result its callers already read.

    ``TimeoutExpired`` is not a variant of "the installer finished badly" that
    every reporting path would have to learn: it carries the output captured
    before the kill, and the only extra fact is WHY that output stops where it
    does. Folding it into a failed :class:`~subprocess.CompletedProcess` with
    that fact in ``stderr`` keeps one shape flowing to the operator.

    The pass POSITION survives when the raise carried it
    (:class:`InstallSequenceTimeout`), so a kill in the write pass is still
    read as an artifact nothing has validated. A kill in the FIRST pass keeps
    the documented answer — judge by where core is — because that pass is the
    resolve, and this is the one place a killed installer and a refused one
    differ: the bound ended a process, it did not make a resolver say no.
    """
    def text(stream) -> str:
        if stream is None:
            return ""
        return stream if isinstance(stream, str) else stream.decode(errors="replace")

    killed = subprocess.CompletedProcess(
        expired.cmd,
        returncode=1,
        stdout=text(expired.stdout),
        stderr=f"timed out after {expired.timeout}s\n{text(expired.stderr)}".strip(),
    )
    if isinstance(expired, InstallSequenceTimeout):
        return InstallSequenceResult(
            killed, index=expired.index, total=expired.total,
        )
    return killed


def _core_install_shape():
    """Snapshot how ``kestrel-sovereign`` is installed in this venv.

    Returns a :class:`~kestrel_sovereign.feature_reconcile.CoreInstallShape`.
    Read through ``cli.`` so the test suite's patch seams apply.

    The version comes from :func:`_installed_version` — the same live read the
    upgrade report uses — because this snapshot is taken *after* installer
    subprocesses have run and has to see what they left behind.
    """
    from kestrel_sovereign.feature_reconcile import CoreInstallShape

    return CoreInstallShape(
        version=_installed_version(CORE_DISTRIBUTION),
        provenance=cli._direct_url_provenance(CORE_DISTRIBUTION),
    )


@dataclass
class CoreGuardOutcome:
    """What the guard found after a batch of installs, and what it did about it.

    ``drift is None`` is the only state that needs no operator attention.
    Otherwise the batch left core somewhere it was not declared to be:
    ``repaired`` says whether reinstalling from the declared source put it back,
    and ``command`` is what the operator must run by hand if it did not.
    """

    drift: Optional[str] = None
    # Core matched its policy at the batch's baseline, so this batch is what
    # moved it — a swap, as opposed to a core that never conformed to begin with.
    replaced: bool = False
    repaired: bool = False
    # Was a repair even ATTEMPTED? An interrupted install is checked but never
    # repaired — an abort must not start installing — so "we tried and failed"
    # and "we did not try" are different facts and must not print the same
    # sentence (issue #2962).
    attempted: bool = True
    # Core IS back from its declared source, but the pass that resolves its
    # dependencies refused. Not a failed restore — saying so would send the
    # operator to redo an install that happened — and not a success either: the
    # host cannot load what it just installed. A third sentence, because it is
    # a third fact (issue #3047).
    unresolved: bool = False
    command: Optional[str] = None
    # Which shell ``command`` is quoted for, captured with it (see
    # :func:`_render_shell`). ``None`` where the platform has only one answer.
    shell: Optional[str] = None
    output: str = ""  # tail of a failed repair's stderr/stdout

    @property
    def conforming(self) -> bool:
        """Does core match its declared source RIGHT NOW?

        The question every caller must answer before reporting success. A drift
        that was repaired is conforming again; one that could not be repaired
        leaves the host running a core nobody declared.
        """
        return self.drift is None or self.repaired

    @property
    def headline(self) -> str:
        return (
            f"{CORE_DISTRIBUTION} was replaced during the install batch" if self.replaced
            else f"{CORE_DISTRIBUTION} is not installed from its declared source"
        )

    @property
    def restore_instruction(self) -> str:
        """The 'we could not put it back, here is how' line — worded once.

        The CLI prints it and the HTTP surface embeds it via :meth:`describe`,
        so it lives here rather than at both call sites: an operator comparing
        the two must not have to wonder whether they mean the same thing. Names
        the shell when :attr:`shell` says the quoting is specific to one, since
        a command pasted into the other shell is not the command that ran.
        """
        if not self.command:
            # No DECLARED source, so no command to offer — which is not the same
            # thing as "the source is unknown". A core installed from a known git
            # ref that no manifest declares reaches here too, and `resolve()`
            # names that ref in the detail immediately above. Claiming ignorance
            # over a line that states the source contradicts itself in front of
            # the operator. The wording has to be true of both cases, and what is
            # true of both is that nothing declared where core belongs.
            return (
                "NOT REPAIRED — no declared source to restore from, so there is "
                "no command that could put it back. See the detail above."
            )
        where = f" in {self.shell}" if self.shell else ""
        if self.unresolved:
            # The restore worked. What failed is the resolve after it, so the
            # operator has to fix the conflict FIRST — re-running the command
            # before that changes nothing and reads as the command being wrong.
            return (
                f"RESTORED, DEPENDENCIES UNRESOLVED — {CORE_DISTRIBUTION} is "
                "back from its declared source, but the pass that validates "
                "its dependencies did not complete. Resolve what failed below, "
                f"then run `{self.command}`{where} by hand."
            )
        if not self.attempted:
            # Nothing was tried, so nothing failed. Saying RESTORE FAILED here
            # would report an attempt that never happened, and an operator who
            # believes a repair ran is exactly the one who will not run this.
            return (
                "NOT RESTORED — the install was interrupted, so no repair was "
                f"attempted. Run `{self.command}`{where} by hand."
            )
        return f"RESTORE FAILED — run `{self.command}`{where} by hand."

    def describe(self) -> str:
        """One operator-readable block: what moved, and what was done about it.

        Same facts the CLI prints to stderr, flattened for a log line or an
        HTTP ``detail`` — including why a failed repair failed, which is the
        one thing the operator cannot see from the response otherwise.
        """
        if self.drift is None:
            return ""
        lines = [f"{self.headline}.", self.drift]
        if self.repaired:
            lines.append(f"restored: {self.command}")
        else:
            lines.append(self.restore_instruction)
            lines.extend(self.output.splitlines()[-3:])
        return "\n".join(lines)


class CoreInstallGuard:
    """The one way to install a feature package (issue #2949).

    Every ``kestrel-feature-*`` depends on ``kestrel-sovereign``, and the
    installer is project-blind (see :func:`_extension_install_run`), so any
    unguarded feature install can resolve core from the index and replace the
    operator's declared core — most visibly by dropping an editable checkout for
    a wheel copy, invisibly, because ``cwd=checkout`` keeps shadowing
    ``site-packages`` for anything started from inside it.

    Guarding is four steps that only work together, so one object owns all four
    and every install command holds one. A caller that reaches for the raw
    installer instead gets prevention without detection — the half-fix that let
    #2949 stay silent:

    1. **snapshot** how core is installed, before anything runs;
    2. **resolve the policy** — the source map's explicit ``kestrel-sovereign``
       entry if there is one, else the live editable link;
    3. **constrain** every install in the batch to that policy;
    4. **verify** afterwards, and repair + fail loudly if core moved anyway.

    Step 3 cannot cover a path that bypasses the constraint file (a feature's
    own build step, a direct ``pip`` call), which is exactly why step 4 is not
    optional.
    """

    def __init__(self, policy, before):
        self.policy = policy
        # ``before`` is the pre-batch state, reported verbatim so the operator
        # sees where core started. ``_baseline`` is the last state the batch
        # deliberately put core in (it advances when the batch installs core
        # itself) and is what decides whether a later drift is "replaced" or
        # "never got there".
        self.before = before
        self._baseline = before
        self._constraints = self._derive(before)
        # A SECOND set of lines, kept apart from core's on purpose: core's pin
        # encodes a source policy and is re-derived whenever core moves, while
        # these are the operator's declared windows for everything else and do
        # not change during a batch. Folding them together would re-derive the
        # manifest's bounds from core's shape, which is not where they come
        # from (issue #3106).
        self._manifest_bounds: list = []

    @classmethod
    def snapshot(cls, source_index=None):
        """Capture the live core install and the policy that governs it.

        *source_index* is the ``{package: SourceEntry}`` map from the host
        manifest when the caller has one (``feature sync``, ``update``'s
        reconcile). Commands with no manifest in their contract
        (``feature install`` / ``upgrade``, the install endpoint) pass nothing
        and guard whatever the venv actually has.
        """
        from kestrel_sovereign import feature_reconcile as fr

        before = cli._core_install_shape()
        # The whole shape, not just its editable path: "core is a plain index
        # wheel with nothing to protect" and "core's provenance would not read"
        # both reduce to editable_path is None, and only the resolver can tell
        # them apart. Passing the narrowed value let a damaged venv run entirely
        # unguarded — no constraint, no verification (issue #2949).
        policy = fr.resolve_core_policy(source_index or {}, before)
        guard = cls(policy, before)
        # Every install this guard performs is bounded by the WHOLE manifest,
        # not only by core's pin. Without it, remediating one entry moves
        # another outside its declared window and the run reports success over
        # the violation (issue #3106).
        guard._manifest_bounds = fr.manifest_version_constraints(source_index or {})
        return guard

    @classmethod
    def unguarded(cls):
        """A guard that constrains and verifies nothing.

        For callers that provably have no core to protect — and for tests that
        exercise install plumbing rather than the guard itself.
        """
        from kestrel_sovereign.feature_reconcile import CoreInstallShape, CoreSourcePolicy

        return cls(CoreSourcePolicy(), CoreInstallShape())

    def _derive(self, shape) -> list:
        from kestrel_sovereign import feature_reconcile as fr

        return fr.core_install_constraints(shape, self.policy)

    @property
    def constraints(self) -> list:
        """CORE's constraint lines, and only core's (may be empty).

        Every caller reads this to say "core is pinned to ``constraints[0]``",
        so it stays what that sentence claims. Folding the manifest's other
        windows in here made that message name a FEATURE's window on a host
        with no core pin — two facts in one list, and the caller reading it
        positionally could not tell.
        """
        return list(self._constraints)

    @property
    def manifest_bounds(self) -> list:
        """The manifest's declared windows for everything EXCEPT core.

        Read by the failure reports, which have to name the right thing: a
        conflict against one of these is not a reason to move core.
        """
        return list(self._manifest_bounds)

    def manifest_bound_note(self) -> Optional[str]:
        """The sentence naming the windows in force for everything but core.

        Owned by the guard because EVERY surface that reports a guarded
        install's failure has to say it: a note added at one of them is a note
        the other three do not have, and an operator on the wrong surface is
        told to move core over a conflict core is not in (#3106).

        ``None`` when the manifest declares no windows, so a caller can ask
        without knowing whether there is anything to say.
        """
        if not self._manifest_bounds:
            return None
        windows = ", ".join(self._manifest_bounds)
        return (
            f"the manifest also bounds {windows} for this install, so "
            "satisfying one entry cannot move another out of its declared "
            "window. If the conflict names one of these, that entry is the "
            "one to change."
        )

    @property
    def install_constraints(self) -> list:
        """Every line a guarded install carries: core's pin, then the manifest's.

        The other fact, under its own name. Core's pin bounds core's source and
        version; the manifest's windows bound everything else it declares, and
        an install that has to satisfy one entry must not move another out of
        its own (issue #3106).
        """
        return list(self._constraints) + list(self._manifest_bounds)

    def run(self, pip_args: list, *, reinstall=None, timeout=None):
        """Install a FEATURE package with core held inside the policy.

        *reinstall* names the feature package to force-reinstall when the
        install is a source switch (see :func:`_extension_install_run`). It is
        the guard's business because a bare ``--force-reinstall`` in *pip_args*
        would reinstall core from the index straight through the constraint —
        same version, wheel instead of link.
        """
        return self._install(
            pip_args,
            constraints=self.install_constraints or None,
            reinstall=reinstall,
            timeout=timeout,
        )

    def install_core(self, pip_args: list, *, reinstall=None, timeout=None):
        """Install CORE itself — the operator deliberately moving core's source.

        Never constrained against itself, and the constraints for the rest of
        the batch are re-derived afterwards: a batch that switches core to a new
        checkout (or a new pin) must go on to pin the version core actually
        became, not the one it was before the switch.

        *reinstall* is scoped the same way as :meth:`run`'s, even though core is
        the target here: reinstalling core does not require rebuilding core's
        own dependency tree, and a blanket flag would.
        """
        result = self._install(
            pip_args,
            # Never core's own pin — this IS the operator moving core — but the
            # rest of the manifest still binds: a core install must not drag a
            # declared feature outside its window either (issue #3106).
            constraints=self._manifest_bounds or None,
            reinstall=reinstall,
            timeout=timeout,
        )
        self.refresh()
        return result

    def refresh(self):
        """Re-read the live core install; re-derive constraints and baseline."""
        shape = cli._core_install_shape()
        self._baseline = shape
        self._constraints = self._derive(shape)
        return shape

    def conforms_now(self) -> bool:
        """Is core on its declared source at this moment? Non-mutating.

        A deliberate exception to :meth:`_check`'s privacy, and narrow: this
        answers a BATCH-CONTROL question — "may the rest of these installs run
        against this core?" — not a detection question. Detection is unaffected;
        :meth:`verify` still runs unconditionally at the end of the batch and is
        still the only thing that reports and repairs.

        Needed because "the core action failed" and "core is off its declared
        source" are not the same thing. A conforming core that declares extras
        yields an `ensure` action, and a failed optional extra says nothing
        about core's source — treating that as a failed transition skipped every
        remaining entry in the batch.
        """
        return self._check() is None

    def _check(self) -> Optional[str]:
        """Describe how core drifted from the policy, or None if it conforms.

        Private on purpose: asking without acting is prevention without
        detection, the exact half-fix that let #2949 stay silent. Callers go
        through :meth:`resolve` (or :meth:`verify`), which cannot look without
        also repairing and reporting.
        """
        from kestrel_sovereign.feature_reconcile import describe_core_change

        if not self.policy.guarded:
            return None
        return describe_core_change(self.before, cli._core_install_shape(), self.policy)

    def _unrestorable(self, drift, replaced, *, attempted=True):
        """The outcome for a drift with no declared source to restore from.

        A hold policy means core did not come from an index and NOBODY declared
        where it should come from. The drift is real and must be reported, but
        there is no declared source to reinstall from — so refuse to guess one.
        Reinstalling core from a source nobody declared is the retargeting this
        guard exists to prevent, and committing it *as the repair* would be the
        worst possible place.

        The two cases say different things to an operator: one can be told
        exactly where core came from, the other cannot be told anything.

        Shared by :meth:`resolve` and the interrupt path so both surfaces say it
        the same way; only *attempted* differs between them.
        """
        held = self.policy.hold_provenance
        cause = (
            f"core was installed from {held.describe()}, which no manifest "
            "entry declares"
            if held is not None and held.known
            else "core's install source could not be read"
        )
        return CoreGuardOutcome(
            drift=drift,
            replaced=replaced,
            repaired=False,
            attempted=attempted,
            command="",
            shell="",
            output=(
                f"{cause}, so there is no declared source to restore from. "
                "Reinstall core yourself, or declare its source in "
                ".kestrel-host-features.toml."
            ),
        )

    def _install(self, pip_args, *, constraints, reinstall, timeout, repairing=False):
        """Run the installer — the ONE place a subprocess is started.

        Every install this guard performs goes through here (:meth:`run`,
        :meth:`install_core`, :meth:`_repair`), which is what makes interrupt
        handling a property of the guard rather than something each of the four
        CLI paths has to remember (issue #2962).

        A ``KeyboardInterrupt`` never returns, so neither the success nor the
        failure branch above it runs and the post-check is skipped entirely —
        while a killed installer can already have replaced core. That is the
        #2949 failure with nobody told. Report what moved, then let the
        interrupt continue on its way.

        *repairing* is passed by :meth:`_repair` alone, and only it can know:
        a Ctrl-C that lands inside the automatic restore interrupted a repair
        that WAS attempted, and telling the operator "no repair was attempted"
        there is the same false report this path exists to prevent.
        """
        try:
            return cli._extension_install_run(
                pip_args,
                constraints=constraints,
                reinstall=reinstall,
                timeout=timeout,
            )
        except KeyboardInterrupt:
            self._report_interrupt(repairing=repairing)
            raise

    def _interrupt_outcome(self, *, attempted):
        """Non-mutating counterpart to :meth:`resolve`: describe, never repair.

        Returns None when core still conforms — an interrupt is not by itself
        evidence that anything moved.
        """
        from kestrel_sovereign.feature_reconcile import core_install_matches

        drift = self._check()
        if drift is None:
            return None
        replaced = core_install_matches(self._baseline, self.policy)
        if not self.policy.source_is_verifiable:
            return self._unrestorable(drift, replaced, attempted=attempted)
        _, _, rendered, shell = self._restore_plan()
        return CoreGuardOutcome(
            drift=drift,
            replaced=replaced,
            repaired=False,
            attempted=attempted,
            command=rendered,
            shell=shell,
        )

    def _report_interrupt(self, *, repairing=False) -> None:
        """Name any drift an interrupted installer left behind. REPORTS ONLY.

        Deliberately does not repair. A repair runs another installer, and an
        abort must not start unbounded work: the operator's second Ctrl-C would
        land inside it, and a repair that hangs turns an abort into a wedge.
        Everything here is a metadata read, so it is bounded by construction —
        which is why this is the one place allowed to ask :meth:`_check` without
        also acting on the answer.

        Never raises. A diagnostic that fails must not replace the interrupt
        with its own traceback — the operator pressed Ctrl-C and is entitled to
        get Ctrl-C. ``KeyboardInterrupt`` is a ``BaseException``, so a SECOND
        one passes straight through this handler rather than being swallowed.
        """
        if not self.policy.guarded:
            return
        try:
            outcome = self._interrupt_outcome(attempted=repairing)
            if outcome is None:
                return
            # The WRITES are inside the guard too, not just the lookup: stderr
            # can be a closed fd or a pipe whose reader already exited, and a
            # BrokenPipeError escaping here would replace the operator's Ctrl-C
            # with a traceback about the diagnostic that was trying to help.
            print(
                f"• core: INTERRUPTED — {outcome.headline}.",
                file=sys.stderr,
            )
            for line in (outcome.drift or "").splitlines():
                print(f"    {line}", file=sys.stderr)
            print(f"    {outcome.restore_instruction}", file=sys.stderr)
            for line in outcome.output.splitlines()[-3:]:
                print(f"      {line}", file=sys.stderr)
        except Exception:  # noqa: BLE001 - see docstring: never mask the interrupt
            return

    def _restore_plan(self):
        """The exact install that would put core back — rendered, NOT run.

        Returns ``(pip_args, reinstall, rendered, shell)``.

        One derivation with two consumers: :meth:`_repair` runs it, and the
        interrupt path prints it without running anything (issue #2962).
        Deriving it twice is how a printed restore command starts describing a
        different install than the one the guard would actually perform — the
        precise failure :meth:`_repair` already documents for the *rendering*,
        now true of the argv as well.
        """
        if self.policy.editable:
            checkout = str(Path(self.policy.editable).expanduser())
            pip_args = ["-e", checkout]
            # Installing from a local path replaces whatever holds the name, so
            # the link comes back without forcing anything.
            reinstall = None
        else:
            # Declared from the index. A reinstall is needed to displace any
            # install the resolver would otherwise consider done: an editable
            # link, and equally a non-editable direct-URL copy (VCS, local path,
            # archive) whose version already satisfies the spec — pip and uv
            # judge "already satisfied" by VERSION, so re-resolving the spec is
            # a no-op and the wrong source survives.
            #
            # Detection and repair must ask the SAME question. Asking only
            # "is it editable?" here while _check() asks "is it from an index?"
            # makes a drift this guard now names one it can never fix: a
            # permanent CORE_UNSAFE, plus a printed manual command that no-ops
            # for the operator exactly as the automatic repair did.
            #
            # A version outside the window is still fixed by resolving the spec
            # again. Scoped to core, never blanket: a repair that reinstalls
            # "everything resolved" would drop an editable SDK (or any other
            # editable dependency of core) for an index wheel — issue #2949
            # committed by the code that exists to undo it.
            spec = f"{CORE_DISTRIBUTION}{self.policy.pypi}"
            pip_args = [spec]
            reinstall = (
                None if cli._core_install_shape().from_index else CORE_DISTRIBUTION
            )
        return (
            pip_args,
            reinstall,
            _render_commands(_install_commands(pip_args, reinstall=reinstall)),
            _render_shell(),
        )

    def _repair(self, *, timeout=None):
        """Reinstall core from its declared source.

        Returns ``(ok, command, shell, result)`` — ``shell`` naming what
        ``command`` is quoted for, captured here so the label can never describe
        a different rendering than the one it travels with.

        ``ok`` is decided by RE-READING core afterwards, in both directions —
        the installer's exit code is evidence about the installer, not about
        where core ended up. A nonzero exit does not mean core is still wrong:
        a pip repair is two passes (:func:`_install_commands`) and the first
        one restores core before the second can fail, and a subprocess killed
        by *timeout* is killed wherever it had got to, which may be after the
        write that put core back. Trusting the code there reports RESTORE
        FAILED over a core that is already home — sending an operator to redo
        an install that happened, and failing an HTTP install for a host that
        conforms.

        ``command`` is rendered from the SAME argv sequence the repair just ran
        (:func:`_install_commands`) — backend, interpreter, and reinstall
        scoping included — quoted for THIS host's shell
        (:func:`_render_commands`). An operator only
        sees it when the automatic restore failed, which is precisely when a
        command naming a uv this host lacks — or omitting ``--python`` and so
        landing in whichever environment is active — would retarget core
        somewhere new instead of putting it back, and when a command their shell
        will not parse at all leaves them with nothing.

        *timeout* bounds the repair's own installer subprocess — see
        :meth:`resolve`. A repair killed by it is reported as an ordinary failed
        repair (``ok=False``) rather than raised: the operator still needs the
        restore command, and the caller still needs to report whatever brought
        it here.
        """
        pip_args, reinstall, rendered, shell = self._restore_plan()
        try:
            result = self._install(
                pip_args,
                # Core is never constrained against itself, here least of all —
                # this install exists to put core back. The rest of the manifest
                # still binds: a repair that resolves a dependency outside
                # another entry's declared window has moved it, and this rechecks
                # only core (issue #3106).
                constraints=self._manifest_bounds or None,
                reinstall=reinstall,
                timeout=timeout,
                repairing=True,
            )
        except subprocess.TimeoutExpired as expired:
            result = _killed_process(expired)
        restored = self._check() is None
        # Two independent facts, so two variables: WHERE core is, and whether
        # the dependencies of what landed have a solution. Re-reading core
        # answers the first and cannot answer the second, and a repair whose
        # final resolving pass refused leaves a host that is off neither its
        # declared source nor able to load — reporting it by core's location
        # alone downgrades the stronger fact to the weaker one.
        unresolved = restored and getattr(result, "resolve_incomplete", False)
        ok = restored and not unresolved
        if restored:
            # Core is where the policy wants it: that is the state this batch
            # now measures later drift against.
            self.refresh()
        return ok, rendered, shell, result, unresolved

    def resolve(self, *, timeout=None) -> CoreGuardOutcome:
        """Check core against the policy, and repair it if it drifted.

        The one place that decides what a drift means and what to do about it,
        so the CLI (:meth:`verify`) and the HTTP install endpoint cannot answer
        that question differently — a divergence there is how one surface ends
        up reporting success over a core the other would have refused.

        Repair success is judged by RE-CHECKING core, not by the installer's
        exit code, and that cuts both ways: a reinstall that returned 0 and
        still left core off its declared source is not a repair, and one that
        exited nonzero — or was killed by *timeout* — after core was already
        back is not a failure. The installer's output is diagnostic; core
        itself is the authority.

        *timeout* bounds the repair's installer subprocess. It defaults to
        unlimited for the CLI, where an operator is watching the run and can
        interrupt it; a caller with nobody at the other end MUST pass one. The
        HTTP install endpoint is the case in point: it bounds the install
        itself, and the resolver or network problem that hung that install hangs
        the repair the same way — an unbounded repair turns the intended 504
        into a request that never returns. A repair that hits the bound without
        restoring core is an unrepaired core
        (:attr:`CoreGuardOutcome.conforming` is False) carrying the manual
        restore command, exactly like any other failed repair.
        """
        from kestrel_sovereign.feature_reconcile import core_install_matches

        drift = self._check()
        if drift is None:
            return CoreGuardOutcome()

        # Distinguish "the batch broke it" from "it never got there": both are
        # failures, but only the first is a swap.
        replaced = core_install_matches(self._baseline, self.policy)

        if not self.policy.source_is_verifiable:
            return self._unrestorable(drift, replaced)

        repaired, rendered, shell, result, unresolved = self._repair(timeout=timeout)
        return CoreGuardOutcome(
            drift=drift,
            replaced=replaced,
            repaired=repaired,
            unresolved=unresolved,
            command=rendered,
            shell=shell,
            output="" if repaired else (result.stderr or result.stdout or "").strip(),
        )

    def verify(self) -> int:
        """Assert core survived the batch; repair and report if it did not.

        Three states, because two of them are not equally safe:

        * ``0`` — core conforms (or nothing is guarded).
        * ``1`` — core moved and was **repaired**. Still an error, so no command
          reports success over a core that was replaced; but the venv is correct
          again, so a caller told to tolerate failures may go on.
        * :data:`CORE_UNSAFE` — core moved and the repair **failed**. The venv is
          left running a core the manifest does not declare.
        * :data:`CORE_UNRESOLVED` — core was put back, and the pass that
          resolves its dependencies refused. Nothing to restore; a conflict to
          fix (#3047).

        The last one is a safety assertion, not a package-level failure, and
        must not be reachable through ``--continue-on-error`` — a flag whose
        contract is "tolerate an optional package that would not install".
        Folding it into ``1`` let ``kestrel update --continue-on-error`` restart
        every agent onto an undeclared core and exit 0 (issue #2949).
        """
        outcome = self.resolve()
        if outcome.drift is None:
            return 0

        print(f"• core: ERROR — {outcome.headline}.", file=sys.stderr)
        for line in outcome.drift.splitlines():
            print(f"    {line}", file=sys.stderr)
        if outcome.repaired:
            print(f"    restored: {outcome.command}", file=sys.stderr)
            return 1
        print(f"    {outcome.restore_instruction}", file=sys.stderr)
        for line in outcome.output.splitlines()[-3:]:
            print(f"      {line}", file=sys.stderr)
        return CORE_UNRESOLVED if outcome.unresolved else CORE_UNSAFE


def _capture_host_manifest(path: Path) -> int:
    """Snapshot currently-installed extension packages into a manifest.

    Reuses the same live-entry-point discovery as ``feature upgrade`` so the
    host never hand-authors the file. Editable installs are recorded with
    their checkout path; extras can't be recovered from metadata, so the user
    adds those by hand if needed.

    ``kestrel-sovereign`` itself is captured too when it is editable-installed:
    core registers no extension entry-points, so live discovery cannot see it,
    yet every feature depends on it and can pull a wheel over the link. A
    first-class entry makes the intended source of core explicit rather than
    assumed (issue #2949).
    """
    dists = cli._installed_extension_distributions()

    lines = [
        "# Host-local feature manifest for `kestrel feature sync`.",
        "# Lists the extension packages THIS host should keep installed so they",
        "# can be restored after `uv sync` (or any rebuild) prunes them.",
        "#",
        "# Regenerate with:  kestrel feature sync --capture",
        "# Restore with:     kestrel feature sync",
        "",
    ]

    captured = 0
    core = cli._core_install_shape()
    if core.is_editable:
        captured += 1
        lines.extend([
            "# The core distribution. Declared so feature installs cannot",
            "# quietly resolve it from PyPI over the editable checkout.",
            "[[feature]]",
            f"name = {_toml_basic_string(CORE_DISTRIBUTION)}",
            f"editable = {_toml_basic_string(core.editable_path)}",
            "",
        ])

    for d in dists:
        if d["dist"] == CORE_DISTRIBUTION:
            continue  # already emitted above
        captured += 1
        lines.append("[[feature]]")
        lines.append(f'name = {_toml_basic_string(d["dist"])}')
        if d["editable_path"]:
            lines.append(f'editable = {_toml_basic_string(d["editable_path"])}')
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return captured


def _registry_info_for(label: str, registry: dict):
    """Resolve a manifest entry name to its registry info.

    Handles both registry short names ("voice") and raw dist names
    ("kestrel-feature-voice", which manifests commonly use). Returns the
    FeaturePackageInfo or None.
    """
    pkg = _resolve_feature_name(label, registry)
    return registry[pkg] if pkg else None


def _version_satisfies(version: str, spec: str) -> bool:
    """Does *version* satisfy the PEP 440 *spec* (e.g. ``>=0.3,<0.4``)?

    One implementation, shared with the core-guard's policy check — a second
    copy is how "installed satisfies the pin" and "core satisfies the pin" drift
    apart.
    """
    from kestrel_sovereign.feature_reconcile import version_satisfies

    return version_satisfies(version, spec)


def _resolve_manifest_action(entry: dict, registry: dict):
    """Resolve a manifest entry to ``(target_package, current_version, action)``.

    ``action`` is exactly what ``kestrel feature sync`` would do:
    ``install`` (not present), ``reinstall`` (editable path mismatch, or a
    ``pypi`` pin the installed version violates), ``ensure`` (present but
    declared extras can't be probed), or ``present``. Shared by ``sync`` and
    ``status`` so the two never report different drift.
    """
    import importlib.metadata as md

    # Same resolver the source index keys on, so "this entry is core" and
    # "this is core's declared source" can never disagree (issue #2949).
    target = package_for_label(entry["name"], registry)
    extras = entry["extras"]
    editable_want = entry["editable"]
    pypi_want = entry.get("pypi")
    try:
        current = md.version(target)
    except md.PackageNotFoundError:
        current = None

    if current is None:
        action = "install"
    elif editable_want:
        have = cli._editable_install_path(target)
        matched = bool(have) and (
            Path(have).resolve() == Path(editable_want).expanduser().resolve()
        )
        action = "reinstall" if not matched else ("ensure" if extras else "present")
    elif pypi_want is not None:
        # The source map declares a PyPI source. Re-install (switch source)
        # when the installed copy is editable — otherwise sync/status would
        # leave the venv pointed at a local checkout the manifest no longer
        # declares (codex round 8 P2) — or when a non-empty pin is violated
        # (codex round 2 P2). Both rather than a false `present`.
        #
        # "Not from an index" covers the editable case and the direct-URL one
        # (VCS ref, local path, remote archive) in a single question — the same
        # question execution and the core guard ask, so a preview can never
        # promise `present` for something the run reinstalls.
        #
        # This applies to FEATURES as well as core, and deliberately so: sync's
        # git fallback is gated on `pypi_want is None`, so a declared entry
        # never falls back and there is no reinstall loop to avoid. Without it,
        # a feature that reached the venv from a git URL by some other path
        # would sit there forever while the manifest said it comes from the
        # index.
        if not cli._from_index(target) or not _version_satisfies(current, pypi_want):
            action = "reinstall"
        elif extras:
            action = "ensure"
        else:
            action = "present"
    elif extras:
        action = "ensure"
    else:
        action = "present"
    if action == "present" and _unsatisfied_requirements(target, extras):
        # `present` is a claim about the VENV, and source+version answers only
        # half of it: this copy is the declared version from the declared
        # source AND cannot load, because something it declares is missing or
        # too old. Left as `present`, sync converges on nothing and reports
        # success — the state a half-completed install leaves behind, never
        # retried even after the operator fixes the conflict (issue #3080).
        #
        # `ensure`, not `reinstall`: the artifact is the right one, so nothing
        # needs forcing. What is missing is a resolve, and an ordinary install
        # is exactly that.
        action = "ensure"
    # The demotion is not reported back. The caller that needs to know whether
    # a package's dependencies resolve asks at the moment it needs the answer
    # — installers are not transactional, so an answer taken here has already
    # gone stale by the time an install has failed (#3080).
    return target, current, action


def _core_entry_first(entries: list, registry: dict) -> list:
    """Manifest order, with any ``kestrel-sovereign`` entry moved to the front.

    Core's declared source has to be applied before the batch derives its pin
    from the installed state, and before any feature can drag core somewhere the
    manifest didn't declare. Order among the remaining entries is preserved so
    the operator's file still reads the way it runs (issue #2949).
    """
    def is_core(entry) -> bool:
        return package_for_label(entry["name"], registry) == CORE_DISTRIBUTION

    core = [e for e in entries if is_core(e)]
    return core + [e for e in entries if not is_core(e)] if core else entries


def cmd_feature_sync(args) -> int:
    """Restore the host's declared feature packages (counterpart to upgrade)."""
    from kestrel_sovereign.feature_registry import load_registry

    path = _host_manifest_path(args)

    if getattr(args, "capture", False):
        count = _capture_host_manifest(path)
        print(f"Captured {count} installed extension package(s) to {path}")
        return 0

    if not path.exists():
        print(f"No host manifest at {path}")
        print("Create one from your current install with:")
        print("  kestrel feature sync --capture")
        return 1

    try:
        entries = _load_host_manifest(path)
    except (ValueError, OSError) as exc:
        print(f"Failed to read manifest {path}: {exc}")
        return 1

    if not entries:
        print(f"Manifest {path} declares no features. Nothing to sync.")
        return 0

    registry = load_registry()
    # Keyed on the same canonical identity ``_resolve_manifest_action`` resolves
    # an entry's target to, so the lookup can't miss on spelling alone.
    git_urls = {
        canonical_package(info.package): info.git
        for info in registry.values() if info.package
    }
    dry_run = getattr(args, "dry_run", False)

    # Every kestrel-feature-* depends on kestrel-sovereign, so any of these
    # installs can resolve core from the index and replace the declared core.
    # Hold core inside the manifest's policy for the whole batch (prevention)
    # and re-check it afterwards (detection) — issue #2949.
    from kestrel_sovereign import feature_reconcile as fr

    guard = CoreInstallGuard.snapshot(fr.build_source_index(entries, registry))

    # The manifest is a declaration, not a program: its order says nothing about
    # dependency order. Core is what every other entry depends on, so its
    # declared source must be applied BEFORE the batch derives a pin from the
    # installed state — otherwise an operator who moves core to a new checkout
    # and lists it last pins every earlier install to the version core had
    # *before* the switch.
    entries = _core_entry_first(entries, registry)

    print()
    print(f"  {'PACKAGE':<34} {'CURRENT':<10} {'ACTION'}")
    print(f"  {'-' * 62}")

    # Two INDEPENDENT facts, deliberately not sharing one int (see
    # `_worst_rc`): `package_rc` is "some package failed to install",
    # `core_state` is "core is in a state a restart must not proceed over".
    package_rc = 0
    core_state = 0
    installed = 0
    for entry in entries:
        target, current, action = _resolve_manifest_action(entry, registry)
        extras = entry["extras"]
        editable_want = entry["editable"]
        pypi_want = entry.get("pypi")

        if action == "present":
            print(f"  {target:<34} {current or '-':<10} present")
            continue

        # A declared PyPI version spec pins the install (``pkg>=0.3,<0.4``);
        # an empty spec means "any version", same as the legacy bare form.
        # Extras go BEFORE the version spec (``pkg[local]>=0.3``) — pip rejects
        # ``pkg>=0.3[local]``.
        install_target = f"{_pip_spec(target, extras)}{pypi_want}" if pypi_want \
            else _pip_spec(target, extras)

        if dry_run:
            how = f"-e {editable_want}" if editable_want else f"pip {install_target}"
            print(f"  {target:<34} {current or '-':<10} would {action} ({how})")
            continue

        reinstall = None
        if editable_want:
            checkout = Path(editable_want).expanduser()
            if canonical_package(target) == CORE_DISTRIBUTION:
                # A declared editable CORE must be pulled before it is linked,
                # or `kestrel update` restarts the fleet on stale contents.
                #
                # Nothing else covers it. Step 1 of `update` pulls the checkout
                # the *currently installed* core lives in — which is None when
                # core is an index wheel, and the wrong checkout when the
                # manifest declares a different one. Reconcile pulls every
                # editable entry it installs, but core is excluded from
                # reconcile because it is bundled. So core is the one editable
                # source on the host that nothing pulls.
                #
                # A failed pull does NOT fail the sync: a dirty core checkout is
                # an ordinary dev state, and refusing to link over it would be
                # hostile. It is printed instead, so the operator learns the
                # code did not move rather than being told nothing.
                from kestrel_sovereign import cli_lifecycle

                rc_pull, out_pull = cli_lifecycle._editable_git_pull(
                    checkout, allow_dirty=getattr(args, "allow_dirty", False),
                )
                for line in (out_pull or "").strip().splitlines():
                    print(f"      {line}")
                if rc_pull != 0:
                    # Reported AND non-zero. Linking an unpulled checkout is
                    # still the right end state — refusing over local edits
                    # would be hostile, and a dirty core checkout is an ordinary
                    # dev state — but `kestrel update` must not then restart and
                    # return SUCCESS over code that never moved. The operator
                    # asked for an update; they got a link. Those are different
                    # outcomes and the exit status has to say so.
                    #
                    # `--allow-dirty` is honoured above, so the operator who
                    # means "link over my edits" has a way to say it.
                    # CORE_STALE, not the generic 1: `--continue-on-error`
                    # ignores 1, restarts, and returns success — which is the
                    # exact outcome this was added to prevent. Reporting is not
                    # enough if the report is then discarded.
                    core_state = CORE_STALE
                    print(
                        f"      note: {target} was linked but NOT updated — the "
                        "checkout above is what will run after a restart."
                    )
            pip_args = ["-e", _pip_spec(str(checkout), extras)]
        else:
            pip_args = [install_target]
            if pypi_want is not None and not cli._from_index(target):
                # Switching to a PyPI source needs a scoped reinstall: pip and
                # uv judge "already satisfied" by VERSION, so a plain install
                # leaves the satisfying non-index copy in place and the source
                # switch never takes effect (codex round 10 P2).
                #
                # "Not from an index", not "is editable" — a direct-URL copy
                # (VCS, local path, archive) is equally satisfying and equally
                # wrong. Asking the narrower question here made sync's own
                # install a no-op, left the final guard to do the real repair,
                # and so failed `feature sync` and `kestrel update` with rc=1
                # over a run that had reached exactly the declared state.
                #
                # Scoped to THIS package: a blanket --force-reinstall cascades
                # to core and swaps its editable link for a same-version index
                # wheel, straight through the pin.
                reinstall = target

        # Core is never constrained against itself — a `kestrel-sovereign`
        # manifest entry IS the operator deliberately moving core's source.
        # `install_core` re-derives the batch pin from what core became.
        is_core = target == fr.CORE_DISTRIBUTION
        run = functools.partial(
            guard.install_core if is_core else guard.run, reinstall=reinstall,
        )
        result = run(pip_args)
        # Registry-backed packages can fall back to their git URL (mirrors
        # `feature install`) — but ONLY the legacy entry that declares no source
        # at all, which has always meant "wherever this package comes from".
        # The git URL installs repo HEAD with no version constraint, so it is
        # both a different SOURCE and an unpinned one: substituting it for a
        # declared source is the same silent retargeting this ticket exists to
        # stop. `editable` has no remote form; `pypi = ">=x,<y"` would be moved
        # outside its pin; and `pypi = ""` names the index as the source — "any
        # version" is a statement about the VERSION, not a waiver of where the
        # package comes from — so `""` must be told apart from "absent", which a
        # truthiness test cannot do. (An empty `editable` is the other way
        # round: it names no checkout, so it declares nothing — which is how the
        # rest of this function reads it too.)
        # Carry extras across via PEP 508 form (``pkg[extra] @ git+url``) so a
        # git fallback doesn't silently drop the local-pipeline deps.
        if (
            result.returncode != 0
            and not editable_want
            and pypi_want is None
            and git_urls.get(target)
        ):
            git_ref = f"git+{git_urls[target]}"
            git_spec = f"{_pip_spec(target, extras)} @ {git_ref}" if extras else git_ref
            result = run([git_spec])

        if result.returncode != 0:
            # A package failure NEVER touches `core_state`: it must not be able
            # to downgrade a CORE_STALE recorded earlier in this same batch.
            package_rc = 1
            print(f"  {target:<34} {current or '-':<10} FAILED")
            for line in (result.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}")
            if not is_core and guard.constraints:
                print(
                    f"      note: core is pinned to {guard.constraints[0]} for "
                    "this install so a feature cannot silently replace it. If "
                    "this is a version conflict, move core to a version the "
                    "feature accepts — do not remove the pin."
                )
            # Not gated on `is_core`: a core install carries these bounds too,
            # and a conflict against one of them is not a reason to move core.
            bound_note = guard.manifest_bound_note()
            if bound_note:
                print(f"      note: {bound_note}")
            # A failed core action stops the batch only when core is ACTUALLY
            # off its declared source. `_resolve_manifest_action` returns
            # `ensure` for a conforming core that declares extras, and a failed
            # optional extra says nothing about core's source — treating it as a
            # failed transition skipped every remaining manifest entry, so
            # `--continue-on-error` restarted with packages still pruned.
            if is_core and guard.conforms_now() and (
                getattr(result, "resolve_incomplete", False)
                # A single-command install has no incomplete SEQUENCE — but
                # when core's dependencies were already unmet, a failed resolve
                # here is the same state by a different route, and
                # `resolve_incomplete` was derived for the three-pass switch
                # alone (#3080).
                #
                # RE-READ rather than reusing the pre-install answer: installers
                # are not transactional, so this run may have installed the
                # missing dependency and then failed on something else. Blocking
                # a restart over a closure that now conforms is the false
                # refusal this check was scoped to avoid.
                or bool(_unsatisfied_requirements(target, extras))
            ):
                # Core reached its declared source and version — so
                # `conforms_now()` is True and says nothing about whether the
                # host can LOAD it. The sequence ended without the pass that
                # validates core's dependencies succeeding, which is a core
                # state a restart must not proceed over; leaving it as the
                # generic package `1` let `--continue-on-error` restart the
                # fleet onto it (#3047).
                core_state = _worst_rc(core_state, CORE_UNRESOLVED)
                print(
                    "      note: core is the declared install, but the pass "
                    "that validates its dependencies did not complete — the "
                    "rest of this batch is skipped and a restart must not "
                    "proceed. Resolve what failed above, then re-run."
                )
                break
            if is_core and not guard.conforms_now():
                # `install_core` has
                # already refreshed the guard from whatever core is NOW — the
                # old version, or a partial write — so every remaining feature
                # would resolve and pin against a core the manifest says is
                # wrong. The final repair may then move core to the declared
                # version those features were never resolved against, and
                # `--continue-on-error` would restart the fleet onto that
                # combination. Verification still runs below; it is installing
                # everything else that must not proceed.
                print(
                    "      note: core did not reach its declared source, so the "
                    "rest of this batch is skipped — installing features "
                    "against the wrong core is what produces a venv no single "
                    "step reports as broken."
                )
                break
            continue

        installed += 1
        done = {"install": "installed", "reinstall": "reinstalled", "ensure": "ensured"}[action]
        print(f"  {target:<34} {current or '-':<10} {done}")

    print()
    if not dry_run:
        print(f"  {installed} package(s) installed/restored.")
        if installed:
            print("  Restart the host/agents to load the synced features.")
        # Constraints prevent the common swap; this catches every other path
        # (a feature's own build step, a direct pip call) so sync can never
        # report success over a core that was replaced (issue #2949).
        # `verify()` is a second reading of CORE state, so it is RANKED against
        # the first rather than assigned over it — a repaired-core `1` must not
        # erase a CORE_STALE recorded by the pull above.
        core_state = _worst_rc(core_state, guard.verify())
    # No gate here, deliberately: `_RC_SEVERITY` already ranks every core state
    # above a package failure, so the fold below returns the core state
    # verbatim and `kestrel update` gates its restart on it exactly as
    # reconcile does. An `if core_state == ...: return core_state` in front of
    # this is dead — it was, and a mutation test that could not kill it is what
    # said so. The ranking is the mechanism; a comparison beside it is a second
    # place to forget a code.
    return _worst_rc(core_state, package_rc)


# ---------------------------------------------------------------------------
# `kestrel feature status` — answer "is my environment set up right, per agent?"
#
# Combines two views the operator otherwise has to assemble by hand:
#   1. Host venv: are the manifest's packages actually installed? (drift vs
#      `.kestrel-host-features.toml` — the `sync --dry-run` answer)
#   2. Per running agent: is each feature actually LOADED? An installed package
#      that an agent hasn't loaded means the host needs a restart (entry-points
#      are read at startup) or the agent's `features` allowlist excludes it.
# ---------------------------------------------------------------------------


def _query_agent_feature_catalog(base_url: str, api_key: str):
    """GET ``/api/features`` from a running agent → ``{package: status}``.

    Status is resolved per the routed agent, so ``enabled`` means loaded on
    *that* agent. Returns None on any network/parse failure (agent treated as
    unreachable, not as "nothing loaded").
    """
    import httpx

    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        resp = httpx.get(f"{base_url}/api/features", headers=headers, timeout=5.0)
    except httpx.RequestError:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    feats = data.get("features")
    if not isinstance(feats, list):
        return None
    out = {}
    for f in feats:
        pkg = f.get("package") if isinstance(f, dict) else None
        if isinstance(pkg, str) and pkg:
            # Keyed on the canonical spelling, because the lookup side is.
            # The agent reports whatever its installed METADATA says, and
            # `Kestrel_Feature_Voice` is a legal Name: for the distribution the
            # registry calls `kestrel-feature-voice`. `_resolve_manifest_action`
            # canonicalizes its target, so keying this side raw makes the two
            # miss each other and `feature status` shows an enabled feature as
            # not loaded. Two spellings that canonicalize alike ARE one
            # distribution, so collapsing them is the right answer rather than a
            # collision (issue #2949).
            out[canonical_package(pkg)] = f.get("status")
    return out


def cmd_feature_status(args) -> int:
    """Show host install state + per-agent loaded state for manifest features."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    path = _host_manifest_path(args)

    entries = []
    if path.exists():
        try:
            entries = _load_host_manifest(path)
        except (ValueError, OSError) as exc:
            print(f"Failed to read manifest {path}: {exc}")
            return 1

    # Resolve each manifest entry to its install action (shared with `sync`)
    # plus whether it participates in the Feature lifecycle. The registry's
    # explicit package boundary owns this decision; class-name suffixes do not.
    from kestrel_sovereign.feature_registry import PackageBoundary

    targets = []
    for entry in entries:
        info = _registry_info_for(entry["name"], registry)
        target, current, action = _resolve_manifest_action(entry, registry)
        if target == CORE_DISTRIBUTION:
            # A `kestrel-sovereign` manifest entry (written by `sync --capture`
            # since #2949) declares where CORE comes from. It is a source
            # policy, not one lifecycle feature, and it cannot be rendered as
            # one: dozens of bundled features share the `kestrel-sovereign`
            # package, so `_registry_info_for` resolves the entry to whichever
            # bundled row the registry happens to list first (`identity`) and
            # the agent's catalog collapses every bundled feature's status under
            # that one package key, last-writer-wins. A per-agent column here
            # would report an arbitrary bundled feature's state under core's
            # name. Its install state IS shown — in the host table below, which
            # is the level a source policy lives at.
            display, is_feature = CORE_DISTRIBUTION, False
        else:
            display = info.name if info else entry["name"]
            is_feature = bool(info) and info.boundary in {
                PackageBoundary.BUNDLED,
                PackageBoundary.FEATURE_PACKAGE,
            }
        targets.append(
            {
                "display": display,
                "package": target,
                "current": current,
                "action": action,
                "is_feature": is_feature,
            }
        )

    # --- Host install level ---
    print()
    if entries:
        print(f"Host venv (declared in {path.name}):")
        print(f"  {'PACKAGE':<34} {'INSTALLED':<12} STATE")
        print(f"  {'-' * 56}")
        drifted = 0
        for t in targets:
            if t["action"] == "present":
                print(f"  {t['package']:<34} {'✓ ' + t['current']:<12} ok")
            elif t["action"] == "install":
                drifted += 1
                print(f"  {t['package']:<34} {'✗ —':<12} not installed")
            else:  # reinstall / ensure — installed but not matching the manifest
                drifted += 1
                ver = ("✓ " + t["current"]) if t["current"] else "✗ —"
                print(f"  {t['package']:<34} {ver:<12} needs {t['action']}")
        if drifted:
            print(f"\n  → {drifted} package(s) drift from the manifest — run: kestrel feature sync")
        else:
            print("\n  → manifest satisfied")
    else:
        print(f"No host manifest at {path}. Run `kestrel feature sync --capture` to create one.")

    # --- Per-agent loaded level ---
    project_dir = cli._get_project_dir()
    multi_agent = cli.MultiAgentConfig.load(project_dir / MULTI_AGENT_CONFIG_FILENAME)
    local_agents = multi_agent.get_local_agents()

    requested = getattr(args, "agent", None)
    if requested:
        if requested not in local_agents:
            print(f"\nAgent '{requested}' not found in {MULTI_AGENT_CONFIG_FILENAME}")
            return 1
        agent_names = [requested]
    else:
        agent_names = list(local_agents)

    cols = [t for t in targets if t["is_feature"]]
    print(f"\nLoaded per agent (host :{multi_agent.host.port}):")
    if not cols:
        print("  (no loadable feature packages in the manifest to check per agent)")
        return 0

    print("  " + f"{'AGENT':<12}" + "".join(f"{c['display']:<16}" for c in cols))
    print("  " + "-" * (12 + 16 * len(cols)))

    drift = False
    any_online = False
    for name in agent_names:
        cfg = local_agents.get(name)
        endpoint = cli._detect_running_agent_server(name, cfg, multi_agent) if cfg else None
        if endpoint is None:
            print("  " + f"{name:<12}" + f"{'(offline)':<16}")
            continue
        base_url, key = endpoint
        catalog = cli._query_agent_feature_catalog(base_url, key)
        if catalog is None:
            print("  " + f"{name:<12}" + f"{'(unreachable)':<16}")
            continue
        any_online = True
        cells = []
        for c in cols:
            status = catalog.get(c["package"])
            if status == "enabled":
                cells.append("✓ enabled")
            elif status in ("installed", "disabled"):
                cells.append(f"⚠ {status}")
                drift = True
            else:
                cells.append("✗ —")
        print("  " + f"{name:<12}" + "".join(f"{cell:<16}" for cell in cells))

    if drift:
        print("\n  ⚠ installed-but-not-loaded → restart the host to load it, or this")
        print("    agent's `features` allowlist in multi_agent.toml excludes it.")
    if not any_online:
        print("\n  No running host found — start it (or check the port) for per-agent state.")
    return 0


def cmd_feature_enable(args) -> int:
    """Enable a feature (remove from disabled list in kestrel.toml)."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    info = registry[pkg_name]
    project_dir = cli._get_project_dir()
    disabled = cli._get_toml_disabled_features(project_dir)

    # Remove all feature classes for this package from disabled list
    removed = []
    new_disabled = []
    for feat in disabled:
        if feat in info.features:
            removed.append(feat)
        else:
            new_disabled.append(feat)

    if not removed:
        print(f"Feature '{pkg_name}' is not disabled")
        return 0

    cli._set_toml_disabled_features(project_dir, new_disabled)
    print(f"Enabled {pkg_name} (removed {', '.join(removed)} from disabled list)")
    print("Restart the agent for changes to take effect")
    return 0


def cmd_feature_disable(args) -> int:
    """Disable a feature (add to disabled list in kestrel.toml)."""
    from kestrel_sovereign.feature_registry import load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    info = registry[pkg_name]
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

    mandatory = sorted(set(info.features) & MANDATORY_FEATURES)
    if mandatory:
        print(
            "Cannot disable mandatory sovereignty feature(s): "
            + ", ".join(mandatory)
        )
        return 1
    project_dir = cli._get_project_dir()
    disabled = cli._get_toml_disabled_features(project_dir)

    # Add all feature classes for this package to disabled list
    added = []
    for feat in info.features:
        if feat not in disabled:
            disabled.append(feat)
            added.append(feat)

    if not added:
        print(f"Feature '{pkg_name}' is already disabled")
        return 0

    cli._set_toml_disabled_features(project_dir, disabled)
    print(f"Disabled {pkg_name} (added {', '.join(added)} to disabled list)")
    print("Restart the agent for changes to take effect")
    return 0


def cmd_feature_info(args) -> int:
    """Show detailed info about a feature package."""
    from kestrel_sovereign.feature_registry import get_registry, FeatureStatus

    project_dir = cli._get_project_dir()
    toml_disabled = set(cli._get_toml_disabled_features(project_dir))

    registry = get_registry()

    # Override status for kestrel.toml disabled
    for info in registry.values():
        if set(info.features) & toml_disabled:
            info.status = FeatureStatus.DISABLED

    pkg_name = _resolve_feature_name(args.name, registry)
    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    info = registry[pkg_name]

    print()
    print(f"  {pkg_name}")
    print(f"  {'─' * 40}")
    print(f"  Description:  {info.description}")
    print(f"  Package:      {info.package}")
    print(f"  Boundary:     {info.boundary.value}")
    print(f"  Status:       {info.status.value}")
    print(f"  Core:         {'yes' if info.core else 'no'}")
    print(f"  Git:          {info.git}")

    if info.tags:
        print(f"  Tags:         {', '.join(info.tags)}")

    if info.features:
        print(f"  Features:     {', '.join(info.features)}")
    if info.provider_classes:
        print(f"  Providers:    {', '.join(info.provider_classes)}")
    if info.command:
        print(f"  Command:      {info.command}")

    if info.skills:
        print(f"\n  Skills:")
        for skill in info.skills:
            tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
            print(f"    {skill.name:<24} {skill.description}{tags_str}")

    print()
    return 0


def _load_scaffold_template(template_name: str) -> str:
    """Load a scaffold template from the data/feature_template directory."""
    template_dir = Path(__file__).parent / "data" / "feature_template"
    return (template_dir / template_name).read_text()


def _render_template(template: str, variables: dict) -> str:
    """Render a template by replacing ``{{variable}}`` placeholders.

    Double-brace syntax avoids collisions with Python f-strings and
    single-brace dicts that may appear in template content.
    """
    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def cmd_feature_scaffold(args) -> int:
    """Generate a feature package project template."""
    name = args.name.lower().replace("-", "_").replace(" ", "_")
    pkg_name = f"kestrel_feature_{name}"
    name_dashed = name.replace("_", "-")
    dir_name = f"kestrel-feature-{name_dashed}"

    scaffold_dir = Path(dir_name)
    if scaffold_dir.exists():
        print(f"Directory '{dir_name}' already exists")
        return 1

    # Feature class name: my_feature -> MyFeature + "Feature"
    class_name = "".join(word.capitalize() for word in name.split("_")) + "Feature"

    # Template variables
    variables = {
        "name": name,
        "name_dashed": name_dashed,
        "pkg_name": pkg_name,
        "class_name": class_name,
    }

    # Create directory structure: src/ layout
    src_dir = scaffold_dir / "src" / pkg_name
    tests_dir = scaffold_dir / "tests"

    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    # Render templates
    templates = {
        src_dir / "__init__.py": "__init__.py.tpl",
        src_dir / "feature.py": "feature.py.tpl",
        scaffold_dir / "pyproject.toml": "pyproject.toml.tpl",
        tests_dir / f"test_{name}.py": "test_feature.py.tpl",
        tests_dir / "conftest.py": "conftest.py.tpl",
        scaffold_dir / "SKILL.md": "SKILL.md.tpl",
        scaffold_dir / "README.md": "README.md.tpl",
    }

    for output_path, template_name in templates.items():
        template = _load_scaffold_template(template_name)
        output_path.write_text(_render_template(template, variables))

    # tests/__init__.py (empty)
    (tests_dir / "__init__.py").write_text("")

    print(f"Scaffolded feature package: {dir_name}/")
    print(f"  src/{pkg_name}/")
    print(f"    __init__.py")
    print(f"    feature.py          <- implement {class_name} here")
    print(f"  tests/")
    print(f"    test_{name}.py      <- uses MockAgent fixture")
    print(f"    conftest.py")
    print(f"  pyproject.toml        <- entry_point registered")
    print(f"  SKILL.md              <- skill descriptions")
    print(f"  README.md")
    print()
    print(f"Install in dev mode:  pip install -e ./{dir_name}")
    return 0


def cmd_feature_skills(args) -> int:
    """List skills for a specific feature package."""
    from kestrel_sovereign.feature_registry import get_skills_for_package, load_registry

    registry = load_registry()
    pkg_name = _resolve_feature_name(args.name, registry)

    if pkg_name is None:
        print(f"Unknown feature: {args.name}")
        return 1

    skills = get_skills_for_package(pkg_name)
    if not skills:
        print(f"No skills declared for '{pkg_name}'")
        return 0

    print()
    print(f"  Skills for {pkg_name}:")
    print(f"  {'─' * 50}")
    for skill in skills:
        tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
        print(f"  {skill.name:<24} {skill.description}{tags_str}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Skills CLI commands
# ---------------------------------------------------------------------------



def add_feature_subparser(subparsers) -> None:
    """Register the ``kestrel feature ...`` subcommands on ``subparsers``."""
    # kestrel feature {list|inventory|install|upgrade|sync|status|enable|disable|info|scaffold|skills}
    feature_p = subparsers.add_parser("feature", help="Manage features")
    feature_sub = feature_p.add_subparsers(dest="feature_command")

    feature_sub.add_parser("list", help="List features with status")

    feat_inventory = feature_sub.add_parser(
        "inventory",
        help="Generate discovered feature/tool/endpoint/command inventory",
    )
    feat_inventory.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    feat_inventory.add_argument(
        "--output",
        default=None,
        help="Write output to a file instead of stdout",
    )
    feat_inventory.add_argument(
        "--write",
        action="store_true",
        help="Regenerate KESTREL_FEATURES.md in place",
    )

    feat_install = feature_sub.add_parser("install", help="Install a feature package")
    feat_install.add_argument("name", help="Feature name (e.g. cloud, voice)")

    feat_upgrade = feature_sub.add_parser(
        "upgrade",
        help="Upgrade installed feature/provider packages via pip",
    )
    feat_upgrade.add_argument(
        "names",
        nargs="*",
        help="Optional package/feature names to upgrade (default: all installed extensions)",
    )
    feat_upgrade.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be upgraded without changing anything",
    )

    feat_sync = feature_sub.add_parser(
        "sync",
        help="Restore the host's declared feature packages (counterpart to upgrade)",
    )
    feat_sync.add_argument(
        "--manifest",
        default=None,
        help=f"Path to the host manifest (default: ./{DEFAULT_HOST_MANIFEST})",
    )
    feat_sync.add_argument(
        "--capture",
        action="store_true",
        help="Write the manifest from currently-installed extension packages",
    )
    feat_sync.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed/restored without changing anything",
    )
    feat_sync.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Pull a declared editable checkout even when it has modified "
            "tracked files (same meaning as on `kestrel update`)"
        ),
    )

    feat_status = feature_sub.add_parser(
        "status",
        help="Show host install state + per-agent loaded state for manifest features",
    )
    feat_status.add_argument(
        "--agent",
        default=None,
        help="Only show this agent (default: all local agents)",
    )
    feat_status.add_argument(
        "--manifest",
        default=None,
        help=f"Path to the host manifest (default: ./{DEFAULT_HOST_MANIFEST})",
    )

    feat_enable = feature_sub.add_parser("enable", help="Enable a disabled feature")
    feat_enable.add_argument("name", help="Feature or package name")

    feat_disable = feature_sub.add_parser("disable", help="Disable a feature")
    feat_disable.add_argument("name", help="Feature or package name")

    feat_info = feature_sub.add_parser("info", help="Show feature details")
    feat_info.add_argument("name", help="Feature or package name")

    feat_scaffold = feature_sub.add_parser("scaffold", help="Generate feature project template")
    feat_scaffold.add_argument("name", help="Feature name (e.g. myfeature)")

    feat_skills = feature_sub.add_parser("skills", help="List skills in a feature")
    feat_skills.add_argument("name", help="Feature or package name")


# Imported at module bottom (not top) to break the cli <-> cli_features import
# cycle: cli.py re-exports the public handlers above, so when this module is
# imported first it must finish defining them before `cli` (which imports
# them back) loads. Every `cli.<name>` reference above resolves at call time,
# so binding `cli` here at end-of-module is safe.
from kestrel_sovereign import cli  # noqa: E402
