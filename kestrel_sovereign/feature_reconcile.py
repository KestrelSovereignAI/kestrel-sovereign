"""Reconcile the host venv against the union of per-agent feature allowlists.

The per-agent ``[agents.*].features`` allowlist in ``multi_agent.toml`` decides
*which* features each agent loads, but an allowlist is a **filter**, not an
**installer**: naming a feature class an agent doesn't have installed in the
venv just makes it silently never load. This module closes that gap by deriving
what the *host* must provide from the union of all agent allowlists, and then
installing / updating exactly those packages.

Three distinct concerns (issue #1788):

1. **WHAT the host needs** = ``union(all [agents.*].features) + MANDATORY_FEATURES``,
   derived from ``multi_agent.toml`` — never hand-maintained.
2. **WHERE each comes from** = ``.kestrel-host-features.toml`` as a source map:
   one entry per package recording install mode — ``editable = "/path"`` or
   ``pypi = ">=x,<y"``.
3. **UPDATE mode** = per-feature, read from the source map: editable → ``git
   pull --ff-only`` the checkout; pypi → ``pip install --upgrade`` (respecting
   floor pins). ``--prefer-source`` / ``--prefer-pypi`` bulk-override the mode.

This module holds the **pure planning** logic (no subprocesses, no printing) so
it is trivially testable; ``cli_lifecycle.cmd_update`` owns execution + display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

# The core distribution. Features whose registry ``package`` is this are bundled
# into kestrel-sovereign itself (step 1 of ``kestrel update`` pulls + reinstalls
# it) — reconcile must NOT try to pip-install them as separate extensions.
CORE_DISTRIBUTION = "kestrel-sovereign"


@dataclass
class SourceEntry:
    """A resolved source-map entry: where a package comes from + how to update.

    ``editable`` and ``pypi`` are mutually exclusive (enforced by the manifest
    loader). ``pypi`` is a PEP 440 version spec; the empty string means "any
    version / latest". When neither is set the entry is legacy "install latest
    from PyPI" — preserved for back-compat with pre-#1788 manifests.
    """

    package: str
    editable: Optional[str] = None
    pypi: Optional[str] = None
    extras: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return "editable" if self.editable else "pypi"


@dataclass
class ReconcileAction:
    """One planned reconcile operation for a single package."""

    package: str
    op: str  # "install" | "update" | "present"
    mode: str  # "editable" | "pypi"
    source: Optional[str]  # checkout path (editable) or version spec (pypi)
    required_by: List[str] = field(default_factory=list)
    current_version: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    # Editable only: the venv must be (re)linked via ``pip install -e`` because
    # the package is absent OR currently linked to a different checkout (or to a
    # non-editable PyPI build). A git pull alone would leave the venv stale.
    relink: bool = False
    # PyPI only: the source carries an explicit version pin, so the executor
    # must NOT fall back to an unpinned git URL on failure (that would silently
    # move the feature outside the operator's declared pin).
    pinned: bool = False
    # PyPI only: the package is CURRENTLY installed editable but the source map
    # now declares PyPI, so the executor must ``--force-reinstall`` to replace
    # the editable link with the wheel — a plain ``--upgrade`` is a no-op when
    # the editable version already satisfies the spec, leaving it linked.
    force_reinstall: bool = False
    note: str = ""


def compute_required_classes(multi_agent) -> Tuple[set, List[str]]:
    """Derive the set of feature classes the host must provide.

    ``required = union(each LOCAL agent's explicit [agents.*].features)
    + MANDATORY_FEATURES``.

    An agent with ``features = None`` (no allowlist) loads *every* installed
    feature and so imposes no *specific* requirement — the host cannot install
    "everything". Such agents are returned separately so the caller can note
    that the reconcile only guarantees the explicitly-named union.

    Returns ``(required_classes, load_all_agent_names)``.
    """
    required = set(MANDATORY_FEATURES)
    load_all: List[str] = []
    for name, cfg in multi_agent.get_local_agents().items():
        if cfg.features is None:
            load_all.append(name)
            continue
        required.update(cfg.features)
    return required, sorted(load_all)


def resolve_packages(
    required_classes,
    registry,
    entrypoint_dists: Optional[Dict[str, str]] = None,
    local_core_classes: Optional[set] = None,
) -> Tuple[Dict, Dict, List[str]]:
    """Resolve feature *class* names to *packages*.

    Allowlists name classes; install works on packages. Resolution consults, in
    order, the most authoritative source first — **live discovery is the source
    of truth**, the static catalog is only a fallback / metadata source. The
    order also mirrors the runtime loader's precedence (local features win over
    duplicate-named entry points):

      1. ``local_core_classes`` — in-tree bundled Feature classes. Checked
         first: the loader discovers them before entry-point features and skips
         duplicates, so a bundled class is what the agent actually loads.
         Provisioned by ``kestrel update`` step 1 (core reinstall) — no
         separate extension install.
      2. ``entrypoint_dists`` — installed external feature packages, mapped
         class → distribution from live entry-point metadata. The truth for
         what's actually loadable from a pip package right now.
      3. The static registry catalog — an external package that's *catalogued*
         but not yet installed (the legitimate "install the missing feature"
         case).
      4. ``MANDATORY_FEATURES`` — guaranteed by core even if uncatalogued.

    Anything that matches none of these is a class an agent allowlist names but
    the host cannot provide — the motivating silent-no-load bug — returned in
    ``unresolved`` as a hard error (no blind fallback).

    Returns ``(pkg_infos, class_to_pkg, unresolved)`` where ``pkg_infos`` is
    ``{package: FeaturePackageInfo}`` for every required, non-core, installable
    package.
    """
    entrypoint_dists = entrypoint_dists or {}
    local_core_classes = local_core_classes or set()

    reg_by_class: Dict[str, "FeaturePackageInfo"] = {}
    reg_by_package: Dict[str, "FeaturePackageInfo"] = {}
    for info in registry.values():
        if info.package:
            reg_by_package.setdefault(info.package, info)
        for cls in info.features:
            reg_by_class.setdefault(cls, info)

    pkg_infos: Dict = {}
    class_to_pkg: Dict[str, str] = {}
    unresolved: List[str] = []

    for cls in sorted(required_classes):
        # 1. Bundled in-tree class — no separate extension to provision. Checked
        #    FIRST to match the runtime loader, which discovers local features
        #    before entry-point ones and skips a duplicate-named entry point;
        #    so when a class is both bundled AND shipped by an installed
        #    package, the agent loads the bundled one and reconcile must not
        #    touch the external package (codex round 6 P2).
        if cls in local_core_classes:
            class_to_pkg[cls] = CORE_DISTRIBUTION
            continue

        # 2. Installed external package (live entry-point truth).
        dist = entrypoint_dists.get(cls)
        if dist:
            class_to_pkg[cls] = dist
            if dist != CORE_DISTRIBUTION:
                # Prefer registry metadata (git URL, extras) when catalogued;
                # otherwise synthesize a minimal info from the live dist.
                pkg_infos[dist] = reg_by_package.get(dist) or FeaturePackageInfo(
                    name=dist, package=dist, git="", features=[cls],
                    description="", core=False,
                )
            continue

        # 3. Catalogued external package (maybe not yet installed → install).
        info = reg_by_class.get(cls)
        if info is not None:
            class_to_pkg[cls] = info.package
            if info.package and info.package != CORE_DISTRIBUTION:
                pkg_infos[info.package] = info
            continue

        # 4. Mandatory features are guaranteed by core even when uncatalogued.
        if cls in MANDATORY_FEATURES:
            continue

        # 5. Named by an allowlist but the host can't provide it — the bug.
        unresolved.append(cls)

    return pkg_infos, class_to_pkg, unresolved


def build_source_index(manifest_entries, registry) -> Dict[str, SourceEntry]:
    """Map manifest entries to ``{package: SourceEntry}``.

    Manifest entries name a registry short-name ("voice") or a raw dist name
    ("kestrel-feature-voice"); both resolve to the package (dist) name so the
    index keys line up with :func:`resolve_packages`.
    """
    index: Dict[str, SourceEntry] = {}
    for entry in manifest_entries:
        package = _entry_package(entry, registry)
        index[package] = SourceEntry(
            package=package,
            editable=entry.get("editable"),
            pypi=entry.get("pypi"),
            extras=list(entry.get("extras", []) or []),
        )
    return index


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    """True only when both are non-empty paths that resolve to the same place."""
    if not a or not b:
        return False
    try:
        from pathlib import Path as _Path

        return _Path(a).expanduser().resolve() == _Path(b).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return a == b


def _pypi_requirement(package: str, spec: Optional[str], extras: List[str]) -> str:
    """Render a pip requirement with extras BEFORE the version spec.

    pip requires ``pkg[extra]>=x,<y`` — ``pkg>=x,<y[extra]`` is rejected. Used
    for the PyPI source form so a pinned-with-extras feature installs cleanly.
    """
    base = f"{package}[{','.join(extras)}]" if extras else package
    return f"{base}{spec}" if spec else base


def _entry_package(entry: dict, registry) -> str:
    """Resolve a manifest entry's ``name`` to its package (dist) name."""
    label = entry["name"]
    # registry short-name -> package
    if label in registry and registry[label].package:
        return registry[label].package
    # already a dist name present in some registry entry
    for info in registry.values():
        if info.package == label:
            return info.package
    # unknown to the registry: trust the manifest's name verbatim
    return label


def plan_reconcile(
    pkg_infos: Dict,
    source_index: Dict[str, SourceEntry],
    installed_versions: Dict[str, Optional[str]],
    editable_paths: Dict[str, Optional[str]],
    class_to_pkg: Dict[str, str],
    prefer: Optional[str] = None,
) -> Tuple[List[ReconcileAction], List[str]]:
    """Plan the reconcile for every required, installable package.

    Args:
        pkg_infos: required packages → FeaturePackageInfo (from resolve_packages).
        source_index: package → SourceEntry (from the source map).
        installed_versions: package → installed version (or None if absent).
        editable_paths: package → editable checkout path (or None) as the venv
            *currently* records it (PEP 660 ``direct_url.json``).
        class_to_pkg: class → package, to populate ``required_by``.
        prefer: ``"source"`` forces editable update mode where a checkout path
            is known; ``"pypi"`` forces pip-upgrade mode even for editable
            installs; ``None`` honours the source map.

    Returns ``(actions, no_source)`` where ``no_source`` lists required packages
    with neither a source-map entry nor a registry-resolvable install source —
    a hard error (no blind fallback).
    """
    # Invert class_to_pkg → package: [classes]
    required_by: Dict[str, List[str]] = {}
    for cls, pkg in class_to_pkg.items():
        required_by.setdefault(pkg, []).append(cls)

    actions: List[ReconcileAction] = []
    no_source: List[str] = []

    for package in sorted(pkg_infos):
        info = pkg_infos[package]
        src = source_index.get(package)
        current = installed_versions.get(package)
        detected_editable = editable_paths.get(package)

        # Decide the install/update MODE and SOURCE.
        editable_path = None
        pypi_spec = None
        extras: List[str] = []

        if src is not None:
            extras = src.extras
            if src.editable:
                editable_path = src.editable
            if src.pypi is not None:
                pypi_spec = src.pypi

        # An install already recorded as editable counts as a known checkout
        # even if the manifest didn't spell one out (keeps a hand-installed dev
        # checkout from being clobbered with a PyPI build).
        if editable_path is None and detected_editable:
            editable_path = detected_editable

        # Resolve mode, honouring the prefer override.
        if prefer == "pypi":
            mode = "pypi"
        elif prefer == "source" and editable_path:
            mode = "editable"
        elif editable_path and (src is None or src.mode == "editable"):
            mode = "editable"
        elif pypi_spec is not None or current is not None or info.package:
            mode = "pypi"
        else:
            mode = "pypi"

        relink = False
        pinned = False
        force_reinstall = False
        op = "update" if current is not None else "install"
        if mode == "editable":
            if not editable_path:
                # prefer=source asked for editable but no checkout is known.
                no_source.append(package)
                continue
            source = editable_path
            # The venv must be re-linked via ``pip install -e`` (not just
            # git-pulled) when the package is absent, is installed from a
            # non-editable PyPI build, or is linked to a DIFFERENT checkout
            # than the source map now wants. Otherwise a git pull would leave
            # the venv pointing at the stale install (codex round 3 P2).
            relink = current is None or not _same_path(detected_editable, editable_path)
        else:
            # PyPI mode. A declared spec always wins. Otherwise we can only pip
            # from PyPI when the package has a known remote source — i.e. it is
            # catalogued in the registry (``info.git`` is set for every catalog
            # entry; synthesized live-only infos have it empty). Render extras
            # BEFORE the version spec (``pkg[extra]>=x``) — pip rejects the
            # reverse.
            # Switching a currently-editable install to PyPI requires a force
            # reinstall: a plain --upgrade is a no-op when the editable version
            # already satisfies the spec, leaving the checkout linked (codex
            # round 9 P2).
            force_reinstall = bool(detected_editable)
            if pypi_spec is not None:
                source = _pypi_requirement(package, pypi_spec, extras)
                # A non-empty spec is a real pin (``""`` means "any version").
                pinned = bool(pypi_spec)
            elif info.git:
                source = _pypi_requirement(package, None, extras)
            elif current is not None:
                # Installed and loadable but no known remote source (a local /
                # private / direct-URL install with no source-map entry). Leave
                # it untouched — the allowlist requirement is already satisfied;
                # fabricating a PyPI upgrade would fail and abort the update
                # (codex round 4 P2). Add a source-map entry to manage it.
                op = "present"
                source = None
            else:
                no_source.append(package)
                continue

        actions.append(
            ReconcileAction(
                package=package,
                op=op,
                mode=mode,
                source=source,
                required_by=sorted(required_by.get(package, [])),
                current_version=current,
                extras=extras,
                relink=relink,
                pinned=pinned,
                force_reinstall=force_reinstall and op != "present",
            )
        )

    return actions, sorted(no_source)
