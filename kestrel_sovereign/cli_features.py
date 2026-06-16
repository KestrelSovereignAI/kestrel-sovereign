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
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Non-patched constant — import directly. (``MultiAgentConfig`` itself is
# referenced as ``cli.MultiAgentConfig`` because the test suite patches it.)
from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME


def _resolve_feature_name(name: str, registry: dict) -> Optional[str]:
    """
    Resolve a user-provided name to a registry package name.

    Accepts: package name ("voice"), feature class name ("VoiceFeature"),
    or human-friendly name (case-insensitive match).
    """
    # Direct package name match
    if name in registry:
        return name

    # Case-insensitive package name match
    lower = name.lower()
    for pkg_name in registry:
        if pkg_name.lower() == lower:
            return pkg_name

    # Feature class name match
    for pkg_name, info in registry.items():
        if name in info.features:
            return pkg_name

    return None


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
            "{list|install|upgrade|sync|status|enable|disable|info|scaffold|skills}"
        )
        return 1
    return handler(args)


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

    cmd = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try git URL as fallback
        if info.git:
            print(f"pip install failed, trying git: {info.git}")
            cmd = [sys.executable, "-m", "pip", "install", f"git+{info.git}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Installed {package}")
        return 0
    else:
        print(f"Failed to install {package}")
        if result.stderr:
            # Show last few lines of error
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        return 1


# Entry-point groups that back installable Kestrel extensions. A feature or
# provider package registers itself here; core/in-repo features do not, which is
# exactly why they are excluded from `feature upgrade` (they ship with, and
# upgrade with, kestrel-sovereign itself).
_EXTENSION_ENTRY_POINT_GROUPS = (
    "kestrel_sovereign.features",
    "kestrel_sovereign.cloud_providers",
    "kestrel_sovereign.voice_providers",
    "kestrel_sovereign.storage_providers",
)


def _editable_install_path(dist_name: str) -> Optional[str]:
    """Return the local source path if *dist_name* is an editable install.

    PEP 660 editable installs record ``direct_url.json`` with
    ``dir_info.editable == true``. Editable installs track a local checkout, so
    pip cannot meaningfully "upgrade" them.
    """
    import importlib.metadata as md
    import json

    try:
        raw = md.distribution(dist_name).read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not data.get("dir_info", {}).get("editable"):
        return None
    url = data.get("url", "")
    return url[len("file://") :] if url.startswith("file://") else url or None


def _installed_extension_distributions() -> list:
    """Enumerate distinct pip distributions backing installed extension entry-points.

    Kestrel already knows what is loaded — each feature/provider class is wired
    through an entry-point whose distribution we can resolve — so no curated
    requirements file is needed. Returns a list of dicts sorted by name:
    ``{dist, version, editable_path, entries: [group:name, ...]}``.
    """
    import importlib.metadata as md

    by_dist: dict = {}
    for group in _EXTENSION_ENTRY_POINT_GROUPS:
        try:
            eps = md.entry_points(group=group)
        except TypeError:  # Python <3.10 selection API
            eps = md.entry_points().get(group, [])
        for ep in eps:
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


def _parse_pip_installed_version(stdout: str, dist_name: str) -> Optional[str]:
    """Extract the post-upgrade version from pip's 'Successfully installed' line."""
    normalized = dist_name.lower().replace("_", "-")
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("Successfully installed"):
            continue
        for token in line.split()[2:]:
            pkg, _, ver = token.rpartition("-")
            if pkg.lower().replace("_", "-") == normalized:
                return ver
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
    git_urls = {info.package: info.git for info in registry.values() if info.package}

    if requested:
        wanted = set()
        for name in requested:
            pkg = _resolve_feature_name(name, registry)
            wanted.add(registry[pkg].package if pkg else name)
        unmatched = wanted - {d["dist"] for d in dists}
        dists = [d for d in dists if d["dist"] in wanted]
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

        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and git_urls.get(name):
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", f"git+{git_urls[name]}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            rc = 1
            print(f"  {name:<34} {current:<10} FAILED")
            for line in (result.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}")
            continue

        new_version = _parse_pip_installed_version(result.stdout, name)
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


def _extension_install_run(pip_args: list):
    """Install via uv when available, else the interpreter's own pip.

    uv-created virtualenvs frequently ship without ``pip``, so a bare
    ``python -m pip`` would fail. Prefer ``uv pip install --python <interp>``
    (pinned to *this* interpreter so a worktree can't retarget the wrong env),
    and fall back to ``python -m pip install`` only when uv isn't on PATH.
    """
    import shutil

    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, *pip_args]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *pip_args]
    return subprocess.run(cmd, capture_output=True, text=True)


def _capture_host_manifest(path: Path) -> int:
    """Snapshot currently-installed extension packages into a manifest.

    Reuses the same live-entry-point discovery as ``feature upgrade`` so the
    host never hand-authors the file. Editable installs are recorded with
    their checkout path; extras can't be recovered from metadata, so the user
    adds those by hand if needed.
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
    for d in dists:
        lines.append("[[feature]]")
        lines.append(f'name = {_toml_basic_string(d["dist"])}')
        if d["editable_path"]:
            lines.append(f'editable = {_toml_basic_string(d["editable_path"])}')
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return len(dists)


def _registry_info_for(label: str, registry: dict):
    """Resolve a manifest entry name to its registry info.

    Handles both registry short names ("voice") and raw dist names
    ("kestrel-feature-voice", which manifests commonly use). Returns the
    FeaturePackageInfo or None.
    """
    pkg = _resolve_feature_name(label, registry)
    if pkg:
        return registry[pkg]
    for info in registry.values():
        if info.package == label:
            return info
    return None


def _version_satisfies(version: str, spec: str) -> bool:
    """Does *version* satisfy the PEP 440 *spec* (e.g. ``>=0.3,<0.4``)?

    An empty spec means "any version". When the spec can't be evaluated
    (``packaging`` missing or a malformed value) we conservatively return True
    so sync doesn't churn-reinstall on an unparseable pin.
    """
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(version) in SpecifierSet(spec)
    except Exception:  # noqa: BLE001
        return True


def _resolve_manifest_action(entry: dict, registry: dict):
    """Resolve a manifest entry to ``(target_package, current_version, action)``.

    ``action`` is exactly what ``kestrel feature sync`` would do:
    ``install`` (not present), ``reinstall`` (editable path mismatch, or a
    ``pypi`` pin the installed version violates), ``ensure`` (present but
    declared extras can't be probed), or ``present``. Shared by ``sync`` and
    ``status`` so the two never report different drift.
    """
    import importlib.metadata as md

    info = _registry_info_for(entry["name"], registry)
    target = info.package if info else entry["name"]
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
        if cli._editable_install_path(target) or not _version_satisfies(current, pypi_want):
            action = "reinstall"
        elif extras:
            action = "ensure"
        else:
            action = "present"
    elif extras:
        action = "ensure"
    else:
        action = "present"
    return target, current, action


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
    git_urls = {info.package: info.git for info in registry.values() if info.package}
    dry_run = getattr(args, "dry_run", False)

    print()
    print(f"  {'PACKAGE':<34} {'CURRENT':<10} {'ACTION'}")
    print(f"  {'-' * 62}")

    rc = 0
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

        if editable_want:
            pip_args = ["-e", _pip_spec(str(Path(editable_want).expanduser()), extras)]
        else:
            pip_args = [install_target]

        result = cli._extension_install_run(pip_args)
        # Registry-backed packages can fall back to their git URL (mirrors
        # `feature install`). Editable installs have no remote fallback. A
        # PyPI-pinned entry is excluded too: the git URL installs repo HEAD
        # with no version constraint, which would silently violate the pin.
        # Carry extras across via PEP 508 form (``pkg[extra] @ git+url``) so a
        # git fallback doesn't silently drop the local-pipeline deps.
        if (
            result.returncode != 0
            and not editable_want
            and not pypi_want
            and git_urls.get(target)
        ):
            git_ref = f"git+{git_urls[target]}"
            git_spec = f"{_pip_spec(target, extras)} @ {git_ref}" if extras else git_ref
            result = cli._extension_install_run([git_spec])

        if result.returncode != 0:
            rc = 1
            print(f"  {target:<34} {current or '-':<10} FAILED")
            for line in (result.stderr or "").strip().splitlines()[-3:]:
                print(f"      {line}")
            continue

        installed += 1
        done = {"install": "installed", "reinstall": "reinstalled", "ensure": "ensured"}[action]
        print(f"  {target:<34} {current or '-':<10} {done}")

    print()
    if not dry_run:
        print(f"  {installed} package(s) installed/restored.")
        if installed:
            print("  Restart the host/agents to load the synced features.")
    return rc


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
            out[pkg] = f.get("status")
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
    # plus whether it's a loadable Feature vs a provider. Feature-vs-provider
    # is decided from the registry by class-name convention (Feature classes
    # end in "Feature"; providers like OpenAITTSProvider do not) — independent
    # of install state, so a not-yet-installed feature still gets an agent row.
    targets = []
    for entry in entries:
        info = _registry_info_for(entry["name"], registry)
        target, current, action = _resolve_manifest_action(entry, registry)
        is_feature = bool(info) and any(
            c.endswith("Feature") for c in (info.features or [])
        )
        targets.append(
            {
                "display": info.name if info else entry["name"],
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
    print(f"  Status:       {info.status.value}")
    print(f"  Core:         {'yes' if info.core else 'no'}")
    print(f"  Git:          {info.git}")

    if info.tags:
        print(f"  Tags:         {', '.join(info.tags)}")

    if info.features:
        print(f"  Features:     {', '.join(info.features)}")

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
    # kestrel feature {list|install|upgrade|sync|status|enable|disable|info|scaffold|skills}
    feature_p = subparsers.add_parser("feature", help="Manage features")
    feature_sub = feature_p.add_subparsers(dest="feature_command")

    feature_sub.add_parser("list", help="List features with status")

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

