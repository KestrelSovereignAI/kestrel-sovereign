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
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

# Non-patched constants — import directly. (``MultiAgentConfig`` itself is
# referenced as ``cli.MultiAgentConfig`` because the test suite patches it.)
# ``feature_reconcile`` holds the pure planning types and imports nothing from
# the CLI, so naming the core distribution here is not a cycle.
from kestrel_sovereign.feature_reconcile import CORE_DISTRIBUTION
from kestrel_sovereign.multi_agent.config import MULTI_AGENT_CONFIG_FILENAME


def _resolve_feature_name(name: str, registry: dict) -> Optional[str]:
    """
    Resolve a user-provided name to a registry short name.

    Accepts a registry short name ("voice"), registered distribution name
    ("kestrel-feature-voice"), or feature/provider class name
    ("VoiceFeature"). Package-style names use Python's normalized-name rules,
    so case and runs of ``-``, ``_``, or ``.`` are equivalent.
    """
    # Preserve the cheapest and most common exact registry-key lookup.
    if name in registry:
        return name

    normalized = canonicalize_name(name)
    for pkg_name, info in registry.items():
        if (
            canonicalize_name(pkg_name) == normalized
            or canonicalize_name(info.package) == normalized
        ):
            return pkg_name

    # Feature lifecycle or provider implementation class-name match.
    for pkg_name, info in registry.items():
        components = getattr(info, "runtime_components", info.features)
        if name in components:
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
        # A failed install can be the very thing that broke the link.
        guard.verify()
        return 1


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

        result = guard.run(["--upgrade", name])
        if result.returncode != 0 and git_urls.get(name):
            result = guard.run(["--upgrade", f"git+{git_urls[name]}"])

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
        # Detection half: an upgrade that bypassed the pin can't leave the
        # command reporting success over a replaced core.
        if guard.verify():
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


def _extension_install_run(pip_args: list, *, constraints, timeout=None):
    """Install via uv when available, else the interpreter's own pip.

    Prefer :meth:`CoreInstallGuard.run` — it decides ``constraints`` for you and
    verifies the result. ``constraints`` is keyword-only and REQUIRED so a new
    caller has to make the core-guard decision explicitly instead of inheriting
    a silent default that reopens issue #2949; pass ``None`` only when core
    itself is the install target (it is never constrained against itself).

    uv-created virtualenvs frequently ship without ``pip``, so a bare
    ``python -m pip`` would fail. Prefer ``uv pip install --python <interp>``
    (pinned to *this* interpreter so a worktree can't retarget the wrong env),
    and fall back to ``python -m pip install`` only when uv isn't on PATH.

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
    constraints file and passed as ``-c``. Callers pin core to the checkout's
    version so the index is not a candidate; a real version skew then fails the
    resolve loudly instead of swapping the install underneath the operator.

    ``--no-deps`` is NOT the fix: it stops feature dependencies resolving at all
    and hides the version-skew signal rather than surfacing it.
    """
    import shutil
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
        if shutil.which("uv"):
            cmd = [
                "uv", "pip", "install", "--python", sys.executable,
                *extra_args, *pip_args,
            ]
        else:
            cmd = [sys.executable, "-m", "pip", "install", *extra_args, *pip_args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    finally:
        if constraint_file:
            try:
                os.unlink(constraint_file)
            except OSError:
                pass


def _core_install_shape():
    """Snapshot how ``kestrel-sovereign`` is installed in this venv.

    Returns a :class:`~kestrel_sovereign.feature_reconcile.CoreInstallShape`.
    Read through ``cli.`` so the test suite's patch seams apply.
    """
    import importlib.metadata as md

    from kestrel_sovereign.feature_reconcile import CoreInstallShape

    try:
        version = md.version(CORE_DISTRIBUTION)
    except md.PackageNotFoundError:
        version = None
    return CoreInstallShape(
        version=version,
        editable_path=cli._editable_install_path(CORE_DISTRIBUTION),
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
    command: Optional[str] = None
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
            lines.append(f"RESTORE FAILED — run `{self.command}` by hand.")
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
        policy = fr.resolve_core_policy(source_index or {}, before.editable_path)
        return cls(policy, before)

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
        """The constraint lines applied to each guarded install (may be empty)."""
        return list(self._constraints)

    def run(self, pip_args: list, *, timeout=None):
        """Install a FEATURE package with core held inside the policy."""
        return cli._extension_install_run(
            pip_args, constraints=self._constraints or None, timeout=timeout,
        )

    def install_core(self, pip_args: list, *, timeout=None):
        """Install CORE itself — the operator deliberately moving core's source.

        Never constrained against itself, and the constraints for the rest of
        the batch are re-derived afterwards: a batch that switches core to a new
        checkout (or a new pin) must go on to pin the version core actually
        became, not the one it was before the switch.
        """
        result = cli._extension_install_run(pip_args, constraints=None, timeout=timeout)
        self.refresh()
        return result

    def refresh(self):
        """Re-read the live core install; re-derive constraints and baseline."""
        shape = cli._core_install_shape()
        self._baseline = shape
        self._constraints = self._derive(shape)
        return shape

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

    def _repair(self):
        """Reinstall core from its declared source. Returns ``(ok, command, result)``."""
        if self.policy.editable:
            checkout = str(Path(self.policy.editable).expanduser())
            pip_args = ["-e", checkout]
        else:
            # Declared from the index. A --force-reinstall is only needed to
            # displace an editable link; a version outside the window is fixed
            # by resolving the declared spec again.
            spec = f"{CORE_DISTRIBUTION}{self.policy.pypi}"
            pip_args = (
                ["--force-reinstall", spec]
                if cli._editable_install_path(CORE_DISTRIBUTION) else [spec]
            )
        result = cli._extension_install_run(pip_args, constraints=None)
        rendered = "uv pip install " + " ".join(pip_args)
        if result.returncode == 0:
            self.refresh()
        return result.returncode == 0, rendered, result

    def resolve(self) -> CoreGuardOutcome:
        """Check core against the policy, and repair it if it drifted.

        The one place that decides what a drift means and what to do about it,
        so the CLI (:meth:`verify`) and the HTTP install endpoint cannot answer
        that question differently — a divergence there is how one surface ends
        up reporting success over a core the other would have refused.

        Repair success is judged by RE-CHECKING core, not by the installer's
        exit code: a reinstall that returned 0 and still left core off its
        declared source is not a repair.
        """
        from kestrel_sovereign.feature_reconcile import core_install_matches

        drift = self._check()
        if drift is None:
            return CoreGuardOutcome()

        # Distinguish "the batch broke it" from "it never got there": both are
        # failures, but only the first is a swap.
        replaced = core_install_matches(self._baseline, self.policy)
        ok, rendered, result = self._repair()
        repaired = ok and self._check() is None
        return CoreGuardOutcome(
            drift=drift,
            replaced=replaced,
            repaired=repaired,
            command=rendered,
            output="" if repaired else (result.stderr or result.stdout or "").strip(),
        )

    def verify(self) -> int:
        """Assert core survived the batch; repair and report if it did not.

        Returns 0 when core conforms (or nothing is guarded), 1 when it moved —
        so no command can report success over a core that was replaced, even
        when the re-link succeeded.
        """
        outcome = self.resolve()
        if outcome.drift is None:
            return 0

        print(f"• core: ERROR — {outcome.headline}.", file=sys.stderr)
        for line in outcome.drift.splitlines():
            print(f"    {line}", file=sys.stderr)
        if outcome.repaired:
            print(f"    restored: {outcome.command}", file=sys.stderr)
        else:
            print(
                f"    RESTORE FAILED — run `{outcome.command}` by hand.",
                file=sys.stderr,
            )
            for line in outcome.output.splitlines()[-3:]:
                print(f"      {line}", file=sys.stderr)
        return 1


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


def _core_entry_first(entries: list, registry: dict) -> list:
    """Manifest order, with any ``kestrel-sovereign`` entry moved to the front.

    Core's declared source has to be applied before the batch derives its pin
    from the installed state, and before any feature can drag core somewhere the
    manifest didn't declare. Order among the remaining entries is preserved so
    the operator's file still reads the way it runs (issue #2949).
    """
    def is_core(entry) -> bool:
        info = _registry_info_for(entry["name"], registry)
        return (info.package if info else entry["name"]) == CORE_DISTRIBUTION

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
    git_urls = {info.package: info.git for info in registry.values() if info.package}
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
        elif pypi_want is not None and cli._editable_install_path(target):
            # Switching a currently-editable install to a PyPI source needs a
            # force reinstall; a plain pip install leaves the satisfying
            # editable link in place, so the source switch never takes effect
            # (codex round 10 P2).
            pip_args = ["--force-reinstall", install_target]
        else:
            pip_args = [install_target]

        # Core is never constrained against itself — a `kestrel-sovereign`
        # manifest entry IS the operator deliberately moving core's source.
        # `install_core` re-derives the batch pin from what core became.
        is_core = target == fr.CORE_DISTRIBUTION
        run = guard.install_core if is_core else guard.run
        result = run(pip_args)
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
            result = run([git_spec])

        if result.returncode != 0:
            rc = 1
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
        if guard.verify():
            rc = 1
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
