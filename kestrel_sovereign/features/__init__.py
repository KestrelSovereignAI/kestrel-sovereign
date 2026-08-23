"""
Feature Discovery for Kestrel Agent.

Auto-discovers and registers Feature classes from:
1. The local features/ directory (core features)
2. Installed pip packages via entry_points (external feature packages)
"""

import importlib
import importlib.metadata
import inspect
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Dict, Iterable, List, Mapping, Optional, Set, Type

from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
# Discovery checks against the SDK base so extracted package features
# (which inherit from kestrel_sdk.features.base.Feature) are recognized.
# Sovereign's Feature is itself a subclass of _SdkFeature, so internal
# features continue to pass the issubclass check too.
from kestrel_sdk.features.base import Feature as _SdkFeature

logger = logging.getLogger(__name__)

# Features directory location
FEATURES_DIR = Path(__file__).parent

# Environment variable for disabling features
DISABLED_FEATURES_ENV = "KESTREL_DISABLED_FEATURES"

# Entry point group for external feature packages
FEATURE_ENTRY_POINT_GROUP = "kestrel_sovereign.features"


def _safe_exception_type_name(error: BaseException) -> str:
    """Return one Core-selected label without reflecting subclass metadata."""

    if type(error) is ModuleNotFoundError:
        return "ModuleNotFoundError"
    if type(error) is ImportError:
        return "ImportError"
    return "Exception"


def _sanitized_isolated_runtime_import_exc_info(
    error: BaseException,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """Keep Core import frames while replacing dependency-controlled text."""

    core_frames = []
    current = error.__traceback__
    while current is not None:
        module_name = current.tb_frame.f_globals.get("__name__")
        if type(module_name) is str and module_name.startswith(
            "kestrel_sovereign."
        ):
            core_frames.append(current)
        current = current.tb_next
    safe_traceback: TracebackType | None = None
    for frame in reversed(core_frames):
        safe_traceback = TracebackType(
            safe_traceback,
            frame.tb_frame,
            frame.tb_lasti,
            frame.tb_lineno,
        )
    safe_error = ImportError(
        "the Core isolated-runtime module could not be imported; verify the "
        "installed Core and SDK dependencies"
    )
    safe_error.__cause__ = None
    safe_error.__context__ = None
    safe_error.__suppress_context__ = True
    return (ImportError, safe_error, safe_traceback)


class DuplicateFeatureEntryPointError(RuntimeError):
    """Raised when multiple distributions claim one external feature class."""


class FeatureDiscoveryAmbiguityError(RuntimeError):
    """A bundled and external class collide without an authorized migration."""


@dataclass(frozen=True)
class FeatureDiscoverySelection:
    """Selected in-process implementation and its inspectable provenance."""

    class_name: str
    feature_class: Type[_SdkFeature]
    source: str
    implementation_module: str
    discovery_location: str
    distribution: Optional[str] = None


@dataclass(frozen=True)
class _EntrypointFeatureCandidate:
    feature_class: Type[_SdkFeature]
    distribution: str
    entry_point: str


class MandatoryFeatureReadinessError(RuntimeError):
    """A sovereignty feature could not satisfy the readiness contract.

    The public message is assembled only from controlled class, stage, and
    problem labels. The underlying exception remains available as ``__cause__``
    for protected logs, while ``/health`` can safely expose ``str(error)``
    without copying credentials or paths out of a dependency exception.
    """

    _SAFE_STAGES = frozenset({
        "agent readiness",
        "configuration",
        "construction",
        "contribution registration",
        "discovery",
        "enablement",
        "import",
        "initialization",
        "persistent enablement",
        "post-load wiring",
        "registration",
        "runtime disable",
    })
    _SAFE_PROBLEMS = frozenset({
        "cannot be disabled",
        "could not be constructed",
        "could not be enabled",
        "could not be imported",
        "could not be inspected",
        "could not register its SDK contributions",
        "could not finish cross-feature wiring",
        "could not initialize",
        "could not register its hooks",
        "could not register its tools",
        "does not export its canonical class",
        "has a non-canonical runtime name",
        "is explicitly disabled",
        "is missing",
        "is not registered under its canonical name",
        "was loaded more than once",
    })

    def __init__(self, feature_name: str, stage: str, problem: str) -> None:
        from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

        self.feature_name = (
            feature_name
            if feature_name in MANDATORY_FEATURES
            else "unknown mandatory feature"
        )
        self.stage = stage if stage in self._SAFE_STAGES else "readiness"
        self.problem = problem if problem in self._SAFE_PROBLEMS else "failed"
        super().__init__(
            f"Mandatory feature '{self.feature_name}' {self.problem} during "
            f"{self.stage}; "
            "agent readiness refused. Repair the kestrel-sovereign "
            "installation or remove the invalid disable configuration, then "
            "restart the agent."
        )


def verify_mandatory_feature_set(
    features: Iterable[Feature] | Mapping[str, Feature],
    *,
    stage: str,
) -> None:
    """Require one canonical instance of every sovereignty feature.

    Discovery and the fully registered agent both call this postcondition. The
    second check is deliberate: tests, embedders, and future loaders may replace
    ``discover_features``, and registration keys come from mutable instance
    state rather than the class name.
    """
    from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

    feature_mapping = features if isinstance(features, Mapping) else None
    instances = list(
        feature_mapping.values() if feature_mapping is not None else features
    )
    counts = Counter(type(feature).__name__ for feature in instances)

    for feature_name in sorted(MANDATORY_FEATURES):
        count = counts[feature_name]
        if count == 0:
            raise MandatoryFeatureReadinessError(
                feature_name, stage, "is missing"
            )
        if count != 1:
            raise MandatoryFeatureReadinessError(
                feature_name, stage, "was loaded more than once"
            )

        instance = next(
            feature
            for feature in instances
            if type(feature).__name__ == feature_name
        )
        if getattr(instance, "name", feature_name) != feature_name:
            raise MandatoryFeatureReadinessError(
                feature_name, stage, "has a non-canonical runtime name"
            )
        if (
            feature_mapping is not None
            and feature_mapping.get(feature_name) is not instance
        ):
            raise MandatoryFeatureReadinessError(
                feature_name,
                stage,
                "is not registered under its canonical name",
            )


def _reject_duplicate_feature_entry_points(feature_eps) -> None:
    """Fail deterministically before entry-point enumeration can pick a winner."""
    claims: Dict[str, List[str]] = {}
    for ep in feature_eps:
        class_name = _entrypoint_class_name(
            getattr(ep, "value", "") or "", ep.name
        )
        dist = getattr(ep, "dist", None)
        owner = getattr(dist, "name", None) or getattr(ep, "value", None) or ep.name
        claims.setdefault(class_name, []).append(str(owner))

    distinct_claims = {
        name: sorted(set(owners)) for name, owners in claims.items()
    }
    conflicts = {
        name: owners
        for name, owners in distinct_claims.items()
        if len(owners) > 1
    }
    if conflicts:
        details = "; ".join(
            f"{name}: {', '.join(owners)}"
            for name, owners in sorted(conflicts.items())
        )
        remediation = ""
        if any(
            "kestrel-feature-intelligence" in owners
            for owners in conflicts.values()
        ):
            remediation = (
                " The archived kestrel-feature-intelligence bundle publishes "
                "stale Reflection/Council entry points; uninstall it and use "
                "the canonical standalone packages."
            )
        raise DuplicateFeatureEntryPointError(
            "Conflicting external feature entry points detected (each class "
            f"must have one owning distribution): {details}.{remediation}"
        )


def get_disabled_features() -> Set[str]:
    """
    Get the set of disabled feature names from environment and kestrel.toml.

    Sources (merged):
    1. KESTREL_DISABLED_FEATURES env var (comma-separated class names)
    2. [features].disabled list in kestrel.toml (project-level config)

    Example env: KESTREL_DISABLED_FEATURES=VoiceFeature,CreativeFeature
    Example toml:
        [features]
        disabled = ["VoiceFeature", "CreativeFeature"]
    """
    disabled: Set[str] = set()

    # Source 1: environment variable
    env_val = os.environ.get(DISABLED_FEATURES_ENV, "")
    if env_val:
        disabled.update(name.strip() for name in env_val.split(",") if name.strip())

    # Source 2: kestrel.toml [features].disabled
    try:
        # Resolve via the central paths module so pip-installed users land
        # on their KESTREL_HOME / ~/.kestrel project root instead of the
        # package's site-packages parent.
        from kestrel_sovereign.paths import project_dir as _resolve_project_dir
        toml_path = _resolve_project_dir() / "kestrel.toml"
        if toml_path.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            toml_disabled = data.get("features", {}).get("disabled", [])
            if isinstance(toml_disabled, list):
                disabled.update(toml_disabled)
    except Exception:
        pass  # Don't break feature loading if toml parsing fails

    return disabled


def discover_feature_modules() -> List[str]:
    """
    Discover all feature module paths in the features directory.
    
    Looks for:
    - features/<name>/__init__.py with a Feature subclass
    - features/<name>/feature.py with a Feature subclass
    - features/<name>.py with a Feature subclass
    
    Returns:
        List of module paths like ["features.sovereignty", "features.mcp"]
    """
    candidate_modules = []

    # Scan for subdirectories with feature modules
    for item in FEATURES_DIR.iterdir():
        if item.name.startswith("_") or item.name.startswith("."):
            continue

        if item.is_dir():
            # Check for __init__.py or feature.py
            init_file = item / "__init__.py"
            feature_file = item / "feature.py"

            if feature_file.exists():
                candidate_modules.append(f"kestrel_sovereign.features.{item.name}.feature")
            elif init_file.exists():
                candidate_modules.append(f"kestrel_sovereign.features.{item.name}")
        elif item.is_file() and item.suffix == ".py" and item.name != "base.py":
            # Single-file features like features/constitution.py
            module_name = item.stem
            candidate_modules.append(f"kestrel_sovereign.features.{module_name}")

    # Only keep modules that actually expose a discoverable Feature subclass.
    modules = []
    for module_path in candidate_modules:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Failed to import feature module {module_path}: {e}")
            continue

        if find_feature_class(module) is not None:
            modules.append(module_path)

    return modules


def find_feature_class(module) -> Optional[Type[Feature]]:
    """
    Find the Feature subclass in a module.

    Returns the first class that:
    1. Is a subclass of Feature
    2. Is not Feature itself
    3. Is defined in this module OR is explicitly exported via __all__
    4. Is not ProxyFeature (which is only created programmatically)

    The __all__ check allows features to be split across multiple files
    while still being discoverable.
    """
    module_all = getattr(module, '__all__', [])

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Accept any subclass of the SDK Feature base (covers both sovereign
        # internal features and extracted package features)
        if not (issubclass(obj, _SdkFeature) and obj is not _SdkFeature and obj is not Feature):
            continue

        # Skip ProxyFeature - it's only created programmatically for isolated-runtime features
        if name == "ProxyFeature":
            continue

        # Check if defined in this module
        if obj.__module__ == module.__name__:
            return obj

        # Check if explicitly exported via __all__
        if name in module_all:
            return obj

    return None


def _discover_entrypoint_feature_candidates(
) -> Dict[str, _EntrypointFeatureCandidate]:
    """Load valid in-process entry points while retaining owner metadata."""

    candidates: Dict[str, _EntrypointFeatureCandidate] = {}
    try:
        from kestrel_sovereign.feature_registry import discover_installed_feature_runtimes

        isolated_classes = {
            name for name, runtime in discover_installed_feature_runtimes().items()
            if runtime.runtime == "isolated-venv"
        }
    except Exception:  # noqa: BLE001
        isolated_classes = set()

    try:
        eps = importlib.metadata.entry_points()
    except Exception as e:
        logger.warning(f"Failed to read entry_points: {e}")
        return candidates

    if hasattr(eps, "select"):
        feature_eps = list(eps.select(group=FEATURE_ENTRY_POINT_GROUP))
    else:
        feature_eps = list(eps.get(FEATURE_ENTRY_POINT_GROUP, []))

    _reject_duplicate_feature_entry_points(feature_eps)

    for ep in feature_eps:
        try:
            declared_class_name = _entrypoint_class_name(
                getattr(ep, "value", "") or "",
                ep.name,
            )
            if declared_class_name in isolated_classes:
                logger.info(
                    "Entry point feature %s uses isolated-venv runtime; "
                    "skipping in-process import",
                    declared_class_name,
                )
                continue
            cls = ep.load()
            if not (
                isinstance(cls, type)
                and issubclass(cls, _SdkFeature)
                and cls is not _SdkFeature
                and cls is not Feature
            ):
                logger.warning(
                    f"Entry point '{ep.name}' does not point to a Feature subclass, skipping"
                )
                continue
            dist = getattr(ep, "dist", None)
            distribution = str(
                getattr(dist, "name", None) or "unknown distribution"
            )
            candidates[cls.__name__] = _EntrypointFeatureCandidate(
                feature_class=cls,
                distribution=distribution,
                entry_point=str(getattr(ep, "value", "") or ep.name),
            )
            logger.info(
                "Discovered entry_point feature: %s from %s",
                cls.__name__,
                getattr(ep, "value", ep.name),
            )
        except Exception as e:
            logger.warning(f"Failed to load entry_point feature '{ep.name}': {e}")

    return candidates


def discover_entrypoint_feature_classes() -> Dict[str, Type[Feature]]:
    """
    Discover Feature classes from installed pip packages via entry_points.

    External feature packages register entry points in their pyproject.toml:

        [project.entry-points."kestrel_sovereign.features"]
        GreeterFeature = "kestrel_feature_greeter.feature:GreeterFeature"

    Returns:
        Dict mapping class name to Feature class.
    """
    return {
        name: candidate.feature_class
        for name, candidate in _discover_entrypoint_feature_candidates().items()
    }


def _normalized_distribution_name(name: str) -> str:
    """Apply Python distribution-name normalization for owner comparisons."""

    return re.sub(r"[-_.]+", "-", name).lower()


def _module_matches_prefix(module: str, prefix: str) -> bool:
    """Return whether ``module`` is exactly inside an authorized package."""

    return module == prefix or module.startswith(f"{prefix}.")


def _raise_feature_ambiguity(
    class_name: str,
    local_module: str,
    external_module: str,
    distribution: str,
    *,
    reason: str,
) -> None:
    raise FeatureDiscoveryAmbiguityError(
        f"Feature class '{class_name}' is provided by bundled module "
        f"'{local_module}' and external distribution '{distribution}' "
        f"(module '{external_module}'), but {reason}. Uninstall the conflicting "
        "external package or add a reviewed extracted-replacement migration "
        "to kestrel_sovereign/data/feature_registry.toml."
    )


def discover_feature_selections() -> Dict[str, FeatureDiscoverySelection]:
    """Resolve all importable Feature classes with explicit collision policy.

    Bundled/external collisions fail closed unless the bundled registry row
    explicitly names the external distribution and implementation-module
    prefix. The returned diagnostic records expose the selected module to
    operators and tests without requiring Feature instantiation.
    """

    selections: Dict[str, FeatureDiscoverySelection] = {}
    for module_path in discover_feature_modules():
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Failed to import feature module {module_path}: {e}")
            continue
        feature_class = find_feature_class(module)
        if feature_class is None:
            continue
        class_name = feature_class.__name__
        selections[class_name] = FeatureDiscoverySelection(
            class_name=class_name,
            feature_class=feature_class,
            source="bundled",
            implementation_module=feature_class.__module__,
            discovery_location=module_path,
            distribution="kestrel-sovereign",
        )

    external_candidates = _discover_entrypoint_feature_candidates()
    from kestrel_sovereign.feature_registry import (
        discover_installed_feature_runtimes,
        get_extracted_feature_replacements,
    )

    replacement_rules = get_extracted_feature_replacements()
    try:
        external_runtimes = discover_installed_feature_runtimes()
    except Exception:  # noqa: BLE001
        external_runtimes = {}

    # Metadata still constitutes an ownership claim when the implementation is
    # isolated, broken, or invalid. An authorized isolated migration suppresses
    # the bundled class here and is instantiated as ProxyFeature in phase 2;
    # every other non-importable collision fails rather than falling back.
    for class_name in set(selections) & set(external_runtimes):
        if class_name in external_candidates:
            continue
        runtime = external_runtimes[class_name]
        external_module = runtime.entry_point.split(":", 1)[0]
        rule = replacement_rules.get(class_name)
        if getattr(runtime, "runtime", None) == "isolated-venv" and rule is not None:
            if (
                _normalized_distribution_name(runtime.distribution)
                != _normalized_distribution_name(rule.extracted_distribution)
            ):
                _raise_feature_ambiguity(
                    class_name,
                    selections[class_name].implementation_module,
                    external_module,
                    runtime.distribution,
                    reason=(
                        "the registry authorizes only distribution "
                        f"'{rule.extracted_distribution}'"
                    ),
                )
            if not _module_matches_prefix(external_module, rule.module_prefix):
                _raise_feature_ambiguity(
                    class_name,
                    selections[class_name].implementation_module,
                    external_module,
                    runtime.distribution,
                    reason=(
                        "the implementation module is outside the registered "
                        f"prefix '{rule.module_prefix}'"
                    ),
                )
            del selections[class_name]
            continue
        _raise_feature_ambiguity(
            class_name,
            selections[class_name].implementation_module,
            external_module,
            runtime.distribution,
            reason=(
                "the external claim did not provide a valid importable "
                "in-process replacement"
            ),
        )

    for class_name, candidate in external_candidates.items():
        local = selections.get(class_name)
        if local is None:
            selections[class_name] = FeatureDiscoverySelection(
                class_name=class_name,
                feature_class=candidate.feature_class,
                source="entry-point",
                implementation_module=candidate.feature_class.__module__,
                discovery_location=candidate.entry_point,
                distribution=candidate.distribution,
            )
            continue

        rule = replacement_rules.get(class_name)
        if rule is None:
            _raise_feature_ambiguity(
                class_name,
                local.implementation_module,
                candidate.feature_class.__module__,
                candidate.distribution,
                reason="no extracted-over-bundled migration is registered",
            )
        if (
            _normalized_distribution_name(candidate.distribution)
            != _normalized_distribution_name(rule.extracted_distribution)
        ):
            _raise_feature_ambiguity(
                class_name,
                local.implementation_module,
                candidate.feature_class.__module__,
                candidate.distribution,
                reason=(
                    "the registry authorizes only distribution "
                    f"'{rule.extracted_distribution}'"
                ),
            )
        if not _module_matches_prefix(
            candidate.feature_class.__module__, rule.module_prefix
        ):
            _raise_feature_ambiguity(
                class_name,
                local.implementation_module,
                candidate.feature_class.__module__,
                candidate.distribution,
                reason=(
                    "the implementation module is outside the registered "
                    f"prefix '{rule.module_prefix}'"
                ),
            )

        selections[class_name] = FeatureDiscoverySelection(
            class_name=class_name,
            feature_class=candidate.feature_class,
            source="entry-point",
            implementation_module=candidate.feature_class.__module__,
            discovery_location=candidate.entry_point,
            distribution=candidate.distribution,
        )

    return selections


def discover_local_feature_class_names() -> Set[str]:
    """Class names of the in-tree (bundled) Feature subclasses.

    The authoritative "ships with core, needs no install" set used by host
    reconcile (issue #1788) to distinguish a bundled class that needs no
    provisioning from an allowlist entry that names a feature the venv does not
    actually provide (the motivating silent-no-load bug).
    """
    names: Set[str] = set()
    for module_path in discover_feature_modules():
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        feature_class = find_feature_class(module)
        if feature_class is not None:
            names.add(feature_class.__name__)
    return names


def _entrypoint_class_name(ep_value: str, ep_name: str) -> str:
    """The Feature CLASS name an entry point resolves to.

    Allowlists and the agent loader key features by ``cls.__name__``, but an
    entry point may be registered under an alias
    (``github = "kestrel_feature_github.feature:GitHubFeature"``). The class
    name is the attribute after ``:`` (its innermost segment), which equals the
    loaded ``cls.__name__``; fall back to the entry-point name when the value
    has no attribute part. (issue #1788, codex round 5)
    """
    if ep_value and ":" in ep_value:
        attr = ep_value.split(":", 1)[1].strip()
        if attr:
            return attr.split(".")[-1]
    return ep_name


def discover_entrypoint_feature_dists() -> Dict[str, str]:
    """Map each entry-point Feature CLASS name to its owning distribution.

    Lightweight: reads entry-point *metadata* (``ep.value`` / ``ep.dist.name``)
    without importing the feature modules. The class name is derived from
    ``ep.value`` so it matches the allowlist namespace even when the entry
    point uses an alias (see :func:`_entrypoint_class_name`). This is the live,
    authoritative class → package map for installed external feature packages,
    used by host reconcile (issue #1788) to resolve allowlist classes to the
    packages that must be updated.
    """
    dist_by_class: Dict[str, str] = {}
    try:
        eps = importlib.metadata.entry_points()
    except Exception:  # noqa: BLE001
        return dist_by_class

    if hasattr(eps, "select"):
        feature_eps = list(eps.select(group=FEATURE_ENTRY_POINT_GROUP))
    else:
        feature_eps = list(eps.get(FEATURE_ENTRY_POINT_GROUP, []))

    _reject_duplicate_feature_entry_points(feature_eps)

    for ep in feature_eps:
        dist = getattr(ep, "dist", None)
        if dist is None:
            continue
        class_name = _entrypoint_class_name(getattr(ep, "value", "") or "", ep.name)
        if class_name:
            dist_by_class[class_name] = dist.name
    return dist_by_class


def discover_feature_class_by_name(name: str) -> Optional[Type[Feature]]:
    """Resolve a discoverable feature class by class name, module name, or shorthand.

    Uses the same explicit bundled/external collision resolution as
    :func:`discover_features`.
    """

    target = _normalize_feature_lookup(name)
    if not target:
        return None

    for selection in discover_feature_selections().values():
        feature_class = selection.feature_class
        # Bundled discovery may intentionally happen through a public
        # re-export module whose class is implemented by a nested module. That
        # public module owns the historic shorthand. External entry points cannot
        # trust their entry-point name/value as an alias, so they continue to
        # derive aliases from the loaded implementation module.
        alias_module = (
            selection.discovery_location
            if selection.source == "bundled"
            else selection.implementation_module
        )
        aliases = _feature_lookup_aliases(
            feature_class,
            alias_module,
        )
        if target in aliases:
            return feature_class

    return None


def resolve_feature_canonical_name(name: str) -> Optional[str]:
    """Resolve a feature name (class name / module / shorthand) to the canonical
    class name the loader (``discover_features``) filters an ``allowed_features``
    set by, or ``None`` if no discoverable feature matches.

    Unlike :func:`discover_feature_class_by_name`, this ALSO resolves
    isolated-venv feature packages, whose class is not importable in-process —
    ``discover_features`` loads them as ``ProxyFeature`` and filters them by the
    exact ``runtime.class_name``. Callers validating a requested feature set
    (e.g. spawn's mandate allowlist) must use this so an installed isolated
    feature is not wrongly rejected as unknown.
    """
    target = _normalize_feature_lookup(name)
    if not target:
        return None

    # In-process (local + entry-point) classes — alias-aware.
    feature_class = discover_feature_class_by_name(name)
    if feature_class is not None:
        return feature_class.__name__

    # Isolated-venv runtimes: match against class-name-derived aliases. The
    # loader keys these by the exact ``class_name`` string, so that is the
    # canonical form to return.
    try:
        from kestrel_sovereign.feature_registry import discover_installed_feature_runtimes

        for class_name, runtime in discover_installed_feature_runtimes().items():
            if getattr(runtime, "runtime", None) != "isolated-venv":
                continue
            aliases = {
                _normalize_feature_lookup(class_name),
                _normalize_feature_lookup(class_name.removesuffix("Feature")),
            }
            if target in {a for a in aliases if a}:
                return class_name
    except Exception as e:  # discovery is best-effort — never block a spawn on it
        logger.warning("Failed to inspect isolated feature runtimes: %s", e)

    return None


def discover_features(agent, allowed_features: Optional[Set[str]] = None) -> List[Feature]:
    """
    Discover and instantiate Feature classes from local directory and entry_points.

    Bundled and entry-point classes share one explicit resolution pass. A
    collision fails unless the static registry authorizes that exact external
    distribution and implementation-module prefix as a migration replacement.

    Args:
        agent: The KestrelAgent instance to pass to feature constructors
        allowed_features: If provided, only load features whose class name
            is in this set. Mandatory features (from MANDATORY_FEATURES) are
            always loaded regardless. If None, all features are loaded
            (backward-compatible default).

    Returns:
        List of instantiated Feature objects
    """
    from kestrel_sovereign.multi_agent.config import (
        MANDATORY_FEATURE_MODULES,
        MANDATORY_FEATURES,
    )

    disabled = get_disabled_features()
    disabled_mandatory = sorted(disabled & MANDATORY_FEATURES)
    if disabled_mandatory:
        raise MandatoryFeatureReadinessError(
            disabled_mandatory[0],
            "configuration",
            "is explicitly disabled",
        )
    selected_implementations = discover_feature_selections()
    features = []
    discovered_names = set()

    def _feature_allowed(class_name: str) -> bool:
        if class_name in disabled:
            logger.info(f"Feature '{class_name}' is disabled via {DISABLED_FEATURES_ENV}")
            return False

        if allowed_features is not None:
            if class_name not in allowed_features and class_name not in MANDATORY_FEATURES:
                logger.debug(f"Feature '{class_name}' not in agent's allowed profile, skipping")
                return False

        if class_name in discovered_names:
            logger.debug(f"Skipping duplicate feature: {class_name}")
            return False

        return True

    def _try_add(
        feature_class: Type[Feature],
        source: str,
        *,
        mandatory: bool = False,
    ) -> None:
        """Attempt to instantiate and add a feature class."""
        class_name = feature_class.__name__

        if not _feature_allowed(class_name):
            return

        # External (SDK-base) features lack the runtime-coupled subagent-dispatch
        # methods that live on the sovereign Feature base, so the orchestrator
        # would silently skip them. Inject the dispatch cluster so in-tree and
        # external features are dispatchable identically. No-op for in-tree.
        try:
            feature_class = ensure_subagent_dispatch(feature_class)
            feature = feature_class(agent)
        except Exception as exc:
            if mandatory:
                raise MandatoryFeatureReadinessError(
                    class_name,
                    "construction",
                    "could not be constructed",
                ) from exc
            raise
        features.append(feature)
        discovered_names.add(class_name)
        logger.info(f"Discovered feature: {class_name} from {source}")

    # Phase 0: import the sovereignty foundation explicitly and fail closed.
    # Generic directory discovery is intentionally best-effort for optional
    # features and therefore cannot own this invariant: it logs import errors
    # and omits those modules before the main loop ever sees them.
    for expected_name, module_path in MANDATORY_FEATURE_MODULES.items():
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            raise MandatoryFeatureReadinessError(
                expected_name,
                "import",
                "could not be imported",
            ) from exc

        try:
            feature_class = find_feature_class(module)
        except Exception as exc:
            raise MandatoryFeatureReadinessError(
                expected_name,
                "discovery",
                "could not be inspected",
            ) from exc
        if feature_class is None or feature_class.__name__ != expected_name:
            raise MandatoryFeatureReadinessError(
                expected_name,
                "discovery",
                "does not export its canonical class",
            )
        _try_add(feature_class, f"mandatory:{module_path}", mandatory=True)

    # Phase 1: Resolved in-process features. Mandatory classes appear in this
    # mapping too, but ``discovered_names`` guarantees their strict Phase 0
    # instances appear exactly once. An authorized extracted replacement for a
    # mandatory class is intentionally not supported by the current registry.
    for selection in selected_implementations.values():
        try:
            _try_add(
                selection.feature_class,
                f"{selection.source}:{selection.implementation_module}",
            )
        except ImportError as e:
            logger.warning(
                "Failed to import selected feature %s: %s",
                selection.class_name,
                e,
            )
        except Exception as e:
            logger.error(
                "Error loading selected feature %s: %s",
                selection.class_name,
                e,
            )

    # Phase 2: Entry-point features (external packages). Keep the isolated
    # runtime itself optional: a core-only profile, or a profile that filters
    # out every isolated feature, must never import its SDK-coupled module.
    try:
        from kestrel_sovereign.feature_registry import discover_installed_feature_runtimes

        installed_runtimes = discover_installed_feature_runtimes()
    except Exception as e:
        logger.warning("Failed to inspect entry_point feature runtime metadata: %s", e)
        installed_runtimes = {}

    for class_name, runtime in installed_runtimes.items():
        if runtime.runtime != "isolated-venv" or not _feature_allowed(class_name):
            continue
        try:
            isolated_runtime = importlib.import_module(
                "kestrel_sovereign.features.isolated_runtime"
            )
        except Exception as e:
            exception_type = _safe_exception_type_name(e)
            logger.error(
                "Error loading isolated entry_point feature %s: the Core "
                "isolated-runtime module could not be imported; verify the "
                "installed Core and SDK dependencies (exception type: %s)",
                class_name,
                exception_type,
                exc_info=_sanitized_isolated_runtime_import_exc_info(e),
            )
            recorder = getattr(agent, "record_feature_unavailable", None)
            if callable(recorder):
                recorder(
                    feature=None,
                    feature_name=class_name,
                    reason="the optional isolated runtime could not be imported",
                )
            continue

        try:
            # Resolve scope during discovery. A selected hosted feature with an
            # unsafe/missing namespace is a tenant-boundary violation and must
            # still escape this optional-feature containment path.
            feature = isolated_runtime.ProxyFeature(agent, runtime)
        except isolated_runtime.IsolatedRuntimeNamespaceError:
            raise
        except Exception as e:
            safe_exc_info = None
            unexpected_type = None
            if isinstance(e, isolated_runtime.IsolatedRuntimeConfigurationError):
                reason = (
                    isolated_runtime.IsolatedRuntimeConfigurationError.safe_diagnostic(
                        e
                    )
                )
            elif isinstance(e, isolated_runtime.IsolatedRuntimePreparationError):
                reason = isolated_runtime.safe_isolated_runtime_preparation_diagnostic(
                    e
                )
                safe_exc_info = (
                    isolated_runtime.sanitized_isolated_runtime_preparation_exc_info(
                        e
                    )
                )
            else:
                reason = "the isolated feature could not be prepared for discovery"
                unexpected_type = (
                    isolated_runtime.safe_isolated_runtime_exception_type_name(e)
                )
                safe_exc_info = (
                    isolated_runtime.sanitized_isolated_runtime_preparation_exc_info(
                        e
                    )
                )
            logger.error(
                "Error loading isolated entry_point feature %s: %s%s",
                class_name,
                reason,
                (
                    f" (unexpected exception type: {unexpected_type})"
                    if unexpected_type is not None
                    else ""
                ),
                exc_info=safe_exc_info,
            )
            recorder = getattr(agent, "record_feature_unavailable", None)
            if callable(recorder):
                recorder(
                    feature=None,
                    feature_name=class_name,
                    reason=reason,
                )
            continue

        features.append(feature)
        discovered_names.add(class_name)
        logger.info(
            "Discovered isolated feature: %s from entry_point:%s",
            class_name,
            runtime.entry_point,
        )

    verify_mandatory_feature_set(features, stage="discovery")
    return features


def get_feature_by_name(features: List[Feature], name: str) -> Optional[Feature]:
    """
    Get a feature by its class name.
    
    Args:
        features: List of Feature instances
        name: The class name to search for
        
    Returns:
        The matching Feature or None
    """
    for feature in features:
        if feature.name == name or feature.__class__.__name__ == name:
            return feature
    return None


def _normalize_feature_lookup(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _feature_lookup_aliases(feature_class: Type[Feature], module_path: str) -> Set[str]:
    class_name = feature_class.__name__
    aliases = {
        _normalize_feature_lookup(class_name),
        _normalize_feature_lookup(class_name.removesuffix("Feature")),
    }
    module_name = module_path.split(".")[-2] if module_path.endswith(".feature") else module_path.split(".")[-1]
    aliases.add(_normalize_feature_lookup(module_name))
    aliases.add(_normalize_feature_lookup(module_name.removesuffix("_feature")))
    return {alias for alias in aliases if alias}
