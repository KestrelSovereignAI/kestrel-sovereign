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
from pathlib import Path
from typing import Dict, List, Type, Optional, Set

from kestrel_sovereign.features.base import Feature

logger = logging.getLogger(__name__)

# Features directory location
FEATURES_DIR = Path(__file__).parent

# Environment variable for disabling features
DISABLED_FEATURES_ENV = "KESTREL_DISABLED_FEATURES"

# Entry point group for external feature packages
FEATURE_ENTRY_POINT_GROUP = "kestrel_sovereign.features"


def get_disabled_features() -> Set[str]:
    """
    Get the set of disabled feature names from environment and kestrel.toml.

    Sources (merged):
    1. KESTREL_DISABLED_FEATURES env var (comma-separated class names)
    2. [features].disabled list in kestrel.toml (project-level config)

    Example env: KESTREL_DISABLED_FEATURES=RunPodFeature,CreativeFeature
    Example toml:
        [features]
        disabled = ["RunPodFeature", "VoiceFeature"]
    """
    disabled: Set[str] = set()

    # Source 1: environment variable
    env_val = os.environ.get(DISABLED_FEATURES_ENV, "")
    if env_val:
        disabled.update(name.strip() for name in env_val.split(",") if name.strip())

    # Source 2: kestrel.toml [features].disabled
    try:
        # Walk up from features/ to find project root (where kestrel.toml lives)
        project_dir = Path(__file__).parent.parent.parent.resolve()
        toml_path = project_dir / "kestrel.toml"
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

    The __all__ check allows features to be split across multiple files
    while still being discoverable (e.g., WalletFeature in wallet_feature.py
    imported into feature.py).
    """
    module_all = getattr(module, '__all__', [])

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if not (issubclass(obj, Feature) and obj is not Feature):
            continue

        # Check if defined in this module
        if obj.__module__ == module.__name__:
            return obj

        # Check if explicitly exported via __all__
        if name in module_all:
            return obj

    return None


def discover_entrypoint_feature_classes() -> Dict[str, Type[Feature]]:
    """
    Discover Feature classes from installed pip packages via entry_points.

    External feature packages register entry points in their pyproject.toml:

        [project.entry-points."kestrel_sovereign.features"]
        RunPodFeature = "kestrel_feature_cloud.runpod.feature:RunPodFeature"

    Returns:
        Dict mapping class name to Feature class.
    """
    classes: Dict[str, Type[Feature]] = {}

    try:
        eps = importlib.metadata.entry_points()
    except Exception as e:
        logger.warning(f"Failed to read entry_points: {e}")
        return classes

    # Python 3.12+ returns a SelectableGroups; 3.9-3.11 returns a dict
    if hasattr(eps, "select"):
        feature_eps = eps.select(group=FEATURE_ENTRY_POINT_GROUP)
    else:
        feature_eps = eps.get(FEATURE_ENTRY_POINT_GROUP, [])

    for ep in feature_eps:
        try:
            cls = ep.load()
            if not (isinstance(cls, type) and issubclass(cls, Feature) and cls is not Feature):
                logger.warning(
                    f"Entry point '{ep.name}' does not point to a Feature subclass, skipping"
                )
                continue
            classes[cls.__name__] = cls
            logger.info(f"Discovered entry_point feature: {cls.__name__} from {ep.value}")
        except Exception as e:
            logger.warning(f"Failed to load entry_point feature '{ep.name}': {e}")

    return classes


def discover_features(agent, allowed_features: Optional[Set[str]] = None) -> List[Feature]:
    """
    Discover and instantiate Feature classes from local directory and entry_points.

    Discovery order:
    1. Local features/ directory (core features) — these take priority
    2. Installed pip packages via entry_points (external feature packages)

    On duplicate class names, local wins and the entry_point is skipped.

    Args:
        agent: The KestrelAgent instance to pass to feature constructors
        allowed_features: If provided, only load features whose class name
            is in this set. Mandatory features (from MANDATORY_FEATURES) are
            always loaded regardless. If None, all features are loaded
            (backward-compatible default).

    Returns:
        List of instantiated Feature objects
    """
    from kestrel_sovereign.rookery.config import MANDATORY_FEATURES

    disabled = get_disabled_features()
    features = []
    discovered_names = set()

    def _try_add(feature_class: Type[Feature], source: str) -> None:
        """Attempt to instantiate and add a feature class."""
        class_name = feature_class.__name__

        if class_name in disabled:
            logger.info(f"Feature '{class_name}' is disabled via {DISABLED_FEATURES_ENV}")
            return

        if allowed_features is not None:
            if class_name not in allowed_features and class_name not in MANDATORY_FEATURES:
                logger.debug(f"Feature '{class_name}' not in agent's allowed profile, skipping")
                return

        if class_name in discovered_names:
            logger.debug(f"Skipping duplicate feature: {class_name} (from {source})")
            return

        feature = feature_class(agent)
        features.append(feature)
        discovered_names.add(class_name)
        logger.info(f"Discovered feature: {class_name} from {source}")

    # Phase 1: Local features (priority — these win on duplicates)
    for module_path in discover_feature_modules():
        try:
            module = importlib.import_module(module_path)
            feature_class = find_feature_class(module)
            if feature_class is None:
                logger.debug(f"No Feature class found in {module_path}")
                continue
            _try_add(feature_class, f"local:{module_path}")
        except ImportError as e:
            logger.warning(f"Failed to import feature module {module_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading feature from {module_path}: {e}")

    # Phase 2: Entry-point features (external packages)
    for class_name, feature_class in discover_entrypoint_feature_classes().items():
        try:
            _try_add(feature_class, f"entry_point:{feature_class.__module__}")
        except Exception as e:
            logger.error(f"Error loading entry_point feature {class_name}: {e}")

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
