"""
Feature Discovery for Kestrel Agent.

Auto-discovers and registers Feature classes from the features/ directory.
"""

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import List, Type, Optional, Set

from kestrel_sovereign.features.base import Feature

logger = logging.getLogger(__name__)

# Features directory location
FEATURES_DIR = Path(__file__).parent

# Environment variable for disabling features
DISABLED_FEATURES_ENV = "KESTREL_DISABLED_FEATURES"


def get_disabled_features() -> Set[str]:
    """
    Get the set of disabled feature names from environment variable.
    
    The KESTREL_DISABLED_FEATURES env var should be a comma-separated list
    of feature class names to disable.
    
    Example: KESTREL_DISABLED_FEATURES=RunPodFeature,CreativeFeature
    """
    disabled = os.environ.get(DISABLED_FEATURES_ENV, "")
    if not disabled:
        return set()
    return {name.strip() for name in disabled.split(",") if name.strip()}


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


def discover_features(agent) -> List[Feature]:
    """
    Discover and instantiate all Feature classes.
    
    Args:
        agent: The KestrelAgent instance to pass to feature constructors
        
    Returns:
        List of instantiated Feature objects
    """
    disabled = get_disabled_features()
    features = []
    discovered_names = set()
    
    for module_path in discover_feature_modules():
        try:
            module = importlib.import_module(module_path)
            feature_class = find_feature_class(module)
            
            if feature_class is None:
                logger.debug(f"No Feature class found in {module_path}")
                continue
                
            class_name = feature_class.__name__
            
            # Check if disabled
            if class_name in disabled:
                logger.info(f"Feature '{class_name}' is disabled via {DISABLED_FEATURES_ENV}")
                continue
                
            # Avoid duplicates (same class from different import paths)
            if class_name in discovered_names:
                logger.debug(f"Skipping duplicate feature: {class_name}")
                continue
                
            # Instantiate the feature
            feature = feature_class(agent)
            features.append(feature)
            discovered_names.add(class_name)
            logger.info(f"Discovered feature: {class_name} from {module_path}")
            
        except ImportError as e:
            logger.warning(f"Failed to import feature module {module_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading feature from {module_path}: {e}")
    
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
