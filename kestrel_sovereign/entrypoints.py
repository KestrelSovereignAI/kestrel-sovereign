"""
Entry Point Discovery Utilities.

Shared helper for scanning Python entry_points across all Kestrel provider
registries (LLM, Voice, Channels, Storage, Search, Deploy).

External packages register entry points in their pyproject.toml::

    [project.entry-points."kestrel_sovereign.llm_providers"]
    MyProvider = "my_package.provider:MyProviderClass"

The ``discover_entry_point_classes`` function loads and validates these.
"""

import importlib.metadata
import logging
from typing import Dict, List, Tuple, Type

logger = logging.getLogger(__name__)


def discover_entry_point_classes(
    group: str,
    base_class: type,
) -> Dict[str, Type]:
    """
    Discover classes registered via entry_points for a given group.

    Loads each entry point, validates it is a subclass of ``base_class``,
    and returns a dict mapping entry point name to class.

    Args:
        group: Entry point group name (e.g. ``"kestrel_sovereign.llm_providers"``).
        base_class: Required base class/ABC that loaded classes must subclass.

    Returns:
        Dict mapping entry point name to the loaded class.
        Only successfully loaded, validated classes are included.
    """
    classes: Dict[str, Type] = {}

    try:
        eps = importlib.metadata.entry_points()
    except Exception as e:
        logger.warning("Failed to read entry_points: %s", e)
        return classes

    # Python 3.12+ returns SelectableGroups; 3.9-3.11 returns a dict
    if hasattr(eps, "select"):
        group_eps = eps.select(group=group)
    else:
        group_eps = eps.get(group, [])

    for ep in group_eps:
        try:
            cls = ep.load()
            if not (isinstance(cls, type) and issubclass(cls, base_class) and cls is not base_class):
                logger.warning(
                    "Entry point '%s' in group '%s' does not point to a %s subclass, skipping",
                    ep.name, group, base_class.__name__,
                )
                continue
            classes[ep.name] = cls
            logger.info(
                "Discovered entry_point %s: %s from %s",
                group, ep.name, ep.value,
            )
        except Exception as e:
            logger.warning("Failed to load entry_point '%s' from group '%s': %s", ep.name, group, e)

    return classes
