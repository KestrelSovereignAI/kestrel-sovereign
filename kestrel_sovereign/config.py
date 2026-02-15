"""
Configuration constants for the Kestrel project.
"""
import os
import toml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping of individual config files to their unified kestrel.toml section paths
_UNIFIED_CONFIG_MAPPING = {
    "llm_config.toml": "llm",
    "model_catalog.toml": "llm.catalog",
    "model_mandate.toml": "llm.mandate",
    "constitutional_profiles.toml": "constitution.profiles",
    "council_config.toml": "council",
}

def load_config(file_name: str, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a TOML configuration file from the project root.

    This function now supports unified configuration via kestrel.toml:
    - First tries to load from kestrel.toml using the mapped section
    - Falls back to individual config files for backward compatibility
    - Logs deprecation warning when individual files are used

    Args:
        file_name: The name of the configuration file (e.g., 'llm_config.toml').
        section: The specific section to load from the TOML file. If None, loads the whole file.

    Returns:
        A dictionary containing the configuration.
    """
    # Try unified config first
    unified_path = Path("kestrel.toml")
    if unified_path.exists() and file_name in _UNIFIED_CONFIG_MAPPING:
        try:
            with open(unified_path, 'r') as f:
                unified_data = toml.load(f)

            # Navigate to the mapped section in unified config
            section_path = _UNIFIED_CONFIG_MAPPING[file_name]
            config_data = unified_data

            # Handle nested sections (e.g., "llm.catalog" -> llm -> catalog)
            for key in section_path.split('.'):
                config_data = config_data.get(key, {})
                if not config_data:
                    break

            # If we found the config in unified file, return it
            if config_data:
                if section:
                    result = config_data.get(section, {})
                else:
                    result = config_data

                # Special handling for llm_config.toml: needs provider_priority at root
                if file_name == "llm_config.toml" and not section:
                    # Ensure provider_priority is at root level for backward compat
                    if "provider_priority" not in result and "provider_priority" in unified_data.get("llm", {}):
                        result = unified_data["llm"].copy()

                logger.debug(f"Loaded '{file_name}' from unified config (kestrel.toml)")
                return result
        except Exception as e:
            logger.warning(f"Error loading from unified config, falling back to individual file: {e}")

    # Fall back to individual config file (backward compatibility)
    config_path = Path(file_name)

    # Create the config file from the example if it doesn't exist
    if not config_path.exists():
        example_path = Path(f"{file_name}.example")
        if example_path.exists():
            logger.info(f"'{file_name}' not found. Copying from '{example_path}'.")
            config_path.write_text(example_path.read_text())
        else:
            logger.warning(f"'{file_name}' and '{example_path}' not found.")
            return {}

    # Log deprecation warning if using individual file when unified exists
    if unified_path.exists() and file_name in _UNIFIED_CONFIG_MAPPING:
        logger.warning(
            f"DEPRECATION: Loading from '{file_name}' directly. "
            f"Consider migrating to unified 'kestrel.toml' configuration. "
            f"Individual config files will be removed in a future version."
        )

    try:
        with open(config_path, 'r') as f:
            config_data = toml.load(f)
            if section:
                return config_data.get(section, {})
            return config_data
    except Exception as e:
        logger.error(f"Error loading configuration from '{file_name}': {e}")
        return {}

# --- Core Paths ---
# Trusted agent keys need to live in a writable location.
# - In the sovereign Docker image, code lives under /app (read-only for non-root), while data lives under /data.
# - In local dev, KESTREL_DB_PATH is often unset, so we keep the historical default under the repo.
_KESTREL_DB_PATH = os.environ.get("KESTREL_DB_PATH")
TRUSTED_AGENTS_DIR = (
    os.path.join(_KESTREL_DB_PATH, "trusted_agents")
    if _KESTREL_DB_PATH
    else os.path.join(os.path.dirname(__file__), "trusted_agents")
)
# The constitution is stored in the package's data directory for portability
_PACKAGE_DIR = os.path.dirname(__file__)
CONSTITUTION_PATH = os.path.join(_PACKAGE_DIR, 'data', 'KESTREL_CONSTITUTION.md')
DEFAULT_LLM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'llm_config.toml')

# --- Inception Service ---
# Ensure the directory for trusted agents' keys exists
os.makedirs(TRUSTED_AGENTS_DIR, exist_ok=True) 