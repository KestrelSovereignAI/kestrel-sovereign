"""
Configuration constants for the Kestrel project.
"""
import os
import toml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_name: str, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a TOML configuration file from the project root.

    Args:
        file_name: The name of the configuration file (e.g., 'llm_config.toml').
        section: The specific section to load from the TOML file. If None, loads the whole file.

    Returns:
        A dictionary containing the configuration.
    """
    config_path = Path(file_name)
    
    # Create the config file from the example if it doesn't exist
    if not config_path.exists():
        example_path = Path(f"{file_name}.example")
        if example_path.exists():
            logging.info(f"'{file_name}' not found. Copying from '{example_path}'.")
            config_path.write_text(example_path.read_text())
        else:
            logging.warning(f"'{file_name}' and '{example_path}' not found.")
            return {}

    try:
        with open(config_path, 'r') as f:
            config_data = toml.load(f)
            if section:
                return config_data.get(section, {})
            return config_data
    except Exception as e:
        logging.error(f"Error loading configuration from '{file_name}': {e}")
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