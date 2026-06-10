"""
Configuration constants for the Kestrel project.
"""
import os
import re
import toml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default CORS origins for the console/API. Override via the comma-separated
# KESTREL_CORS_ORIGINS env var. Shared by server.py and host.py so the
# allowlist and the wildcard guard live in exactly one place.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8888",
    "http://127.0.0.1:8888",
    "https://kestrelsovereignai.github.io",
]


def build_cors_origins() -> list:
    """Resolve the CORS ``allow_origins`` list from ``KESTREL_CORS_ORIGINS``.

    Rejects a wildcard at startup: the app always sends credentialed CORS
    (``allow_credentials=True``), and browsers forbid a wildcard origin on
    credentialed requests. So ``KESTREL_CORS_ORIGINS="*"`` does not "open"
    CORS — it silently breaks credentialed cross-origin requests while making
    the operator believe access was granted. Fail closed with an actionable
    message instead of mis-securing quietly.
    """
    raw = os.environ.get("KESTREL_CORS_ORIGINS", "")
    origins = (
        [o.strip() for o in raw.split(",") if o.strip()]
        if raw
        else list(_DEFAULT_CORS_ORIGINS)
    )
    if "*" in origins:
        raise RuntimeError(
            'KESTREL_CORS_ORIGINS="*" cannot be combined with credentialed CORS '
            "(the app sends allow_credentials=True). Browsers reject a wildcard "
            "origin on credentialed requests. Set an explicit comma-separated "
            "origin allowlist instead."
        )
    return origins

# Mapping of legacy standalone config files to their unified kestrel.toml
# section paths. Callers using load_config(file_name) get the unified path
# transparently when kestrel.toml is present.
#
# llm_config.toml was removed from this map in #940 — LLMService now reads
# the [llm] section directly via load_section("llm"). The migration tool
# kestrel migrate-llm-config (#939) folds legacy llm_config.toml files into
# kestrel.toml [llm] in one shot.
_UNIFIED_CONFIG_MAPPING = {
    "model_catalog.toml": "llm.catalog",
    "model_mandate.toml": "llm.mandate",
    # NOTE: constitutional_profiles.toml is intentionally NOT mapped here.
    # ConstitutionalProfileManager.load() (llm/constitutional_profile.py)
    # reads it directly via tomllib, never through load_config(), so a
    # mapping entry was dead. (That manager reimplements the unified-config
    # fallback instead of reusing this module — a separate consolidation.)
    "council_config.toml": "council",
}

def _project_root_for_config() -> Path:
    """Anchor for ``kestrel.toml`` lookups.

    Imported lazily so this module stays cheap to import (``paths`` is
    cheap, but the import dance keeps the cycle-risk surface small).
    """
    from kestrel_sovereign.paths import project_dir

    return project_dir()


def load_config(file_name: str, section: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a TOML configuration file from the project root.

    This function supports unified configuration via kestrel.toml:
    - First tries to load from kestrel.toml using the mapped section
    - Falls back to individual config files for backward compatibility
    - Logs deprecation warning when individual files are used

    For LLM config, prefer ``load_section("llm")`` directly. ``llm_config.toml``
    was removed from the unified-mapping in #940; use the migration command
    ``kestrel migrate-llm-config`` to fold a legacy file into kestrel.toml.

    Args:
        file_name: The name of the configuration file (e.g., 'model_mandate.toml').
        section: The specific section to load from the TOML file. If None, loads the whole file.

    Returns:
        A dictionary containing the configuration.
    """
    project_root = _project_root_for_config()
    # Try unified config first
    unified_path = project_root / "kestrel.toml"
    if unified_path.exists() and file_name in _UNIFIED_CONFIG_MAPPING:
        try:
            with open(unified_path, 'r', encoding='utf-8') as f:
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

                logger.debug(f"Loaded '{file_name}' from unified config (kestrel.toml)")
                return result
        except Exception as e:
            logger.warning(f"Error loading from unified config, falling back to individual file: {e}")

    # Fall back to individual config file (backward compatibility)
    config_path = project_root / file_name

    # Create the config file from the example if it doesn't exist
    if not config_path.exists():
        example_path = project_root / f"{file_name}.example"
        if example_path.exists():
            logger.info(f"'{file_name}' not found. Copying from '{example_path}'.")
            config_path.write_text(example_path.read_text(encoding='utf-8'), encoding='utf-8')
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
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = toml.load(f)
            if section:
                return config_data.get(section, {})
            return config_data
    except Exception as e:
        logger.error(f"Error loading configuration from '{file_name}': {e}")
        return {}

# --- Duration Parsing ---

def parse_duration(value: str) -> int:
    """Parse a human-readable duration string into seconds.

    Supports: "30s", "5m", "1h", "2h30m", "90m", "1h30m15s".
    Plain integers are treated as minutes for backward compatibility.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    value = value.strip()
    if not value:
        raise ValueError("Empty duration string")

    # Plain integer → minutes
    if value.isdigit():
        return int(value) * 60

    total = 0
    pattern = re.compile(r'(\d+)\s*([hms])', re.IGNORECASE)
    matches = pattern.findall(value)
    if not matches:
        raise ValueError(f"Cannot parse duration: '{value}'")

    for amount, unit in matches:
        unit = unit.lower()
        if unit == 'h':
            total += int(amount) * 3600
        elif unit == 'm':
            total += int(amount) * 60
        elif unit == 's':
            total += int(amount)

    return total


def load_section(section: str) -> Dict[str, Any]:
    """Load a top-level section from kestrel.toml.

    Args:
        section: Section name (e.g. 'heartbeat', 'bootstrap', 'agent').

    Returns:
        Dict of config values, or empty dict if section/file not found.
    """
    # Check agent-specific config first (KESTREL_DB_PATH/kestrel.toml)
    db_path = os.environ.get("KESTREL_DB_PATH")
    search_paths = []
    if db_path:
        search_paths.append(Path(db_path) / "kestrel.toml")
    search_paths.append(_project_root_for_config() / "kestrel.toml")

    for config_path in search_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
                result = data.get(section, {})
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Error reading {config_path}: {e}")

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

# --- Inception Service ---
# Ensure the directory for trusted agents' keys exists
os.makedirs(TRUSTED_AGENTS_DIR, exist_ok=True) 