"""Backward-compat shims for the Rookery -> MultiAgent rename.

The internal Python identifiers were hard-renamed; the operator-facing
surface (env var name, config filename, deployment_mode value) accepts
both forms with a one-time deprecation warning per process.

When the deprecation period ends, drop the OLD-form branches here and
delete this module — every caller already imports from this module, so
removing the compat is a single edit.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

NEW_CONFIG_ENV_VAR = "KESTREL_MULTI_AGENT_CONFIG"
LEGACY_CONFIG_ENV_VAR = "KESTREL_ROOKERY_CONFIG"

NEW_CONFIG_FILENAME = "multi_agent.toml"
LEGACY_CONFIG_FILENAME = "rookery.toml"

NEW_DEPLOYMENT_MODE = "multi_agent"
LEGACY_DEPLOYMENT_MODE = "rookery"

# Module-level latches so we warn once per process per surface, not per
# call site. Tests can clear them via reset_deprecation_latches().
_warned_env_var = False
_warned_filename = False
_warned_deployment_mode = False


def reset_deprecation_latches() -> None:
    """Test hook: clear the once-per-process warning latches."""
    global _warned_env_var, _warned_filename, _warned_deployment_mode
    _warned_env_var = False
    _warned_filename = False
    _warned_deployment_mode = False


def get_config_env_value(env: dict | os._Environ | None = None) -> str | None:
    """Return the value of the multi-agent config env var, or None.

    Prefers ``KESTREL_MULTI_AGENT_CONFIG``. Falls back to the legacy
    ``KESTREL_ROOKERY_CONFIG`` and emits a deprecation warning once.
    """
    if env is None:
        env = os.environ
    new = env.get(NEW_CONFIG_ENV_VAR)
    if new is not None:
        return new
    legacy = env.get(LEGACY_CONFIG_ENV_VAR)
    if legacy is not None:
        _warn_legacy_env_var()
        return legacy
    return None


def is_config_env_var_set(env: dict | os._Environ | None = None) -> bool:
    """True if either the new or legacy config env var is set."""
    if env is None:
        env = os.environ
    return NEW_CONFIG_ENV_VAR in env or LEGACY_CONFIG_ENV_VAR in env


def write_config_env_value(env: dict | os._Environ, value: str) -> None:
    """Set the config env var on a target env dict.

    Writes the NEW name (canonical). Also clears the legacy name to
    avoid stale fallback values surviving in subprocess environments
    (relevant for ProcessManager which spawns child agents).
    """
    env[NEW_CONFIG_ENV_VAR] = value
    if LEGACY_CONFIG_ENV_VAR in env:
        try:
            del env[LEGACY_CONFIG_ENV_VAR]
        except (KeyError, TypeError):
            pass


def find_existing_config(project_dir: Path) -> Path | None:
    """Return the path of an existing multi-agent config file, or None.

    Prefers ``multi_agent.toml``; falls back to ``rookery.toml`` (legacy)
    with a deprecation warning. Returns None if neither exists.
    """
    new_path = project_dir / NEW_CONFIG_FILENAME
    if new_path.exists():
        return new_path
    legacy_path = project_dir / LEGACY_CONFIG_FILENAME
    if legacy_path.exists():
        _warn_legacy_filename(legacy_path)
        return legacy_path
    return None


def normalize_deployment_mode(value: str | None) -> str:
    """Map a deployment_mode string to its canonical value.

    ``"rookery"`` is canonicalised to ``"multi_agent"`` with a one-time
    deprecation warning. Anything else passes through.
    """
    if value == LEGACY_DEPLOYMENT_MODE:
        _warn_legacy_deployment_mode()
        return NEW_DEPLOYMENT_MODE
    return value or "agent"


def _warn_legacy_env_var() -> None:
    global _warned_env_var
    if _warned_env_var:
        return
    _warned_env_var = True
    msg = (
        f"DEPRECATED: {LEGACY_CONFIG_ENV_VAR} env var is set; rename to "
        f"{NEW_CONFIG_ENV_VAR}. The legacy name will be removed in a "
        "future release."
    )
    logger.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def _warn_legacy_filename(path: Path) -> None:
    global _warned_filename
    if _warned_filename:
        return
    _warned_filename = True
    msg = (
        f"DEPRECATED: loading multi-agent config from legacy {path.name!r}; "
        f"rename to {NEW_CONFIG_FILENAME!r}. The legacy filename will be "
        "removed in a future release."
    )
    logger.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def _warn_legacy_deployment_mode() -> None:
    global _warned_deployment_mode
    if _warned_deployment_mode:
        return
    _warned_deployment_mode = True
    msg = (
        f"DEPRECATED: deployment_mode={LEGACY_DEPLOYMENT_MODE!r} is the "
        f"legacy name; use {NEW_DEPLOYMENT_MODE!r} instead. The legacy "
        "value will be removed in a future release."
    )
    logger.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
