"""Retention helpers for the soft-delete migration (#764).

The retention sweep itself runs on the existing per-agent cron scheduler
(``SchedulerFeature``) — see the ``trash_retention`` built-in there.
This module supplies just the supporting pieces: the config-resolver
that maps an agent's privacy mode to a retention window, and the small
constants the scheduler handler reaches for.

Operators tune the rail through ``[trash]`` in ``kestrel.toml``::

    [trash]
    conversation_history_days = 30  # default for any privacy mode

    [trash.privacy_overrides]
    isolated  = 7
    anonymous = 7
    normal    = 30
    public    = 30

EPHEMERAL is intentionally absent — that mode never persists. See #767
for the EPHEMERAL hard-purge defense-in-depth.

Resolution priority (most specific wins):

    1. ``[trash.privacy_overrides].<mode>`` if the mode matches
    2. ``[trash].conversation_history_days`` (global default)
    3. The compiled-in fallback (30 days)

A resolved value of zero or negative is treated as "skip this agent" —
purging on a same-day cutoff would scrub rows the user might still be
reaching for.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_RETENTION_DAYS = 30
DEFAULT_MAX_ROWS_PER_SWEEP = 10_000
# Default cron — 6h cadence matches the ticket spec. Operators
# override per-agent via `!schedule add` after the seed runs.
DEFAULT_RETENTION_CRON = "0 */6 * * *"

# Privacy modes that the override table understands. EPHEMERAL is
# excluded — that mode has its own hard-purge contract (#767) and is
# never expected to have rows in Trash.
SUPPORTED_PRIVACY_MODES = ("isolated", "anonymous", "normal", "public")


def load_trash_config() -> Dict[str, Any]:
    """Read the merged ``[trash]`` section from ``kestrel.toml``.

    Lazily imports ``kestrel_sovereign.config`` to keep this module
    cheap to import in tests that don't need TOML parsing.
    """
    try:
        from kestrel_sovereign.config import load_section
        return load_section("trash") or {}
    except Exception as e:
        logger.debug("[retention] config load failed (using defaults): %s", e)
        return {}


def resolve_retention_days(
    *,
    config: Dict[str, Any],
    privacy_mode: Optional[str],
    fallback: int = DEFAULT_RETENTION_DAYS,
) -> Optional[int]:
    """Compute the retention window for an agent.

    Priority (most specific wins):

      1. ``[trash.privacy_overrides].<mode>`` if the mode matches
      2. ``[trash].conversation_history_days`` (global default)
      3. ``fallback`` (the compiled-in default)

    Returns ``None`` to signal "skip this agent" if the resolved value
    is zero or negative.
    """
    days: Optional[int] = None

    overrides = config.get("privacy_overrides") or {}
    if privacy_mode:
        mode_key = privacy_mode.lower()
        if mode_key in overrides:
            try:
                days = int(overrides[mode_key])
            except (TypeError, ValueError):
                logger.warning(
                    "[retention] override for '%s' is not an int: %r",
                    mode_key, overrides[mode_key],
                )
                days = None

    if days is None and "conversation_history_days" in config:
        try:
            days = int(config["conversation_history_days"])
        except (TypeError, ValueError):
            logger.warning(
                "[retention] conversation_history_days is not an int: %r",
                config["conversation_history_days"],
            )
            days = None

    if days is None:
        days = fallback

    if days <= 0:
        return None
    return days


def agent_privacy_mode(agent: Any) -> Optional[str]:
    """Resolve the agent's current privacy mode as a lowercase string.

    Tolerates the various shapes the codebase uses for privacy mode:
    ``PrivacyMode`` enum, ``PrivacyConfig`` dataclass, plain string,
    or missing entirely. Returns ``None`` when nothing fits — the
    handler then falls back to the global default.
    """
    mode = getattr(agent, "_privacy_mode", None)
    if mode is None:
        mode = getattr(agent, "privacy_mode", None)
    if mode is None:
        return None
    val = getattr(mode, "value", None)
    if isinstance(val, str):
        return val.lower()
    if isinstance(mode, str):
        return mode.lower()
    return None
