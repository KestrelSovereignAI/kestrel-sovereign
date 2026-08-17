"""Canonical command policy for readiness and authority gates."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


@dataclass(frozen=True, slots=True)
class RecoveryCommandRule:
    """Readiness and authority policy for one non-cognitive command."""

    requires_sovereign: bool = False
    allowed_in_safe_mode: bool = False


# Keeping routing and authority in one immutable mapping prevents a one-shot
# iterable from producing different answers when separate gates inspect it.
RECOVERY_COMMAND_POLICY: Final[Mapping[str, RecoveryCommandRule]] = MappingProxyType(
    {
        "!verify-constitution": RecoveryCommandRule(allowed_in_safe_mode=True),
        "!status": RecoveryCommandRule(allowed_in_safe_mode=True),
        "!help": RecoveryCommandRule(allowed_in_safe_mode=True),
        "!safe-mode": RecoveryCommandRule(
            requires_sovereign=True,
            allowed_in_safe_mode=True,
        ),
        "!reanchor-constitution": RecoveryCommandRule(
            requires_sovereign=True,
            allowed_in_safe_mode=True,
        ),
        "!get-privacy-mode": RecoveryCommandRule(),
        "!privacy-status": RecoveryCommandRule(),
        "!bootstrap-status": RecoveryCommandRule(),
    }
)

RECOVERY_COMMANDS: Final[frozenset[str]] = frozenset(RECOVERY_COMMAND_POLICY)
SAFE_MODE_COMMANDS: Final[frozenset[str]] = frozenset(
    command
    for command, rule in RECOVERY_COMMAND_POLICY.items()
    if rule.allowed_in_safe_mode
)
SOVEREIGN_COMMANDS: Final[frozenset[str]] = frozenset(
    command
    for command, rule in RECOVERY_COMMAND_POLICY.items()
    if rule.requires_sovereign
)

BOOTSTRAP_CONTROL_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "!skip-discovery",
        "!restart-discovery",
    }
)
BOOTSTRAP_ALLOWED_COMMANDS: Final[frozenset[str]] = (
    BOOTSTRAP_CONTROL_COMMANDS | RECOVERY_COMMANDS
)

GENESIS_AUDIT_BYPASS_COMMANDS: Final[frozenset[str]] = RECOVERY_COMMANDS


def prefixed_command_token(user_input: str) -> str | None:
    """Return a normalized command token only for exclamation-prefixed input."""
    if not user_input.startswith("!"):
        return None
    return user_input.split(maxsplit=1)[0].lower()


def requires_sovereign_authority(command: str) -> bool:
    """Return whether a canonical recovery command is sovereign-only."""
    rule = RECOVERY_COMMAND_POLICY.get(command)
    return rule is not None and rule.requires_sovereign
