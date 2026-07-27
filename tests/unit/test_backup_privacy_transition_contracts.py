"""Backup caller contracts for retryable privacy-transition refusals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.backup import BackupMixin
from kestrel_sovereign.storage.privacy_wrapper import (
    PRIVACY_TRANSITION_RETRY_MESSAGE,
)


@pytest.mark.asyncio
async def test_promote_backup_does_not_backup_after_transition_refusal():
    """A retryable mode conflict is non-applied and cannot claim a backup."""
    agent = BackupMixin()
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.privacy_config.uses_temp_storage.return_value = True
    agent.privacy_agent.save_isolated_session = AsyncMock(
        return_value="Isolated session saved."
    )
    agent.set_privacy_mode_with_effects = AsyncMock(
        side_effect=(
            SimpleNamespace(
                applied=False,
                retryable_conflict=True,
                message=PRIVACY_TRANSITION_RETRY_MESSAGE,
            ),
            SimpleNamespace(
                applied=True,
                retryable_conflict=False,
                message="Privacy mode changed to normal.",
            ),
        )
    )
    agent._command_backup = AsyncMock(return_value="Backup created.")

    refused = await agent._command_promote_backup(
        "!promote-backup --tier local"
    )
    assert refused == PRIVACY_TRANSITION_RETRY_MESSAGE
    agent._command_backup.assert_not_awaited()

    applied = await agent._command_promote_backup(
        "!promote-backup --tier local"
    )
    assert applied == "Backup created."
    agent._command_backup.assert_awaited_once_with(
        "!backup --tier local"
    )
