import asyncio
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.github_app.feature import GitHubAppFeature


@pytest.mark.asyncio
async def test_webhook_event_tasks_are_removed_after_completion():
    feature = GitHubAppFeature(MagicMock())
    await feature.initialize()

    async def handle_event(event, payload):
        return None

    feature._safe_handle_event = handle_event

    feature._schedule_event_handling("issues", {"action": "opened"})
    tasks = set(feature._event_tasks)

    assert len(tasks) == 1

    await asyncio.gather(*tasks)
    await asyncio.sleep(0)

    assert feature._event_tasks == set()


@pytest.mark.asyncio
async def test_shutdown_cancels_in_flight_webhook_event_tasks():
    feature = GitHubAppFeature(MagicMock())
    await feature.initialize()
    started = asyncio.Event()

    async def handle_event(event, payload):
        started.set()
        await asyncio.Event().wait()

    feature._safe_handle_event = handle_event

    feature._schedule_event_handling("issues", {"action": "opened"})
    tasks = set(feature._event_tasks)
    await started.wait()

    await feature.shutdown()

    assert all(task.done() for task in tasks)
    assert feature._event_tasks == set()
