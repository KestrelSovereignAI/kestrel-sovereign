"""Contracts for StrategicMemory async offloading boundaries."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature


@pytest.mark.asyncio
async def test_github_api_offloads_urlopen_via_to_thread():
    feature = StrategicMemoryFeature(agent=MagicMock())

    real_to_thread = asyncio.to_thread
    calls = []

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    with patch(
        "kestrel_sovereign.features.strategic_memory.urllib.request.urlopen",
        return_value=MagicMock(read=lambda: json.dumps({"ok": True}).encode("utf-8")),
    ), patch(
        "kestrel_sovereign.features.strategic_memory.asyncio.to_thread",
        side_effect=tracking_to_thread,
    ):
        result = await feature._github_api("/repos/test/repo/issues", "token")

    assert result == {"ok": True}
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_github_post_offloads_urlopen_via_to_thread():
    feature = StrategicMemoryFeature(agent=MagicMock())

    real_to_thread = asyncio.to_thread
    calls = []

    async def tracking_to_thread(func, *args, **kwargs):
        calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    with patch(
        "kestrel_sovereign.features.strategic_memory.urllib.request.urlopen",
        return_value=MagicMock(read=lambda: json.dumps({"created": True}).encode("utf-8")),
    ), patch(
        "kestrel_sovereign.features.strategic_memory.asyncio.to_thread",
        side_effect=tracking_to_thread,
    ):
        result = await feature._github_api_post("/repos/test/repo/issues", "token", {"title": "hello"})

    assert result == {"created": True}
    assert len(calls) == 1
