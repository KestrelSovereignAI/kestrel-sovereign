"""Heartbeat bootstrap timeout checks (#378)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.bootstrap.service import BootstrapService
from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner


@dataclass
class _Node:
    node_id: str
    properties: dict


class _Storage:
    def __init__(self, node):
        self.node = node
        self.saved = None

    async def get_node(self, node_id):
        return self.node if node_id == self.node.node_id else None

    async def add_node(self, node):
        self.saved = node
        self.node = node


class _MetadataDB:
    def __init__(self):
        self.data = {}

    async def fetchall(self, query, params=None):
        key = (params[0], params[1]) if params and len(params) >= 2 else None
        if key in self.data:
            return [(self.data[key],)]
        return []

    async def execute(self, query, params=None):
        if params and len(params) >= 4:
            self.data[(params[0], params[1])] = params[2]


@pytest.mark.asyncio
async def test_heartbeat_alerts_on_stale_pending_bootstrap(tmp_path):
    agent_id = "did:test:stale-heartbeat"
    node = _Node(
        node_id=agent_id,
        properties={
            "bootstrap_state": "pending",
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        },
    )
    db = _MetadataDB()
    service = BootstrapService(
        db=db,
        agent_id=agent_id,
        agent_name="Stale",
        llm_service=object(),
        agent_data_path=tmp_path,
    )
    agent = SimpleNamespace(
        agent_id=agent_id,
        did=agent_id,
        _agent_name="Stale",
        bootstrap_service=service,
        storage=_Storage(node),
    )
    runner = HeartbeatRunner(agent, HeartbeatConfig())

    result = await runner._check_stale_bootstrap(
        datetime.now(timezone.utc).isoformat(),
        time.monotonic(),
    )

    assert result is not None
    assert result.status == "alert"
    assert result.reason == "stale_bootstrap"
    assert "stale_bootstrap" in result.message
    assert db.data[(agent_id, BootstrapService.BOOTSTRAP_STATUS_KEY)] == "stale_bootstrap"
