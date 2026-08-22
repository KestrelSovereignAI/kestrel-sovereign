"""#2878 — boot must refuse (never fabricate a placeholder) when on-disk
identity material is present but the agent node is absent from the runtime
database.

When an agent's identity node is missing from the runtime database,
``KestrelAgent`` previously fabricated an ``Agent <did>`` placeholder and
booted. That placeholder cannot be told apart from a real identity by anything
downstream, so ``/health`` reports ok while the agent has no name, no
``bootstrap_state`` and zero constitution chunks.

The distinguishing signal needs no new state:

    on-disk identity material  |  node in runtime DB  |  meaning
    absent                     |  absent              |  genuinely new — fabricate
    present                    |  absent              |  inconsistent — REFUSE
    present                    |  present             |  normal boot

``self.identity is None`` is the guard for the first row, so a legitimately new
agent is unaffected.

Mutation trap (per the issue): a test asserting merely "an agent node exists
after boot" PASSES on the broken code, because the fabrication path creates
exactly such a node. These tests instead assert that boot REFUSES, or assert
the returned node's real label/properties, so a fabricated stub cannot satisfy
them.

Replicating the birth record into the runtime database is deliberately out of
scope here — that is #2871.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.identity.runtime_identity import IdentityReadinessError
from kestrel_sovereign.inception_service import (
    DID_WEB_DOMAIN_ENV,
    IDENTITY_METHOD_ENV,
    create_kestrel_identity_async,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage import GraphNode

TEST_DOMAIN = "agents.kestrel-sovereign.test"
TEST_DATA_KEY = "test-master-key-for-encryption-32chars!"


@pytest.fixture
def hybrid_env(monkeypatch):
    """Born-hybrid inception needs encrypted key storage + a did:web domain.
    Clear any PostgreSQL selection so a leaked env can't perturb SQLite tests."""
    monkeypatch.setenv("KESTREL_DATA_KEY", TEST_DATA_KEY)
    monkeypatch.setenv(DID_WEB_DOMAIN_ENV, TEST_DOMAIN)
    monkeypatch.delenv(IDENTITY_METHOD_ENV, raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)


def _empty_runtime_storage():
    """A runtime storage handle whose database holds no birth record."""
    storage = MagicMock()
    storage.get_node = AsyncMock(return_value=None)
    storage.add_node = AsyncMock()
    return storage


# ---------------------------------------------------------------------------
# Boot refuses to fabricate when the birth record is elsewhere
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_refuses_when_identity_present_but_node_absent(tmp_path, hybrid_env):
    """On-disk identity material + no agent node in the runtime database means
    inception wrote the birth record to a different database. Boot must refuse,
    not fabricate a placeholder."""
    creds = await create_kestrel_identity_async(
        str(tmp_path), None, agent_name="Custody Bird",
    )
    agent = KestrelAgent(
        did=creds.agent_did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
    )
    # The runtime just proved this DID by loading on-disk identity material.
    assert agent.identity is not None
    agent.storage = _empty_runtime_storage()

    with pytest.raises(IdentityReadinessError) as exc_info:
        await agent._ensure_agent_node_present()

    assert exc_info.value.failure == "birth_record"
    assert exc_info.value.error_code == "identity_birth_record"
    # A fabricated placeholder was never written.
    agent.storage.add_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_fabricates_node_for_genuinely_new_agent(tmp_path, hybrid_env):
    """No on-disk identity material means a genuinely new agent with no prior
    inception — creating its node here is correct and must NOT refuse."""
    did = f"did:web:{TEST_DOMAIN}:brand-new-agent"
    agent = KestrelAgent(
        did=did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
    )
    assert agent.identity is None
    agent.storage = _empty_runtime_storage()

    node = await agent._ensure_agent_node_present()

    agent.storage.add_node.assert_awaited_once()
    assert node.label == f"Agent {did}"
    assert node.properties["initialBalance"] == "100.0"


@pytest.mark.asyncio
async def test_boot_returns_existing_node_without_refusing(tmp_path, hybrid_env):
    """Normal boot: the birth record is present in the runtime database, so the
    node is returned untouched and nothing is fabricated."""
    did = f"did:web:{TEST_DOMAIN}:already-here"
    existing = GraphNode(
        node_id=did,
        node_type="agent",
        label="AlreadyHere",
        properties={"name": "AlreadyHere", "bootstrap_state": "pending"},
    )
    agent = KestrelAgent(
        did=did,
        storage_path=str(tmp_path / "kestrel_prime.db"),
    )
    storage = MagicMock()
    storage.get_node = AsyncMock(return_value=existing)
    storage.add_node = AsyncMock()
    agent.storage = storage

    node = await agent._ensure_agent_node_present()

    assert node is existing
    storage.add_node.assert_not_awaited()


def test_birth_record_refusal_message_is_public_safe(tmp_path, hybrid_env):
    """The refusal names the directory + backend in the LOG, but the public-safe
    IdentityReadinessError message must not leak the agent directory path."""
    async def _make_agent():
        creds = await create_kestrel_identity_async(
            str(tmp_path), None, agent_name="Quiet Bird",
        )
        return KestrelAgent(
            did=creds.agent_did,
            storage_path=str(tmp_path / "kestrel_prime.db"),
        )

    agent = asyncio.run(_make_agent())
    with pytest.raises(IdentityReadinessError) as exc_info:
        agent._refuse_if_birth_record_in_another_database()

    err = exc_info.value
    assert err.failure == "birth_record"
    assert err.cause_type == "BirthRecordDatabaseMismatch"
    assert str(tmp_path) not in str(err)
