"""Canonical external pair atomicity on real SQLite and PostgreSQL connections."""

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from kestrel_sovereign.agent.invocation import invocation_scope, request_provenance
from kestrel_sovereign.auth import AuthMethod, CallerContext, caller_context_scope
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature
from kestrel_sovereign.features.privacy.feature import PrivacyAgent
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
from tests.integration.test_conversation_write_receipts import (
    receipt_storage as _receipt_storage,
)
from tests.utils.conversation_write_faults import fail_external_conversation_insert
from tests.utils.native_a2a import (
    native_a2a_backend_factory as _native_a2a_backend_factory,
)

receipt_storage = _receipt_storage
native_a2a_backend_factory = _native_a2a_backend_factory


@pytest_asyncio.fixture
async def external_feature(receipt_storage):
    raw = receipt_storage
    agent = SimpleNamespace(did=raw.agent_id, _raw_storage=raw,
        storage=PrivacyEnforcingStorage(raw), privacy_agent=PrivacyAgent(raw))
    feature = MemoryAgencyFeature(agent)
    await feature.initialize()
    return feature


@contextmanager
def original_external_invocation():
    caller = CallerContext.authenticated("mcp:" + "a" * 64, auth_method=AuthMethod.MCP_OAUTH)
    with caller_context_scope(caller), invocation_scope(uuid4().hex, provenance=request_provenance(
            actor=caller.identity, source_kind="http_request", source_locator="https://owned.example/mcp")):
        yield


class OwnedExchangeRollback(Exception):
    """The original transaction owner deliberately refuses its own commit."""


@pytest.mark.asyncio
@pytest.mark.parametrize("commit", [True, False])
async def test_external_pair_receipts_remain_provisional_inside_original_outer_transaction(
    external_feature, native_a2a_backend_factory, commit,
):
    raw = external_feature.agent._raw_storage
    observer = native_a2a_backend_factory()
    await observer.connect()
    try:
        try:
            async with raw.db.transaction():
                with original_external_invocation():
                    result = await external_feature.log_turn("Supplied user", "Supplied assistant", uuid4().hex)
                ids = (result.data["user_message_id"], result.data["assistant_message_id"])
                assert all(type(row_id) is int and row_id > 0 for row_id in ids) and ids[0] != ids[1]
                assert await observer.fetch_all(
                    "SELECT id FROM conversation_history WHERE id IN (?,?)", ids) == [], (
                    "External pair receipt committed its caller's original transaction")
                if not commit:
                    raise OwnedExchangeRollback()
        except TransactionError as error:
            assert not commit and any(isinstance(cause, OwnedExchangeRollback)
                for cause in execution_evidence_carriers(error))
        rows = await observer.fetch_all(
            "SELECT id,role,model,provider FROM conversation_history WHERE id IN (?,?) ORDER BY id", ids)
        if commit:
            assert rows == [(ids[0], "user", None, None), (ids[1], "external_assistant", None, None)]
        else:
            assert rows == [], "Original caller rollback retained an external exchange row"
    finally:
        await observer.close()


@pytest.mark.asyncio
async def test_standalone_external_pair_native_failure_never_commits_its_first_row(
    external_feature, native_a2a_backend_factory, monkeypatch,
):
    raw = external_feature.agent._raw_storage
    fault = fail_external_conversation_insert(raw.conversation, monkeypatch)
    with original_external_invocation(), pytest.raises((QueryError, TransactionError)):
        await external_feature.log_turn("Supplied user", "Supplied assistant", uuid4().hex)
    if raw.backend_type == "sqlite":
        native_error = sqlite3.IntegrityError
    elif raw.backend_type == "postgres":
        from asyncpg import UniqueViolationError
        native_error = UniqueViolationError
    else:
        raise AssertionError("Unsupported original native test backend")
    assert fault.user_row_ids and fault.native_errors
    assert any(isinstance(cause, native_error) for error in fault.native_errors
        for cause in execution_evidence_carriers(error))
    observer = native_a2a_backend_factory()
    await observer.connect()
    try:
        assert await observer.fetch_all("SELECT id FROM conversation_history WHERE agent_id=?", (raw.agent_id,)) == [], (
            "A standalone external pair committed its first row after a real native uniqueness failure")
    finally:
        await observer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_method,actor", [
    (AuthMethod.JWT, "mcp:" + "a" * 64),
    (AuthMethod.MCP_OAUTH, "a-different-verified-principal"),
])
async def test_canonical_external_exchange_refuses_foreign_caller_or_provenance(
    external_feature, auth_method, actor,
):
    caller = CallerContext.authenticated("mcp:" + "a" * 64, auth_method=auth_method)
    with caller_context_scope(caller), invocation_scope(uuid4().hex, provenance=request_provenance(
            actor=actor, source_kind="http_request", source_locator="https://owned.example/mcp")):
        result = await external_feature.log_turn("Supplied user", "Supplied assistant", uuid4().hex)
    assert result.status.value == "error", "A foreign canonical caller forged external exchange provenance"
    raw = external_feature.agent._raw_storage
    assert await raw.db.fetchone("SELECT COUNT(*) FROM conversation_history WHERE agent_id=?", (raw.agent_id,)) == (0,)
