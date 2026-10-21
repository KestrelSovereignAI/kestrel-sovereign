"""Notice audit failure cannot replace an actual native COMMIT receipt."""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.operator_signals import (
    OperatorSignalProducer,
    OperatorTurnInjectionResult,
)
from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    execution_commit_outcome,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.integration.test_execution_custody_review56_pg import lose_native_ack
from tests.unit.test_execution_custody_review43_native import carries


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["turn", "stream"])
async def test_original_native_unknown_survives_guarded_notice_settlement(
    native_pg, tmp_path, monkeypatch, surface
):
    backend, observer, schema = native_pg
    audit_backend = SQLiteBackend(str(tmp_path / "operator-original.db"))
    await audit_backend.connect()
    audit = AsyncDatabase(audit_backend)
    await audit._init_schema()
    producer = OperatorSignalProducer(
        SimpleNamespace(
            did="did:test:original-notice", _raw_storage=SimpleNamespace(db=audit)
        )
    )
    producer.enqueue_auto_mode("session")
    batch = await producer.collect_for_turn(
        session_id="native-original",
        llm_service=None,
        model_override=None,
        force_local_only=False,
        budget_summary=None,
        state_of_mind=None,
    )
    injection = OperatorTurnInjectionResult(batch, None)
    backend._execution_custody = ExecutionCustody(GenerationFence())
    fault = OSError("actual original native COMMIT response lost")
    receipts = lose_native_ack(monkeypatch, boundary="commit", original=fault)
    try:
        with pytest.raises(BaseException) as failed:
            await backend.execute(
                "INSERT INTO effects VALUES (103, 'original provider effect')"
            )
        primary = failed.value
        assert receipts == [True] and execution_commit_outcome(primary) == "unknown"
        assert (
            await observer.fetchval(
                f'SELECT COUNT(*) FROM "{schema}".effects WHERE id=103'
            )
            == 1
        )
        denied = ExecutionCustody(ProcessRuntimeExecutionFence("original audit owner"))
        denied.revoke("audit owner independently retired")
        audit_backend._execution_custody = denied
        with pytest.raises(BaseException) as propagated:
            if surface == "turn":
                await injection.settle_interrupted(primary)
            else:

                async def original_stream():
                    raise primary
                    yield  # pragma: no cover - original provider never yielded

                async for _ in injection.watch_stream(original_stream()):
                    raise AssertionError("failed provider must not yield output")
        assert carries(propagated.value, primary), (
            "notice settlement discarded the original explicit control owner"
        )
        assert execution_commit_outcome(propagated.value) == "unknown"
        assert producer._pending_auto_mode == ["session"]
    finally:
        backend._execution_custody = None
        audit_backend._execution_custody = None
        await audit.close()
