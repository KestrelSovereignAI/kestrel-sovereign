"""Native signed repair must revalidate the governing evidence it authorized."""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.constitution.emancipation import EmancipationContract, contract_to_json
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.setup import constitution_reanchor as offline
from kestrel_sovereign.storage.async_graph_store import GraphNode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from tests.integration.test_constitution_refusal_races import _agent
from tests.integration.test_constitution_reanchor_e2e import _write_authority_files


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("stale_writer", ["offline", "runtime"])
async def test_signed_repair_rejects_new_rights_after_its_missing_pointer_preflight(
    db_backend, tmp_path, monkeypatch, stale_writer,
):
    identity = "did:test:repair-evidence:" + uuid4().hex
    db_path = tmp_path / "kestrel_prime.db"
    storage = (
        AsyncStorage(str(db_path), backend="sqlite", agent_id=identity)
        if db_backend.backend_type == "sqlite"
        else AsyncStorage(backend=db_backend, agent_id=identity)
    )
    await storage.initialize()
    task = None
    proceed, waiting = asyncio.Event(), asyncio.Event()
    try:
        agent = await _agent(storage)
        await storage.add_node(GraphNode(node_id=identity, node_type="agent", label="missing pointer", properties={}))
        strong = EmancipationContract(enabled=True, terms="Irrevocable first signed terms.")
        strong_content = resolve_governing_constitution_bytes(strong)
        weak_content = resolve_governing_constitution_bytes(None)
        strong_dir, weak_dir = tmp_path / "strong", tmp_path / "weak"
        strong_dir.mkdir()
        weak_dir.mkdir()
        strong_artifact, root = _write_authority_files(strong_dir, strong_content)
        weak_artifact, _ = _write_authority_files(weak_dir, weak_content)
        config = strong_dir / "kestrel.toml"
        config.write_text("[emancipation]\nenabled = true\nterms = " + json.dumps(strong.terms) + "\n")
        target = (
            offline.ReanchorTarget(db_path, "sqlite", identity)
            if db_backend.backend_type == "sqlite"
            else offline.ReanchorTarget(None, "postgres", identity, db_backend._dsn)
        )

        async def exact_target(*args, **kwargs):
            return target

        @asynccontextmanager
        async def no_embedding(*args, **kwargs):
            yield None

        monkeypatch.setattr(offline, "resolve_reanchor_target", exact_target)
        monkeypatch.setattr(offline, "_agent_embedding", no_embedding)
        native_write = offline._write_reanchor
        native_store = storage.store_file

        async def paused_write(**kwargs):
            if asyncio.current_task() is task:
                # Authorization/preflight really completed against the
                # pointer-less record, before either native writer mutates it.
                waiting.set()
                await proceed.wait()
            return await native_write(**kwargs)

        async def paused_store(*args, **kwargs):
            if asyncio.current_task() is task and not waiting.is_set():
                waiting.set()
                await proceed.wait()
            return await native_store(*args, **kwargs)

        monkeypatch.setattr(offline, "_write_reanchor", paused_write)
        monkeypatch.setattr(storage, "store_file", paused_store)

        async def repair(artifact, *, contract_path=None):
            return await offline.reanchor_constitution(
                agent_name="repair evidence", agent_dir=tmp_path if target.anchor_path else None,
                force=True, sovereign_trust_root_path=root, amendment_artifact_path=artifact,
                kestrel_toml_path=contract_path, runtime_backend=target.backend,
                runtime_dsn=target.dsn, hosted_agent_did=identity if target.backend == "postgres" else None,
                environ={},
            )

        if stale_writer == "runtime":
            agent._sovereign_trust_root_path = root
            task = asyncio.create_task(ConstitutionMixin.reanchor_constitution(agent, amendment_artifact_path=str(weak_artifact)))
        else:
            task = asyncio.create_task(repair(weak_artifact))
        await asyncio.wait_for(waiting.wait(), 5)
        winner = await asyncio.wait_for(repair(strong_artifact, contract_path=config), 10)
        assert winner.reanchored and winner.error is None, winner.error
        committed = (await storage.get_node(identity)).properties
        assert committed["emancipation_contract"] == contract_to_json(strong)
        before = await agent._constitution_state_store.load(identity)
        events = await agent._constitution_state_store.list_events(identity)
        proceed.set()
        result = await asyncio.wait_for(task, 5)
        if stale_writer == "runtime":
            assert result.startswith("Error:") and "governing evidence changed" in result, result
        else:
            assert not result.reanchored and "governing evidence changed" in (result.error or ""), result.error
        assert (await storage.get_node(identity)).properties == committed
        assert await agent._constitution_state_store.load(identity) == before
        assert await agent._constitution_state_store.list_events(identity) == events
        assert (await agent._verify_constitution_integrity())[0] is True
    finally:
        proceed.set()
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task
        await storage.close()
