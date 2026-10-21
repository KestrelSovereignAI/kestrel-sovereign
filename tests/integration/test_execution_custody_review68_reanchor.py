"""Reanchor inspections preserve errors from real guarded storage reads."""

import os
from hashlib import sha256

import pytest

from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.execution_custody import (
    bind_execution_custody, execution_commit_outcome, execution_denial_scopes,
    is_execution_control_error,
)
from kestrel_sovereign.setup import constitution_reanchor as reanchor
from kestrel_sovereign.storage import AsyncStorage, GraphNode
from tests.integration.test_constitution_reanchor_postgres import _make_local_anchor
from tests.integration.test_execution_authority_postgres import GenerationFence, native_pg as native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.mark.parametrize("boundary", ["initial", "retargeted", "bootstrap"])
@pytest.mark.parametrize("control_kind", ["authority", "unknown-commit"])
async def test_reanchor_inspection_preserves_original_native_control(
    native_pg, tmp_path, monkeypatch, boundary, control_kind,
):
    backend, _observer, schema = native_pg
    identity = "did:test:review68:" + schema
    dsn = os.environ["TEST_POSTGRES_URL"]
    dsn += ("&" if "?" in dsn else "?") + f"options=-csearch_path%3D{schema}"
    content = resolve_governing_constitution_bytes(None)
    digest = sha256(content).hexdigest()
    storage = AsyncStorage(backend=backend, agent_id=identity)
    await storage.initialize()
    try:
        if boundary != "retargeted":
            await storage.store_file(content, "KESTREL_CONSTITUTION.md")
            await storage.add_node(GraphNode(
                node_id=identity, node_type="agent", label="Native reanchor inspection",
                properties={"constitution_hash": digest},
            ))
            await storage.add_node(GraphNode(
                node_id=digest, node_type="document", label="Constitution",
                properties={"hash": digest},
            ))
            await storage.add_edge(identity, digest, "governed_by")
        else:
            _make_local_anchor(tmp_path, identity)

        original_read = reanchor._read_agent_anchor
        original_bootstrap = reanchor._initial_anchor_custody_pending
        original_errors, entered = [], []

        async def deny_native(operation, target):
            entered.append(target.backend)
            if control_kind == "authority":
                scope.revoke("native reanchor inspection admission revoked")
            else:
                scope.preserve_commit_uncertainty("unknown")
            try:
                return await operation(target)
            except Exception as error:
                original_errors.append(error)
                raise

        async def read_anchor(target):
            if boundary == "initial" or (boundary == "retargeted" and target.writes_to_anchor):
                return await deny_native(original_read, target)
            return await original_read(target)

        async def inspect_bootstrap(target):
            return await deny_native(original_bootstrap, target)

        monkeypatch.setattr(reanchor, "_read_agent_anchor", read_anchor)
        if boundary == "bootstrap":
            monkeypatch.setattr(reanchor, "_initial_anchor_custody_pending", inspect_bootstrap)
        with bind_execution_custody(GenerationFence()) as scope:
            with pytest.raises(Exception) as caught:
                await reanchor.reanchor_constitution(
                    agent_name="native-inspection", agent_dir=tmp_path if boundary == "retargeted" else None,
                    force=False, runtime_backend="postgres", runtime_dsn=dsn,
                    hosted_agent_did=None if boundary == "retargeted" else identity, environ={},
                )
            assert entered == ["sqlite" if boundary == "retargeted" else "postgres"]
            assert len(original_errors) == 1
            assert caught.value is original_errors[0], "inspection replaced the native evidence carrier"
            assert is_execution_control_error(caught.value)
            if control_kind == "authority":
                assert execution_denial_scopes(caught.value) == (scope,)
            assert execution_commit_outcome(caught.value) == ("unknown" if control_kind == "unknown-commit" else None)
    finally:
        await storage.close()
