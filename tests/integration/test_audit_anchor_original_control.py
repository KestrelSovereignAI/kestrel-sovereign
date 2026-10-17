"""Native audit anchoring boundaries, not Agent/LLM readiness acceptance."""

from dataclasses import dataclass, field
from uuid import uuid4

import asyncpg
import pytest
from cryptography.fernet import Fernet

from kestrel_sovereign.execution_custody import (
    ExecutionCustody,
    _explicit_control_chain,
    execution_commit_outcome,
    is_execution_control_error,
)
from kestrel_sovereign.features.audit_anchor.feature import AuditAnchorFeature
from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook
from kestrel_sovereign.features.security.permissions import PermissionStore
from kestrel_sovereign.storage import AsyncStorage
from tests.integration.test_execution_authority_postgres import GenerationFence
from tests.integration.test_execution_authority_postgres import native_pg as native_pg


@dataclass
class SecurityFeature:
    permission_store: PermissionStore


@dataclass
class NativeAuditInput:
    did: str
    storage: AsyncStorage
    _raw_storage: AsyncStorage
    features: dict = field(default_factory=dict)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [
        "initialize",
        "timestamp",
        "count",
        "enumerate",
        "file",
        "record",
        "graph",
        "auto",
        "notify",
    ],
)
async def test_native_audit_control_is_not_fallback_or_anchor_success(
    native_pg, tmp_path, monkeypatch, route
):
    backend, observer, schema = native_pg
    # A fresh disposable data key, never the user's keys/configuration.
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    did = "did:web:native-audit:" + uuid4().hex
    storage = AsyncStorage.from_backend(backend, agent_id=did)
    await storage.initialize()
    permissions = PermissionStore(str(tmp_path / "actual-permissions.db"))
    await permissions.initialize()
    for _ in range(50 if route in {"auto", "notify"} else 1):
        await permissions.log_decision(
            "NativeAuditInput", "native_audit", "tool_execution", "allowed"
        )
    runtime = NativeAuditInput(did, storage, storage)
    runtime.features["SecurityFeature"] = SecurityFeature(permissions)
    feature = AuditAnchorFeature(runtime)
    runtime.features["AuditAnchorFeature"] = feature
    if route != "initialize":
        await feature.initialize()
    original = OSError("actual audit native physical release acknowledgement lost")
    native_exit = asyncpg.pool.PoolAcquireContext.__aexit__
    selected = route not in {"file", "record", "graph"}
    faulted = []
    source_errors = []

    async def lose_actual_ack(checkout, typ, error, tb):
        result = await native_exit(checkout, typ, error, tb)
        if selected and not faulted and typ is None:
            faulted.append(checkout)
            raise original
        return result

    def track(name, operation):
        async def actual(*args, **kwargs):
            nonlocal selected
            if name == "execute" and args:
                query = args[0].upper()
                if route == "file" and ("INTO FILES" in query):
                    selected = True
                elif route == "record" and ("INTO AUDIT_ANCHORS" in query):
                    selected = True
                elif route == "graph" and ("INTO GRAPH_NODES" in query):
                    selected = True
            try:
                return await operation(*args, **kwargs)
            except BaseException as error:
                source_errors.append(error)
                raise

        return actual

    for name in ("execute", "fetch_one", "fetch_val", "fetch_all"):
        monkeypatch.setattr(backend, name, track(name, getattr(backend, name)))
    for name in ("store_file", "add_node"):
        monkeypatch.setattr(storage, name, track(name, getattr(storage, name)))
    backend._execution_custody = ExecutionCustody(GenerationFence())
    monkeypatch.setattr(asyncpg.pool.PoolAcquireContext, "__aexit__", lose_actual_ack)
    failure, result = None, None
    try:
        try:
            if route == "initialize":
                result = await feature.initialize()
            elif route == "timestamp":
                result = await feature._get_last_anchor_timestamp()
            elif route == "count":
                result = await feature._count_anchors()
            elif route == "enumerate":
                result = await feature._get_all_anchors()
            elif route == "auto":
                result = await feature.on_audit_complete(
                    {"is_valid": False, "message": "native audit input"}
                )
            elif route == "notify":
                result = await ResponseAuditHook(
                    runtime, mode="strict"
                )._notify_audit_anchor(3, "native audit input")
            else:
                result = await feature.anchor_audit()
        except BaseException as error:
            failure = error
        assert faulted and source_errors, (
            "test did not reach an actual guarded native failure"
        )
        assert any(
            is_execution_control_error(error)
            and original in tuple(_explicit_control_chain(error))
            for error in source_errors
        )
        assert failure is not None and original in tuple(
            _explicit_control_chain(failure)
        ), f"native audit control became an ordinary fallback/anchor result: {result!r}"
        assert execution_commit_outcome(failure) == "committed"
    finally:
        backend._execution_custody = None
        await storage.close()
