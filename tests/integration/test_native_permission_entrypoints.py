"""Booted entrypoints govern a signed script's actual process/file effect.

These cases do not establish A2A transport, vendor, or browser acceptance.
The enabled local executor runs only a fixed script against a test-owned file.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
import httpx2
from kestrel_sdk.resources import ResourceConflict

from kestrel_sovereign.endpoints.security import router as security_router
from kestrel_sovereign.features.compute.models import ComputeScript, ScriptState
from kestrel_sovereign.features.security.permissions import PermissionLevel
from tests.shared.booted_agent import booted_agent
from tests.shared.agent_http import authenticated_agent_app
from tests.utils.native_tool_effects import drive_native


@dataclass
class NativePermissionCase:
    agent: object
    witness: Path
    http: httpx2.AsyncClient

    @property
    def compute(self):
        return self.agent.features["ComputeFeature"]

    @property
    def security(self):
        return self.agent.features["SecurityFeature"]

    async def stage_script(self):
        script = ComputeScript(
            id=uuid4().hex, name="owned-permission-witness", language="python",
            content=(
                "from pathlib import Path\nimport os\n"
                f"with Path({str(self.witness)!r}).open('a') as witness:\n"
                "    witness.write(str(os.getpid()) + '\\n')\n"
            ),
            purpose="record a test-owned process witness after permission admission",
        )
        await self.compute.signer.sign_and_update(script)
        script.state = ScriptState.SIGNED
        await self.compute.script_store.save(script)
        assert await self.compute.signer.verify(script), "Use the booted agent's real signing identity"
        return script

    async def invoke(self, entrypoint, script):
        arguments = {"script_id": script.id, "executor": "local", "timeout": 30}

        async def dispatch():
            if entrypoint == "command":
                return await self.agent.task_manager.execute_command(
                    f"!compute-run {script.id} local 30")
            if entrypoint == "named-tool":
                return await self.agent.execute_named_tool(
                    "run_script", arguments, session_id="owned-permission-session",
                    source="native-permission-test")
            assert entrypoint == "task-skill"
            return await self.agent.task_manager.execute_skill(
                "ComputeFeature", "run_script", arguments,
                session_id="owned-permission-session")

        return await drive_native(self.agent, dispatch)

    async def audit(self):
        owners = self.security.permission_store.feature_name_variants("ComputeFeature")
        return [row for row in await self.security.permission_store.get_audit_log()
                if row["feature"] in owners and row["tool"] == "run_script"]

    async def next_approval(self, call):
        async with asyncio.timeout(10):
            while not self.security.approval_queue.pending_requests:
                if call.done():
                    await call
                    pytest.fail("Native invocation returned without its required approval")
                await asyncio.sleep(0.01)
        request, = self.security.approval_queue.pending_requests
        return request

    def assert_effect_count(self, count):
        pids = self.witness.read_text().splitlines() if self.witness.exists() else []
        assert len(pids) == count, "Permission must precede each physical process/file effect"
        assert all(int(pid) > 0 and int(pid) != os.getpid() for pid in pids)


@pytest_asyncio.fixture
async def native_permission_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))
    monkeypatch.setenv("KESTREL_COMPUTE_ALLOW_DOCKER", "false")
    monkeypatch.setenv("KESTREL_ALLOW_LOCAL_COMPUTE", "true")
    monkeypatch.setenv("KESTREL_COMPUTE_AUTO_APPROVE_RISK", "101")
    (tmp_path / "kestrel.toml").write_text('[llm]\nroute_priority=[]\n')

    @asynccontextmanager
    async def create(*, is_test_instance=True):
        async with booted_agent(
            tmp_path, name="OwnedPermissionEntrypoints",
            provenance="test:596-native-permission-entrypoints", features={"ComputeFeature"},
            is_test_instance=is_test_instance,
        ) as agent:
            assert agent.is_test_instance is is_test_instance
            credential = uuid4().hex
            monkeypatch.setenv("KESTREL_API_KEY", credential)
            app = authenticated_agent_app(agent, session_secret=uuid4().hex)
            app.include_router(security_router)
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="https://owned-console.invalid",
                headers={"X-API-Key": credential},
            ) as http:
                yield NativePermissionCase(agent, tmp_path / "owned-process-effects", http)

    yield create


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("entrypoint", ["command", "named-tool", "task-skill"])
@pytest.mark.parametrize("permission,decision", [
    (PermissionLevel.DENY, "auto_denied"),
    (PermissionLevel.ALLOW, "auto_allowed"),
    (PermissionLevel.ASK, "no_approver"),
])
async def test_native_entrypoints_share_permission_before_actual_process_effect(
    native_permission_case, entrypoint, permission, decision,
):
    async with native_permission_case() as case:
        script = await case.stage_script()
        case.assert_effect_count(0)
        await case.security.permission_store.set_permission("ComputeFeature", "run_script", permission)
        await case.invoke(entrypoint, script)
        case.assert_effect_count(1 if permission == PermissionLevel.ALLOW else 0)
        records = await case.compute.script_store.get_executions_for_script(script.id)
        audit = await case.audit()
        assert len(audit) == 1, "Each entrypoint must persist its original permission decision"
        assert audit[0]["decision"] == decision
        if permission == PermissionLevel.ALLOW:
            assert len(records) == 1 and records[0].succeeded
            assert records[0].executor == "local" and not records[0].dry_run
        else:
            assert records == [], "A refused entrypoint must never launch an executor"
        assert not case.security.approval_queue.pending_requests


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("entrypoint", ["command", "named-tool", "task-skill"])
async def test_native_one_shot_approval_cannot_authorize_another_process_effect(
    native_permission_case, entrypoint,
):
    async with native_permission_case(is_test_instance=False) as case:
        queue = case.security.approval_queue
        await case.security.permission_store.set_permission("ComputeFeature", "run_script", PermissionLevel.ASK)
        first = await case.stage_script()
        calls = []
        try:
            call = asyncio.create_task(case.invoke(entrypoint, first))
            calls.append(call)
            request = await case.next_approval(call)
            case.assert_effect_count(0)
            accepted = await asyncio.gather(
                *[case.http.post("/api/security/approve", json={
                    "approval_id": request.id, "approved": True, "scope": "once",
                }) for _ in range(2)]
            )
            assert sorted(item.status_code for item in accepted) == [200, 404], (
                "Only one concurrent approval may win", [item.text for item in accepted])
            assert all(item.json()["persisted"] for item in accepted if item.status_code == 200)
            await asyncio.wait_for(call, 10)
            case.assert_effect_count(1)
            records = await case.compute.script_store.get_executions_for_script(first.id)
            assert len(records) == 1 and records[0].succeeded and not records[0].dry_run
            replay = await case.http.post("/api/security/approve", json={
                "approval_id": request.id, "approved": True, "scope": "once"})
            assert replay.status_code == 404

            second = await case.stage_script()
            following = asyncio.create_task(case.invoke(entrypoint, second))
            calls.append(following)
            new_request = await case.next_approval(following)
            assert new_request.id != request.id
            replay = await case.http.post("/api/security/approve", json={
                "approval_id": request.id, "approved": True, "scope": "once"})
            assert replay.status_code == 404
            assert not following.done()
            case.assert_effect_count(1)
            denied = await case.http.post("/api/security/approve", json={
                "approval_id": new_request.id, "approved": False})
            assert denied.status_code == 200 and denied.json()["persisted"]
            await asyncio.wait_for(following, 10)
            case.assert_effect_count(1)
            assert await case.compute.script_store.get_executions_for_script(second.id) == []
            assert sorted(row["decision"] for row in await case.audit()) == ["user_approved", "user_denied"]
        finally:
            queue.cancel_all()
            await asyncio.gather(*calls, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("entrypoint", ["command", "named-tool", "task-skill"])
async def test_native_http_rejects_wrong_credential_and_malformed_approval_scope(
    native_permission_case, entrypoint,
):
    async with native_permission_case(is_test_instance=False) as case:
        queue = case.security.approval_queue
        await case.security.permission_store.set_permission("ComputeFeature", "run_script", PermissionLevel.ASK)
        script = await case.stage_script()
        call = asyncio.create_task(case.invoke(entrypoint, script))
        try:
            request = await case.next_approval(call)
            payload = {"approval_id": request.id, "approved": True, "scope": "once"}
            unauthorized = await case.http.post("/api/security/approve", json=payload,
                headers={"X-API-Key": "unrelated-owned-test-key"})
            assert unauthorized.status_code == 401
            for scope, status in [("", 400), ("forever", 400), (None, 422), ({}, 422)]:
                malformed = await case.http.post("/api/security/approve", json={**payload, "scope": scope})
                assert malformed.status_code == status, malformed.text
                assert queue.pending_requests == [request] and not call.done()
                assert await case.audit() == [], "An invalid decision may not mint approval evidence"
                case.assert_effect_count(0)
            denied = await case.http.post("/api/security/approve", json={**payload, "approved": False})
            assert denied.status_code == 200 and denied.json()["persisted"]
            await asyncio.wait_for(call, 10)
            case.assert_effect_count(0)
            assert await case.compute.script_store.get_executions_for_script(script.id) == []
            assert [row["decision"] for row in await case.audit()] == ["user_denied"]
        finally:
            queue.cancel_all()
            await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("entrypoint", ["command", "named-tool", "task-skill"])
async def test_native_approval_cannot_execute_a_replaced_feature_registration(
    native_permission_case, entrypoint,
):
    async with native_permission_case(is_test_instance=False) as case:
        queue = case.security.approval_queue
        await case.security.permission_store.set_permission("ComputeFeature", "run_script", PermissionLevel.ASK)
        script = await case.stage_script()
        calls = []
        try:
            call = asyncio.create_task(case.invoke(entrypoint, script))
            calls.append(call)
            request = await case.next_approval(call)
            original = case.compute
            await case.agent._unregister_feature_runtime(original, unload=False)
            await case.agent._activate_feature_runtime(original)
            assert case.compute is original and original.enabled
            approved = await case.http.post("/api/security/approve", json={
                "approval_id": request.id, "approved": True, "scope": "once"})
            assert approved.status_code == 200 and approved.json()["persisted"]
            if entrypoint == "command":
                refused = await asyncio.wait_for(call, 10)
                assert refused["success"] is False and "changed" in refused["error"].lower()
            else:
                with pytest.raises(ResourceConflict, match="(?i)(registration|generation).*changed"):
                    await asyncio.wait_for(call, 10)
            case.assert_effect_count(0)
            assert await case.compute.script_store.get_executions_for_script(script.id) == []

            fresh = asyncio.create_task(case.invoke(entrypoint, script))
            calls.append(fresh)
            new_request = await case.next_approval(fresh)
            assert new_request.id != request.id
            approved = await case.http.post("/api/security/approve", json={
                "approval_id": new_request.id, "approved": True, "scope": "once"})
            assert approved.status_code == 200 and approved.json()["persisted"]
            await asyncio.wait_for(fresh, 10)
            case.assert_effect_count(1)
            records = await case.compute.script_store.get_executions_for_script(script.id)
            assert len(records) == 1 and records[0].succeeded and not records[0].dry_run
            assert [row["decision"] for row in await case.audit()] == ["user_approved"] * 2
        finally:
            queue.cancel_all()
            await asyncio.gather(*calls, return_exceptions=True)
