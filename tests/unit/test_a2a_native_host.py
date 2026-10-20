"""Native manager construction and the real host's signal boot phase."""

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.task_manager import TaskManager, create_task_manager
from kestrel_sovereign.storage.db.interface import QueryError


@pytest.mark.parametrize("owns_backend", [False, True])
async def test_native_factory_initializes_one_backend_and_honors_its_owner(
    native_a2a_backend_factory,
    owns_backend,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    submitted = []
    scope = NativeRecipientScope("did:kestrel:recipient", "tenant")
    manager = TaskManager.from_backend(
        backend,
        scope=scope,
        owns_backend=owns_backend,
        on_task_submitted=submitted.append,
    )
    try:
        stores = [
            manager.task_store,
            manager.context_store,
            manager.session_service,
            manager.observability_store,
            manager.memory_service,
            manager.feedback_store,
        ]
        assert all(store.backend is backend for store in stores)
        await manager.initialize()
        task = await manager.create_task(
            SendMessageRequest(
                tenant="tenant",
                message=new_text_message("native request", role=Role.ROLE_USER),
            ),
            creator_agent_id="did:kestrel:sender",
        )
        assert submitted[0].task == task
        assert submitted[0].ownership.recipient_agent_id == scope.recipient_id
        assert submitted[0].ownership.tenant == scope.tenant
        await manager.close()
        assert backend.is_connected is (not owns_backend)
        if not owns_backend:
            assert await backend.fetch_one("SELECT 1") == (1,)
    finally:
        await manager.close()
        await backend.close()


async def test_owned_sqlite_factory_uses_the_same_native_admission_contract(tmp_path):
    scope = NativeRecipientScope("did:kestrel:standalone")
    manager = await create_task_manager(str(tmp_path / "standalone.db"), scope=scope)
    backend = manager.task_store.backend
    try:
        task = await manager.create_task(
            SendMessageRequest(
                message=new_text_message("native request", role=Role.ROLE_USER)
            ),
            creator_agent_id="did:kestrel:sender",
        )
        view = await manager.get_cognition_wake_candidate(
            task.id, recipient_agent_id=scope.recipient_id
        )
        assert view.ownership.creator_agent_id == "did:kestrel:sender"
    finally:
        await manager.close()
    assert not backend.is_connected


@pytest.mark.parametrize("broken_session_schema", [False, True])
async def test_actual_host_signal_boot_borrows_native_storage_and_rolls_back(
    native_a2a_backend_factory,
    tmp_path,
    broken_session_schema,
):
    # Imports happen after the suite has isolated every runtime/config path.
    from kestrel_sovereign.agent.boot import BootContext
    from kestrel_sovereign.features.base import Feature, tool
    from kestrel_sovereign.features.privacy import PrivacyAgent
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage, PrivacyEnforcingStorage

    backend = native_a2a_backend_factory()
    storage = AsyncStorage(backend=backend, agent_id="did:kestrel:host")
    await storage.initialize()
    llm = LLMService(usage_db=storage.db, agent_data_dir=tmp_path)
    agent = KestrelAgent(
        "did:kestrel:host",
        storage_path=str(tmp_path / "host.db"),
        llm_service=llm,
        db_backend=backend.backend_type,
        sync_enabled=False,
        allowed_features=set(),
    )
    agent._raw_storage = storage
    agent.storage = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    agent.privacy_agent = PrivacyAgent(storage, PrivacyMode.NORMAL)
    ctx = BootContext(custody=agent._custody)
    try:
        if broken_session_schema:
            # A real SQL failure after task/context initialization proves the
            # rollback owner was registered before initialize's first await.
            await backend.execute_script(
                "CREATE TABLE a2a_sessions (id TEXT PRIMARY KEY)"
            )
            with pytest.raises(QueryError, match="agent_name"):
                await agent._boot_phase_a2a_observability_signals(ctx)
            assert "task_manager" in ctx.rollback_labels
            assert agent.task_manager is not None
        else:
            await agent._boot_phase_a2a_observability_signals(ctx)
            manager = agent.task_manager
            assert manager.task_store.scope == NativeRecipientScope(agent.did)
            assert (
                manager.context_store.backend is manager.task_store.backend is backend
            )
            assert manager.session_service.backend is backend
            assert manager.feedback_store is agent.feedback_store
            assert manager.observability_store is agent.observability_store
            assert agent.dispatcher is not None
            assert agent.signal_registry.get("a2a.task_submitted") is not None
            assert agent.signal_registry.get("a2a.task_complete") is not None

            class NativeHostFeature(Feature):
                @property
                def tool_description(self):
                    return "Native host skill contribution"

                async def initialize(self):
                    pass

                @tool(
                    name="native_host_echo",
                    description="Echo JSON unchanged",
                    command_prefix="!native_host_echo",
                )
                async def echo(self, value: dict) -> dict:
                    return value

            feature = NativeHostFeature(agent)
            await agent._register_feature(feature)
            assert [skill.id for skill in manager.get_agent_skills()] == [
                "native_host_echo"
            ]
            assert manager.get_agent_for_skill("native_host_echo") == feature.name
            assert manager.get_agent_for_command("!native_host_echo") == (
                feature.name,
                "native_host_echo",
            )
            await agent._unregister_feature_runtime(feature)
            assert manager.get_agent_skills() == []
            assert manager.get_agent_for_command("!native_host_echo") is None
        await ctx.run_rollback()
        assert not ctx.rollback_errors
        assert agent.task_manager is None
        assert backend.is_connected
        assert await backend.fetch_one("SELECT 1") == (1,)
    finally:
        await ctx.run_rollback()
        await llm.close()
        await storage.close()
