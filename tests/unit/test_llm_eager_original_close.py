"""Actual eager SDK close settles its original native resource registry."""

import asyncio

import httpx2 as httpx
import openai
import pytest


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="original eager factory requires Python 3.12+",
)
async def test_original_eager_remote_close_reconciles_actual_sdk_resource(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.llm import service as service_module

    monkeypatch.setattr(service_module, "load_section", lambda _name: {})
    service = service_module.LLMService(agent_data_dir=tmp_path)
    transport = httpx.AsyncClient(trust_env=False)
    client = openai.AsyncOpenAI(
        api_key="inert-local-native-close-test", http_client=transport
    )
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        await service._close_remote_client(client, lease_id="actual-idle-sdk-owner")
        loop.set_task_factory(previous)
        assert transport.is_closed
        await asyncio.sleep(0)
        assert not service.has_pending_close_owners(), (
            "completed eager SDK close retained false native resource debt"
        )
        assert not any(
            resource is client
            for resource in service._llm_native_retirement_resources.values()
        )
        await service.wait_for_close_completion()
        assert not service.has_pending_close_owners()
    finally:
        loop.set_task_factory(previous)
        await client.close()
        await service.close()
