"""Actual shipped console and real initialized host; no substituted auth/router."""

import asyncio
import os
from pathlib import Path
import shutil
import socket

from cryptography.fernet import Fernet
import pytest

from tests.shared.native_shutdown_budget import shutdown_fixture_agent


@pytest.mark.asyncio
async def test_production_console_artifact_upload_edit_recovery_discovery_and_conflict(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    if not shutil.which('node') or not (project_root / 'node_modules/playwright/package.json').exists():
        pytest.skip('Browser evidence requires the project Node/Playwright runtime')
    # The test conftest owns all runtime paths. Drop deployment and provider
    # configuration before initializing this disposable host; preserve HOME.
    for name in tuple(os.environ):
        if name.startswith(('KESTREL_', 'OTEL_')) or name.endswith(('_API_KEY', '_TOKEN', '_SECRET')):
            monkeypatch.delenv(name, raising=False)
    from kestrel_sovereign.cli_demo import _build_smoke_env
    from kestrel_sovereign import server
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec
    import uvicorn

    (tmp_path / 'home').mkdir()
    (tmp_path / 'data').mkdir()
    isolated = _build_smoke_env(os.environ, home=tmp_path / 'home', data_dir=tmp_path / 'data', data_key=Fernet.generate_key().decode())
    for name, value in isolated.items():
        monkeypatch.setenv(name, value)
    key = 'artifact-browser-disposable-test-key'
    monkeypatch.setenv('KESTREL_API_KEY', key)
    monkeypatch.setenv('KESTREL_DATA_KEY', Fernet.generate_key().decode())
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setenv('KESTREL_SKIP_DOTENV', '1')
    monkeypatch.setenv('KESTREL_TRACING_ENABLED', '0')
    llm = LLMService()
    host_policy = PayerPolicy.host_env_default()
    policy = PayerPolicy(**{field: PayerSpec(vendor=getattr(host_policy, field).vendor, kind=PayerKind.NONE)
                            for field in ('llm', 'storage', 'compute', 'tools', 'comms')})
    agent = KestrelAgent(did='did:test:artifact-browser', storage_path=str(tmp_path / 'agent.db'),
                         llm_service=llm, payer_policy=policy, allowed_features={'ArtifactsFeature'})
    app = server.app
    lifespan = app.router.lifespan_context
    monkeypatch.setattr(app.state, 'agent', agent, raising=False)
    monkeypatch.setattr(app.state, 'agent_manager', None, raising=False)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    serving = None
    runner = None
    try:
        await agent.initialize()
        assert {'artifact_publish_text', 'artifact_get', 'artifact_list', 'artifact_publication_status'} <= {
            item['function']['name'] for item in agent._build_all_tools()}
        server._mount_feature_routers(app)
        runner = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False, lifespan='off'))
        serving = asyncio.create_task(runner.serve(sockets=[sock]))
        async with asyncio.timeout(15):
            while not runner.started:
                await asyncio.sleep(.01)
        child = await asyncio.create_subprocess_exec(
            'node', 'tests/frontend/artifact_product.browser.cjs',
            cwd=project_root,
            env={**os.environ, 'ARTIFACT_TEST_URL': f'http://127.0.0.1:{port}', 'ARTIFACT_TEST_KEY': key},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(child.communicate(), 90)
        except BaseException:
            child.kill()
            await child.communicate()
            raise
        assert child.returncode == 0, (stdout + stderr).decode()
        assert '"verdict":"PASS"' in stdout.decode()
        print(stdout.decode(), end="")
    finally:
        if runner is not None:
            runner.should_exit = True
        if serving is not None:
            await asyncio.wait_for(serving, 15)
        sock.close()
        server._unmount_feature_routers(app)
        app.router.lifespan_context = lifespan
        await shutdown_fixture_agent(agent)
        await llm.close()
        assert serving is None or serving.done()
        assert sock.fileno() == -1
