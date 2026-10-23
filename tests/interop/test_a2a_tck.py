"""Drive real Kestrel through the pinned TCK with its reviewed spec corrections."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.utils.native_go_peer import retire_peer
from tests.utils.native_tck_business import execute_tck_task


@pytest.mark.timeout(300)
@pytest.mark.parametrize("level", ["must", "full"])
async def test_tck_native_requirements(native_go_environment, a2a_tck_source, level):
    environment = native_go_environment
    environment.set_business_handler(partial(
        execute_tck_task, manager=environment.recipient.task_manager,
    ))
    tck_root = a2a_tck_source
    origin = environment.config["coreCardUrl"].split("/agents/", 1)[0]
    # The real standalone publication also serves the selected card at the
    # standard root well-known URL; no proxy protocol handler is introduced.
    parsed = urlsplit(origin)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    env = {key: value for key, value in os.environ.items() if key not in {
        "VIRTUAL_ENV", "PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS",
        "DATABASE_URL", "TEST_POSTGRES_URL",
    }}
    env.update(
        KESTREL_TCK_PEER_CONFIG=str(environment.config_path),
        KESTREL_TCK_GO_PEER=str(environment.binary),
        PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(tck_root)]),
    )
    log_path = environment.root / f"tck-{level}.log"
    arguments = [
        "uv", "run", "--no-sync", "./run_tck.py", "--sut-host", origin,
        "--transport", "jsonrpc,http_json",
        "--http-auth-provider", "kestrel_tck_auth:create_auth", "-v",
    ]
    if level == "must":
        arguments.extend(["--level", "must"])
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=tck_root, env=env, stdout=log, stderr=log, start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), 240)
        finally:
            await retire_peer(process, process_group=True)
    reports = environment.root / f"tck-{level}-reports"
    shutil.copytree(tck_root / "reports", reports)
    assert process.returncode == 0, log_path.read_text()
    report = json.loads((reports / "compatibility.json").read_text())
    failures = {
        key: value for key, value in report["per_requirement"].items()
        if value["level"] == "MUST" and value["status"] == "FAIL"
    }
    assert not failures, failures
    assert report["summary"]["score_scope"] == "automated"
    assert report["summary"]["must_compatibility"] == "100.0%", report["summary"]
    for requirement in ("DM-MSG-001", "STREAM-SUB-002"):
        for transport in ("jsonrpc", "http_json"):
            assert report["per_requirement"][requirement]["transports"][transport] == "PASS"
