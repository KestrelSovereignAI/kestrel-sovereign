"""The independent TCK launcher must join its entire owned process session."""

import asyncio
import json
import os
import signal
import sys

import psutil
import pytest

from tests.utils.native_go_peer import retire_peer


@pytest.mark.skipif(os.name != "posix", reason="TCK launcher process-session ownership requires POSIX")
async def test_retiring_the_tck_launcher_also_terminates_its_actual_child():
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c",
        "import subprocess,sys,json,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); sys.stdout.write(json.dumps({'child':p.pid})+'\\n'); sys.stdout.flush(); time.sleep(60)",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    child = None
    try:
        line = await asyncio.wait_for(process.stdout.readline(), 5)
        child = psutil.Process(json.loads(line)["child"])
        assert child.is_running()
        await retire_peer(process, process_group=True)
        assert process.returncode is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        # Also owns cleanup when the deliberate group-producer mutation fails.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
