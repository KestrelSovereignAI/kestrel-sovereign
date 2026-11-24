"""Bounded derivative wire format and process-wide admission, with no decoder."""
import struct
import zlib
from types import SimpleNamespace

import pytest

from kestrel_sovereign.artifacts.preview import (
    ArtifactPreviewPool, ArtifactPreviewRefused, ArtifactPreviewUnavailable,
    DockerArtifactRenderer, MAX_PREVIEW_PIXELS, encode_preview, host_preview_pool,
)


def test_png_encoder_emits_only_rgb_ihdr_idat_iend_with_exact_bounded_pixels():
    raw = b'KAP1' + struct.pack('!II', 2, 1) + b'\xff\0\0\0\xff\0'
    result = encode_preview(raw)
    assert (result.width, result.height) == (2, 1)
    offset, chunks = 8, []
    while offset < len(result.png):
        length = struct.unpack('!I', result.png[offset:offset + 4])[0]
        kind = result.png[offset + 4:offset + 8]
        data = result.png[offset + 8:offset + 8 + length]
        crc = struct.unpack('!I', result.png[offset + 8 + length:offset + 12 + length])[0]
        assert zlib.crc32(kind + data) == crc
        chunks.append((kind, data)); offset += length + 12
    assert [kind for kind, _ in chunks] == [b'IHDR', b'IDAT', b'IEND']
    assert zlib.decompress(chunks[1][1]) == b'\0\xff\0\0\0\xff\0'


@pytest.mark.parametrize('raw', [b'', b'KAP1', b'BAD1' + bytes(8),
    b'KAP1' + struct.pack('!II', 0, 1), b'KAP1' + struct.pack('!II', 1, 0),
    b'KAP1' + struct.pack('!II', 1025, 1), b'KAP1' + struct.pack('!II', 2**32 - 1, 2**32 - 1),
    b'KAP1' + struct.pack('!II', 1, 1) + b'\0\0',
    b'KAP1' + struct.pack('!II', 1, 1) + b'\0\0\0extra'])
def test_malformed_or_oversized_raw_output_never_reaches_a_decoder(raw):
    with pytest.raises(ArtifactPreviewRefused):
        encode_preview(raw)


def test_host_pool_is_shared_across_agents_and_unknown_cleanup_keeps_capacity():
    app = SimpleNamespace(state=SimpleNamespace())
    pool = host_preview_pool(app)
    assert host_preview_pool(app) is pool
    first, second = pool.reserve(), pool.reserve()
    with pytest.raises(ArtifactPreviewUnavailable): pool.reserve()
    first.creation_started = True
    with pytest.raises(ArtifactPreviewUnavailable): pool.complete(first)
    assert pool.active_count == 2
    first.cleanup_confirmed = True; pool.complete(first); pool.complete(second)
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('image', [None, 'renderer:latest', 'renderer@sha256:bad'])
async def test_missing_or_mutable_image_fails_before_docker_or_slot_admission(image):
    pool = ArtifactPreviewPool()
    with pytest.raises(ArtifactPreviewUnavailable):
        await DockerArtifactRenderer(image).render(pool, b'x', 'image/png', lambda: None)
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["reap-error", "refuse-cleanup", "refuse-drain"])
async def test_failed_original_preview_retirement_is_bounded_and_never_loses_its_child(monkeypatch, fault):
    import asyncio
    import signal
    import sys
    from kestrel_sovereign import async_ownership as ownership
    from kestrel_sovereign._async_process import _signal_process_tree, _process_tree_alive, SubprocessCleanupError
    from kestrel_sovereign.artifacts import preview
    from kestrel_sovereign.execution_custody import execution_evidence_carriers, execution_cleanup_debts
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    from tests.unit.test_owned_task_lending import CapacityLoan

    original_start = preview.start_async_process
    children = []
    original = SubprocessCleanupError("original private CLI reap failed")
    async def launch(arguments, **options):
        child = await original_start([sys.executable, "-B", "-c", "import time;time.sleep(30)"], **options)
        children.append(child)
        return child
    async def failed_reap(*args, **kwargs):
        raise original
    monkeypatch.setattr(preview, "start_async_process", launch)
    if fault == "reap-error":
        monkeypatch.setattr(preview, "_terminate_and_await", failed_reap)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **options):
        operation = _actual_operation(coroutine)
        name = getattr(getattr(operation, "cr_code", None), "co_name", None)
        if ((fault == "refuse-cleanup" and name == "_terminate_and_await")
                or (fault == "refuse-drain" and name == "_drain_and_wait")):
            raise original
        return previous(loop, coroutine, **options) if previous else asyncio.Task(coroutine, loop=loop, **options)
    async def run():
        host = CapacityLoan(capacity=0 if fault == "refuse-drain" else 8)
        with ownership.bind_owned_task_lender(host):
            return await DockerArtifactRenderer(None)._command_native(
                ["version"], environment=DockerArtifactRenderer._env(), timeout=.01)
    command = None
    try:
        loop.set_task_factory(factory)
        command = asyncio.create_task(run())
        done, _ = await asyncio.wait({command}, timeout=.25)
        loop.set_task_factory(previous)
        assert command in done, "failed cleanup entered an unbounded original waiter join"
        error = command.exception()
        assert original in execution_evidence_carriers(error)
        assert children and any(resource is children[0] for resource, _ in execution_cleanup_debts(error))
        if fault != "reap-error":
            # A refused closer still synchronously signals its exact private
            # group; the test owner observes the real exit, never fakes an ACK.
            await asyncio.wait_for(children[0].wait(), 2)
            assert not _process_tree_alive(children[0])
    finally:
        loop.set_task_factory(previous)
        for child in children:
            if _process_tree_alive(child):
                _signal_process_tree(child, signal.SIGKILL)
            await asyncio.wait_for(child.wait(), 3)
        if command is not None:
            await asyncio.wait_for(asyncio.gather(command, return_exceptions=True), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [0, 1, 2])
async def test_preview_admission_refusal_reaps_the_already_acquired_private_process(monkeypatch, capacity):
    import asyncio
    import signal
    import sys
    from kestrel_sovereign import async_ownership as ownership
    from kestrel_sovereign._async_process import _signal_process_tree, _process_tree_alive
    from kestrel_sovereign.artifacts import preview
    from tests.unit.test_owned_task_lending import CapacityLoan

    original = preview.start_async_process
    processes = []
    async def launch(arguments, **options):
        # An idle, bounded private child replaces only the external executable;
        # the actual native process, pipes, waiter and cleanup stay production.
        process = await original([sys.executable, "-B", "-c", "import time; time.sleep(30)"], **options)
        processes.append(process)
        return process
    monkeypatch.setattr(preview, "start_async_process", launch)
    previous = set(asyncio.all_tasks())
    host = CapacityLoan(capacity=capacity)
    error = None
    try:
        with ownership.bind_owned_task_lender(host):
            try:
                await DockerArtifactRenderer(None)._command_native(
                    ["version"], environment=DockerArtifactRenderer._env(), timeout=.02)
            except BaseException as failure:
                error = failure
        assert error is not None, "refused/deadline-bound command reported success"
        assert len(processes) == 1
        assert processes[0].returncode is not None, "admission refusal orphaned its original process"
        assert not _process_tree_alive(processes[0]), "original private process group survived"
    finally:
        for process in processes:
            if _process_tree_alive(process):
                _signal_process_tree(process, signal.SIGKILL)
            await asyncio.wait_for(process.wait(), 3)
        owned = [task for task in asyncio.all_tasks() - previous
                 if task.get_name() in {"preview-cli-exit", "preview-cli-io", "preview-cli-cleanup"}]
        if owned:
            await asyncio.wait_for(asyncio.gather(*owned, return_exceptions=True), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("message,rejected", [
    ("Error response from daemon: no command specified\n", True),
    ("Error response from daemon: unknown or invalid runtime name: owned-test-123\n", True),
    ("Error response from daemon: internal server error\n", False),
    ("Error response from daemon: EOF\n", False),
    ("Error response from daemon: unknown or invalid runtime name: \n", False),
    ("Error response from daemon: unknown or invalid runtime name: valid\nEOF\n", False),
    ("connection reset by peer\n", False),
])
async def test_only_known_precreation_daemon_refusals_are_definitive_in_owned_native_process(monkeypatch, message, rejected):
    import sys
    from kestrel_sovereign.artifacts import preview
    original = preview.start_async_process
    children = []
    async def launch(arguments, **kwargs):
        child = await original([sys.executable, "-B", "-c", "import sys;sys.stderr.write("+repr(message)+");sys.exit(1)"], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(preview, "start_async_process", launch)
    renderer = DockerArtifactRenderer("sha256:"+"1"*64)
    if rejected:
        with pytest.raises(preview._CreateRejected, match="renderer creation was refused") as error:
            await renderer._command(["create"], classify_create=True)
        assert message.strip() not in str(error.value)
    else:
        status, output = await renderer._command(["create"], classify_create=True)
        assert status == 1 and not output
    assert len(children) == 1 and children[0].returncode == 1


@pytest.mark.asyncio
async def test_known_rejection_still_retains_slot_without_successful_absence_oracle(monkeypatch):
    renderer = DockerArtifactRenderer("sha256:"+"1"*64);pool = ArtifactPreviewPool();job = pool.reserve()
    job.creation_started = True;job.create_rejected = True;job.finished = True
    job.endpoint = "unix:///owned-test.sock";job.image_id = renderer.image
    calls = []
    async def unavailable(arguments, **kwargs):
        calls.append(arguments)
        return 1, b""
    monkeypatch.setattr(renderer, "_command", unavailable)
    await renderer._cleanup_native(pool, job)
    assert pool.active_count == 1 and not job.cleanup_confirmed, "failed absence oracle released capacity"
    assert any("ls" in call for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["label", "image", "name", "id"])
async def test_recovery_never_removes_a_foreign_or_malformed_container(monkeypatch, damage):
    renderer=DockerArtifactRenderer("sha256:"+"1"*64);pool=ArtifactPreviewPool();job=pool.reserve()
    job.creation_started=True;job.finished=True;job.endpoint="unix:///owned-test.sock";job.image_id=renderer.image
    value={"Id":"2"*64,"Name":"/"+job.name,"Config":{"Image":job.image_id,"Labels":{"io.kestrel.artifact-preview.job":job.name}}}
    if damage=="label":value["Config"]["Labels"]={}
    if damage=="image":value["Config"]["Image"]="sha256:"+"3"*64
    if damage=="name":value["Name"]="/foreign"
    if damage=="id":value["Id"]="not-a-container-id"
    calls=[]
    async def inspect_only(arguments, **kwargs):
        import json
        calls.append(arguments)
        return 0,json.dumps([value]).encode()
    monkeypatch.setattr(renderer,"_command",inspect_only)
    await renderer._cleanup_native(pool,job)
    assert pool.active_count==1 and not job.cleanup_confirmed, "foreign container reclaimed preview slot"
    assert not any("rm" in args for args in calls), "foreign cleanup was dispatched"


@pytest.mark.asyncio
async def test_every_engine_child_has_a_fresh_empty_config_until_native_join(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    import sys
    from kestrel_sovereign.artifacts import preview
    operator = tmp_path / "operator-config"; operator.mkdir()
    (operator / "config.json").write_text('{"proxies":{"default":{"httpProxy":"synthetic-private"}}}')
    monkeypatch.setenv("DOCKER_CONFIG", str(operator))
    original = preview.start_async_process
    commands = []
    async def launch(arguments, **kwargs):
        assert arguments[:2] == ["docker", "--config"]
        config = Path(arguments[2])
        assert config.is_dir() and not list(config.iterdir()) and config != operator
        assert "DOCKER_CONFIG" not in kwargs["env"]
        code = "from pathlib import Path;import sys;assert Path(sys.argv[1]).is_dir();print('native-config-present')"
        child = await original([sys.executable, "-B", "-c", code, str(config)], **kwargs)
        commands.append((config, child))
        return child
    monkeypatch.setattr(preview, "start_async_process", launch)
    renderer = DockerArtifactRenderer(None)
    for command in (["--host", "unix:///owned.sock", "info"], ["--host", "unix:///owned.sock", "create"]):
        status, output = await renderer._command(command)
        assert status == 0 and output.strip() == b"native-config-present"
    assert len({config for config, _ in commands}) == 2
    assert all(child.returncode == 0 and not config.exists() for config, child in commands)
    assert json.loads((operator / "config.json").read_text())["proxies"]["default"]["httpProxy"] == "synthetic-private"


@pytest.mark.parametrize("entry", ["HTTP_PROXY=synthetic-private", "https_proxy=synthetic-private",
    "ALL_PROXY=synthetic-private", "ftp_proxy=synthetic-private", "No_Proxy=synthetic-private", 42])
def test_native_descriptor_with_proxy_environment_refuses_before_decoder_start(entry):
    value = {"Config": {"OpenStdin": True, "User": "65532:65532", "Env": [entry]}, "Mounts": [],
        "HostConfig": {"Memory": 512 * 1024 * 1024, "MemorySwap": 512 * 1024 * 1024,
            "NanoCpus": 500000000, "PidsLimit": 16, "NetworkMode": "none", "ReadonlyRootfs": True,
            "IpcMode": "none", "PidMode": "", "CgroupnsMode": "private", "Runtime": "runc",
            "Privileged": False, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
            "LogConfig": {"Type": "none"}, "Tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=64m,mode=1777"}}}
    with pytest.raises(ArtifactPreviewUnavailable, match="sandbox limits"):
        DockerArtifactRenderer._verify_container(value)
    value["Config"]["Env"] = ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8"]
    DockerArtifactRenderer._verify_container(value)
