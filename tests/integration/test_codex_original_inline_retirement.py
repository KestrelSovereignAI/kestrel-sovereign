"""Actual bridge Tasks/stdio/PG keep accepted tools inside original turn close.

The owned peer speaks only declared JSON-RPC protocol input. No Codex binary,
authentication file, model call or external provider is used. Effects and
checkpoints are real local PG writes, not returned success stand-ins.
"""

import asyncio
import base64
import contextvars
import inspect
from pathlib import Path
import sys

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign import execution_custody
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.llm import codex_adapter as adapter_module
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cleanup_task_admission import _operation_name


PEER = """
import json, sys
def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "config/read":
        send({"id": message["id"], "result": {"config": {}}})
    elif method == "thread/start":
        send({"id": message["id"], "result": {"thread": {"id": "local-inline-thread"}}})
    elif method == "turn/start":
        send({"id": message["id"], "result": {"turn": {"id": "local-inline-turn"}}})
        send({"id": 17923, "method": "item/tool/call", "params": {
            "threadId": "local-inline-thread", "callId": "local-effect-call",
            "tool": "write_local_effect", "arguments": {}}})
    elif method is not None and "id" in message:
        send({"id": message["id"], "result": {}})
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "eager"), [(mode, False) for mode in ["collector", "collector-native", "checkpoint", "checkpoint-observer", "native-tool", "tool-observer", "deferred-completion", "image", "image-native", "checkpoint-publish-before", "checkpoint-publish-after"]] + [(mode, True) for mode in ["checkpoint-publish-before", "checkpoint-publish-after"]]
)
async def test_original_inline_tool_and_checkpoint_retire_before_thread_release(
    native_pg, monkeypatch, tmp_path, mode, eager
):
    backend, observer, schema = native_pg
    await observer.execute(f'CREATE TABLE "{schema}".inline_checkpoints (id INTEGER PRIMARY KEY)')
    async with backend.advisory_locks([(3569, 17923)]) as lease:
        lease.require_live()
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", PEER,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    app = CodexAppServerClient(binary="owned-local-protocol-only")
    app._proc, app._initialized = process, True
    app._reader_task = create_owned_task(app._read_loop(), context=contextvars.Context())
    adapter = CodexAdapter()
    adapter._client = app
    started, release_tool, admission_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    checkpoint_started, release_checkpoint, allocation_attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tools, checkpoints, withdrawals, source_errors, original_chains, refused = [], [], [], [], [], []
    caller_receipts = []
    retirement_observers = []
    handler_sets, join_counts = [], {}
    image_paths, image_errors, image_owners, image_tasks = [], [], [], []
    content = "local fixture"
    if mode in {"image", "image-native"}:
        payload = b"\x89PNG\r\n\x1a\nlocal-declared-image-input"
        content = [{"type": "text", "text": "local fixture"}, {
            "type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(payload).decode("ascii"),
            },
        }]
        materialize = adapter_module._materialize_app_server_data_images
        native_unlink = Path.unlink

        def materialize_then_replace(turn_input, **kwargs):
            materialized, paths = materialize(turn_input, **kwargs)
            image_owners.append(kwargs.get("retirement_owner"))
            image_tasks.append(asyncio.current_task())
            for path in paths:
                assert path.read_bytes() == payload
                native_unlink(path)
                path.mkdir()
                image_paths.append(path)
            return materialized, paths

        def observe_native_unlink(path, *args, **kwargs):
            try:
                return native_unlink(path, *args, **kwargs)
            except OSError as error:
                if path in image_paths:
                    image_errors.append(error)
                raise

        monkeypatch.setattr(adapter_module.tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(adapter_module, "_materialize_app_server_data_images", materialize_then_replace)
        monkeypatch.setattr(Path, "unlink", observe_native_unlink)
    refusal = RuntimeError("original inline retirement Task allocation refused")
    publication_tasks, checkpoint_operations = [], []
    loop, factory = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    actual_register, actual_events = app.register_server_request_handler, app.iter_turn_events
    actual_terminal = execution_custody.execution_terminal_error
    actual_retirement_join = adapter_module.await_owned_task

    def remember_withdrawal(error):
        if isinstance(error, asyncio.CancelledError) and error.args in {
            ("actual inline turn observer withdrawal",),
            ("actual later inline checkpoint observer withdrawal",),
            ("actual later inline tool observer withdrawal",),
        } and all(error is not prior for prior in caller_receipts):
            caller_receipts.append(error)

    def terminal(*errors, **kwargs):
        for error in errors:
            remember_withdrawal(error)
        return actual_terminal(*errors, **kwargs)

    async def retirement_join(task, *args, **kwargs):
        if mode == "deferred-completion" and task in tools:
            join_counts[task] = join_counts.get(task, 0) + 1
            assert join_counts[task] == 1, "completed original inline task was rejoined without active-set progress"
        retirement_observers.append((task, asyncio.current_task()))
        outcome = await actual_retirement_join(task, *args, **kwargs)
        remember_withdrawal(outcome.cancellation)
        return outcome

    monkeypatch.setattr(execution_custody, "execution_terminal_error", terminal)
    monkeypatch.setattr(adapter_module, "await_owned_task", retirement_join)
    # Checkpoint admission now joins through the shared admission-safe helper;
    # observe that same real join as well as the direct handler-join alias.
    from kestrel_sovereign import _async_ownership as ownership_module
    monkeypatch.setattr(ownership_module, "await_owned_task", retirement_join)

    def register(method, handler, **kwargs):
        marked = getattr(handler, "_codex_active_tasks", None)
        if isinstance(marked, set) and all(marked is not prior for prior in handler_sets):
            handler_sets.append(marked)
        unregister = actual_register(method, handler, **kwargs)

        def close_original_admission():
            unregister()
            if method == "item/tool/call":
                admission_closed.set()

        return close_original_admission

    async def events(*args, **kwargs):
        try:
            async for event in actual_events(*args, **kwargs):
                yield event
        except asyncio.CancelledError as error:
            # The owned stream may cancel its queue reader as a PRIVATE close
            # interrupt. That is not the original caller's named withdrawal.
            withdrawals.append(error)
            raise

    monkeypatch.setattr(app, "register_server_request_handler", register)
    monkeypatch.setattr(app, "iter_turn_events", events)

    async def execute(name, arguments):
        assert name == "write_local_effect" and arguments == {}
        tools.append(asyncio.current_task())
        async with backend.transaction():
            await backend.execute("INSERT INTO effects VALUES (179, 'original inline effect')")
            started.set()
            await release_tool.wait()
        if mode in {"collector-native", "native-tool", "tool-observer", "image-native"}:
            try:
                lease.require_live()
            except BaseException as error:
                source_errors.append(error)
                original_chains.append((error, tuple(_explicit_control_chain(error))))
                raise
        return {"success": True, "result": "local PG write completed"}

    async def checkpoint(executed):
        assert len(executed) == 1 and executed[0]["id"] == "local-effect-call"
        checkpoints.append(asyncio.current_task())
        checkpoint_started.set()
        await release_checkpoint.wait()
        await backend.execute("INSERT INTO inline_checkpoints VALUES (179)")
        if mode == "checkpoint-observer":
            try:
                lease.require_live()
            except BaseException as error:
                source_errors.append(error)
                original_chains.append((error, tuple(_explicit_control_chain(error))))
                raise

    execute.persist_completed_effects = checkpoint

    class PublicationTask(asyncio.Task):
        failed = False

        def add_done_callback(self, callback, *args, **kwargs):
            if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                self.failed = True
                if mode == "checkpoint-publish-after":
                    super().add_done_callback(callback, *args, **kwargs)
                allocation_attempted.set()
                raise refusal
            return super().add_done_callback(callback, *args, **kwargs)

    def refuse(loop, coroutine, **kwargs):
        name = _operation_name(coroutine)
        if mode.startswith("checkpoint-publish-") and name.endswith(".checkpoint"):
            from tests.unit.test_host_context_native_cleanup import _actual_operation
            checkpoint_operations.append(_actual_operation(coroutine))
            task = PublicationTask(coroutine, loop=loop, eager_start=eager, **kwargs)
            publication_tasks.append(task)
            return task
        selected = (
            mode in {"collector", "collector-native"} and name.endswith(".settle_inline_tools")
        ) or (mode == "checkpoint" and name.endswith(".checkpoint"))
        if selected:
            refused.append(coroutine)
            allocation_attempted.set()
            raise refusal
        return factory(loop, coroutine, **kwargs) if factory else asyncio.Task(coroutine, loop=loop, **kwargs)

    owner = create_owned_task(
        adapter.get_response(
            client="declared-local-protocol", model="auto",
            messages=[{"role": "user", "content": content}],
            tools=[{"type": "function", "function": {"name": "write_local_effect", "parameters": {"type": "object"}}}],
            session_id="local-inline-session", tool_executor=execute,
        ), retain_originals=True,
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert len(tools) == 1 and not tools[0].done()
        if mode in {"image", "image-native"}:
            assert len(image_paths) == 1 and image_paths[0].is_dir()
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 0
        loop.set_task_factory(refuse)
        owner.cancel("actual inline turn observer withdrawal")
        try:
            await asyncio.wait_for(admission_closed.wait(), 5)
        except TimeoutError:
            assert not image_errors, "image cleanup error bypassed original handler admission closure"
            raise
        await asyncio.sleep(0)
        lock = adapter._thread_locks["local-inline-thread"]
        assert lock.locked() and "local-inline-thread" in app._turn_sinks, (
            "inline cleanup admission released original mutex/sink before accepted tool retirement"
        )
        assert not owner.done() and not tools[0].done()
        if mode == "deferred-completion":
            markers = [marked for marked in handler_sets if tools[0] in marked]
            assert len(markers) == 1
            # Fault only the deferred membership bookkeeping, not the actual
            # Task, PG effect or outcome. The strict join-count guard above
            # catches non-progress before a callback-starving loop can run.
            tools[0].add_done_callback(markers[0].add)
        if mode == "tool-observer":
            observers = [observing for target, observing in retirement_observers if target is tools[0]]
            assert len(observers) == 1
            observers[0].cancel("actual later inline tool observer withdrawal")
            await asyncio.sleep(0)
        release_tool.set()
        if mode == "checkpoint" or mode.startswith("checkpoint-publish-"):
            await asyncio.wait_for(allocation_attempted.wait(), 5)
        elif mode not in {"collector-native", "native-tool", "tool-observer", "image-native"}:
            try:
                await asyncio.wait_for(checkpoint_started.wait(), 5)
            except TimeoutError:
                if mode == "deferred-completion":
                    assert join_counts.get(tools[0], 0) == 1, (
                        "completed original inline task was rejoined without active-set progress"
                    )
                raise
            assert lock.locked(), "thread mutex released before its accepted effect checkpoint"
            if mode == "checkpoint-observer":
                observers = [observing for target, observing in retirement_observers if target is checkpoints[0]]
                assert len(observers) == 1
                observers[0].cancel("actual later inline checkpoint observer withdrawal")
                await asyncio.sleep(0)
            release_checkpoint.set()
        outcome = await asyncio.wait_for(await_owned_task(owner), 5)
        loop.set_task_factory(factory)
        assert outcome.error is not None and all(task.done() for task in (*tools, *checkpoints))
        assert not lock.locked() and "local-inline-thread" not in app._turn_sinks
        assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".effects') == 1
        chain = tuple(_explicit_control_chain(outcome.error))
        assert caller_receipts and all(error in chain for error in (*caller_receipts, *source_errors)), (
            "inline retirement replaced actual caller withdrawal or native tool receipt"
        )
        assert all(tuple(_explicit_control_chain(error)) == before for error, before in original_chains)
        if mode in {"image", "image-native"}:
            assert image_errors and all(error in chain for error in image_errors)
            assert all(error.filename == str(image_paths[0]) for error in image_errors)
            assert is_execution_control_error(outcome.error), "failed image retirement became ordinary provider availability"
            assert len(image_owners) == 1 and image_owners[0] is image_tasks[0] and image_owners[0].done()
            assert any(
                resource is image_paths[0] and attempt is image_owners[0]
                for resource, attempt in execution_cleanup_debts(outcome.error)
            ), "failed image unlink lost its actual artifact/original turn owner"
        if mode == "checkpoint" or mode.startswith("checkpoint-publish-"):
            assert refusal in chain and all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in refused)
            expected = publication_tasks[0] if publication_tasks else None
            assert (execute, expected) in execution_cleanup_debts(outcome.error)
            assert all(task.done() for task in publication_tasks)
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for coro in checkpoint_operations)
            assert not checkpoints, "failed start gate admitted the native checkpoint body"
            assert is_execution_control_error(outcome.error)
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".inline_checkpoints') == 0
        elif mode in {"collector-native", "native-tool", "tool-observer", "image-native"}:
            assert source_errors and is_execution_control_error(outcome.error)
            assert not refused and not checkpoints
        else:
            assert not refused and len(checkpoints) == 1
            assert await observer.fetchval(f'SELECT count(*) FROM "{schema}".inline_checkpoints') == 1
    finally:
        loop.set_task_factory(factory)
        release_tool.set()
        release_checkpoint.set()
        if not owner.done():
            owner.cancel()
        await await_owned_task(owner)
        for task in tools:
            await await_owned_task(task)
        for task in publication_tasks:
            await await_owned_task(task)
        for coroutine in refused:
            coroutine.close()
        await await_owned_task(create_owned_task(app.aclose(), retain_originals=True))
        assert process.returncode is not None
        for path in image_paths:
            path.rmdir()
