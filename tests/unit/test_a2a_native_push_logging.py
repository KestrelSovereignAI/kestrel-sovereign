"""Actual callback headers stay on the wire and out of logs and error text."""

import asyncio
import logging
import traceback
from datetime import timedelta

import httpx
import pytest
from kestrel_sovereign._vendor.a2a.types import Task

from kestrel_sovereign.a2a.push_logging import PushLogRedaction
from kestrel_sovereign.a2a.push_transport import NativePushTransport
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
from tests.unit.test_a2a_native_push_config_store import config
from tests.unit.test_a2a_native_push_config_store import (
    push_store as push_store,  # noqa: PLC0414 - pytest fixture re-export
)
from tests.utils.outbound_tls import (
    outbound_tls as outbound_tls,  # noqa: PLC0414 - pytest fixture re-export
)


@pytest.mark.parametrize("path_secret", ["private-path-secret", "private path secret"])
async def test_actual_callback_secrets_are_absent_from_logs_and_retained_error_text(
    push_store, outbound_tls, caplog, monkeypatch, path_secret
):
    store, task = push_store
    value = config(task)
    value.url = outbound_tls.url + "/failure?secret=" + path_secret
    value.token = "private-notification-secret"
    value.authentication.credentials = "private-authentication-secret"
    created = await store.create(value, creator_id="did:creator")
    native_init = httpx.AsyncClient.__init__
    logger = logging.getLogger("httpx")
    caplog.set_level(logging.INFO, logger="httpx")

    async def request_diagnostic(request):
        logger.info(
            "Callback diagnostic url=%s token=%s auth=%s",
            request.url,
            request.headers.get("X-A2A-Notification-Token"),
            request.headers.get("Authorization"),
        )

    def diagnostics(client, *args, **kwargs):
        native_init(client, *args, **kwargs)
        client.event_hooks["request"].append(request_diagnostic)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", diagnostics)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await NativePushTransport(store, timeout=timedelta(seconds=5)).send(
            created, task, executor_id="original-callback-runtime"
        )
    error = caught.value
    logging.getLogger("kestrel.test.callback").warning(
        "Retained callback failure: %s",
        error,
        exc_info=(type(error), error, error.__traceback__),
    )
    rendered = "\n".join(record.getMessage() for record in caplog.records) + "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert error.response.status_code == 503
    assert error.request.url.params["secret"] == path_secret
    assert (
        outbound_tls.posts[0][1]["Authorization"]
        == "Bearer private-authentication-secret"
    )
    assert outbound_tls.posts[0][1]["X-A2A-Notification-Token"] == value.token
    assert "Callback diagnostic" in rendered and "503" in rendered
    for secret in (
        path_secret,
        str(error.request.url),
        value.token,
        value.authentication.credentials,
    ):
        assert secret not in rendered


async def test_filter_is_local_to_original_context_and_inherited_children(caplog):
    value = config(Task(id="task"))
    logger = logging.getLogger("httpx")
    caplog.set_level(logging.INFO, logger="httpx")
    release = asyncio.Event()

    async def unrelated():
        await release.wait()
        logger.info("unrelated %s", value.token)

    outside = asyncio.create_task(unrelated())
    before = list(logger.filters)
    try:
        with PushLogRedaction(value):
            release.set()
            await outside

            async def original_child():
                logger.info("original %s", value.token)

            await asyncio.create_task(original_child())
    finally:
        await outside
    assert [record.getMessage() for record in caplog.records] == [
        "unrelated " + value.token,
        "original [redacted]",
    ]
    assert logger.filters == before


@pytest.mark.parametrize("kind", ["error", "cancellation", "group"])
def test_same_original_control_and_causes_survive_redaction(kind):
    value = config(Task(id="task"))
    original = ExecutionCommitOutcomeError("unknown")
    original.add_note(value.token)
    context = ValueError(value.url)
    original.__context__ = context
    cause = RuntimeError(value.authentication.credentials)
    original.__cause__ = cause
    error = (
        original
        if kind == "error"
        else asyncio.CancelledError(value.token)
        if kind == "cancellation"
        else BaseExceptionGroup("original errors", [original])
    )
    if kind == "cancellation":
        error.__cause__ = original
    before = list(logging.getLogger("httpx").filters)
    with pytest.raises(type(error)) as caught, PushLogRedaction(value):
        raise error
    assert caught.value is error
    assert original.__context__ is context and original.__cause__ is cause
    assert execution_commit_outcome(error) == "unknown"
    assert is_execution_control_error(error)
    assert value.token not in str(error)
    assert value.token not in "".join(original.__notes__)
    assert value.url not in str(context)
    assert value.authentication.credentials not in str(cause)
    assert logging.getLogger("httpx").filters == before


async def test_filter_stays_with_actual_pending_cleanup_owner(caplog):
    value = config(Task(id="task"))
    logger = logging.getLogger("httpx")
    caplog.set_level(logging.INFO, logger="httpx")
    release, emitted = asyncio.Event(), asyncio.Event()
    before = list(logger.filters)
    error = RuntimeError(value.token)
    with pytest.raises(RuntimeError) as caught, PushLogRedaction(value) as redaction:

        async def cleanup():
            await release.wait()
            logger.info("original cleanup %s", value.token)
            emitted.set()

        owner = asyncio.create_task(cleanup())
        resource = object()
        error._kestrel_native_retirement_debts = ((resource, owner),)
        raise error
    try:
        assert caught.value is error
        assert execution_cleanup_debts(error) == ((resource, owner),)
        assert redaction in logger.filters
        release.set()
        await emitted.wait()
        await owner
        await asyncio.sleep(0)  # original owner's registered done callback
        assert caplog.records[-1].getMessage() == "original cleanup [redacted]"
        assert logger.filters == before
    finally:
        release.set()
        await owner


def test_nested_redaction_scopes_and_short_credentials(caplog):
    value, inner = config(Task(id="task")), config(Task(id="task"))
    value.token = "a"
    inner.token = "inner-secret"
    logger = logging.getLogger("httpx")
    caplog.set_level(logging.INFO, logger="httpx")
    before = list(logger.filters)
    with PushLogRedaction(value):
        with PushLogRedaction(inner):
            logger.info("%s %s", value.token, inner.token)
        assert len(logger.filters) == len(before) + 1
    assert all(value.token not in record.getMessage() for record in caplog.records)
    assert all(inner.token not in record.getMessage() for record in caplog.records)
    assert logger.filters == before


def test_root_handler_redacts_cached_traceback_and_log_arguments(caplog):
    value = config(Task(id="task"))
    logger = logging.getLogger("kestrel.test.callback")
    with PushLogRedaction(value):
        logger.warning("Callback %s", value.token)
        record = logging.LogRecord(
            logger.name, logging.WARNING, __file__, 1, "failure", (), None
        )
        record.exc_text = value.url
        caplog.handler.handle(record)
    assert all(value.token not in record.getMessage() for record in caplog.records)
    assert value.url not in caplog.records[-1].exc_text


def test_partial_filter_registration_restores_context_and_bindings(caplog, monkeypatch):
    value = config(Task(id="task"))
    redaction = PushLogRedaction(value)
    logger = logging.getLogger("httpx")
    before = list(logger.filters)
    error = RuntimeError("handler refused filter registration")

    def refuse(_filter):
        raise error

    monkeypatch.setattr(caplog.handler, "addFilter", refuse)
    with pytest.raises(RuntimeError) as caught, redaction:
        pytest.fail("failed registration cannot admit the context")
    assert caught.value is error
    assert logger.filters == before
    assert not redaction._active.get()
