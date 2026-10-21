"""Own google-genai's otherwise unclosed async forwarding layers.

The maintained SDK returns several generators whose bare ``async for`` does
not close its child. Record each acquired handle in a request-local lexical
owner, including the HTTP response before any read. Retirement joins every
layer and the native body; async-generator finalization is never the owner.
"""

import inspect
import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar, copy_context
from functools import partial, wraps

from kestrel_sovereign._async_ownership import create_owned_iterator, await_owned_task
from kestrel_sovereign.execution_custody import (
    NativeExecutionStreamOwner as _StreamOwner,
    _NATIVE_RESPONSE_SINK,
    close_native_execution_stream,
    execution_terminal_error,
    retain_native_execution_http_client_close,
)

_OWNER = ContextVar("kestrel_genai_stream_owner", default=None)


def _track_iterator_factory(instance, name):
    original = getattr(instance, name)

    @wraps(original)
    def acquire(*args, **kwargs):
        handle = original(*args, **kwargs)
        owner = _OWNER.get()
        return owner.retain(handle) if owner is not None else handle

    setattr(instance, name, acquire)


def is_genai_client(client):
    try:
        from google.genai import Client
    except ImportError:
        return False
    return isinstance(client, Client)


def prepare_genai_client(client):
    """Pin native async transport and install request-local handle custody.

    Core uses the SDK's HTTPX transport, whose pre-body response hooks permit
    ownership BEFORE error-body reads as well as streaming. Existing HTTPX
    settings/client are retained. The SDK then treats that handle as injected;
    Core records its explicit close responsibility instead of leaking it.
    Legacy in-memory adapter doubles have no native transport and are untouched.
    """
    if not is_genai_client(client) or getattr(client, "_kestrel_genai_custody", False):
        return
    api = client._api_client
    if api._use_google_auth_async():
        raise RuntimeError(
            "Core google-genai HTTPX custody does not support automatic Vertex mTLS; "
            "configure an explicit certificate-capable HTTPX transport"
        )
    http = api._async_httpx_client
    if http is None:
        raise RuntimeError("Core google-genai requires an owned HTTPX async transport")
    if api._http_options.aiohttp_client is not None:
        raise RuntimeError("Core google-genai cannot own an explicitly injected aiohttp transport")
    if api._http_options.httpx_async_client is None:
        api._http_options.httpx_async_client = http
        client._kestrel_genai_httpx_owned = http
    retain_native_execution_http_client_close(http)

    # SDK-version-specific integration points, proved against real SDKs by
    # offline transport tests. No frame inspection or finalizer dependence.
    native_request = api._async_request

    @wraps(native_request)
    async def request(*args, **kwargs):
        response = await native_request(*args, **kwargs)
        if _OWNER.get() is not None:
            _track_iterator_factory(response, "async_segments")
            _track_iterator_factory(response, "_aiter_response_stream")
        return response

    api._async_request = request
    for instance, name in ((api, "async_request_streamed"),
                           (client.aio.models, "_generate_content_stream"),
                           (client.aio.models, "generate_content_stream")):
        original = getattr(instance, name)

        def tracked(original):
            @wraps(original)
            async def acquire(*args, **kwargs):
                handle = await original(*args, **kwargs)
                owner = _OWNER.get()
                return owner.retain(handle) if owner is not None else handle
            return acquire

        setattr(instance, name, tracked(original))
    client._kestrel_genai_custody = True


@asynccontextmanager
async def genai_stream_owner(client):
    prepare_genai_client(client)
    owner = _StreamOwner()
    token = _OWNER.set(owner)
    sink_token = _NATIVE_RESPONSE_SINK.set(owner)
    try:
        yield owner
    finally:
        original = sys.exc_info()[1]
        try:
            await owner.close(original)
        finally:
            _NATIVE_RESPONSE_SINK.reset(sink_token)
            _OWNER.reset(token)


@asynccontextmanager
async def joined_iterator(stream):
    """Explicitly retire a Core forwarding child before its parent returns."""
    try:
        yield stream
    finally:
        await close_native_execution_stream(stream)


def genai_operation(function=None, *, resolve_client=True):
    if function is None:
        return partial(genai_operation, resolve_client=resolve_client)
    @wraps(function)
    async def owned(self, client=None, *args, **kwargs):
        if resolve_client and client is None:
            client = self._get_client()
        async with genai_stream_owner(client):
            return await function(self, client, *args, **kwargs)
    return owned


def genai_stream(function):
    @wraps(function)
    def owned(self, client, *args, **kwargs):
        client = client if client is not None else self._get_client()
        owner_context = copy_context()
        async def produce():
            async with genai_stream_owner(client) as owner:
                source = owner.retain(function(self, client, *args, **kwargs))
                async for item in source:
                    yield item
        # Acquisition, advancement and closure share ONE task/context even
        # when the caller abandons a read or closes from another task.
        async def forward():
            iterator = await create_owned_iterator(produce, operation="genai-native-stream", owner_context=owner_context)
            async with joined_iterator(iterator):
                async for item in iterator:
                    yield item
        return forward()
    return owned


async def close_genai_client(client):
    """Join the SAME original SDK/session/pool close on every observer."""
    from .lifecycle import harvest_llm_close_task, llm_close_observation_control, start_llm_close_task

    original = sys.exc_info()[1]
    try:
        task = start_llm_close_task(
            client, "genai-native-client", lambda: _close_genai_layers(client), resource=client,
        )
    except BaseException as error:
        # Allocation refusal precedes the normal join/composer. Its actual
        # client debt must not erase a body cancellation into implicit context.
        raise execution_terminal_error(
            original, error, retain_originals=True, isolate_control_originals=True,
        )
    if task is None:
        if vars(client).get("_llm_close_control") is not None:
            raise execution_terminal_error(
                original, client._llm_close_control, retain_originals=True,
                isolate_control_originals=True,
            )
        return
    outcome = await await_owned_task(task)
    terminal = execution_terminal_error(
        original, harvest_llm_close_task(client, "genai-native-client", task, observation_error=llm_close_observation_control(outcome)),
        vars(client).get("_llm_close_control"), outcome.error, outcome.cancellation,
        retain_originals=True, isolate_control_originals=True,
    )
    if terminal is not None:
        raise terminal


async def _close_genai_layers(client):
    """One physical owner attempts independent async, sync and pinned pools."""
    terminal = None
    for close in (client.aio.aclose, client.close,
                  getattr(getattr(client, "_kestrel_genai_httpx_owned", None), "aclose", None)):
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except BaseException as error:
            terminal = execution_terminal_error(
                terminal, error, retain_originals=True, isolate_control_originals=True,
            )
    if terminal is not None:
        raise terminal
