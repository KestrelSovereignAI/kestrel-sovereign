"""One bounded retrieval hook and the physical lifetime of actual threads."""

import asyncio
import threading
from functools import partial
from types import SimpleNamespace

import cbor2
import pytest

from kestrel_sovereign._async_ownership import run_blocking_operation
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
)
from kestrel_sovereign.storage.car_builder import compute_raw_cid
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter


def adapter(storage, adapter_type=SovereignStorageAdapter):
    return adapter_type(SimpleNamespace(backend_type="sqlite"), "owned-secret", storage)


def external_link():
    raw, cid = compute_raw_cid(b"owned external payload; not GPU evidence")
    return cbor2.CBORTag(42, b"\0" + raw), cid


@pytest.mark.asyncio
async def test_external_ref_honors_the_canonical_subclass_bounded_retrieval_hook():
    calls = []

    class Storage:
        def retrieve_content(self, content_hash, **kwargs):
            calls.append((content_hash, kwargs))
            assert kwargs["max_output_bytes"] == 8
            assert kwargs["allow_local_cache"] is False
            return b"bounded"

    class Bounded(SovereignStorageAdapter):
        async def _download_content(self, cid):
            return await run_blocking_operation(partial(
                self.adapter.retrieve_content, cid, ipfs_cid=cid,
                max_output_bytes=8, allow_local_cache=False,
            ))

    link, cid = external_link()
    assert await adapter(Storage(), Bounded)._fetch_external_ref_bytes(link) == (
        b"bounded", "external_ref_fetched"
    )
    assert calls == [(cid, {"ipfs_cid": cid, "max_output_bytes": 8, "allow_local_cache": False})]


@pytest.mark.asyncio
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_repeated_cancellation_cannot_leave_the_original_download_thread(external, fails):
    entered, finish, exited = threading.Event(), threading.Event(), threading.Event()

    class Storage:
        def retrieve_content(self, **kwargs):
            entered.set()
            try:
                assert finish.wait(5), "test must release its actual download thread"
                if fails:
                    raise OSError("owned transfer failed")
                return b"owned bytes"
            finally:
                exited.set()

    instance = adapter(Storage())
    link, cid = external_link()
    operation = instance._fetch_external_ref_bytes(link) if external else instance._download_content(cid)
    task = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not exited.is_set()
        finish.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert exited.is_set()
        if fails:
            assert isinstance(caught.value.__cause__, OSError)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("outcome", ["denied", "committed", "unknown"])
async def test_original_execution_control_is_not_an_optional_skip_or_fetch_wrapper(external, outcome):
    original = ExecutionAuthorityError("original denied") if outcome == "denied" else ExecutionCommitOutcomeError(outcome)

    class Storage:
        def retrieve_content(self, **kwargs):
            raise original

    instance = adapter(Storage())
    link, cid = external_link()
    with pytest.raises(type(original)) as caught:
        if external:
            await instance._fetch_external_ref_bytes(link)
        else:
            await instance.import_agent(cid)
    assert caught.value is original


@pytest.mark.asyncio
async def test_ordinary_external_fetch_failure_remains_an_explicit_optional_skip():
    class Storage:
        def retrieve_content(self, **kwargs):
            raise OSError("owned offline fixture")

    payload, reason = await adapter(Storage())._fetch_external_ref_bytes(external_link()[0])
    assert payload is None and reason == "external_ref_fetch_failed: owned offline fixture"
