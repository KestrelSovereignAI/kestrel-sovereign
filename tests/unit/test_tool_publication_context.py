"""A private publisher receives only its live host-selected chat destination."""

import asyncio
import copy
import pickle
from types import SimpleNamespace

import pytest

from kestrel_sdk.tools import ToolPublicationContext
from kestrel_sovereign.agent.parts import (
    bind_part_collector,
    current_part_collector,
    current_tool_publication_context,
    current_tool_publication_epoch,
    part_collector,
)


def test_publication_epoch_stays_with_original_live_turn_and_cannot_be_copied():
    epoch = ["normal:original:0"]
    host = SimpleNamespace(did="did:test:chat-owner", _get_privacy_policy_epoch=lambda: epoch[0])
    with part_collector(host=host, session_id="chat-1"):
        collector = current_part_collector()
        assert current_tool_publication_epoch(host, "chat-1") == epoch[0]
        epoch[0] = "normal:original:2"
        assert current_tool_publication_epoch(host, "chat-1") == "normal:original:0"
        assert current_tool_publication_epoch(host, "chat-2") is None
        assert current_tool_publication_epoch(SimpleNamespace(did=host.did), "chat-1") is None
        with bind_part_collector(copy.copy(collector)):
            assert current_tool_publication_epoch(host, "chat-1") is None
    with bind_part_collector(collector):
        assert current_tool_publication_epoch(host, "chat-1") is None
    with part_collector(host=SimpleNamespace(did=host.did), session_id="chat-1"):
        assert current_tool_publication_epoch(host, "chat-1") is None


@pytest.mark.parametrize("epoch", [None, "", " ", 1, False])
def test_invalid_host_epoch_is_refused(epoch):
    host = SimpleNamespace(did="did:test:chat-owner", _get_privacy_policy_epoch=lambda: epoch)
    with pytest.raises(ValueError, match="original privacy policy epoch"):
        with part_collector(host=host, session_id="chat-1"):
            pytest.fail("invalid epoch was accepted")


def test_publication_destination_requires_exact_owner_and_session():
    host = SimpleNamespace(did="did:test:chat-owner")
    other = SimpleNamespace(did=host.did)
    assert current_tool_publication_context(host, "chat-1") is None
    with part_collector(host=host, session_id="chat-1"):
        assert current_tool_publication_context(host, "chat-1") == ToolPublicationContext(host.did, "chat-1")
        assert current_tool_publication_context(other, "chat-1") is None
        assert current_tool_publication_context(host, "chat-2") is None
        host._active_session_id = "chat-2"
        assert current_tool_publication_context(host, "chat-1").session_id == "chat-1"
        with bind_part_collector([]):
            assert current_tool_publication_context(host, "chat-1") is None
        with part_collector(host=other, session_id="chat-2"):
            assert current_tool_publication_context(other, "chat-2").session_id == "chat-2"
            assert current_tool_publication_context(host, "chat-1") is None
        assert current_tool_publication_context(host, "chat-1").session_id == "chat-1"


def test_generic_chat_host_does_not_invent_a_privacy_interface_from_dynamic_attributes():
    class GenericHost:
        did = "did:test:generic-chat"
        def __getattr__(self, name):
            if name == "_get_privacy_policy_epoch":
                raise AssertionError("generic chat queried an undeclared privacy interface")
            raise AttributeError(name)
    host = GenericHost()
    with part_collector(host=host, session_id="chat-1"):
        assert current_tool_publication_epoch(host, "chat-1") is None
        assert current_tool_publication_context(host, "chat-1") == ToolPublicationContext(host.did, "chat-1")


@pytest.mark.parametrize("session_id", [None, "", " ", "original", "orchestrator"])
def test_sessionless_collector_does_not_invent_a_chat_destination(session_id):
    host = SimpleNamespace(did="did:test:chat-owner",
        _get_privacy_policy_epoch=lambda: pytest.fail("sessionless turn queried private publication policy"))
    with part_collector(host=host, session_id=session_id):
        assert current_tool_publication_context(host, session_id) is None
    with part_collector(session_id="real-chat"):
        assert current_tool_publication_context(host, "real-chat") is None


@pytest.mark.parametrize("transform", [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
                         ids=["copy", "deepcopy", "pickle"])
def test_copied_part_data_cannot_copy_or_extend_a_chat_destination(transform):
    host = SimpleNamespace(did="did:test:chat-owner")
    with part_collector(host=host, session_id="chat-1"):
        collector = current_part_collector()
        collector.append({"type": "notice", "data": {"value": "public-part"}})
        copied = transform(collector)
        assert type(copied) is list
        assert copied == list(collector)
        with bind_part_collector(copied):
            assert current_tool_publication_context(host, "chat-1") is None
    with bind_part_collector(copied):
        assert current_tool_publication_context(host, "chat-1") is None


@pytest.mark.asyncio
async def test_copied_and_borrowed_turn_context_expires_on_owning_scope_exit():
    host = SimpleNamespace(did="did:test:chat-owner")
    observed = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def foreign_task():
        observed.append(current_tool_publication_context(host, "chat-1"))
        entered.set()
        await release.wait()
        observed.append(current_tool_publication_context(host, "chat-1"))

    with part_collector(host=host, session_id="chat-1"):
        borrowed = current_part_collector()
        task = asyncio.create_task(foreign_task())
        await entered.wait()
    release.set()
    await task
    assert observed == [ToolPublicationContext(host.did, "chat-1"), None]
    with bind_part_collector(borrowed):
        assert current_tool_publication_context(host, "chat-1") is None


@pytest.mark.asyncio
async def test_concurrent_chat_destinations_on_the_same_host_stay_separate():
    host = SimpleNamespace(did="did:test:chat-owner")
    entered = [asyncio.Event(), asyncio.Event()]
    observed = []

    async def turn(index):
        session = f"chat-{index}"
        with part_collector(host=host, session_id=session):
            entered[index].set()
            await entered[1 - index].wait()
            observed.append(current_tool_publication_context(host, session))
            assert current_tool_publication_context(host, f"chat-{1 - index}") is None

    await asyncio.gather(turn(0), turn(1))
    assert set(observed) == {ToolPublicationContext(host.did, "chat-0"), ToolPublicationContext(host.did, "chat-1")}
    assert current_tool_publication_context(host, "chat-0") is None
