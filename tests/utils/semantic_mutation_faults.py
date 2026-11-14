"""Fault the original semantic revision boundary after real predecessor writes."""

from contextlib import contextmanager

from kestrel_sovereign.knowledge import AssertionStatus
from kestrel_sovereign.storage.async_assertion_store import AssertionConflictError


@contextmanager
def immutable_revision_conflict(monkeypatch, store, *, object_value):
    """Refuse one exact store's replacement revision without replacing storage."""
    original = type(store)._write_revision
    conflicts = []

    async def collide(self, assertion, sources):
        if self is store and assertion.status is AssertionStatus.ACTIVE and assertion.object.value == object_value:
            conflicts.append(assertion.revision_id)
            raise AssertionConflictError("revision_id is immutable and already stores a different assertion")
        return await original(self, assertion, sources)

    with monkeypatch.context() as patch:
        patch.setattr(type(store), "_write_revision", collide)
        yield conflicts
