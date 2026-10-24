"""Source identity and physical-retirement assertions shared by custody tests."""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass

import aiosqlite

from kestrel_sovereign.execution_custody import _explicit_control_chain, _native_source_originals, execution_cleanup_debts


@dataclass(frozen=True)
class OriginalSQLiteCommitAck:
    entered: asyncio.Event
    release: asyncio.Event


@contextmanager
def hold_original_sqlite_commit_ack(monkeypatch, owns_commit):
    """Hold a selected real worker's successful COMMIT reply, never its SQL."""
    acknowledgement = OriginalSQLiteCommitAck(asyncio.Event(), asyncio.Event())
    original = aiosqlite.Connection._execute

    async def hold(connection, function, *args, **kwargs):
        result = await original(connection, function, *args, **kwargs)
        # Selection may observe the real INSERT before choosing its owner's
        # COMMIT; earlier schema/initialization commits are not that receipt.
        selected = owns_commit(connection, function, args)
        if function.__name__ == "commit" and selected:
            acknowledgement.entered.set()
            await acknowledgement.release.wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(aiosqlite.Connection, "_execute", hold)
        try:
            yield acknowledgement
        finally:
            acknowledgement.release.set()


@dataclass(frozen=True)
class OriginalFailure:
    error: BaseException
    chain: tuple
    debts: tuple

    @classmethod
    def capture(cls, error):
        return cls(error, tuple(_explicit_control_chain(error)), tuple(execution_cleanup_debts(error)))

    @classmethod
    def capture_sources(cls, terminal):
        """Snapshot genuine sources, not a replaceable canonical finite envelope."""
        sources = _native_source_originals(terminal)
        assert sources, "terminal does not retain a genuine native source"
        return tuple(cls.capture(source) for source in sources)

    def assert_retained(self, terminal, *, resource=None, task=None):
        assert any(error is self.error for error in _explicit_control_chain(terminal)), "original failure disappeared from finite retirement evidence"
        assert tuple(_explicit_control_chain(self.error)) == self.chain, "retirement rewrote the original source error graph"
        assert tuple(execution_cleanup_debts(self.error)) == self.debts, "private retirement debt was appended to the shared original source"
        if resource is not None:
            assert any(native is resource and original is not None and original.done()
                       and (task is None or original is task)
                       for native, original in execution_cleanup_debts(terminal)), "retirement lost its exact native resource/original settled Task"
