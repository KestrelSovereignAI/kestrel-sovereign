"""Source identity and physical-retirement assertions shared by custody tests."""
import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NoReturn

import aiosqlite
import httpx

from kestrel_sovereign.execution_custody import _explicit_control_chain, _native_source_originals, execution_cleanup_debts


def fail_original_file_read(observed: list[OSError]) -> NoReturn:
    """Retain an actual native read failure and close its original descriptor."""
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        try:
            os.read(descriptor, 1)
        except OSError as error:
            observed.append(error)
            raise
        raise AssertionError("A write-only native descriptor unexpectedly permitted a read")
    finally:
        os.close(descriptor)


class OriginalUploadDescriptor(httpx.AsyncByteStream):
    """An actual caller-owned descriptor and original HTTPX stream, not a mock."""

    def __init__(self, original=None):
        self.original = httpx.ByteStream(b"original") if original is None else original
        self.descriptor = os.open(os.devnull, os.O_RDONLY)
        self.entered = False
        self.closed = False
        self.closers = []

    async def __aiter__(self):
        self.entered = True
        async for chunk in self.original:
            yield chunk

    async def aclose(self):
        os.close(self.descriptor)
        self.closed = True
        self.closers.append(asyncio.current_task())
        await self.original.aclose()


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
        # Owned lexical transactions send COMMIT through execute; standalone
        # backend writes use the driver's commit method. Both must have
        # actually returned from this exact native worker before holding ACK.
        native_commit = function.__name__ == "commit" or (
            function.__name__ == "execute" and args
            and isinstance(args[0], str) and args[0].strip().upper() == "COMMIT"
        )
        if native_commit and selected:
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


@contextmanager
def lost_original_database_commit_ack(monkeypatch, store, *, accept_commit=None):
    """Fail a genuine driver COMMIT acknowledgment after its actual success."""
    originals = []
    with monkeypatch.context() as patch:
        if store.is_sqlite:
            import aiosqlite
            native_execute = aiosqlite.Connection._execute

            async def lost_sqlite_ack(connection, function, *args, **kwargs):
                result = await native_execute(connection, function, *args, **kwargs)
                committed = function.__name__ == 'commit' or (
                    function.__name__ == 'execute' and args
                    and isinstance(args[0], str) and args[0].strip().upper() == 'COMMIT')
                if not originals and committed and (accept_commit is None or await accept_commit()):
                    fail_original_file_read(originals)
                return result

            patch.setattr(aiosqlite.Connection, '_execute', lost_sqlite_ack)
        else:
            import asyncpg
            native_exit = asyncpg.transaction.Transaction.__aexit__

            async def lost_postgres_ack(transaction, error_type, error, traceback):
                result = await native_exit(transaction, error_type, error, traceback)
                if (error_type is None and not transaction._nested and not originals
                        and (accept_commit is None or await accept_commit())):
                    fail_original_file_read(originals)
                return result

            patch.setattr(asyncpg.transaction.Transaction, '__aexit__', lost_postgres_ack)
        yield originals
