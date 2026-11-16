"""Physical retirement of the candidate tests' original usage workers."""

import threading
from contextlib import ExitStack

import pytest

from tests.unit import test_candidate_attempt_bounds as bounds
from tests.utils.aiosqlite_workers import (
    CloseStage,
    aiosqlite_worker,
    delay_aiosqlite_worker_exit,
    observe_bounded_close_stages,
)
from tests.utils.virtual_clock import HANG_GUARD_SECONDS


def test_candidate_fixture_runs_original_worker_poll_without_clock_rescue():
    release, delayed = threading.Event(), threading.Event()
    stop_rescue, rescued = threading.Event(), threading.Event()
    targets, poll_witness = [], []

    with pytest.MonkeyPatch.context() as monkeypatch, ExitStack() as owners:
        fixture = bounds.loop.__wrapped__(monkeypatch)
        owners.callback(fixture.close)
        owners.callback(release.set)
        loop = next(fixture)
        with delay_aiosqlite_worker_exit(
            release, delayed, should_delay=lambda worker: worker in targets,
        ):
            service = bounds._service([], config={})
            loop.run_until_complete(service._ensure_db_initialized())
            backend = service._usage_db.backend
            connection = backend._connection
            worker = aiosqlite_worker(connection)
            targets.append(worker)

            def rescue_original_worker():
                if stop_rescue.wait(HANG_GUARD_SECONDS):
                    return
                # A failed regression must still retire this SAME original
                # worker and let its SAME polling owner finish before closing
                # the loop. This is never the successful cleanup path.
                rescued.set()
                release.set()
                worker.join(HANG_GUARD_SECONDS)
                loop.call_soon_threadsafe(loop.advance, 1.0)

            rescuer = threading.Thread(target=rescue_original_worker)
            rescuer.start()
            try:
                with observe_bounded_close_stages() as stages:
                    def release_after_original_poll():
                        if stages.has_reached(connection, CloseStage.WORKER_EXIT):
                            poll_witness.append(worker.is_alive())
                            release.set()
                        else:
                            loop.call_later(0.01, release_after_original_poll)

                    loop.call_later(0.01, release_after_original_poll)
                    fixture.close()
            finally:
                stop_rescue.set()
                release.set()
                rescuer.join(HANG_GUARD_SECONDS)
                worker.join(HANG_GUARD_SECONDS)
            assert not rescuer.is_alive() and not worker.is_alive()
            assert service._usage_db is None and not backend.is_connected
            assert delayed.is_set(), "the actual original worker never reached its exit gate"
            assert not rescued.is_set(), "the original worker poll required manual virtual-clock rescue"
            assert poll_witness == [True], "the loop must observe and release the still-live original worker"
