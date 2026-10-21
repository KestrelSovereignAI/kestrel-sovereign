"""One local original Task admission fault; never a replacement native owner."""
import asyncio
import inspect

from tests.unit.test_host_context_native_cleanup import _actual_operation


def create_admission_task(task_type, coroutine, *, loop, eager, **kwargs):
    """Select the requested actual constructor without changing lazy semantics."""
    if eager:
        return task_type(coroutine, loop=loop, eager_start=True, **kwargs)
    return task_type(coroutine, loop=loop, **kwargs)


class NativeAdmissionFault:
    def __init__(self, predicate, error, mode, *, eager=False):
        self.predicate, self.error, self.mode, self.eager = predicate, error, mode, eager
        self.accepted, self.refused, self.operations = [], [], []
        self.loop = asyncio.get_running_loop()
        self.previous = self.loop.get_task_factory()

    def factory(self, loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if not self.operations and self.predicate(operation):
            self.operations.append(operation)
            if self.mode == "refuse":
                self.refused.extend((coroutine, operation))
                raise self.error
            fault = self

            class PublicationTask(asyncio.Task):
                failed = False

                def add_done_callback(self, callback, *args, **options):
                    if not self.failed and getattr(callback, "__name__", "") == "close_unstarted":
                        self.failed = True
                        if fault.mode == "publish-after":
                            super().add_done_callback(callback, *args, **options)
                        raise fault.error
                    return super().add_done_callback(callback, *args, **options)

            task = create_admission_task(PublicationTask, coroutine, loop=loop, eager=self.eager, **kwargs)
            self.accepted.append(task)
            return task
        return self.previous(loop, coroutine, **kwargs) if self.previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    def __enter__(self):
        self.loop.set_task_factory(self.factory)
        return self

    def __exit__(self, *_):
        self.loop.set_task_factory(self.previous)

    @property
    def task(self):
        return self.accepted[0] if self.accepted else None

    def assert_terminal_unstarted(self):
        assert len(self.operations) == 1, "fault did not hit the intended original operation"
        assert all(task.done() for task in self.accepted), "accepted original was not joined"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                   for coro in (*self.operations, *self.refused))
