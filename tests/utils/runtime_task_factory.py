"""Finite native Task publication refusal, shared by SQL and context tests."""

import asyncio


def refusing_runtime_completion_factory(opening):
    class RefusingTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if (
                self.get_name().startswith("durable_cognition_owner:")
                and callback.__qualname__.endswith(
                    "KestrelAgent._track_runtime_task.<locals>.completed"
                )
            ):
                raise opening
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **kwargs):
        return RefusingTask(coroutine, loop=loop, **kwargs)

    return factory
