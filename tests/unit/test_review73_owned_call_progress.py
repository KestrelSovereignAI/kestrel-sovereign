"""One lazy source pins one complete call-observation scope, never a mix."""

import contextvars

from kestrel_sovereign.llm.call_progress import (
    bound_attempt_stream,
    call_progress_listener,
    inherit_call_progress,
    report_call_attempt,
    report_call_progress,
)


class _Bound:
    def __init__(self):
        self.calls = []

    def attempt_stream(self, route, provider, model, stream):
        self.calls.append((route, provider, model))
        return stream


def test_lazy_observation_scope_is_pinned_once():
    original_calls, foreign_calls = [], []
    original, foreign = _Bound(), _Bound()
    context = contextvars.Context()
    with call_progress_listener(
        lambda: original_calls.append("progress"),
        lambda provider, model: original_calls.append((provider, model)),
        original,
    ):
        inherit_call_progress(context)
    with call_progress_listener(
        lambda: foreign_calls.append("progress"), attempt_bound=foreign,
    ):
        inherit_call_progress(context)
    context.run(report_call_progress)
    context.run(report_call_attempt, "original", "model")
    stream = object()
    assert context.run(bound_attempt_stream, {"name": "route"}, "original", "model", stream) is stream
    assert original_calls == ["progress", ("original", "model")]
    assert original.calls == [({"name": "route"}, "original", "model")]
    assert foreign_calls == [] and foreign.calls == []


def test_original_progress_only_scope_cannot_borrow_foreign_attempt_bound():
    original_calls = []
    with call_progress_listener(lambda: original_calls.append("progress")):
        context = contextvars.copy_context()
    foreign = _Bound()
    with call_progress_listener(lambda: None, attempt_bound=foreign):
        inherit_call_progress(context)
    context.run(report_call_progress)
    stream = object()
    assert context.run(bound_attempt_stream, {"name": "route"}, "foreign", "model", stream) is stream
    assert original_calls == ["progress"] and foreign.calls == []
