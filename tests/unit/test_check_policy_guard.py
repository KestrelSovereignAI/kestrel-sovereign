"""Reflection test for the LLMService._check_policy() guard.

Phase 3b of the PayerPolicy foundation work.

Plan v11 §"LLM path" requires that EVERY public generation entry
point on `LLMService` (and the mixins it composes) calls
`_check_policy()` at the top, so that an agent with policy
`llm.kind = NONE` (which sets `LLMService.disabled = True`) is
guaranteed to raise `PolicyDeniedError` rather than silently fall
through to a shared host key.

The risk this test guards against: future commits that add a new
generation method (especially streaming, which is in a mixin) and
forget the guard. Without an automated check, the new method becomes
a silent bypass of the NONE policy.

The reflection sweep walks `LLMService` AND its mixin bases (so
`StreamingMixin`'s methods are caught), collects every method that
is `inspect.iscoroutinefunction` OR `inspect.isasyncgenfunction`
AND whose name matches a generation-entry pattern, then asserts each
collected method calls `_check_policy()` somewhere in its source.

Both predicate halves are critical:
- Without the name filter, we'd pull in management/discovery methods
  (`discover_all_models`, `get_storage_info`, `pull_model`,
  `use_agent_key`, `close`) that are correctly NOT policy-gated.
- Without `isasyncgenfunction`, the streaming entry points
  (`generate_stream`, `get_streaming_response`,
  `stream_with_messages`, `stream_with_tool_detection`) — which are
  async generators — would silently slip past the sweep.
"""
from __future__ import annotations

import inspect
import re
import textwrap

import pytest

from kestrel_sovereign.llm.service import (
    LLMService,
    PolicyDeniedError,
)


# Generation-entry name patterns. New methods on LLMService that
# generate text (or stream tool calls) MUST land in one of these.
# Adding a new pattern is a deliberate plan-side decision; the test
# catches missed guards under the EXISTING patterns.
_GENERATION_PATTERNS = (
    re.compile(r"^generate(_|$)"),
    re.compile(r"^get_response($|_)"),
    re.compile(r"^get_audit_response$"),
    re.compile(r"^stream_"),
    re.compile(r"_streaming_response$"),
)


def _is_generation_name(name: str) -> bool:
    return any(p.search(name) for p in _GENERATION_PATTERNS)


def _walk_methods(cls: type):
    """Yield (name, method, defining-class) across cls and its MRO,
    skipping `object`. Includes mixin bases so StreamingMixin methods
    are caught.
    """
    seen: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            continue
        for name, member in klass.__dict__.items():
            if name in seen:
                continue
            if not callable(member):
                continue
            if name.startswith("_"):
                continue
            seen.add(name)
            yield name, member, klass


def _is_generation_method(member) -> bool:
    return inspect.iscoroutinefunction(member) or inspect.isasyncgenfunction(member)


def _collect_generation_entry_points():
    found = []
    for name, member, klass in _walk_methods(LLMService):
        if not _is_generation_method(member):
            continue
        if not _is_generation_name(name):
            continue
        found.append((name, member, klass))
    return found


class TestPolicyGuardCoverage:
    def test_collects_at_least_the_known_entry_points(self) -> None:
        """The sweep should find AT LEAST these names today. If a future
        refactor renames or splits one, this assertion drives the test
        author to update both the sweep predicate and the production
        guards together."""
        found_names = {name for name, _, _ in _collect_generation_entry_points()}
        expected_subset = {
            "get_audit_response",
            "get_response",
            "get_response_with_model",
            "generate",
            "generate_with_messages",
            "get_streaming_response",
            "generate_stream",
            "stream_with_messages",
            "stream_with_tool_detection",
        }
        missing = expected_subset - found_names
        assert not missing, (
            f"Reflection sweep missed known generation entry points: {missing}. "
            "If you renamed/removed one of these, update _GENERATION_PATTERNS."
        )

    def test_every_collected_method_calls_check_policy(self) -> None:
        """The load-bearing assertion: each collected method's source
        contains a call to `self._check_policy()`. Catches the future
        bug where a new entry point is added without the guard.
        """
        offenders = []
        for name, member, klass in _collect_generation_entry_points():
            try:
                source = inspect.getsource(member)
            except (TypeError, OSError):
                # If we can't read source, we can't verify — record and fail.
                offenders.append((name, klass.__name__, "<source unavailable>"))
                continue
            if "self._check_policy()" not in source:
                offenders.append(
                    (name, klass.__name__, textwrap.shorten(source, width=80))
                )
        assert not offenders, (
            "The following LLMService generation entry points do not call "
            "self._check_policy() at the top of their body. Each one is a "
            "silent bypass of `llm.kind = NONE`:\n  "
            + "\n  ".join(f"{c}.{n}" for n, c, _ in offenders)
        )


class TestCheckPolicyBehavior:
    """Direct unit tests of `_check_policy` without exercising every
    generation method — the reflection test above asserts coverage.
    """

    def test_disabled_false_does_not_raise(self) -> None:
        svc = LLMService()
        assert svc.disabled is False
        # No exception.
        svc._check_policy()

    def test_disabled_true_raises_policy_denied(self) -> None:
        svc = LLMService()
        svc.disabled = True
        with pytest.raises(PolicyDeniedError):
            svc._check_policy()

    def test_policy_denied_is_llm_service_error(self) -> None:
        from kestrel_sovereign.llm.service import LLMServiceError
        assert issubclass(PolicyDeniedError, LLMServiceError)
