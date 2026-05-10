"""Unit tests for LLMService.attach_to_agent.

Phase 2 of the PayerPolicy foundation work.

Enforces the per-agent LLMService invariant: each KestrelAgent must hold
its own service instance, because `use_agent_key()` mutates
`self.providers` in place. Sharing a service across agents would let
the last-loaded agent silently steal the others' OpenRouter clients.

Production code (`kestrel_sovereign/multi_agent/agent_manager.py:90-91`)
already constructs one service per agent. These tests pin the
invariant at construction time so it cannot regress.
"""
from __future__ import annotations

import pytest

from kestrel_sovereign.llm.service import (
    LLMService,
    LLMServiceAlreadyAttachedError,
)


class TestAttachToAgent:
    def test_attach_records_owner(self) -> None:
        svc = LLMService()
        svc.attach_to_agent("did:test:agent-a")
        # Internal accessor is fine for a regression-pin test.
        assert svc._owner_agent_did == "did:test:agent-a"

    def test_attach_rejects_empty_did(self) -> None:
        svc = LLMService()
        with pytest.raises(ValueError):
            svc.attach_to_agent("")

    def test_repeated_attach_same_did_is_idempotent(self) -> None:
        svc = LLMService()
        svc.attach_to_agent("did:test:agent-a")
        # No exception; owner unchanged.
        svc.attach_to_agent("did:test:agent-a")
        assert svc._owner_agent_did == "did:test:agent-a"

    def test_second_attach_different_did_raises(self) -> None:
        svc = LLMService()
        svc.attach_to_agent("did:test:agent-a")
        with pytest.raises(LLMServiceAlreadyAttachedError) as excinfo:
            svc.attach_to_agent("did:test:agent-b")
        # The error mentions both DIDs so debuggers can see who took it.
        msg = str(excinfo.value)
        assert "did:test:agent-a" in msg
        assert "did:test:agent-b" in msg

    def test_distinct_services_can_each_attach(self) -> None:
        # Two services, two agents — the supported pattern.
        svc_a = LLMService()
        svc_b = LLMService()
        svc_a.attach_to_agent("did:test:agent-a")
        svc_b.attach_to_agent("did:test:agent-b")
        assert svc_a._owner_agent_did == "did:test:agent-a"
        assert svc_b._owner_agent_did == "did:test:agent-b"

    def test_already_attached_error_is_llm_service_error(self) -> None:
        # Catchable via the LLMService base error class.
        from kestrel_sovereign.llm.service import LLMServiceError

        assert issubclass(LLMServiceAlreadyAttachedError, LLMServiceError)
