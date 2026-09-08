"""An adapter whose audit attach raises is skipped, not fatal (#3261).

``KestrelAgent.__init__`` hands every stateful adapter a reference to the
agent through ``attach_agent_for_audit``, "best-effort". The except branch
logged through a ``logger`` name this module never binds, so the first
adapter that raised turned into a NameError out of the constructor and the
agent did not construct.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock


from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


class _RaisingAdapter:
    def attach_agent_for_audit(self, agent):
        raise RuntimeError("audit attach refused")


class _RecordingAdapter:
    def __init__(self):
        self.attached = None

    def attach_agent_for_audit(self, agent):
        self.attached = agent


def _llm_service(*adapters):
    service = MagicMock()
    service.providers = [{"adapter": adapter} for adapter in adapters]
    return service


def test_a_raising_audit_attach_is_logged_and_the_agent_still_constructs(tmp_path, caplog):
    raising, recording = _RaisingAdapter(), _RecordingAdapter()
    with caplog.at_level(logging.DEBUG, logger="kestrel_sovereign.kestrel_agent"):
        agent = KestrelAgent(
            did="did:test:attach-audit",
            storage_path=str(tmp_path / "kestrel.db"),
            privacy_mode=PrivacyMode.ISOLATED,
            llm_service=_llm_service(raising, recording),
        )
    assert agent.did == "did:test:attach-audit"
    # The adapter after the raising one is still attached: the loop went on.
    assert recording.attached is agent
    messages = [r.getMessage() for r in caplog.records if "attach_agent_for_audit failed" in r.getMessage()]
    assert messages == ["attach_agent_for_audit failed on _RaisingAdapter: audit attach refused"]


def test_adapters_without_the_hook_are_skipped_silently(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger="kestrel_sovereign.kestrel_agent"):
        agent = KestrelAgent(
            did="did:test:attach-audit-none",
            storage_path=str(tmp_path / "kestrel.db"),
            privacy_mode=PrivacyMode.ISOLATED,
            llm_service=_llm_service(SimpleNamespace()),
        )
    assert agent.did == "did:test:attach-audit-none"
    assert not any("attach_agent_for_audit" in r.getMessage() for r in caplog.records)
