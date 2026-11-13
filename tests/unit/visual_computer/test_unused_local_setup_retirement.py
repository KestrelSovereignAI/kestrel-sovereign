"""Unused issued setup follows the existing encrypted no-acquisition CAS."""
import pytest

from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_local_two_domain_admit import (
    local_setup as local_setup,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal", ["grant", "generation", "expiry", "privacy", "key"])
async def test_unused_issued_setup_retires_original_unacquired_row(
    local_setup, monkeypatch, withdrawal,
):
    setup = local_setup
    issue = setup.host._issue_local_setup
    issued, native_entries = [], []

    def withdraw_after_issue(admission):
        original = issue(admission)
        issued.append(original)
        if withdrawal == "grant":
            setup.feature._cfg["granted_actions"].remove(VisualAction.ADMIT.value)
        elif withdrawal == "generation":
            setup.feature._generation += 1
        elif withdrawal == "privacy":
            setup.agent.set_mode(PrivacyMode.ISOLATED)
        elif withdrawal == "key":
            monkeypatch.setenv("KESTREL_API_KEY", "withdrawn-source-test-key")
        else:
            monkeypatch.setattr(setup.runtime.authority, "clock", lambda: admission._context.expires_at)
        return original

    async def forbidden_native_entry(original):
        native_entries.append(original)
        pytest.fail("an unused refused setup entered resource acquisition")

    monkeypatch.setattr(setup.host, "_issue_local_setup", withdraw_after_issue)
    monkeypatch.setattr(setup.host, "_create_original_resources", forbidden_native_entry)
    async with setup.bound() as context:
        request = setup.request(context)
        with pytest.raises(OperatorAuthorizationError):
            await setup.runtime.admit(context, request)
    original, = issued
    assert original._used is False and original._resources is None
    assert native_entries == [] and setup.runtime.egress._calls == {}
    # Restore controlled caller state for inspection. API-key withdrawal is
    # not loss of the repository's original encryption key. The original
    # ADMIT holds its persistence lease; actual privacy transitions wait for
    # that lease. Neither refusal forbids its same-row no-acquisition CAS.
    if withdrawal == "privacy":
        setup.agent.set_mode(PrivacyMode.NORMAL)
    elif withdrawal == "key":
        monkeypatch.setenv("KESTREL_API_KEY", setup.key)
    saved = await setup.runtime.repository.get(request.scope, request_id=request.request_id)
    assert saved.lease is None and saved.connection is None and saved.session is None
    assert saved.phase == "retired", "an unused issued setup kept its no-acquisition row charged"
    assert saved.retired_from == "quoted" and saved.retired_at is not None
    assert saved.request == request and saved.session_id == original._record.session_id
