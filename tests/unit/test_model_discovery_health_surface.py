"""A dead model-discovery credential must be visible on the health surface (#3190).

Before this, a 401 on ``GET /v1/models`` produced exactly one WARNING line in a
22-million-line host log. Nothing on ``/health/detailed`` said anything, so the
condition — and the pinned-model loss it causes at the next restart — was
undetectable in practice. It ran for two boots across four agents unnoticed.

The asymmetry that makes this worth its own check: discovery and chat use
different endpoints and, on plan/OAuth routes, different credentials. A vendor
can serve chat perfectly while its catalog call 401s, so no existing
reachability or provider check notices.
"""

from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.health.checks import (
    check_llm_service,
    check_model_discovery,
    derive_overall_status,
    worst_status,
)


def _agent(*, failures=None, pinned_vendor=None, providers=None, load_error=None):
    llm = MagicMock()
    llm._discovery_failures = failures if failures is not None else {}
    llm._mandate_load_error = load_error
    llm.providers = providers if providers is not None else [
        {"name": "anthropic:plan", "vendor": "anthropic"},
    ]
    llm.get_model_preference = MagicMock(
        return_value={"vendor": pinned_vendor, "model": "claude-opus-5", "route": "plan"}
    )
    llm.get_active_model_id = MagicMock(return_value="claude-opus-5")
    llm.reachability = None
    agent = MagicMock()
    agent.llm_service = llm
    return agent


class TestModelDiscoveryCheck:
    @pytest.mark.asyncio
    async def test_passes_when_every_vendor_discovered(self):
        result = await check_model_discovery(_agent())
        assert result["status"] == "pass"

    @pytest.mark.asyncio
    async def test_warns_and_names_the_failing_vendor(self):
        result = await check_model_discovery(
            _agent(failures={"anthropic": "AuthenticationError: 401 Unauthorized"})
        )
        assert result["status"] == "warn"
        assert "anthropic" in result["message"]
        assert result["details"]["failed_vendors"]["anthropic"].endswith("401 Unauthorized")

    @pytest.mark.asyncio
    async def test_fails_when_the_agents_own_pinned_vendor_is_the_broken_one(self):
        """The exact 2026-08-31 condition: the pin about to be discarded."""
        result = await check_model_discovery(
            _agent(
                failures={"anthropic": "AuthenticationError: 401 Unauthorized"},
                pinned_vendor="anthropic",
            )
        )
        assert result["status"] == "fail"
        assert result["details"]["pinned_vendor_at_risk"] is True
        assert "discarded on the next restart" in result["message"]

    @pytest.mark.asyncio
    async def test_a_failure_in_an_unpinned_vendor_is_only_a_warning(self):
        result = await check_model_discovery(
            _agent(failures={"xai": "timeout"}, pinned_vendor="anthropic")
        )
        assert result["status"] == "warn"
        assert result["details"]["pinned_vendor_at_risk"] is False

    @pytest.mark.asyncio
    async def test_reports_degraded_not_unhealthy(self):
        """A catalog problem must not take the whole host to unhealthy."""
        result = await check_model_discovery(
            _agent(failures={"anthropic": "401"}, pinned_vendor="anthropic")
        )
        assert result["status"] == "fail"
        assert derive_overall_status([result]) == "degraded"


class TestLlmServiceSurfacesDroppedMandate:
    @pytest.mark.asyncio
    async def test_dropped_mandate_is_reported(self):
        result = await check_llm_service(
            _agent(load_error="Cannot set model 'claude-opus-5' on 'anthropic:plan'")
        )
        assert result["status"] == "warn"
        assert "UNPINNED" in result["message"]
        assert "claude-opus-5" in result["details"]["mandate_load_error"]

    @pytest.mark.asyncio
    async def test_no_error_reports_pass(self):
        result = await check_llm_service(_agent())
        assert result["status"] == "pass"
        assert "mandate_load_error" not in result["details"]

    @pytest.mark.asyncio
    async def test_unreachable_route_is_not_downgraded_by_a_clean_mandate(self):
        """Severity writes must escalate only, never overwrite a worse verdict."""
        agent = _agent()
        agent.llm_service.reachability = [
            {"name": "anthropic:plan", "status": "unreachable"}
        ]
        result = await check_llm_service(agent)
        assert result["status"] == "warn"


class TestRegisteredInTheSharedList:
    @pytest.mark.asyncio
    async def test_model_discovery_runs_in_run_standard_checks(self):
        """Both health surfaces share one list; a check absent from it is invisible."""
        import inspect

        from kestrel_sovereign.features.health import checks as mod

        src = inspect.getsource(mod.run_standard_checks)
        assert "check_model_discovery(agent)" in src


class TestWorstStatus:
    """A check whose status has two writers must not let the later one downgrade.

    This is the mechanism behind ``check_llm_service``'s severity. Testing it
    directly is what makes the guard falsifiable: today both of that check's
    conditions happen to yield ``warn``, so a plain assignment would be
    indistinguishable at the call site and the protection would be untested.
    """

    def test_keeps_the_worse_of_two(self):
        assert worst_status("pass", "warn") == "warn"
        assert worst_status("warn", "pass") == "warn"

    def test_a_later_milder_verdict_cannot_downgrade(self):
        assert worst_status("fail", "warn") == "fail"
        assert worst_status("fail", "pass") == "fail"

    def test_all_pass_stays_pass(self):
        assert worst_status("pass", "pass") == "pass"

    def test_unknown_status_cannot_manufacture_a_failure(self):
        assert worst_status("pass", "bogus") == "pass"
        assert worst_status("warn", "bogus") == "warn"
