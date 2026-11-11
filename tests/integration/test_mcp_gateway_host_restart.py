"""Native Docker Gateway keeps governance across actual Core host processes."""
import sys
import pytest
from tests.integration.test_mcp_permission_host_restart import (
    host_campaign as host_campaign, exercise_permission_restart, exercise_approval_restart,
)

pytestmark = [pytest.mark.mcp_integration, pytest.mark.docker_mcp, pytest.mark.timeout(120),
    pytest.mark.skipif(sys.platform == 'win32', reason='Native host restart requires SIGTERM')]

@pytest.mark.parametrize('host_campaign', [{'transport': 'gateway'}], indirect=True)
async def test_native_gateway_host_restores_permission_before_another_effect(host_campaign):
    await exercise_permission_restart(host_campaign)
    host_campaign.peer.assert_restart_observed()

@pytest.mark.parametrize('host_campaign', [{'transport': 'gateway'}], indirect=True)
async def test_native_gateway_host_rejects_old_and_replayed_approvals(host_campaign):
    await exercise_approval_restart(host_campaign)
    host_campaign.peer.assert_restart_observed()
