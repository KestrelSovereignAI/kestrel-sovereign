"""Native App Task results remain bound to the original physical View."""

import pytest

from kestrel_sovereign.auth import caller_identity_digest
from kestrel_sovereign.mcp_task_submission import MCPTaskDelivery, MCPTaskInteractiveOrigin, MCPTaskSubmission
from tests.integration.test_mcp_apps_host import apps_boundary as apps_boundary
from tests.integration.test_mcp_task_host import task_boundary as task_boundary
from tests.shared.mcp_task_custody import attach, binding, observation, submission


@pytest.mark.mcp_integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize('apps_boundary', [
    {'operator_input': 'form', 'input_tool': 'mutate_marker'}], indirect=True)
async def test_task_result_disclosure_requires_the_original_physical_app_view(task_boundary):
    b, host = task_boundary
    publication = await b.publish()
    first = await b.open(publication)
    second = await b.open(publication)
    assert first.status_code == second.status_code == 200
    view = first.json()['view_id']
    replacement = second.json()['view_id']
    assert replacement != view

    physical = b.host._views[view]
    origin = MCPTaskInteractiveOrigin(caller_identity_digest(physical.owner.caller), physical.session_id)
    original = submission(origin=origin)
    delivery = MCPTaskDelivery('app', physical.session_id, view)
    sealed = MCPTaskSubmission.seal(arguments=original.arguments, contract=original.contract,
        authorization=original.authorization, delivery=delivery)
    reservation = await host.store.reserve(binding(agent_did=b.agent.did, origin=origin), sealed)
    attached = await attach(host.store, reservation.issuance)
    result = observation('completed')
    retained = await host.store.observe(attached, result)
    reference = host.results.reference(retained, delivery)

    def path(view_id):
        return f'/api/mcp/apps/views/{view_id}/tasks/{retained.handle}'

    parameters = {'session_id': physical.session_id}
    headers = b.headers() | {'X-Kestrel-Task-Result': reference}
    permitted = await b.client.get(path(view), params=parameters, headers=headers)
    assert permitted.status_code == 200
    assert permitted.json() == {'status': 'completed', 'result': result['result']}
    attempts = [
        await b.client.get(path(replacement), params=parameters, headers=headers),
        await b.client.get(path(view), params=parameters, headers=b.headers()),
        await b.client.get(path(view), params=parameters, headers=b.headers('bob@apps.test') |
            {'X-Kestrel-Task-Result': reference}),
    ]
    assert [refused.status_code for refused in attempts] == [403, 403, 404]
    for refused in attempts:
        assert 'private-terminal-content' not in refused.text

    closed = await b.client.delete(f'/api/mcp/apps/views/{view}', params=parameters, headers=b.headers())
    assert closed.status_code < 300
    after_close = await b.client.get(path(view), params=parameters, headers=headers)
    assert after_close.status_code == 404
