/** Canonical validation for the host cooperative-Stop response envelope. */

const CONFIRMED = new Set(['stopped', 'already_complete']);
const UNCONFIRMED = new Set(['refused', 'unreachable']);

export function validateHostStopEnvelope(response, expectedCorrelationId = null) {
    if (!response || typeof response !== 'object') return null;
    const outcomes = response.stop_outcomes;
    const correlationId = response.correlation_id;
    if (!Array.isArray(outcomes) || outcomes.length === 0
        || typeof correlationId !== 'string' || correlationId.length === 0
        || (expectedCorrelationId !== null && correlationId !== expectedCorrelationId)
        || !Number.isSafeInteger(response.target_count) || response.target_count < 0
        || !Number.isSafeInteger(response.confirmed_count) || response.confirmed_count < 0
        || !Number.isSafeInteger(response.unconfirmed_count) || response.unconfirmed_count < 0) {
        return null;
    }

    let confirmedCount = 0;
    let unconfirmedCount = 0;
    const receiptIds = new Set();
    const agentIds = new Set();
    const resolvedTargets = new Set();
    for (const outcome of outcomes) {
        if (!outcome || typeof outcome !== 'object'
            || outcome.scope !== 'host'
            || outcome.requested_target !== null
            || typeof outcome.agent_id !== 'string' || outcome.agent_id.trim().length === 0
            || typeof outcome.resolved_target !== 'string' || outcome.resolved_target.trim().length === 0
            || typeof outcome.receipt_id !== 'string' || outcome.receipt_id.trim().length === 0
            || outcome.agent_id !== outcome.resolved_target
            || outcome.correlation_id !== correlationId) {
            return null;
        }
        receiptIds.add(outcome.receipt_id);
        agentIds.add(outcome.agent_id);
        resolvedTargets.add(outcome.resolved_target);
        if (CONFIRMED.has(outcome.disposition)) confirmedCount += 1;
        else if (UNCONFIRMED.has(outcome.disposition)) unconfirmedCount += 1;
        else return null;
    }
    if (receiptIds.size !== 1
        || agentIds.size !== outcomes.length
        || resolvedTargets.size !== outcomes.length
        || response.confirmed_count !== confirmedCount
        || response.unconfirmed_count !== unconfirmedCount
        || confirmedCount + unconfirmedCount !== outcomes.length) {
        return null;
    }

    const empty = response.state === 'empty'
        && response.target_count === 0
        && outcomes.length === 1
        && confirmedCount === 0
        && unconfirmedCount === 1
        && outcomes[0].agent_id === 'host'
        && outcomes[0].resolved_target === 'host';
    if (!empty && response.target_count !== outcomes.length) return null;
    const expectedState = empty
        ? 'empty'
        : (confirmedCount > 0 && unconfirmedCount > 0
            ? 'partial'
            : (unconfirmedCount > 0 ? 'unconfirmed' : 'confirmed'));
    const expectedSuccess = !empty && response.target_count > 0 && unconfirmedCount === 0;
    if (response.state !== expectedState || response.success !== expectedSuccess) return null;

    return { correlationId, outcomes, confirmedCount, unconfirmedCount, empty };
}
