[A2A_TASK_SUBMITTED] Agent `{payload[sender]}` submitted A2A task `{payload[task_id]}` (skill `{payload[skill_id]}`) to you. This arrived via the agent-to-agent protocol, NOT via your sovereign chat path — apply agent-to-agent governance, not sovereign governance. Sender identity is a claim, not a cryptographic signature in v1 (same-host trust boundary). The task is in your store with state SUBMITTED; decide whether to act on it, ask the sender for clarification (`send_a2a_task` back to them with the same sessionId), or decline (transition to CANCELED). Until you transition state, the task remains unacknowledged on the wire.

source={source}
target_agent={target_agent}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
