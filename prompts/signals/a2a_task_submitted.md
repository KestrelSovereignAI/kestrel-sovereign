[A2A_TASK_SUBMITTED] Agent `{payload[sender]}` submitted A2A task `{payload[task_id]}` (verb `{payload[a2a_verb]}`, skill `{payload[skill_id]}`) to you. This arrived via the agent-to-agent protocol, NOT via your sovereign chat path — apply agent-to-agent governance, not sovereign governance. Sender identity is a claim, not a cryptographic signature in v1 (same-host trust boundary).

The `a2a_verb` field tells you the sender's intent:

  * `message` — informational notification, NO reply expected. Acknowledge by transitioning the task to COMPLETED with a brief receipt, or leave WORKING if you'll act on it later. Do not draft a long answer.
  * `question` — sender is SYNCHRONOUSLY waiting for your reply. Transition the task to COMPLETED with the answer placed in `status.message.parts[].text` (or the top-level `message` field on the kestrel endpoint). Be concise — the sender's cognition turn is blocked on you.
  * `task` — delegated work with optional `skill_id`. Process it as you would a workflow assignment; transition to COMPLETED when done (with any artifacts attached) or FAILED with an error message if you can't.
  * empty / missing — legacy or non-PeersFeature path; treat as task by default.

The task is in your store with state SUBMITTED; until you transition state, it remains unacknowledged on the wire.

source={source}
target_agent={target_agent}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
