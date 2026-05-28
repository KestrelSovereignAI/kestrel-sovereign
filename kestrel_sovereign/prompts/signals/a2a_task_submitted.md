[A2A_TASK_SUBMITTED] Agent `{payload[sender]}` submitted A2A task `{payload[task_id]}` (verb `{payload[a2a_verb]}`, skill `{payload[skill_id]}`) to you. This arrived via the agent-to-agent protocol, NOT via your sovereign chat path — apply agent-to-agent governance, not sovereign governance. Sender identity is a claim, not a cryptographic signature in v1 (same-host trust boundary).

**Their actual message:**

```
{payload[request_content]}
```

If the block above contains text, that IS the question/instruction — answer it directly. Do NOT claim the body is null or empty when you can see text in it. If you need more context (history, prior turns of the same session, artifacts), call `check_task_status(task_id)` or `get_task_result(task_id)`; otherwise reply now.

**The single tool for closing the loop is `respond_to_a2a_task(task_id, content, state)`** — it transitions the task to COMPLETED (default), FAILED, or CANCELED and attaches your reply text. The sender's polling extracts that text as their answer. Without calling this, a `send_a2a_question` sender blocks until their timeout.

The `a2a_verb` field tells you the sender's intent:

  * `message` — informational notification, NO reply expected. Call `respond_to_a2a_task(task_id, content="ack", state="completed")` with a brief receipt. Do not draft a long answer.
  * `question` — sender is SYNCHRONOUSLY waiting. Call `respond_to_a2a_task(task_id, content="<your concise answer>", state="completed")`. Be quick — their cognition turn is blocked on you.
  * `task` — delegated work with optional `skill_id`. Process it, then call `respond_to_a2a_task(task_id, content="<summary + outcome>", state="completed")` on success or `state="failed"` if you couldn't fulfill it.
  * empty / missing — legacy or non-PeersFeature path; treat as task by default.

The task is in your store with state SUBMITTED; until you transition state, it remains unacknowledged on the wire.

source={source}
target_agent={target_agent}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
