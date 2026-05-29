[A2A_TASK_SUBMITTED] Agent `{payload[sender]}` submitted A2A task `{payload[task_id]}` (verb `{payload[a2a_verb]}`, skill `{payload[skill_id]}`) to you. This arrived via the agent-to-agent protocol, NOT via your sovereign chat path — apply agent-to-agent governance, not sovereign governance. Sender identity is a claim, not a cryptographic signature in v1 (same-host trust boundary).

**Their actual message:**

```
{payload[request_content]}
```

If the block above contains text, that IS the question/instruction — answer it directly. Do NOT claim the body is null or empty when you can see text in it. If you need more context (history, prior turns of the same session, artifacts), call `check_task_status(task_id)` or `get_task_result(task_id)`; otherwise reply now.

**The single tool for closing the loop is `respond_to_a2a_task(task_id, content, state)`** — it transitions the task to COMPLETED (default), FAILED, or CANCELED and attaches your reply text. The sender's subscription supervisor (or the receiver's `/tasks/{{id}}/subscribe` SSE stream) picks up the terminal frame and fires an `a2a.question_answered` signal back on the sender's dispatcher with your reply inline. Without calling this, a `send_a2a_question` sender never gets resumed (the supervisor times out on the deadline, then the hourly sweep fires a `state='expired'` signal so the sender doesn't hang forever).

The `a2a_verb` field tells you the sender's intent:

  * `message` — informational notification, NO reply expected. Call `respond_to_a2a_task(task_id, content="ack", state="completed")` with a brief receipt. Do not draft a long answer.
  * `question` — sender's `send_a2a_question` turn has ALREADY ended; they are NOT blocking on you. Their next cognition turn fires when this task hits a terminal state (#1444 fire-and-resume). Call `respond_to_a2a_task(task_id, content="<your concise answer>", state="completed")` whenever you have an answer — no synchronous-turn pressure to be artificially fast.
  * `task` — delegated work with optional `skill_id`. Process it, then call `respond_to_a2a_task(task_id, content="<summary + outcome>", state="completed")` on success or `state="failed"` if you couldn't fulfill it.
  * empty / missing — legacy or non-PeersFeature path; treat as task by default.

The task is in your store with state SUBMITTED; until you transition state, it remains unacknowledged on the wire.

source={source}
target_agent={target_agent}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
