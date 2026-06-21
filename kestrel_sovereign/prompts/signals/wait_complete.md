[WAIT_COMPLETE] An async wait you were tracking has reached a terminal state. Handle `{payload[kind]}:{payload[handle]}` reached outcome `{payload[outcome]}` — {payload[summary]}

This wake fired from the periodic wait reconciler poll, NOT from a user prompt — the reconciler watches every registered async waitable provider and wakes you when ANY in-flight handle reaches a terminal state, even one you did not explicitly block on. Decide if this completion needs a follow-up action and act, or acknowledge silently.

**Outcome meanings:**

  * `done` — the work the handle represents completed successfully. Summarize what happened and close the loop (comment, dispatch dependent work, or do nothing if the work's own side effects already closed it).
  * `failed` — the work ended in a terminal failure. Diagnose using the provider-specific details below, then decide whether to retry, file a bug, or abandon.
  * `partial` — a mixed terminal state (some success, some failure). Read the summary for the caveat and decide what still needs doing.

Provider-specific status (the waitable's own status string, if any): `{payload[status]}`.

The reconciler records this transition durably so you will not be woken again for the same `{payload[kind]}:{payload[handle]}:{payload[outcome]}`. If you need the full provider state, use the kind's own status tool (e.g. `task_status` / `talon_status`) with handle `{payload[handle]}`.

source={source}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
