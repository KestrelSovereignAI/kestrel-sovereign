# Tasks & Activity Demo — Narration

The Tasks panel is the agent's operational window. Two views:

- **Background Tasks**: long-running A2A work — image generation, training, multi-agent dispatch. Filterable by state (working, completed, failed).
- **Activity Log**: real-time stream of tool calls, LLM invocations, and feature executions.

This demo complements **spawn**: spawn shows the *lifecycle* of delegated work; tasks shows the *observability* of all async work.

## Beats

### Act 1: Empty Tasks view
Open Tasks. Empty queue, "no tasks" message. "Before anything's been asked, the queue is empty."

### Act 2: Trigger a background task
Kick off something that takes time — an image generation call, or a feature that fires async work.

### Act 3: Task appears with working state
The newly queued task shows up with state=working. Status, progress, started-at timestamp visible.

### Act 4: Filter by state
Flip the filter to "Completed". Now only finished tasks are shown.

### Act 5: Activity Log
Switch view to Activity Log. The real-time tick — every tool call, every LLM hop. "This is the truth layer — every call, every return, timestamped."

### Bookend
Return to default Tasks view. "One panel, two perspectives: what's pending, what's happening."
