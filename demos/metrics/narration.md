# Metrics Dashboard Demo — Narration

Observability for sovereign agents. Every tool call, LLM request, and error becomes a timestamped event. The Metrics panel turns those events into KPI cards, a timeline, and a duration distribution — so operators can see, at a glance, *what* the agent has been doing and *how fast*.

## Beats

### Act 1: The empty dashboard
Open a fresh agent's Metrics panel. Zero KPI cards, flat timeline. "Before the agent has done anything, there's nothing to measure."

### Act 2: Generate activity
Send a few chat messages. Issue a tool call. Invoke a privacy-mode change. Each of these emits events.

### Act 3: KPI cards light up
Return to Metrics. KPI cards now show event counts, error rate, average tool duration. "Every number here is derived from the audit stream — no sampling, no aggregation service."

### Act 4: Charts
Timeline chart: events over time, colored by type. Duration chart: p50 / p95 per tool. Distribution pie: event types.

### Act 5: Errors table
If any tool errored, it shows here with timestamp and payload. "The agent doesn't hide failures — they're first-class citizens in the metrics."

### Beat bookend
Close on the full dashboard as a bookend shot. "This is what operational accountability looks like for an autonomous agent."
