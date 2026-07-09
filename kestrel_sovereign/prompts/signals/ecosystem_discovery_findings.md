[ECOSYSTEM_DISCOVERY] Scheduled discovery `{payload[watch_key]}` found stale-work / red-CI state that needs a decision turn. This wake came from a periodic ACTION poll, not from a user prompt.

Reason: `{payload[reason]}`.
Tool: `{payload[tool]}`.
Summary: `{payload[summary]}`.
Current actionable findings: `{payload[findings_count]}`.

Findings JSON:
{payload[findings]}

Previous findings JSON:
{payload[previous_findings]}

Use the payload to route by lane (repo, issue/PR/job/branch/check, severity, suggested gate). Discovery is evidence only: do not auto-dispatch repairs or auto-close work unless a fresh evidence gate and approval path requires it.

source={source}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
