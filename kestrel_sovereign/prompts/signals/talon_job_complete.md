[TALON_JOB_COMPLETE] Background Talon job `{payload[job_id]}` reached terminal state `{payload[status]}` (returncode `{payload[returncode]}`). This wake fired from the periodic `talon_monitor` poll, NOT from a user prompt — decide if this completion needs a follow-up action and act, or acknowledge silently.

Dispatched against: `{payload[repo]}#{payload[issue]}` — label: `{payload[label]}`.

**Status meanings:**

  * `complete` — wrapper recorded exit code 0. The job ran to a clean exit. Open the log, summarize what happened, then close the loop (file a follow-up issue, comment on the original ticket, dispatch dependent work, or do nothing if the job's own side effects — PR open, issue comment, etc. — already closed the loop).
  * `failed` — wrapper recorded a non-zero exit code. The job hit an error before completing its workflow. Read the log tail below + tail the full log via `talon_job_log(job_id="{payload[job_id]}")` to diagnose. Decide whether to retry, file a bug, or abandon.
  * `finished_unknown` — process is gone but no exit-code sidecar was written. Most likely cause: SIGKILL, host reboot, or OOM kill before the wrapper could write. Treat as "outcome lost"; tail the log to see how far the job got and decide whether to re-dispatch.

**Log tail (last lines):**

```
{payload[log_tail]}
```

If the tail is empty, the job exited before any output was produced. Full log path: `{payload[log_path]}`. Time range: `{payload[started_at]}` → `{payload[completed_at]}`.

**Test evidence (implementation side):**

```
{payload[test_evidence]}
```

CI status reported by Talon: `{payload[ci_status]}`.

Test evidence is the implementation-side gate: which targeted tests Talon ran, their exit codes, and the CI status it observed. If it is empty, Talon did not report structured evidence — do NOT assume tests passed. As the reviewer, run independent verification with `talon_verify(...)` (allowlisted test commands run without prompting; the result state distinguishes `passed` / `failed` / `blocked_by_policy` / `blocked_by_user` / `blocked_by_sandbox` / `tooling_error`). If local tests cannot run, say so precisely and treat CI as the remaining hard merge gate — do not describe a sandbox/policy block as the user denying anything unless the approval record explicitly says so.

The job record stays in the durable registry — you can call `talon_status()` to see the full job listing or `talon_job_log(job_id="{payload[job_id]}")` to tail more lines. You can also dispatch a fresh Talon job for the same issue if a retry is warranted.

source={source}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
