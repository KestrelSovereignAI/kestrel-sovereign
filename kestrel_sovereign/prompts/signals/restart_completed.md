[RESTART_COMPLETED] The Kestrel host you requested be restarted has come back up. Your restart request `{payload[request_id]}` is now in terminal state `completed`. This wake fired from the post-restart sweep in `RestartCoordinatorFeature.initialize`, NOT from a user prompt — verify post-restart runtime state and move on.

**The reason you originally requested this restart:**

```
{payload[reason]}
```

Urgency `{payload[urgency]}`, policy `{payload[policy]}`. Requested `{payload[requested_at]}`, restart landed `{payload[completed_at]}`.

If your reason for restarting was a code or config change, verify the new behaviour now — most commonly:
  * a new feature surface should appear in your tool listing,
  * a kestrel.toml value should reflect its new setting via the relevant runtime introspection tool,
  * an upgraded dependency's `__version__` should report the new floor.

If verification fails, file a follow-up ticket explaining what was expected vs. what is observed — the restart itself succeeded; the underlying change may not have.

source={source}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
