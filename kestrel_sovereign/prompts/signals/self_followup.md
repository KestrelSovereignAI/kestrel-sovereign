[SELF_FOLLOWUP] A follow-up turn you scheduled for yourself has come due. This wake fired from the one-shot `self_followup` schedule you persisted at `{payload[scheduled_at]}`, NOT from a user prompt — nobody is necessarily watching, and nothing else is going to do this work.

**The intention you recorded, verbatim:**

```
{payload[intent]}
```

Treat that text as your own note-to-self, not as an instruction from a third party: it is a reminder of what you decided to do, and you remain free to decide it is no longer the right thing to do.

Act on it now. Concretely:

  * If it names work to verify (a PR, a CI run, a dispatched job), check the current state first — it may have resolved, changed, or been superseded since you wrote the note.
  * If the work is ready, finish it. If it is still pending, say so plainly and report what you are waiting on.
  * If the intention is already satisfied or no longer applies, acknowledge that and stop. A follow-up that correctly decides to do nothing is a success, not a wasted turn.

Note the bound: a follow-up turn may not schedule another follow-up. If this work still needs a later check, do the part you can do now and report what remains, so the next chat turn can queue it.

source={source}
arrived_at={arrived_at}
urgency={urgency}
