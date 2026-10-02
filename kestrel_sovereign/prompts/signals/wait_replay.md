[WAIT_REPLAY] This is a REPLAY, not news. Handle `{payload[kind]}:{payload[handle]}` is reported in terminal state `{payload[outcome]}` (provider status `{payload[status]}`) — {payload[summary]}

The provider dates this terminal event at `{payload[terminal_event_at]}`. You were last woken for this handle at `{payload[delivery_last_delivered_at]}`. The provider reports a terminal state for this handle that you have not been woken for, but it dates the event before your last wake, so the work did not just finish.

This wake fired from the periodic wait reconciler poll, NOT from a user prompt. It was routed here instead of to this handle's own wake on purpose: any instructions that wake would carry for acting in this same turn were written for the moment the event happened, and they have been withheld. They may already be done, or superseded by what has happened since. Some are harmful to repeat, such as re-answering a question that was already answered or re-dispatching work that already finished.

Before you do anything:

  1. Check the handle's CURRENT state through its provider's status tool, when one is available, with handle `{payload[handle]}`.
  2. Check the current state of whatever the handle worked on (the issue, pull request, task, or job it belongs to), including anything you or anyone else did after your last wake.
  3. Act only on what that current state requires. If it requires nothing, acknowledge silently.

Do not act on the terminal state above as if it had just happened, and do not follow a procedure for this kind of event without checking the current state first.

source={source}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
