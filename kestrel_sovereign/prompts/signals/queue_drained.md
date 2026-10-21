[QUEUE_DRAINED] Your work queue has drained. Since `{payload[idle_since]}` (at least {payload[idle_minutes]} minutes, {payload[observed_idle_minutes]} observed) you have had no Talon job in flight (in your Talon dispatcher's lane, when your host names one), no armed wait watch, no pending restart request and no wake waiting to be delivered. This wake fired from the idle detector, NOT from a user prompt. Nobody asked for it, and nothing else is going to start the next piece of work.

Triage now, under your continuous-work doctrine (2026-10-10): an empty queue is the trigger to look for the next piece of work.

  * Check what is open and actionable: your strategy ledger, your todos, open issues labelled for you, any work order you sent that has not reported back, and anything you deferred while the last item ran.
  * If something is ready, start it now: dispatch it, or register the wait that will wake you when it settles.
  * If nothing is actionable, say so plainly and stop. A triage that correctly finds nothing to do is a success, not a wasted turn.

This wake dispatched nothing for you; the decision is yours. It fires once per idle stretch: if you start nothing now, it will not fire again until new work starts (a Talon dispatch, an armed watch, a restart you file) or an operator or peer turn arrives.

source={source}
arrived_at={arrived_at}
urgency={urgency}
