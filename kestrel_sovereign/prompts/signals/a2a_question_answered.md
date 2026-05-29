[A2A_QUESTION_ANSWERED] An earlier `send_a2a_question` of yours to `{payload[recipient]}` has been answered. This is a cognition wake firing locally on YOUR dispatcher — `{payload[recipient]}` did not send you anything new; they transitioned the task you sent them to a terminal state, and your subscription on the task picked up the result.

**Your original question to {payload[recipient]}** (task_id `{payload[task_id]}`):

```
{payload[original_question]}
```

**Their reply** (state: `{payload[state]}`):

```
{payload[reply_text]}
```

If the reply block above contains the text `[truncated; call get_peer_task_result(...)`, the full body exceeded the inline cap (8 KiB). Call `get_peer_task_result("{payload[recipient]}", "{payload[task_id]}")` to fetch the complete reply through the host proxy before responding — do NOT pretend the truncated text is the whole answer.

If `state` is `expired`, your subscription deadline passed without `{payload[recipient]}` reaching a terminal state. The reply block will be empty. Decide whether to give up, re-ask via a fresh `send_a2a_question`, or escalate — do NOT claim you got an answer when state is `expired`.

If `state` is `failed` or `canceled`, the recipient transitioned the task into a non-success terminal state. Their reply text (if any) typically carries the reason.

**Now**: integrate the answer into your in-flight context and continue whatever you were doing when you sent the original question. Do NOT re-ask the same question — the response is right here. If the original conversation was with the Sovereign, your next message should fold this answer into your reply to them; if it was internal multi-step work, continue the next step.

source={source}
target_agent={target_agent}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
