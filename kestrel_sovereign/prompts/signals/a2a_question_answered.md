[A2A_QUESTION_ANSWERED] Your question to `{payload[recipient_id]}` has a local settlement event.

Question `{payload[question_id]}`, sender dispatch `{payload[dispatch_id]}`:

```
{payload[original_question]}
```

Original native input: `{payload[request]}`. Read structured Parts as part of the original question.

Peer outcome: `{payload[state]}`. Late answer: `{payload[late]}`. Confirmed withdrawal: `{payload[withdrawn]}`.

```
{payload[reply_text]}
```

Read `get_peer_task_result("{payload[dispatch_id]}")` for the full native response and its standard artifacts. If `requires_full_response` is `{payload[requires_full_response]}` and true, the answer contains structured data, files, or artifacts: fetch it before interpreting or using the answer. An empty text block does not mean an empty native response. If `truncated` is `{payload[truncated]}` and true, fetch the complete result before relying on it. Artifact IDs identify outputs; chunk order comes from the native append and last-chunk events.

`expired` means your local deadline passed. An empty reply is no answer. Only `withdrawn=True` confirms cancellation; otherwise delivery or withdrawal can remain uncertain and the recipient may still answer. Do not resend an uncertain dispatch. A late answer is a separate event after the deadline event; update any conclusions you drew while waiting.

For `failed`, `canceled`, or `rejected`, integrate the failure reason without claiming success. For `completed`, integrate the answer and continue the original conversation. Do not ask the same question again merely because this is a new cognition turn.

source={source}
target_agent={target_agent}
arrived_at={arrived_at}
urgency={urgency}
payload={payload}
