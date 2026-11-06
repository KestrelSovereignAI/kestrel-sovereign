"""Stable tool-operation metadata from the original admitted signal input.

This metadata grants no authority. Lifecycle and execution admission remain
bound to each fresh signal attempt, independently of its retry identity.
"""

from kestrel_sdk.signals import Signal

from kestrel_sovereign.agent.invocation import InvocationProvenance, request_provenance
from kestrel_sovereign.signals.durable import DurableSignalEvent
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    SOURCE_NAME, parse_submitted_input_identity, submitted_input_identity,
)
from kestrel_sovereign.tools.effect_journal import logical_operation_key


def signal_effect_provenance(
    original: Signal | DurableSignalEvent, *, agent_id: str,
    canonical_signal: Signal | None = None,
) -> InvocationProvenance:
    """Fingerprint canonical source input before session/canary generation."""
    if original.target_agent != agent_id:
        raise ValueError('Signal tool operation belongs to another recipient')
    durable = isinstance(original, DurableSignalEvent)
    if durable and original.agent_id != agent_id:
        raise ValueError('Durable signal tool operation belongs to another owner')
    content = canonical_signal if canonical_signal is not None else original
    if content.source != original.source or content.target_agent != agent_id:
        raise ValueError('Canonical signal does not match the original source recipient')
    identity = (original.source_event_id or original.event_id) if durable else original.id
    if original.source == SOURCE_NAME:
        # WORKING wakes may have a fresh lifecycle revision. They still refer
        # to the same admitted input; only a creator continuation changes it.
        identity = original.source_event_id if durable else original.dedupe_key
        task_id, revision = parse_submitted_input_identity(identity)
        payload = content.payload
        if (not isinstance(payload, dict) or payload.get('task_id') != task_id
                or payload.get('input_revision') != revision):
            raise ValueError('A2A tool operation disagrees with its admitted input')
        identity = submitted_input_identity(task_id, revision)
    return request_provenance(
        source_kind='signal', source_locator=original.source,
        operation_key=logical_operation_key(['signal-input-v1', identity]),
        input_digest=logical_operation_key([
            'signal-input-v1', content.kind, content.payload, content.session_id,
        ]),
    )
