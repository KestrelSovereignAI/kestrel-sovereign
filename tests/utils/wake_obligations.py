"""Arm A2A wake obligations in tests that run no TaskStore (#4041)."""

from kestrel_sovereign.a2a.stores.unified.wake_obligation_store import WakeObligationStore, wake_identity


async def arm_wake_obligation(backend, *, agent_id, task_id, input_revision=0):
    """Owe a wake for one input as a task write into SUBMITTED would (I3).

    Returns the source identity its wake is admitted under.
    """
    store = WakeObligationStore(backend)
    await store.ensure_schema()
    async with backend.transaction():
        await store.lock_scope(agent_id=agent_id)
        generation = await store.arm(
            agent_id=agent_id, task_id=task_id, input_revision=input_revision
        )
    return wake_identity(task_id, input_revision, generation)
