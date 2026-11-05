"""Arrange pre-cutover Runpod tails in the SAME real session lifecycle.

Only declared test hooks are exercised. This fixture grants no fresh catalog,
physical-resource or provider authority and is not a production admission API.
"""

from kestrel_sovereign.features.training.adapters.runpod_adapter import RunPodTrainingAdapter
from kestrel_sovereign.features.training.types import TrainingConfig, TrainingJob


async def start_original_runpod_tail(
    adapter: RunPodTrainingAdapter,
    *,
    job_id: str,
    companion_id: str,
    avatar_data: bytes,
    config: TrainingConfig,
) -> TrainingJob:
    """Observe original acquire/submission races admitted before retirement."""
    record = await adapter._lifecycle.start(
        job_id=job_id,
        companion_id=companion_id,
        trigger_word=config.trigger_word or f"TOK{companion_id[:8]}",
        config=config,
        avatar_data=avatar_data,
    )
    return adapter._lifecycle.training_job(record)
