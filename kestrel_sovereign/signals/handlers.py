"""Built-in helpers for source handlers.

`template_artifact_handler` covers the simple ARTIFACT case where a source
just wants to render a prompt template and run a one-shot LLM completion.
Most ARTIFACT sources will write their own handler because their workflow
fetches data, may make multiple LLM calls, or mutates feature state — see
SIGNAL_DISPATCHER.md §"The three modes" and §Concern 9.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from kestrel_sdk.signals import ArtifactHandler, Signal


CompletionFn = Callable[[str], Awaitable[Any]]


def template_artifact_handler(
    template_path: Path, *, complete: CompletionFn
) -> ArtifactHandler:
    """Build an ArtifactHandler that renders the template against the
    Signal envelope and runs `complete(prompt)`.

    `complete` is the LLM completion seam — sovereign supplies one bound
    to its LLM service. Tests can pass a fake.
    """

    async def _handler(signal: Signal) -> Any:
        template = template_path.read_text(encoding="utf-8")
        prompt = template.format(
            source=signal.source,
            kind=signal.kind,
            target_agent=signal.target_agent,
            payload=signal.payload,
            urgency=signal.urgency.value,
            arrived_at=signal.arrived_at.isoformat(),
        )
        return await complete(prompt)

    return _handler
