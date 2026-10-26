"""ToolOutputFeature — read back the whole of a clipped tool result (#3740).

The orchestrator replaces any serialized tool result over its cap
(``MAX_TOOL_RESULT_CHARS``) with a head-and-tail preview. It keeps the whole
text in the agent's :class:`~kestrel_sovereign.agent.tool_output_store.ToolOutputStore`
and names it in the preview by a ``tool_output:<id>`` handle. This feature's
one tool, ``read_tool_output``, reads that text in chunks that each fit under
the cap, so any tool's output — a long repository file, a subagent's long
answer — can be read in full.

It is MANDATORY and promoted at startup, like ``wait``. The cap it answers is
applied by core to every agent's tool results unconditionally, so its escape
hatch must be present wherever the cap is: an agent whose allowlist or spawn
ceiling left this feature out would be back to reading a preview with no way
past it. Loading it widens nothing: it reads only output the agent's own tools
already returned to it, from the conversation that produced it.
"""

from __future__ import annotations

from typing import Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.agent.tool_output_store import (
    ToolOutputUnavailable,
    agent_tool_output_store,
    tool_output_scope,
)
from kestrel_sovereign.features.base import (
    Feature,
    coerce_int_arg,
    largest_fitting_result,
    orchestrator_result_cap,
    tool,
)

_MAX_OFFSET = 2**63 - 1


class ToolOutputFeature(Feature):
    """Provides ``read_tool_output`` over the agent's clipped tool results."""

    @property
    def tool_description(self) -> str:
        return (
            "Read the whole of a tool result that was clipped to a preview, "
            "by the tool_output:<id> handle the preview names"
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    async def initialize(self):
        """Nothing to prepare: the store belongs to the agent."""

    @tool(
        name="read_tool_output",
        description=(
            "Read a tool result that was too long to show whole. When a tool "
            "result is replaced by a '[Large tool output …]' preview, the "
            "preview names a handle like tool_output:<id> and the offset where "
            "the omitted characters start. Pass that handle and offset; each "
            "chunk reports its range, total_chars and next_offset — call again "
            "with offset=next_offset until next_offset is null to read every "
            "character. Offsets index the result's JSON text, the same text "
            "the preview shows. A handle can be read only in the conversation "
            "that produced it."
        ),
        category=ToolCategory.SYSTEM,
    )
    async def read_tool_output(
        self,
        handle: str,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> ToolResult:
        """Read one chunk of a clipped tool result.

        Each chunk is sized so this whole result fits under the orchestrator's
        tool-result cap, so a chunk is never itself clipped.

        Args:
            handle: The tool_output:<id> handle a clipped result's preview named.
            offset: Character offset to start from (default 0).
            length: Most characters to return; the chunk shrinks further to
                fit the cap. Defaults to as much as fits.
        """
        start, error = coerce_int_arg(offset, "offset", lo=0, hi=_MAX_OFFSET)
        if error:
            return ToolResult.failed(error)
        wanted: Optional[int] = None
        if length is not None:
            wanted, error = coerce_int_arg(length, "length", lo=1, hi=_MAX_OFFSET)
            if error:
                return ToolResult.failed(error)

        store = agent_tool_output_store(self.agent)
        if store is None:
            return ToolResult.failed(
                "This agent keeps no tool-output store, so no clipped tool "
                "output can be read back."
            )
        try:
            entry = store.read(handle, scope=tool_output_scope(self.agent))
        except ToolOutputUnavailable as exc:
            return ToolResult.failed(str(exc))

        text = entry.text
        total = len(text)
        if start > total:
            return ToolResult.failed(
                f"offset {start} is past the end of {entry.handle} "
                f"({total} characters)",
                data={"handle": entry.handle, "total_chars": total},
            )
        requested_end = total if wanted is None else min(start + wanted, total)

        def build(end: int) -> ToolResult:
            has_more = end < total
            confirmation = (
                f"Read {entry.handle} (output of {entry.tool_name}) — "
                f"characters {start}–{end} of {total}."
            )
            if has_more:
                confirmation += (
                    f" More remains; call read_tool_output again with "
                    f"offset={end} for the next chunk."
                )
            else:
                confirmation += " End of output."
            chunk = text[start:end]
            return ToolResult.ok(
                confirmation=confirmation,
                data={
                    "handle": entry.handle,
                    "tool": entry.tool_name,
                    "offset": start,
                    "length": len(chunk),
                    "total_chars": total,
                    "next_offset": end if has_more else None,
                    "truncated": has_more,
                    "content": chunk,
                },
            )

        # A chunk carries at least one character while any remain — an empty
        # chunk would page forever — so where not even one fits, the read
        # fails rather than return a result the orchestrator would clip.
        cap = orchestrator_result_cap()
        shortest = start + 1 if start < requested_end else start
        fit = largest_fitting_result(
            build,
            shortest,
            requested_end,
            cap=cap,
            tool_name="read_tool_output",
        )
        if fit is not None:
            return fit[1]
        return ToolResult.failed(
            f"read_tool_output cannot fit the next character of {entry.handle} "
            f"under the {cap:,}-character tool-result cap",
            data={"handle": entry.handle, "offset": start, "total_chars": total},
        )
