"""Clipped tool output (#3740): ``read_tool_output`` reads back, in chunks,
the whole of a tool result the orchestrator replaced with a preview."""
from .feature import ToolOutputFeature

__all__ = ["ToolOutputFeature"]
