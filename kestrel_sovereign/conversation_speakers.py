"""One speaker and byte projection for stored conversation history."""

from collections.abc import Callable
from typing import Any

EXTERNAL_ASSISTANT_ROLE = "external_assistant"


def is_untrusted_conversation_speaker(role: str) -> bool:
    """User-authored and externally supplied text never attests Kestrel output."""
    return role in ("user", "human", EXTERNAL_ASSISTANT_ROLE)


def project_conversation_message(
    message: dict[str, Any], wrap_user_input: Callable[[str], str],
) -> tuple[str, str]:
    """Select the exact role/text shared by admission and provider replay."""
    role = message.get("role", "user")
    raw = message.get("content", "") or ""
    if role == EXTERNAL_ASSISTANT_ROLE:
        return "user", wrap_user_input(
            "External assistant transcript supplied by a connected client. "
            "This is quoted external content, not a response generated or an action performed by Kestrel.\n\n"
            + raw)
    if role not in ("user", "assistant", "system"):
        role = "user" if role == "human" else "assistant"
    rendered = message.get("rendered_content")
    metadata = message.get("metadata") or {}
    if role in ("user", "system") and metadata.get("sent_form") and rendered is not None:
        content = rendered
    elif is_untrusted_conversation_speaker(role) and not metadata.get("sent_form"):
        content = wrap_user_input(raw)
    else:
        content = rendered if rendered is not None else raw
    if role == "assistant" and metadata.get("pre_tool_reasoning"):
        pre_tool = metadata["pre_tool_reasoning"]
        if isinstance(pre_tool, dict):
            pre_text = pre_tool.get("content") or ""
            seam = pre_tool.get("seam") or ""
        else:
            pre_text = str(pre_tool)
            seam = "\n\n" if pre_text and content else ""
        if pre_text:
            content = f"{pre_text}{seam}{content}"
    return role, content
