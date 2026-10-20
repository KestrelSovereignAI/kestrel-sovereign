"""Read-only presentation of official native protocol values."""

from a2a.types import Message


def message_text(message: Message) -> str | None:
    """Keep text parts in order without treating other oneof values as text."""
    if not isinstance(message, Message):
        raise TypeError("Text projection requires an official A2A Message")
    texts = [part.text for part in message.parts if part.HasField("text")]
    return "\n".join(texts) if texts else None
