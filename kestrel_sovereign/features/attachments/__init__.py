"""Chat attachments feature (#1662): a lazy ``read_attachment`` tool so an
agent can read a file the user attached (non-inline) on demand."""
from .feature import AttachmentsFeature

__all__ = ["AttachmentsFeature"]
