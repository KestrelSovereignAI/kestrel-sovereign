"""Lazy attachment reading (#1662 PR C).

When a user attaches a file with the composer's attach button (not a pasted
image), it is staged as a *lazy* reference: the bytes live in the encrypted
file store and the agent reads them on demand via ``read_attachment`` rather
than every attachment being shoved into the model's context. Pasted images
take the *eager* path instead (sent as vision the same turn — see the LLM
streaming layer); this feature is the lazy counterpart.

Security: an agent may only read a hash that actually appears as an attachment
in its OWN conversation history (session-scoped), so the content-addressed
store can't be turned into a read-anything oracle by a hallucinated hash.
"""
import io
import logging
import re
from typing import Any, Dict, List, Optional

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

logger = logging.getLogger(__name__)

_HASH_RE = re.compile(r"^[a-f0-9]{64}$")
# Cap how much extracted text we hand back to the model in one read.
_MAX_TEXT_CHARS = 20000


class AttachmentsFeature(Feature):
    """Read a file the user attached to this conversation, on demand."""

    @property
    def tool_description(self) -> str:
        return (
            "Read a document the user attached to this conversation (text, "
            "markdown, or PDF). Use the attachment id shown next to the file."
        )

    async def initialize(self):
        self.storage = getattr(self.agent, "storage", None)

    def _collect_session_attachments(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Map hash -> attachment ref across the session's user turns."""
        found: Dict[str, Dict[str, Any]] = {}
        for msg in history or []:
            meta = msg.get("metadata")
            if isinstance(meta, dict):
                atts = meta.get("attachments")
            else:
                atts = None
            if not isinstance(atts, list):
                continue
            for att in atts:
                if isinstance(att, dict) and isinstance(att.get("hash"), str):
                    found[att["hash"]] = att
        return found

    @staticmethod
    def _extract_text(data: bytes, mime: Optional[str], name: str) -> Optional[str]:
        """Return extracted text for a supported document, or None if the
        bytes aren't text-extractable (e.g. an image)."""
        mime = (mime or "").lower()
        lower_name = (name or "").lower()
        is_pdf = mime == "application/pdf" or lower_name.endswith(".pdf")
        is_text = (
            mime.startswith("text/")
            or lower_name.endswith((".txt", ".md", ".markdown"))
        )
        if is_pdf:
            try:
                from pypdf import PdfReader
            except Exception:
                return (
                    "[PDF text extraction is unavailable in this deployment "
                    "(pypdf not installed). Ask the user to paste a screenshot "
                    "of the relevant page to send it as an image instead.]"
                )
            try:
                reader = PdfReader(io.BytesIO(data))
                pages = [(p.extract_text() or "") for p in reader.pages]
                return "\n\n".join(pages).strip()
            except Exception as exc:
                logger.warning("read_attachment: PDF parse failed: %s", exc)
                return f"[Could not parse PDF: {exc}]"
        if is_text:
            return data.decode("utf-8", errors="replace")
        return None

    @tool(
        name="read_attachment",
        description=(
            "Read a document the user attached to THIS conversation. Pass the "
            "attachment id (a 64-char hex id shown next to the file). Works for "
            "text, markdown, and PDF documents. Images can't be read as text — "
            "ask the user to paste the image to send it as vision instead."
        ),
        category=ToolCategory.SYSTEM,
        command_prefix="!read-attachment",
    )
    async def read_attachment(
        self,
        attachment_id: str,
        session_id: Optional[str] = None,
    ) -> ToolResult:
        """Read a lazily-attached document by its id (content hash).

        Args:
            attachment_id: The 64-char hex id of an attachment in this
                conversation.
            session_id: Scope the lookup to one conversation thread.
        """
        if not isinstance(attachment_id, str) or not _HASH_RE.match(attachment_id):
            return ToolResult.failed(
                "Invalid attachment id — expected the 64-character hex id shown "
                "next to the attached file.",
                data={"attachment_id": attachment_id},
            )
        if self.storage is None or not hasattr(self.storage, "retrieve_file"):
            return ToolResult.failed("Attachment storage is unavailable.")

        # Authoritative security gate: validate the STORE's own metadata, not
        # the client-supplied conversation refs (a client can post an arbitrary
        # well-formed hash into a turn's `attachments`, which then shows up in
        # history — so history membership is NOT proof of a legitimate upload).
        # The upload endpoint stamps every chat attachment with
        # type="attachment" + the owning agent_id; require both. This keeps the
        # content-addressed store from being read as an oracle for avatars,
        # snapshots, or any other non-attachment file.
        file_store = getattr(self.storage, "files", None)
        meta = None
        if file_store is not None and hasattr(file_store, "get_file_metadata"):
            try:
                meta = await file_store.get_file_metadata(attachment_id)
            except Exception:
                meta = None
        if not isinstance(meta, dict) or meta.get("type") != "attachment":
            return ToolResult.failed(
                "That id isn't a readable chat attachment.",
                data={"attachment_id": attachment_id},
            )
        owner = meta.get("agent_id")
        my_id = getattr(self.agent, "agent_id", None) or getattr(self.agent, "did", None)
        if owner and my_id and owner != my_id:
            return ToolResult.failed(
                "That attachment doesn't belong to this agent.",
                data={"attachment_id": attachment_id},
            )

        # Conversation scope (defense-in-depth): prefer the in-conversation ref
        # for the display name; fall back to the store's recorded name.
        try:
            history = await self.storage.get_conversation_history(
                limit=200, session_id=session_id
            )
        except TypeError:
            history = await self.storage.get_conversation_history(limit=200)
        ref = self._collect_session_attachments(history).get(attachment_id) or {
            "hash": attachment_id,
            "kind": meta.get("kind"),
            "mime": meta.get("mime_type"),
            "name": meta.get("original_name") or "attachment",
        }

        data = await self.storage.retrieve_file(attachment_id)
        if not data:
            return ToolResult.failed(
                "The attachment's bytes are no longer in storage.",
                data={"attachment_id": attachment_id, "name": ref.get("name")},
            )

        name = ref.get("name") or "attachment"
        if ref.get("kind") == "image" or str(ref.get("mime") or "").startswith("image/"):
            return ToolResult.ok(
                confirmation=(
                    f"'{name}' is an image and can't be read as text. Ask the "
                    "user to paste it into the chat so it's sent as vision."
                ),
                data={"attachment_id": attachment_id, "name": name, "kind": "image"},
            )

        text = self._extract_text(data, ref.get("mime"), name)
        if text is None:
            return ToolResult.failed(
                f"'{name}' isn't a readable text/markdown/PDF document.",
                data={"attachment_id": attachment_id, "name": name,
                      "mime": ref.get("mime")},
            )

        truncated = len(text) > _MAX_TEXT_CHARS
        body = text[:_MAX_TEXT_CHARS]
        note = " (truncated)" if truncated else ""
        return ToolResult.ok(
            confirmation=f"Read '{name}'{note} — {len(body)} characters.",
            data={
                "attachment_id": attachment_id,
                "name": name,
                "mime": ref.get("mime"),
                "truncated": truncated,
                "content": body,
            },
        )
