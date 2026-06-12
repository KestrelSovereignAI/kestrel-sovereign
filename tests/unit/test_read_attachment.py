"""Lazy attachment reading (#1662 PR C) — read_attachment tool + context hint."""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock


def _history_with(att):
    return [{"role": "user", "metadata": {"attachments": [att]}}]


def _make(att=None, *, bytes_=b"", file_meta="attachment"):
    """Build (feature, storage). `file_meta` is the store-side metadata the
    provenance gate validates: "attachment" → a real chat-attachment upload,
    None → not an attachment (e.g. an avatar or fabricated hash), or a dict to
    supply custom metadata."""
    from kestrel_sovereign.features.attachments.feature import AttachmentsFeature
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(
        return_value=_history_with(att) if att else [])
    storage.retrieve_file = AsyncMock(return_value=bytes_)
    if file_meta == "attachment":
        meta = {"type": "attachment", "kind": (att or {}).get("kind"),
                "mime_type": (att or {}).get("mime"),
                "original_name": (att or {}).get("name"), "agent_id": "did:test"}
    else:
        meta = file_meta  # None or a custom dict
    storage.files = MagicMock()
    storage.files.get_file_metadata = AsyncMock(return_value=meta)
    agent = MagicMock()
    agent.agent_id = "did:test"
    agent._active_session_id = None
    agent.storage = storage
    feat = AttachmentsFeature(agent)
    feat.storage = storage
    return feat, storage


def _feature(storage):
    from kestrel_sovereign.features.attachments.feature import AttachmentsFeature
    agent = MagicMock()
    agent.agent_id = "did:test"
    agent.storage = storage
    feat = AttachmentsFeature(agent)
    feat.storage = storage
    return feat


# --- session-scoped security gate -------------------------------------------

@pytest.mark.asyncio
async def test_read_attachment_rejects_non_attachment_hash():
    # A hash whose STORE metadata isn't type="attachment" — an avatar, a
    # snapshot, or a fabricated hash coaxed into the turn's attachment metadata
    # — is rejected before any bytes are fetched. Store provenance, not the
    # client-supplied conversation refs, is the authoritative gate.
    feat, storage = _make(
        {"hash": "a" * 64, "kind": "document", "mime": "text/plain", "name": "x"},
        bytes_=b"secret", file_meta=None)
    res = await feat.read_attachment("a" * 64)
    assert res.status == "error"
    storage.retrieve_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_attachment_rejects_other_agents_attachment():
    feat, storage = _make(
        {"hash": "f" * 64, "kind": "document", "mime": "text/plain", "name": "x"},
        bytes_=b"data",
        file_meta={"type": "attachment", "agent_id": "did:someone-else"})
    res = await feat.read_attachment("f" * 64)
    assert res.status == "error"
    storage.retrieve_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_attachment_rejects_hash_not_in_this_conversation():
    # Valid store provenance (a real attachment owned by this agent) but NOT
    # referenced in this conversation's history → rejected; the tool can't pull
    # another thread's attachment by id alone.
    feat, storage = _make(
        att=None, bytes_=b"data",
        file_meta={"type": "attachment", "agent_id": "did:test"})
    res = await feat.read_attachment("a" * 64)
    assert res.status == "error"
    storage.retrieve_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_attachment_scopes_to_agents_active_session():
    # The model omits session_id; the tool must scope to the session the agent
    # recorded for the active turn (not the agent's whole history).
    h = "b" * 64
    feat, storage = _make(
        {"hash": h, "kind": "document", "mime": "text/plain", "name": "n.txt"},
        bytes_=b"hi")
    feat.agent._active_session_id = "sess-active"
    await feat.read_attachment(h)  # no session_id arg
    storage.get_conversation_history.assert_awaited_once()
    assert storage.get_conversation_history.await_args.kwargs["session_id"] == "sess-active"


@pytest.mark.asyncio
async def test_read_attachment_rejects_malformed_id():
    feat = _feature(MagicMock())
    res = await feat.read_attachment("not-a-hash")
    assert res.status == "error"


# --- text + markdown --------------------------------------------------------

@pytest.mark.asyncio
async def test_read_attachment_returns_text_document():
    h = "b" * 64
    feat, _ = _make(
        {"hash": h, "kind": "document", "mime": "text/markdown", "name": "notes.md"},
        bytes_=b"# Title\n\nhello world")
    res = await feat.read_attachment(h)
    assert res.status == "ok"
    assert res.data["content"] == "# Title\n\nhello world"
    assert res.data["name"] == "notes.md"


@pytest.mark.asyncio
async def test_read_attachment_truncates_long_text():
    from kestrel_sovereign.features.attachments.feature import _MAX_TEXT_CHARS
    h = "c" * 64
    feat, _ = _make(
        {"hash": h, "kind": "document", "mime": "text/plain", "name": "big.txt"},
        bytes_=b"x" * (_MAX_TEXT_CHARS + 500))
    res = await feat.read_attachment(h)
    assert res.status == "ok"
    assert res.data["truncated"] is True
    assert len(res.data["content"]) == _MAX_TEXT_CHARS


# --- PDF --------------------------------------------------------------------

def _one_page_pdf(text: str) -> bytes:
    from pypdf import PdfWriter
    # A blank page is enough to exercise the reader path (extract_text may be
    # empty for a programmatically blank page; the tool must still succeed).
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_read_attachment_parses_pdf():
    h = "d" * 64
    feat, _ = _make(
        {"hash": h, "kind": "document", "mime": "application/pdf", "name": "doc.pdf"},
        bytes_=_one_page_pdf("hi"))
    res = await feat.read_attachment(h)
    assert res.status == "ok"
    assert "content" in res.data  # parsed without error (text may be empty)


# --- image: not readable as text --------------------------------------------

@pytest.mark.asyncio
async def test_read_attachment_image_directs_to_paste():
    h = "e" * 64
    feat, _ = _make(
        {"hash": h, "kind": "image", "mime": "image/png", "name": "shot.png"},
        bytes_=b"\x89PNG\r\n")
    res = await feat.read_attachment(h)
    # Not a failure, but tells the agent to use the paste/vision path.
    assert res.status == "ok"
    assert "paste" in res.confirmation.lower()
    assert "content" not in res.data


# --- lazy-attachment context hint -------------------------------------------

def test_lazy_attachment_hint_lists_only_lazy():
    from kestrel_sovereign.agent.streaming import StreamingMixin
    h1, h2, h3 = "a" * 64, "b" * 64, "c" * 64
    hint = StreamingMixin._lazy_attachment_hint([
        {"hash": h1, "kind": "document", "name": "report.pdf", "inline": False},
        {"hash": h2, "kind": "image", "name": "shot.png", "inline": True},  # eager → omit
        {"hash": h3, "kind": "image", "name": "pic.png", "inline": False},  # lazy image → list
    ])
    assert "report.pdf" in hint and h1 in hint
    assert h3 in hint            # a lazy image is still readable/referable
    assert h2 not in hint        # the eager image is folded as vision, not listed
    assert "read_attachment" in hint


def test_lazy_attachment_hint_empty_when_none_or_all_inline():
    from kestrel_sovereign.agent.streaming import StreamingMixin
    assert StreamingMixin._lazy_attachment_hint(None) == ""
    assert StreamingMixin._lazy_attachment_hint([]) == ""
    assert StreamingMixin._lazy_attachment_hint(
        [{"hash": "a" * 64, "kind": "image", "inline": True}]) == ""
