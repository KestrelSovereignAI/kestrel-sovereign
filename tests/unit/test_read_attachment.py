"""Lazy attachment reading (#1662 PR C) — read_attachment tool + context hint."""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock


def _feature(storage):
    from kestrel_sovereign.features.attachments.feature import AttachmentsFeature
    agent = MagicMock()
    agent.storage = storage
    feat = AttachmentsFeature(agent)
    feat.storage = storage
    return feat


def _history_with(att):
    return [{"role": "user", "metadata": {"attachments": [att]}}]


# --- session-scoped security gate -------------------------------------------

@pytest.mark.asyncio
async def test_read_attachment_rejects_hash_not_in_conversation():
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=[])  # no attachments
    storage.retrieve_file = AsyncMock(return_value=b"secret")
    feat = _feature(storage)
    res = await feat.read_attachment("a" * 64)
    assert res.status == "error"
    # The hash was never even fetched — security gate fired first.
    storage.retrieve_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_attachment_rejects_malformed_id():
    feat = _feature(MagicMock())
    res = await feat.read_attachment("not-a-hash")
    assert res.status == "error"


# --- text + markdown --------------------------------------------------------

@pytest.mark.asyncio
async def test_read_attachment_returns_text_document():
    h = "b" * 64
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=_history_with(
        {"hash": h, "kind": "document", "mime": "text/markdown", "name": "notes.md"}))
    storage.retrieve_file = AsyncMock(return_value=b"# Title\n\nhello world")
    feat = _feature(storage)
    res = await feat.read_attachment(h)
    assert res.status == "ok"
    assert res.data["content"] == "# Title\n\nhello world"
    assert res.data["name"] == "notes.md"


@pytest.mark.asyncio
async def test_read_attachment_truncates_long_text():
    from kestrel_sovereign.features.attachments.feature import _MAX_TEXT_CHARS
    h = "c" * 64
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=_history_with(
        {"hash": h, "kind": "document", "mime": "text/plain", "name": "big.txt"}))
    storage.retrieve_file = AsyncMock(return_value=b"x" * (_MAX_TEXT_CHARS + 500))
    feat = _feature(storage)
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
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=_history_with(
        {"hash": h, "kind": "document", "mime": "application/pdf", "name": "doc.pdf"}))
    storage.retrieve_file = AsyncMock(return_value=_one_page_pdf("hi"))
    feat = _feature(storage)
    res = await feat.read_attachment(h)
    assert res.status == "ok"
    assert "content" in res.data  # parsed without error (text may be empty)


# --- image: not readable as text --------------------------------------------

@pytest.mark.asyncio
async def test_read_attachment_image_directs_to_paste():
    h = "e" * 64
    storage = MagicMock()
    storage.get_conversation_history = AsyncMock(return_value=_history_with(
        {"hash": h, "kind": "image", "mime": "image/png", "name": "shot.png"}))
    storage.retrieve_file = AsyncMock(return_value=b"\x89PNG\r\n")
    feat = _feature(storage)
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
