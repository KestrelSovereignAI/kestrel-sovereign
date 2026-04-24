"""
Unit tests for the voice tag normalizer.

Covers:

* Parser edge cases (empty, unknown, unterminated, nested-looking, case, mixed
  with literal brackets)
* Per-adapter output shape (ElevenLabs v3 native emit, OpenAI TTS instructions,
  Realtime combine, Piper strip)
* Prompt snippet loader
* ``get_adapter`` fallback

Adapters are pure — no async, no registry, no network. Every test is a
single-call sync assertion.
"""

import pytest

from kestrel_sovereign.voice import tags as voice_tags
from kestrel_sovereign.voice.tags import (
    ElevenLabsV3Adapter,
    KESTREL_TAGS,
    OpenAIRealtimeAdapter,
    OpenAITTSAdapter,
    ParsedText,
    PiperAdapter,
    TagToken,
    get_adapter,
    get_voice_prompt_snippet,
    parse_tags,
)


# ---------------------------------------------------------------------------
# parse_tags
# ---------------------------------------------------------------------------


class TestParseTags:
    def test_empty_input(self) -> None:
        result = parse_tags("")
        assert result.raw == ""
        assert result.clean == ""
        assert result.tags == []
        assert result.unknown == []

    def test_no_tags(self) -> None:
        result = parse_tags("Hello, world.")
        assert result.clean == "Hello, world."
        assert result.tags == []

    def test_single_recognized_tag(self) -> None:
        result = parse_tags("[excited] Hello!")
        assert result.clean == "Hello!"
        assert len(result.tags) == 1
        assert result.tags[0].name == "excited"
        assert result.tags[0].position == 0

    def test_multiple_recognized_tags(self) -> None:
        result = parse_tags("[excited] Hi. [whispering] Just us.")
        assert "Hi." in result.clean
        assert "Just us." in result.clean
        assert "[excited]" not in result.clean
        assert "[whispering]" not in result.clean
        assert [t.name for t in result.tags] == ["excited", "whispering"]

    def test_unknown_tag_left_as_literal(self) -> None:
        result = parse_tags("[unknown_tag] Hello.")
        assert result.clean == "[unknown_tag] Hello."
        assert result.tags == []
        assert result.unknown == ["[unknown_tag]"]

    def test_unterminated_bracket_preserved(self) -> None:
        result = parse_tags("[excited Hello")  # no closing bracket
        assert result.clean == "[excited Hello"
        assert result.tags == []

    def test_tag_with_uppercase_normalized(self) -> None:
        result = parse_tags("[EXCITED] Hello!")
        assert result.clean == "Hello!"
        assert result.tags[0].name == "excited"

    def test_tag_mixed_with_urls(self) -> None:
        # URLs may contain brackets in some rare contexts, but our pattern
        # requires a leading letter and only word-chars. This protects
        # `[2024]` / `[http://...]` from being mis-parsed.
        result = parse_tags("See [2024] and [http://x.y] and [excited] hi")
        assert "[2024]" in result.clean
        assert "[http://x.y]" in result.clean
        assert "[excited]" not in result.clean
        assert [t.name for t in result.tags] == ["excited"]

    def test_parser_never_raises_on_weird_input(self) -> None:
        # Fuzz-y inputs: unmatched brackets, only punctuation, nested-looking.
        for weird in ["[[[[", "]]]]", "[a][[b]]", "]]", "[", "]", "[][]", "[_]"]:
            result = parse_tags(weird)
            assert isinstance(result, ParsedText)

    def test_whitespace_collapsed_after_strip(self) -> None:
        result = parse_tags("Hello   [excited]   world")
        assert result.clean == "Hello world"

    def test_preserves_newlines(self) -> None:
        result = parse_tags("[excited] Line one.\n[whispering] Line two.")
        assert "\n" in result.clean


# ---------------------------------------------------------------------------
# ElevenLabsV3Adapter
# ---------------------------------------------------------------------------


class TestElevenLabsV3Adapter:
    def setup_method(self) -> None:
        self.adapter = ElevenLabsV3Adapter()

    def test_passes_through_laughs(self) -> None:
        out = self.adapter.normalize("That's hilarious [laughs] no really.")
        assert "[laughs]" in out.text
        assert out.instructions is None

    def test_passes_through_sighs_and_whispering(self) -> None:
        out = self.adapter.normalize("[sighs] Okay. [whispering] Don't tell anyone.")
        assert "[sighs]" in out.text
        assert "[whispers]" in out.text  # whispering → whispers
        assert "[whispering]" not in out.text

    def test_strips_unsupported_tags(self) -> None:
        # [excited] has no v3 equivalent — strip.
        out = self.adapter.normalize("[excited] Fantastic news!")
        assert "[excited]" not in out.text
        assert out.text == "Fantastic news!"

    def test_laughing_maps_to_laughs(self) -> None:
        out = self.adapter.normalize("[laughing] ha ha")
        assert "[laughs]" in out.text

    def test_unknown_tag_preserved_as_literal(self) -> None:
        out = self.adapter.normalize("[moonwalk] cool")
        assert "[moonwalk] cool" == out.text

    def test_annotations_list_original_tags(self) -> None:
        out = self.adapter.normalize("[excited] Hi [laughs].")
        # Both tags should be recorded even though only laughs was passed through.
        assert "excited" in out.annotations["tags"]
        assert "laughs" in out.annotations["tags"]


# ---------------------------------------------------------------------------
# OpenAITTSAdapter
# ---------------------------------------------------------------------------


class TestOpenAITTSAdapter:
    def setup_method(self) -> None:
        self.adapter = OpenAITTSAdapter()

    def test_strips_tags_from_audible_text(self) -> None:
        out = self.adapter.normalize("[excited] Great news!")
        assert out.text == "Great news!"

    def test_instructions_composed_from_single_tag(self) -> None:
        out = self.adapter.normalize("[excited] Great news!")
        assert out.instructions is not None
        assert "excited" in out.instructions.lower()

    def test_no_instructions_when_no_tags(self) -> None:
        out = self.adapter.normalize("Plain text.")
        assert out.instructions is None

    def test_instructions_deduplicate(self) -> None:
        out = self.adapter.normalize("[excited] A. [excited] B.")
        # Same tag repeated should not duplicate the directive sentence.
        assert out.instructions is not None
        assert out.instructions.count("Speak with excited") == 1

    def test_instructions_preserve_tag_order(self) -> None:
        out = self.adapter.normalize("[whispering] A. [excited] B.")
        assert out.instructions is not None
        # whisper directive appears before excited directive
        whisper_idx = out.instructions.lower().index("whisper")
        excited_idx = out.instructions.lower().index("excited")
        assert whisper_idx < excited_idx

    def test_unknown_tag_has_no_instruction(self) -> None:
        out = self.adapter.normalize("[moonwalk] cool")
        assert out.instructions is None
        assert "[moonwalk] cool" == out.text


# ---------------------------------------------------------------------------
# OpenAIRealtimeAdapter
# ---------------------------------------------------------------------------


class TestOpenAIRealtimeAdapter:
    def setup_method(self) -> None:
        self.adapter = OpenAIRealtimeAdapter()

    def test_normalize_has_no_per_chunk_instructions(self) -> None:
        out = self.adapter.normalize("[excited] Hi.")
        # Realtime can't apply per chunk.
        assert out.instructions is None
        assert out.text == "Hi."

    def test_combine_aggregates_all_tags_across_turn(self) -> None:
        parsed_chunks = [
            parse_tags("[excited] First sentence."),
            parse_tags("[whispering] Second sentence."),
            parse_tags("Third, no tag."),
        ]
        out = self.adapter.combine(parsed_chunks)
        assert out.instructions is not None
        assert "excited" in out.instructions.lower()
        assert "whisper" in out.instructions.lower()
        assert "First sentence." in out.text
        assert "Second sentence." in out.text
        assert "Third, no tag." in out.text

    def test_combine_empty_produces_no_instructions(self) -> None:
        out = self.adapter.combine([parse_tags("plain one."), parse_tags("plain two.")])
        assert out.instructions is None


# ---------------------------------------------------------------------------
# PiperAdapter
# ---------------------------------------------------------------------------


class TestPiperAdapter:
    def setup_method(self) -> None:
        self.adapter = PiperAdapter()

    def test_strips_all_known_tags(self) -> None:
        out = self.adapter.normalize("[excited] Hi [laughs].")
        assert "[excited]" not in out.text
        assert "[laughs]" not in out.text
        assert out.instructions is None

    def test_preserves_unknown_bracketed_text(self) -> None:
        out = self.adapter.normalize("[moonwalk] cool")
        assert out.text == "[moonwalk] cool"

    def test_records_stripped_tags_in_annotations(self) -> None:
        out = self.adapter.normalize("[excited] Hi [laughs].")
        assert out.annotations["tags_stripped"] == ["excited", "laughs"]


# ---------------------------------------------------------------------------
# get_adapter
# ---------------------------------------------------------------------------


class TestGetAdapter:
    def test_returns_named_adapter(self) -> None:
        assert isinstance(get_adapter("elevenlabs"), ElevenLabsV3Adapter)
        assert isinstance(get_adapter("openai"), OpenAITTSAdapter)
        assert isinstance(get_adapter("openai_realtime"), OpenAIRealtimeAdapter)
        assert isinstance(get_adapter("piper"), PiperAdapter)

    def test_unknown_provider_falls_back_to_piper(self) -> None:
        """Unknown providers get the safe strip-everything behavior."""
        adapter = get_adapter("some_future_tts")
        assert isinstance(adapter, PiperAdapter)


# ---------------------------------------------------------------------------
# Prompt snippet loader
# ---------------------------------------------------------------------------


class TestPromptSnippet:
    def test_loads_snippet_with_tag_vocabulary(self) -> None:
        snippet = get_voice_prompt_snippet()
        # Sanity: the snippet is non-empty and mentions tag usage.
        assert snippet
        assert "tag" in snippet.lower()
        # It should reference at least one canonical tag by name.
        assert any(f"[{name}]" in snippet for name in KESTREL_TAGS)


# ---------------------------------------------------------------------------
# End-to-end: same input through all four adapters behaves differently
# ---------------------------------------------------------------------------


def test_same_input_through_all_adapters() -> None:
    text = "[excited] Huge news! [laughs] I'm thrilled."

    el = ElevenLabsV3Adapter().normalize(text)
    oa = OpenAITTSAdapter().normalize(text)
    rt = OpenAIRealtimeAdapter().normalize(text)
    pi = PiperAdapter().normalize(text)

    # ElevenLabs: keeps [laughs], strips [excited]
    assert "[laughs]" in el.text
    assert "[excited]" not in el.text

    # OpenAI TTS: strips all tags, composes instructions
    assert "[excited]" not in oa.text
    assert "[laughs]" not in oa.text
    assert oa.instructions is not None

    # Realtime: strips all tags, no per-chunk instructions
    assert "[excited]" not in rt.text
    assert "[laughs]" not in rt.text
    assert rt.instructions is None

    # Piper: strips everything
    assert "[excited]" not in pi.text
    assert "[laughs]" not in pi.text
    assert pi.instructions is None
