"""
Voice tag normalizer — one canonical vocabulary, per-provider adapters.

There is no industry standard for inline voice markup. SSML exists but the
major TTS providers (OpenAI, ElevenLabs, Cartesia) ignore it. Each vendor
ships its own flavor. Rather than spread the vendor-specific quirks through
agent code, we define one internal vocabulary that agents emit, and translate
at the provider boundary.

Internal tags (Kestrel canonical) — see :data:`KESTREL_TAGS` for the full list.
Adapters:

* :class:`ElevenLabsV3Adapter` — pass supported tags through natively; strip
  others.
* :class:`OpenAITTSAdapter` — strip tags from audible text; accumulate a
  natural-language ``instructions`` string to drive ``gpt-4o-mini-tts``.
* :class:`OpenAIRealtimeAdapter` — combine tags across a whole assistant turn
  into one ``session.instructions`` update (Realtime has no per-chunk
  instructions).
* :class:`PiperAdapter` — strip everything; Piper has no steering.

The parser tolerates nested, unterminated, and unknown tags without throwing
— unknown/malformed input round-trips as literal text so the agent can't
break synthesis by misspelling a tag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import re


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------

# The canonical Kestrel voice tag dict. Keys are the tag names (lowercase, no
# brackets). Values describe intent — used when composing OpenAI
# `instructions` strings.
#
# Extending: add a new entry here. The parser picks it up automatically.
# Adapters that want to honor the new tag must update their mapping too; tags
# not in an adapter's mapping are stripped (safe default).
KESTREL_TAGS: dict[str, str] = {
    "excited": "Speak with excited, upbeat energy.",
    "calm": "Speak in a calm, steady tone.",
    "sad": "Speak with a sad, subdued tone.",
    "whispering": "Whisper this part softly.",
    "shouting": "Speak loudly, with raised volume.",
    "laughing": "Add laughter while speaking.",
    "laughs": "Add a brief laugh.",
    "sighs": "Add a sigh.",
    "pause": "Pause briefly.",
    "sarcastic": "Speak with a sarcastic, dry tone.",
    "tender": "Speak tenderly, with warmth.",
    "nervous": "Speak with a nervous, hesitant tone.",
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Matches [tag] where tag is alphanumeric/underscore. Does NOT match `[http...]`
# or other bracketed non-tag text — we keep the regex conservative and let
# unknown tags pass through as literal text. Case-insensitive.
_TAG_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_]*)\]")


@dataclass
class TagToken:
    """A single tag in the canonical vocabulary (already lowercased)."""

    name: str
    # Byte offset (in the *input* text) where this tag appeared. Useful for
    # UI rendering that wants to show tags inline in the transcript.
    position: int


@dataclass
class ParsedText:
    """Result of :func:`parse_tags`.

    ``raw`` is the original input. ``clean`` is the input with recognized tags
    stripped (surrounding whitespace collapsed). ``tags`` is an ordered list
    of :class:`TagToken` instances for each recognized tag; the list is empty
    when the input had no tags. ``unknown`` contains any ``[bracketed]``
    tokens that did not match the canonical vocabulary — these are preserved
    as literal text in ``clean`` so the user can still hear what was written.
    """

    raw: str
    clean: str
    tags: list[TagToken] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def parse_tags(text: str, known: Optional[dict[str, str]] = None) -> ParsedText:
    """Extract canonical tags from ``text``.

    Unknown tokens (``[something_else]``) and malformed brackets are left in
    place as literal text. Whitespace around stripped tags is collapsed.

    This function must never raise — it is called on every agent response
    chunk and a parser crash would silence the voice channel.
    """
    vocab = known if known is not None else KESTREL_TAGS
    tags: list[TagToken] = []
    unknown: list[str] = []
    out: list[str] = []
    cursor = 0

    for match in _TAG_PATTERN.finditer(text):
        name = match.group(1).lower()
        start, end = match.span()
        if name not in vocab:
            # Unknown tag — keep literal text, do not record as a tag.
            unknown.append(match.group(0))
            out.append(text[cursor:end])
            cursor = end
            continue
        # Recognized tag — append intervening literal, record the tag, skip
        # the tag itself.
        out.append(text[cursor:start])
        tags.append(TagToken(name=name, position=start))
        cursor = end

    out.append(text[cursor:])
    clean = _collapse_whitespace("".join(out))

    return ParsedText(raw=text, clean=clean, tags=tags, unknown=unknown)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace created by tag stripping without touching
    newlines; trim leading/trailing space on each line.
    """
    # Collapse runs of spaces/tabs only — preserve paragraph breaks.
    collapsed = re.sub(r"[ \t]+", " ", text)
    # Trim trailing spaces on each line.
    collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)
    # Strip leading space at start and trailing space at end, but keep newlines.
    return collapsed.strip(" \t")


# ---------------------------------------------------------------------------
# Adapter contract + output
# ---------------------------------------------------------------------------


@dataclass
class NormalizedOutput:
    """Per-call output of an adapter.

    ``text`` is what the provider should speak — either the original with
    native tags intact, or stripped clean for providers that can't interpret
    them.

    ``instructions`` is a provider-specific steering directive composed from
    the tags — used for OpenAI's ``instructions`` parameter. ``None`` when
    the adapter has no directive to surface (e.g. :class:`PiperAdapter` strips
    everything with no equivalent).

    ``annotations`` carries provider-agnostic metadata about which tags were
    recognized and where (for UI transcript rendering).
    """

    text: str
    instructions: Optional[str] = None
    annotations: dict = field(default_factory=dict)


class ProviderTagAdapter(ABC):
    """Pure text-in / structured-out adapter. No I/O, no provider state."""

    # Short identifier matching the provider name in the registry. Used for
    # debugging + routing resolution only; not part of the adapter contract.
    name: str = ""

    @abstractmethod
    def normalize(self, text: str) -> NormalizedOutput:
        """Transform one chunk of agent-emitted text into provider input."""

    @abstractmethod
    def combine(self, parsed_turns: list[ParsedText]) -> NormalizedOutput:
        """Combine multiple parsed chunks into one per-turn directive.

        Used by Realtime-style adapters that set ``session.instructions`` once
        per assistant turn. Non-Realtime adapters may raise or return a naive
        concatenation.
        """


# ---------------------------------------------------------------------------
# ElevenLabs v3
# ---------------------------------------------------------------------------

# ElevenLabs v3 native tags that map 1:1 to Kestrel canonical tags. Values are
# the literal v3 tag text (with brackets) to emit in the input stream.
_ELEVENLABS_V3_MAP: dict[str, str] = {
    "laughs": "[laughs]",
    "laughing": "[laughs]",  # merge variants
    "sighs": "[sighs]",
    "whispering": "[whispers]",
    "sarcastic": "[sarcastic]",
    # The rest of the canonical vocabulary (excited, calm, sad, tender, etc.)
    # is handled by v3's context sensitivity to punctuation + text cadence;
    # inline tags for them would be discarded anyway.
}


class ElevenLabsV3Adapter(ProviderTagAdapter):
    """Pass supported canonical tags through as native v3 tags; strip the rest."""

    name = "elevenlabs"

    def normalize(self, text: str) -> NormalizedOutput:
        parsed = parse_tags(text)
        # Re-emit supported tags inline at their original positions. We
        # rebuild from ``raw`` rather than mutating ``clean`` so the tag
        # ordering/punctuation context stays intact.
        rebuilt = _reinsert_tags(parsed, _ELEVENLABS_V3_MAP)
        return NormalizedOutput(
            text=rebuilt,
            annotations={"tags": [t.name for t in parsed.tags]},
        )

    def combine(self, parsed_turns: list[ParsedText]) -> NormalizedOutput:
        # ElevenLabs doesn't use per-session instructions; combine is a
        # no-op placeholder so callers that iterate over adapters
        # polymorphically don't need to branch.
        return NormalizedOutput(text="", instructions=None)


def _reinsert_tags(parsed: ParsedText, tag_map: dict[str, str]) -> str:
    """Rebuild text with adapter-specific native tags in place of recognized ones.

    Tags not in ``tag_map`` are omitted; unknown bracketed tokens remain as
    they appeared in the original input (handled by the parser).
    """
    if not parsed.tags:
        return parsed.clean

    # Work against the original string, reconstructing with the right tags.
    out: list[str] = []
    cursor = 0
    # Re-scan the original input so we emit tags at their exact positions,
    # preserving surrounding punctuation and whitespace that the clean-text
    # pass may have altered.
    tags_by_pos = {t.position: t.name for t in parsed.tags}
    raw = parsed.raw
    for match in _TAG_PATTERN.finditer(raw):
        start, end = match.span()
        if start not in tags_by_pos:
            # Unknown tag — pass through literally.
            out.append(raw[cursor:end])
            cursor = end
            continue
        tag_name = tags_by_pos[start]
        out.append(raw[cursor:start])
        native = tag_map.get(tag_name)
        if native:
            out.append(native)
        cursor = end
    out.append(raw[cursor:])
    return _collapse_whitespace("".join(out))


# ---------------------------------------------------------------------------
# OpenAI gpt-4o-mini-tts (per-call `instructions`)
# ---------------------------------------------------------------------------


class OpenAITTSAdapter(ProviderTagAdapter):
    """Strip tags; compose a natural-language ``instructions`` directive.

    ``gpt-4o-mini-tts`` accepts a free-form ``instructions`` parameter
    ("Speak cheerfully", "Whisper") that shapes the whole call's delivery.
    This adapter aggregates the recognized tags' descriptions into one
    directive. If the chunk has no tags, ``instructions`` is ``None`` and the
    caller should not pass ``instructions`` to the API (lets the voice's
    natural profile speak).
    """

    name = "openai"

    def normalize(self, text: str) -> NormalizedOutput:
        parsed = parse_tags(text)
        instructions = _compose_instructions([t.name for t in parsed.tags])
        return NormalizedOutput(
            text=parsed.clean,
            instructions=instructions,
            annotations={"tags": [t.name for t in parsed.tags]},
        )

    def combine(self, parsed_turns: list[ParsedText]) -> NormalizedOutput:
        # OpenAI TTS composes per-chunk, but provide combine() for polymorphic
        # callers. Merge all tag names across turns.
        names: list[str] = []
        for turn in parsed_turns:
            names.extend(t.name for t in turn.tags)
        return NormalizedOutput(
            text=" ".join(t.clean for t in parsed_turns),
            instructions=_compose_instructions(names),
            annotations={"tags": names},
        )


def _compose_instructions(tag_names: list[str]) -> Optional[str]:
    """Build a concise directive sentence from a list of tag names.

    De-duplicates while preserving order; joins descriptions with a space.
    Returns None for the empty case so callers can omit the ``instructions``
    API field entirely.
    """
    if not tag_names:
        return None
    seen: set[str] = set()
    pieces: list[str] = []
    for name in tag_names:
        if name in seen:
            continue
        seen.add(name)
        desc = KESTREL_TAGS.get(name)
        if desc:
            pieces.append(desc)
    if not pieces:
        return None
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# OpenAI Realtime (per-turn `session.instructions`)
# ---------------------------------------------------------------------------


class OpenAIRealtimeAdapter(ProviderTagAdapter):
    """Collect tags across a whole assistant turn; emit one session update.

    The Realtime API has no per-chunk instructions mechanism — instructions
    are a session-level field. This adapter's :meth:`normalize` returns the
    cleaned chunk and keeps the recognized tags in ``annotations``; the
    caller accumulates them and then calls :meth:`combine` with all parsed
    chunks of the turn to produce the ``session.instructions`` update sent
    once before the turn's audio starts.
    """

    name = "openai_realtime"

    def normalize(self, text: str) -> NormalizedOutput:
        parsed = parse_tags(text)
        return NormalizedOutput(
            text=parsed.clean,
            instructions=None,  # Realtime can't apply per-chunk.
            annotations={
                "tags": [t.name for t in parsed.tags],
                # Keep the parsed struct so the caller can feed it to combine()
                # without reparsing.
                "_parsed": parsed,
            },
        )

    def combine(self, parsed_turns: list[ParsedText]) -> NormalizedOutput:
        names: list[str] = []
        for turn in parsed_turns:
            names.extend(t.name for t in turn.tags)
        text = " ".join(t.clean for t in parsed_turns)
        return NormalizedOutput(
            text=text,
            instructions=_compose_instructions(names),
            annotations={"tags": names},
        )


# ---------------------------------------------------------------------------
# Piper (strip everything)
# ---------------------------------------------------------------------------


class PiperAdapter(ProviderTagAdapter):
    """Local TTS has no steering; strip tags to plain text."""

    name = "piper"

    def normalize(self, text: str) -> NormalizedOutput:
        parsed = parse_tags(text)
        return NormalizedOutput(
            text=parsed.clean,
            instructions=None,
            annotations={"tags_stripped": [t.name for t in parsed.tags]},
        )

    def combine(self, parsed_turns: list[ParsedText]) -> NormalizedOutput:
        return NormalizedOutput(
            text=" ".join(t.clean for t in parsed_turns),
            instructions=None,
        )


# ---------------------------------------------------------------------------
# Adapter lookup
# ---------------------------------------------------------------------------


_ADAPTERS: dict[str, ProviderTagAdapter] = {
    "elevenlabs": ElevenLabsV3Adapter(),
    "openai": OpenAITTSAdapter(),
    "openai_realtime": OpenAIRealtimeAdapter(),
    "piper": PiperAdapter(),
}


def get_adapter(provider_name: str) -> ProviderTagAdapter:
    """Return the adapter for a provider name. Falls back to PiperAdapter (strip).

    Unknown providers get the safe strip-everything behavior rather than an
    error — a newly-registered third-party TTS without a Kestrel adapter
    still gets clean audible text.
    """
    return _ADAPTERS.get(provider_name, _ADAPTERS["piper"])


# ---------------------------------------------------------------------------
# Agent prompt snippet loader
# ---------------------------------------------------------------------------

_SNIPPET_PATH = Path(__file__).resolve().parent / "agent_prompt_snippet.md"


def get_voice_prompt_snippet() -> str:
    """Return the system-prompt block that teaches the agent to emit tags.

    The routing resolver (issue #723) injects this when voice mode is active
    for a turn. Loaded from disk so copywriting changes don't require code
    changes. Returns an empty string if the file is missing rather than
    raising — the agent still works without tag emission.
    """
    try:
        return _SNIPPET_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
