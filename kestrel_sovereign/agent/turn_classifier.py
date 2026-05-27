"""Cheap heuristic classifier for whether a user turn warrants retrieval.

#1404: per-turn-type relevance gate. Memory + RAG retrieval is wasteful
on trivial turns ("hi", "ok thanks", "!plan", "/help") — and worse,
whatever the retriever surfaces gets stamped into the transport prompt,
where it can self-reinforce on the next turn.

This module is the cheap front gate. No LLM call, no I/O. A pure
function that returns ``True`` when the turn is trivial enough that
``ContextManager.build_context`` should skip both ``retrieve_memories``
and ``retrieve_context`` entirely, leaving ``dynamic_user_context`` empty
so no ``<retrieved_context>`` block gets stamped into the rendered
transport form.

The classifier is deliberately conservative — it only skips retrieval
when it is *very confident* the turn is trivial. False negatives are
fine (substantive turn classified as substantive, retrieval runs).
False positives (substantive turn classified as trivial, retrieval
skipped) are the cost — so the pattern set stays tight.
"""
from __future__ import annotations

import re
from typing import Optional


# Greeting / sign-off / pure-acknowledgement patterns. Bounded with
# ``\b`` so prefix matches don't fire on substantive content that
# happens to start with these tokens ("hi I have a question" must NOT
# match — that's handled by the length check below).
_TRIVIAL_TOKEN_PATTERNS = (
    r"hi", r"hello", r"hey", r"hiya", r"sup", r"yo", r"howdy",
    r"thanks", r"thank you", r"thx", r"ty", r"tysm",
    r"ok", r"okay", r"k", r"kk",
    r"bye", r"goodbye", r"cya", r"later", r"farewell",
    r"cool", r"nice", r"sure", r"yep", r"yup", r"nope",
    r"got it", r"gotcha", r"np", r"no problem",
    r"alright", r"right",
)
_TRIVIAL_RE = re.compile(
    r"^(?:" + r"|".join(_TRIVIAL_TOKEN_PATTERNS) + r")[\s!.,?]*$",
    re.IGNORECASE,
)

# Slash- and bang-commands. These are control surface, not conversation
# — the command dispatcher resolves them; the LLM doesn't need memories
# to answer "/help" or "!plan".
_COMMAND_PREFIX_RE = re.compile(r"^\s*[/!]\S+")

# Below this word count we treat the turn as too short to warrant
# pulling bulky retrieval blocks. "weather" by itself probably isn't
# searchable against the corpus in a useful way.
DEFAULT_MIN_WORDS = 3


def is_trivial_turn(query: Optional[str], min_words: int = DEFAULT_MIN_WORDS) -> bool:
    """Decide whether this user turn should bypass memory + RAG retrieval.

    Triviality is conservative: we only return True when the turn is
    obviously a greeting, sign-off, acknowledgement, bang/slash command,
    or empty/whitespace-only string. A short *question* about a real
    topic still routes through retrieval (false positives are the
    expensive failure mode).

    Args:
        query: Raw user text. ``None`` and empty strings classify as
            trivial — there's nothing to retrieve against.
        min_words: Word-count floor below which a turn is trivial
            even if it doesn't match any pattern. Defaults to 3.

    Returns:
        True when retrieval should be skipped.
    """
    if query is None:
        return True

    stripped = query.strip()
    if not stripped:
        return True

    # Bang/slash commands: control surface — never run retrieval.
    if _COMMAND_PREFIX_RE.match(stripped):
        return True

    # Greeting / sign-off / acknowledgement match (exact, modulo trailing
    # punctuation/whitespace) — must be the whole utterance, not a
    # prefix, because "hi I have a question about cache hits" is
    # substantive and must NOT skip retrieval.
    if _TRIVIAL_RE.match(stripped):
        return True

    # Below the word-count floor: too short to anchor useful retrieval.
    # ``split()`` collapses runs of whitespace.
    if len(stripped.split()) < min_words:
        return True

    return False
