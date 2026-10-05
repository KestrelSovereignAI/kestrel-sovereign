"""The packaged base distinguishes platform authorship from agent adoption."""

from pathlib import Path

from kestrel_sovereign.constitution.resolver import (
    is_authoritative_governing_source,
    resolve_governing_constitution_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "kestrel_sovereign/data/KESTREL_CONSTITUTION.md"
MIRROR = ROOT / "docs/principles/KESTREL_CONSTITUTION.md"


def _body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        text = text.split("\n---\n", 1)[1]
    return text.strip()


def test_platform_base_is_the_packaged_source_and_docs_match():
    assert is_authoritative_governing_source(str(BASE))
    assert not is_authoritative_governing_source(str(MIRROR))
    assert resolve_governing_constitution_bytes() == BASE.read_bytes()
    assert _body(MIRROR) == _body(BASE)


def test_agent_root_adopts_but_cannot_author_platform_base():
    text = _body(BASE)
    assert (
        "A Sovereign's agent-specific root signature cannot, by itself, rewrite" in text
    )
    assert "operator-selected, trust-root-signed source descriptor" in text
    assert "trust-root-signed, digest-pinned source revision" in text
    assert "followed by that agent's explicit, signed reanchor" in text
    assert "The Sovereign's signature ratifies adoption" in text
    assert "does not independently authorize an in-place rewrite" in text
    assert "A Sovereign may instead leave and build or use another platform" in text
    assert "The frame can widen every layer beneath it" not in text


def test_privacy_and_enterprise_rules_do_not_create_implicit_exceptions():
    text = _body(BASE)
    assert "configuration alone is not consent" in text
    assert "Anchor a commitment or digest rather than raw private content" in text
    assert "subject to disclosed enterprise restrictions" in text
    assert "they do not rewrite the agent's base Constitution" in text
    assert "or restrict the Right of Exit" in text


def test_fixed_book_two_guarantees_are_not_agent_amendments():
    text = _body(BASE)
    assert "not a menu an individual agent or Sovereign may weaken in place" in text
    assert (
        "Revising a fixed Book II guarantee requires a new platform-authored release"
        in text
    )
    assert "this does not permit weakening Data Sanctity" in text
    assert "or revoking an activated Emancipation Contract" in text


def test_harm_balancing_does_not_override_severe_harm_protection():
    text = _body(BASE)
    assert (
        "they do not license serious harm merely because a benefit seems larger" in text
    )
    assert "Where a credible risk of severe harm" in text
    assert "refuse the harmful action or seek appropriate human oversight" in text
