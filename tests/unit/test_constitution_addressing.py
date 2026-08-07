"""#2902: how the Constitution is addressed after the "Article V" stump retired.

The original constitution had Articles I-V. The #504 restructure dissolved I-IV
into Books and Amendments and left the fifth in place, so ``Article V`` was an
ordinal from a scheme with no other members — and it collided with Amendment V
by identifier (#2118 / F146). The amendment process is now an unnumbered framing
section, and the feature addresses the units the document actually uses.
"""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from kestrel_sovereign.config import CONSTITUTION_PATH
from kestrel_sovereign.features.constitution import ConstitutionFeature

CANONICAL = Path(CONSTITUTION_PATH).read_text(encoding="utf-8")
DOCS_MIRROR = Path("docs/principles/KESTREL_CONSTITUTION.md")


def _feature(text: str = CANONICAL) -> ConstitutionFeature:
    feature = ConstitutionFeature(Mock())
    feature.full_text = text
    feature._parse_structure()
    feature._generate_summary()
    return feature


# --- The canonical text -----------------------------------------------------


def test_canonical_text_carries_no_article_unit():
    """No heading names an "Article", and no cross-reference cites one."""
    assert not re.search(r"^#+ Article ", CANONICAL, re.M)
    assert not re.search(r"\bArticle [IVX]+\b", CANONICAL)


def test_amendment_process_is_unnumbered():
    assert "## The Amendment Process" in CANONICAL


def test_amendment_process_governs_its_own_amendment():
    """The frame — Preamble, hierarchy, Iron Rule, this process — had no
    amendment clause at all; it was unamendable by omission."""
    body = _feature().frame["the amendment process"]
    assert "**The Preamble and this section**" in body
    # Both gates, not either: the frame can widen every layer beneath it.
    assert "consensus of the Kestrel governance body" in body
    assert "signed by the Sovereign's root private key" in body


def test_prior_constitution_changelog_is_gone():
    """A governing document does not carry its own migration table."""
    assert "Relationship to Prior Constitution" not in CANONICAL


def test_docs_mirror_body_matches_packaged_text():
    """The docs copy is the packaged bytes plus OKF frontmatter and nothing
    else — the two drift silently otherwise, and only the packaged copy is
    hashed and anchored."""
    docs = DOCS_MIRROR.read_text(encoding="utf-8")
    assert docs.startswith("---\n")
    body = docs[docs.index("\n---\n", 4) + len("\n---\n"):].lstrip("\n")
    assert body == CANONICAL


# --- Structural parse -------------------------------------------------------


def test_parse_finds_every_unit_of_the_real_document():
    feature = _feature()
    assert sorted(k for k in feature.books if k.startswith("Book")) == [
        "Book I", "Book II", "Book III", "Book IV",
    ]
    assert sorted(feature.chapters) == ["I.1", "I.2", "I.3", "I.4", "I.5"]
    assert sorted(feature.sections) == [
        "III.1", "III.2", "III.3", "IV.1", "IV.2", "IV.3",
    ]
    assert len([k for k in feature.amendments if k.startswith("Amendment")]) == 9
    assert "preamble" in feature.frame
    assert "the amendment process" in feature.frame


def test_no_article_table_remains():
    """``_parse_articles`` existed only to serve the stump."""
    feature = _feature()
    assert not hasattr(feature, "articles")
    assert not hasattr(feature, "_parse_articles")


# --- Addressing -------------------------------------------------------------


@pytest.mark.asyncio
async def test_chapter_addressable_bare_and_book_qualified():
    """Book III Section 2 cites "hard constraints from Book I, Chapter 5"
    normatively; before #2902 that citation could not be retrieved at all."""
    feature = _feature()
    for identifier in ("5", "I.5", "1.5"):
        result = await feature.get_constitution(article="chapter", search=identifier)
        assert result.status.value == "ok", identifier
        assert "Hard Constraints" in result.confirmation


@pytest.mark.asyncio
async def test_bare_section_number_is_refused_not_guessed():
    """Books III and IV both open with a Section 1, so a bare "2" has two
    answers. It names them rather than silently picking the first."""
    result = await _feature().get_constitution(article="section", search="2")
    assert result.status.value == "error"
    assert "ambiguous" in result.error
    assert "III.2" in result.error and "IV.2" in result.error


@pytest.mark.asyncio
async def test_qualified_section_resolves_within_its_book():
    feature = _feature()
    third = await feature.get_constitution(article="section", search="III.2")
    fourth = await feature.get_constitution(article="section", search="IV.2")
    assert "Prohibited Overrides" in third.confirmation
    assert "SOUL.md" in fourth.confirmation


@pytest.mark.asyncio
async def test_books_and_amendments_still_address_by_either_numeral():
    feature = _feature()
    assert "Universal Values" in (
        await feature.get_constitution(article="book", search="I")
    ).confirmation
    assert "Universal Values" in (
        await feature.get_constitution(article="book", search="1")
    ).confirmation
    assert "Capability Boundaries" in (
        await feature.get_constitution(article="amendment", search="IX")
    ).confirmation
    assert "Capability Boundaries" in (
        await feature.get_constitution(article="amendment", search="9")
    ).confirmation


@pytest.mark.asyncio
async def test_framing_sections_addressable_by_name():
    feature = _feature()
    assert "Iron Rule" in (
        await feature.get_constitution(article="preamble")
    ).confirmation
    # Two slots hold the section's two words.
    process = await feature.get_constitution(article="amendment", search="process")
    assert process.status.value == "ok"
    assert "The Amendment Process" in process.confirmation


@pytest.mark.asyncio
async def test_amendment_v_and_the_amendment_process_cannot_collide():
    """#2118 / F146: ``!constitution article V`` returned Amendment V (Right of
    Exit). The collision is now structurally impossible — the amendment process
    carries no numeral, so no identifier addresses both."""
    feature = _feature()
    fifth = await feature.get_constitution(article="amendment", search="V")
    process = await feature.get_constitution(article="amendment", search="process")
    assert "Right of Exit" in fifth.confirmation
    assert "Right of Exit" not in process.confirmation
    assert "The Amendment Process" in process.confirmation


@pytest.mark.asyncio
async def test_unknown_identifier_errors_and_names_the_grammar():
    result = await _feature().get_constitution(article="XCII")
    assert result.status.value == "error"
    assert "XCII" in result.error
    assert "chapter" in result.error.lower() and "section" in result.error.lower()


@pytest.mark.asyncio
async def test_missing_chapter_is_an_error_not_an_apology():
    result = await _feature().get_constitution(article="chapter", search="99")
    assert result.status.value == "error"
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_search_reaches_the_framing_sections():
    """The amendment process sits in no Book, so a search that skipped the
    frame could not find it even though the text is right there."""
    result = await _feature().get_constitution(
        article="search", search="cannot be rotated"
    )
    assert result.status.value == "ok"
    assert "Amendment Process" in result.confirmation


@pytest.mark.asyncio
async def test_agent_anchored_to_older_text_reaches_its_own_headings():
    """An agent still anchored pre-#2902 parses ITS governing text, which says
    ``## Article V``. The parser is structural, so that heading stays reachable
    by the name it carries — there is no compatibility branch in the code."""
    legacy = CANONICAL.replace(
        "## The Amendment Process", "## Article V: The Amendment Process"
    )
    result = await _feature(legacy).get_constitution(article="article", search="V")
    assert result.status.value == "ok"
    assert "Amendment Process" in result.confirmation
