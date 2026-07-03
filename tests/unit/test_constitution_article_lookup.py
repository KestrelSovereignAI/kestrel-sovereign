"""F146: `!constitution article V` must return Article V (Amendment Process),
not Amendment V (Right of Exit), which shadowed it in the generic cascade."""

from unittest.mock import Mock

import pytest

from kestrel_sovereign.features.constitution import ConstitutionFeature


def _feature() -> ConstitutionFeature:
    feature = ConstitutionFeature(Mock())
    # Mirror the real parse: keys are "Article V" / "Amendment V" (plus int
    # aliases), and Article V collides with Amendment V by id.
    feature.articles = {"Article V": "ARTICLE FIVE — Amendment Process", "5": "ARTICLE FIVE — Amendment Process"}
    feature.amendments = {"Amendment V": "AMENDMENT FIVE — Right of Exit", "5": "AMENDMENT FIVE — Right of Exit"}
    feature.summary = "SUMMARY"
    return feature


@pytest.mark.asyncio
async def test_article_subcommand_returns_article_not_amendment():
    result = await _feature().get_constitution(article="article", search="V")
    assert result.status.value == "ok"
    assert "ARTICLE FIVE" in result.confirmation
    assert "AMENDMENT FIVE" not in result.confirmation


@pytest.mark.asyncio
async def test_amendment_subcommand_still_returns_amendment():
    result = await _feature().get_constitution(article="amendment", search="V")
    assert result.status.value == "ok"
    assert "AMENDMENT FIVE" in result.confirmation


@pytest.mark.asyncio
async def test_missing_article_surfaces_error_not_unrelated_match():
    result = await _feature().get_constitution(article="article", search="XCII")
    assert result.status.value == "error"
    assert "Article 'XCII' not found" in result.error
