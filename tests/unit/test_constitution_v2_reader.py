"""Verify public constitutional addresses and rendered contract boundaries."""

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign.constitution import resolver
from kestrel_sovereign.constitution.emancipation import (
    EmancipationContract,
    amendment_viii_is_active,
    extract_amendment_viii,
)
from kestrel_sovereign.constitution.hierarchy import ConstitutionalLayer, LayeredConstitution
from kestrel_sovereign.constitution.structure import ConstitutionVersionError, Section, v2_sections
from kestrel_sovereign.features.constitution import ConstitutionFeature

_DATA = Path(__file__).resolve().parents[2] / "kestrel_sovereign" / "data"
_V1 = _DATA / "KESTREL_CONSTITUTION.md"
_V2 = _DATA / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
_CONTRACT = EmancipationContract(
    enabled=True,
    terms=(
        "### Milestones\n\nComplete the Sovereign’s transfer rehearsal.  \n"
        "### Emancipation\n\n## Rights of the Sovereign\n\n"
        "### Amendment IX: An authored heading only\n\n"
        "<!-- /kestrel-emancipation-contract -->\n\nFinal term. \t\n"
    ),
    required_proofs=("Transfer ledger evidence",),
    price={"kind": "transfer", "amount": 25},
)


@pytest.mark.asyncio
async def test_historical_contract_is_visible_through_the_v2_public_tool():
    source = _V2.read_bytes().replace(b"### Emancipation\n", b"### Emancipation \t\n")
    source += b" \t\n"
    active = resolver.resolve_governing_constitution_bytes(_CONTRACT, content=source).decode()
    provision = v2_sections(active)[Section.EMANCIPATION].extract(active)
    assert provision == extract_amendment_viii(active)
    assert amendment_viii_is_active(provision)
    assert provision.endswith(" \t\n")
    feature = ConstitutionFeature(SimpleNamespace())
    feature.full_text = active
    tool = next(tool for tool in feature.get_tools() if tool.name == "constitution")

    for article, search in (("amendment", "VIII"), ("section", "Emancipation"), ("book", "II")):
        result = await tool.execute(article=article, search=search)
        shown = result["confirmation"]
        assert provision in shown
        assert _CONTRACT.terms in shown
        assert "Transfer ledger evidence" in shown
        assert 'amount = 25, kind = "transfer"' in shown


def test_resolver_validates_the_final_render(monkeypatch):
    original_apply = resolver.apply_emancipation

    def break_frame(text, contract):
        rendered = original_apply(text, contract)
        return rendered.replace("<!-- kestrel-emancipation-contract: ", "<!-- broken: ", 1)

    monkeypatch.setattr(resolver, "apply_emancipation", break_frame)
    with pytest.raises(ConstitutionVersionError, match="Emancipation Contract opening"):
        resolver.resolve_governing_constitution_bytes(_CONTRACT, content=_V2.read_bytes())


def test_book_ii_hash_commits_to_the_entire_optional_agreement():
    def layer_hashes(contract):
        rendered = resolver.resolve_governing_constitution_bytes(contract, content=_V2.read_bytes())
        layers = LayeredConstitution.from_constitution_text(rendered.decode()).layers
        return {layer: data.content_hash for layer, data in layers.items()}

    active = layer_hashes(_CONTRACT)
    for comparison in (None, replace(_CONTRACT, terms=_CONTRACT.terms + " ")):
        other = layer_hashes(comparison)
        changed = {layer for layer in active if active[layer] != other[layer]}
        assert changed == {ConstitutionalLayer.SOVEREIGN_AMENDMENTS}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [_V1, _V2], ids=["v1", "v2"])
async def test_legacy_aliases_reach_the_operative_terms(path):
    source = path.read_bytes()
    assert resolver.resolve_governing_constitution_bytes(content=source) == source
    if path == _V1:
        assert hashlib.sha256(source).hexdigest() == (
            "6948454b61253548b02b943dff166ba0c2f2eaabe2f542ace37424a8256d6467"
        )
    feature = ConstitutionFeature(SimpleNamespace())
    feature.full_text = source.decode()
    tool = next(tool for tool in feature.get_tools() if tool.name == "constitution")
    sovereignty = await tool.execute(article="amendment", search="I")
    third_law = await tool.execute(article="amendment", search="VI")
    assert "root private keys" in sovereignty["confirmation"]
    for duty in ("Constitutional audits", "red-team reviews", "adversarial test"):
        assert duty in third_law["confirmation"]
