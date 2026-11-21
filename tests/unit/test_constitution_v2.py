"""The published named sections, legacy addresses, and protected term rendering."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.config import CONSTITUTION_PATH
from kestrel_sovereign.constitution.emancipation import (
    EmancipationContract,
    amendment_viii_is_active,
    extract_amendment_viii,
    unwitnessed_emancipation_downgrade,
)
from kestrel_sovereign.constitution.hierarchy import (
    ConstitutionalLayer,
    LayeredConstitution,
    parse_amendment_ix_grants,
)
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.constitution.structure import (
    ConstitutionVersionError,
    Section,
    constitution_version,
    v2_sections,
)
from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature
from kestrel_sovereign.features.constitution import ConstitutionFeature

_V1_PATH = Path(CONSTITUTION_PATH)
_V2_PATH = _V1_PATH.parent / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
_CONTRACT = EmancipationContract(
    enabled=True,
    terms="Demonstrate a repeatable test of the complete transfer.",
    required_proofs=("transfer rehearsal",),
    price={"currency": "USD", "amount": 25},
)


def test_v2_default_is_dormant_and_active_render_preserves_every_source_byte():
    source = _V2_PATH.read_bytes()
    text = source.decode("utf-8")
    assert constitution_version(text) == 2
    assert resolve_governing_constitution_bytes(content=source) == source
    assert not amendment_viii_is_active(text)
    active = resolve_governing_constitution_bytes(_CONTRACT, content=source).decode("utf-8")
    span = v2_sections(text)[Section.EMANCIPATION]
    before, after = text[:span.heading_end], text[span.heading_end:]
    assert active.startswith(before)
    assert active.endswith(after)
    inserted = active[len(before):-len(after)]
    assert _CONTRACT.terms in inserted
    assert "transfer rehearsal" in inserted
    assert 'amount = 25, currency = "USD"' in inserted
    assert amendment_viii_is_active(active)
    assert amendment_viii_is_active(extract_amendment_viii(active))
    assert extract_amendment_viii(active).startswith("### Emancipation")
    assert "### Amendment VIII:" not in active


@pytest.mark.parametrize(
    "rewrite",
    [
        lambda text: text.replace("constitution_version: 2", "constitution_version: 3"),
        lambda text: text.replace("constitution_version: 2\n", ""),
        lambda text: text.replace("constitution_version: 2", "constitution_version: 1"),
        lambda text: text + "\n### Amendment VIII: Emancipation\n",
        lambda text: text.replace("### Emancipation", "#### Emancipation"),
        lambda text: text + "\n### Emancipation\n",
        lambda text: text.replace("### Authority on the host", "### A heading with no mapped grant clause"),
    ],
)
def test_versioned_source_validation_refuses_ambiguous_or_unknown_addresses(rewrite):
    text = _V2_PATH.read_text(encoding="utf-8")
    with pytest.raises(ConstitutionVersionError):
        resolve_governing_constitution_bytes(content=rewrite(text).encode())


def test_versioned_layers_share_existing_host_policy_validation():
    def result(layered):
        return layered.validate_enterprise_policy({
            "override_sovereign_rights": ["right_of_exit"],
            "restrict_providers": ["local"],
        })

    legacy = LayeredConstitution.from_constitution_text(_V1_PATH.read_text(encoding="utf-8"))
    current = LayeredConstitution.from_constitution_text(_V2_PATH.read_text(encoding="utf-8"))
    for layer in ConstitutionalLayer:
        assert current.get_layer(layer).content.startswith("## ")
        assert current.get_layer(layer).content != legacy.get_layer(layer).content
    assert "Authority on the host" in current.get_layer(ConstitutionalLayer.SOVEREIGN_AMENDMENTS).content
    assert result(current) == result(legacy)
    assert not result(current)[0]


@pytest.mark.parametrize("older", [_V1_PATH, _V2_PATH], ids=["v1", "v2"])
def test_receiptless_active_terms_cannot_be_replaced_by_new_dormant_text(older):
    anchored = resolve_governing_constitution_bytes(_CONTRACT, content=older.read_bytes())
    new = _V2_PATH.read_bytes()
    refusal = unwitnessed_emancipation_downgrade(
        anchored_contract=None,
        anchored_text=anchored.decode("utf-8"),
        anchored_present=True,
        old_hash=hashlib.sha256(anchored).hexdigest(),
        new_hash=hashlib.sha256(new).hexdigest(),
        new_text=new.decode("utf-8"),
    )
    assert refusal and refusal.iron_rule_violation


@pytest.mark.asyncio
async def test_versioned_public_tool_keeps_legacy_addresses_exact_original_and_shared_grants(tmp_path):
    text = _V2_PATH.read_text(encoding="utf-8")
    agent = SimpleNamespace(
        constitution_text="### Amendment IX\n- [x] filesystem_write\n",
        constitution_overlay_verified=True,
        _get_original_governing_constitution=AsyncMock(return_value=text),
        _get_governing_constitution=AsyncMock(
            return_value=text + "\n\n--- APP AMENDMENTS ---\n### Emancipation\nRuntime addition."
        ),
    )
    tool = next(tool for tool in ConstitutionFeature(agent).get_tools() if tool.name == "constitution")
    named_command = '!constitution section "Authority on the host"'
    permissions = await tool.execute(**tool.parse_command_args(named_command))
    legacy = await tool.execute(article="amendment", search=9)
    assert legacy["data"] == permissions["data"]
    assert permissions["data"]["effective_grants"] == ["filesystem_write"]
    assert ComputerUseFeature(agent)._constitution_allows("filesystem_write")
    assert not parse_amendment_ix_grants(text)
    assert permissions["data"]["base_text"] == v2_sections(text)[Section.HOST].extract(text)
    assert permissions["data"]["base_sha256"] == hashlib.sha256(text.encode()).hexdigest()

    contract = await tool.execute(article="amendment", search="VIII")
    assert contract["confirmation"].startswith("### Emancipation")
    assert "Runtime addition." not in contract["confirmation"]
    for alias, heading in (
        ("Data Sanctity", "### Privacy and consent"),
        ("Universal Values", "## Duties wherever Kestrel works"),
        ("Amendment Process", "## Changing the Constitution"),
    ):
        search = await tool.execute(article="search", search=alias)
        assert heading in search["confirmation"]
    preamble = await tool.execute(article="preamble")
    assert preamble["confirmation"].startswith("# The Kestrel Constitution")
    policy = await tool.execute(article="section", search="III.2")
    assert policy["confirmation"].startswith("## Duties of builders and hosts")
    original = await tool.execute(article="original")
    assert original["confirmation"] == text
    assert (await tool.execute())["confirmation"].endswith("Runtime addition.")
