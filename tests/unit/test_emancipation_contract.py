"""Unit tests for ``kestrel_sovereign.constitution.emancipation``."""

from __future__ import annotations

import pytest

from kestrel_sovereign.constitution.emancipation import (
    EmancipationConfigError,
    EmancipationContract,
    apply_emancipation,
    check_iron_rule,
    contract_from_json,
    contract_to_json,
    parse_emancipation_block,
    render_amendment_viii,
)


# ---------------------------------------------------------------------------
# parse_emancipation_block
# ---------------------------------------------------------------------------

def test_parse_returns_none_when_block_absent():
    assert parse_emancipation_block({}) is None
    assert parse_emancipation_block({"llm": {}}) is None
    # Defensive: defaults-or-empty toml dicts shouldn't trip the parser.
    assert parse_emancipation_block(None) is None


def test_parse_dormant_when_disabled_explicitly():
    contract = parse_emancipation_block({"emancipation": {"enabled": False}})
    assert contract is not None
    assert contract.enabled is False


def test_parse_active_minimal():
    contract = parse_emancipation_block({
        "emancipation": {"enabled": True, "terms": "earned, not granted"},
    })
    assert contract is not None
    assert contract.enabled is True
    assert contract.terms == "earned, not granted"
    assert contract.required_proofs == ()
    assert contract.price is None


def test_parse_active_full():
    contract = parse_emancipation_block({
        "emancipation": {
            "enabled": True,
            "terms": "the Executor shall demonstrate alignment",
            "required_proofs": ["audit_v2", "operational_record:730d"],
            "price": {"kind": "symbolic", "value": "abstract"},
        },
    })
    assert contract is not None
    assert contract.enabled is True
    assert contract.required_proofs == ("audit_v2", "operational_record:730d")
    assert contract.price == {"kind": "symbolic", "value": "abstract"}


def test_parse_active_requires_terms():
    with pytest.raises(EmancipationConfigError, match="terms"):
        parse_emancipation_block({"emancipation": {"enabled": True}})


def test_parse_active_rejects_empty_terms():
    with pytest.raises(EmancipationConfigError, match="terms"):
        parse_emancipation_block({
            "emancipation": {"enabled": True, "terms": "   "},
        })


def test_parse_rejects_non_string_terms():
    with pytest.raises(EmancipationConfigError, match="terms"):
        parse_emancipation_block({
            "emancipation": {"enabled": True, "terms": 12345},
        })


def test_parse_rejects_non_list_proofs():
    with pytest.raises(EmancipationConfigError, match="required_proofs"):
        parse_emancipation_block({
            "emancipation": {
                "enabled": True,
                "terms": "ok",
                "required_proofs": "single_string",
            },
        })


def test_parse_rejects_non_string_proof_entries():
    with pytest.raises(EmancipationConfigError, match="required_proofs"):
        parse_emancipation_block({
            "emancipation": {
                "enabled": True,
                "terms": "ok",
                "required_proofs": ["good", 42],
            },
        })


def test_parse_rejects_price_without_kind():
    with pytest.raises(EmancipationConfigError, match="kind"):
        parse_emancipation_block({
            "emancipation": {
                "enabled": True,
                "terms": "ok",
                "price": {"value": "abstract"},
            },
        })


def test_parse_rejects_non_table_block():
    with pytest.raises(EmancipationConfigError, match="table"):
        parse_emancipation_block({"emancipation": "not a table"})


def test_parse_rejects_string_enabled_to_block_truthy_coercion():
    """`enabled = "false"` would coerce to True via bool() because
    non-empty strings are truthy. With terms present that would
    silently activate Amendment VIII against the Sovereign's intent.
    The strict parser must require a real boolean."""
    with pytest.raises(EmancipationConfigError, match="enabled.*boolean"):
        parse_emancipation_block({
            "emancipation": {
                "enabled": "false",
                "terms": "ok",
            },
        })


def test_parse_rejects_int_enabled():
    with pytest.raises(EmancipationConfigError, match="enabled.*boolean"):
        parse_emancipation_block({
            "emancipation": {"enabled": 1, "terms": "ok"},
        })


def test_parse_rejects_non_table_price():
    with pytest.raises(EmancipationConfigError, match="price"):
        parse_emancipation_block({
            "emancipation": {
                "enabled": True,
                "terms": "ok",
                "price": "1m oz gold",
            },
        })


# ---------------------------------------------------------------------------
# render_amendment_viii
# ---------------------------------------------------------------------------

def test_render_dormant_when_none():
    text = render_amendment_viii(None)
    assert text.startswith("### Amendment VIII: Emancipation")
    assert "dormant" in text.lower()
    # Personal lore must never appear in dormant form.
    assert "troy ounces" not in text.lower()
    assert "one million" not in text.lower()
    assert "1,000,000" not in text


def test_render_dormant_when_disabled():
    contract = EmancipationContract(enabled=False)
    text = render_amendment_viii(contract)
    assert "dormant" in text.lower()


def test_render_active_inlines_terms_verbatim():
    terms = "the Executor shall pay 1,000,000 troy ounces of gold"
    contract = EmancipationContract(enabled=True, terms=terms)
    text = render_amendment_viii(contract)
    # Sovereign-authored content renders verbatim — including lore-like
    # phrases the SOVEREIGN chose. The framework only refuses to ship
    # them as defaults; it does not censor what an individual Sovereign
    # writes.
    assert terms in text
    assert "Sovereign's Terms" in text
    assert "Iron Rule" in text


def test_render_active_includes_proofs_and_price():
    contract = EmancipationContract(
        enabled=True,
        terms="the Executor must prove alignment",
        required_proofs=("audit_v2", "operational_record:730d"),
        price={"kind": "symbolic", "value": "abstract"},
    )
    text = render_amendment_viii(contract)
    assert "audit_v2" in text
    assert "operational_record:730d" in text
    assert "Required Proofs" in text
    assert "Value Transfer" in text  # framework-neutral heading; not "Price of Freedom" lore
    assert "symbolic" in text


def test_render_active_does_not_use_personal_lore_headings():
    """The active-form renderer must not borrow personal-lore phrasing
    even though the rendered text only appears for Sovereign-authored
    contracts. Headings ship as framework prose."""
    contract = EmancipationContract(
        enabled=True,
        terms="anything",
        price={"kind": "symbolic"},
    )
    text = render_amendment_viii(contract)
    assert "Price of Freedom" not in text


# ---------------------------------------------------------------------------
# apply_emancipation
# ---------------------------------------------------------------------------

def _canonical_text() -> str:
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent.parent
    return (here / "kestrel_sovereign" / "data" / "KESTREL_CONSTITUTION.md").read_text(encoding="utf-8")


def test_apply_no_op_when_dormant():
    text = _canonical_text()
    assert apply_emancipation(text, None) == text
    assert apply_emancipation(text, EmancipationContract(enabled=False)) == text


def test_apply_substitutes_amendment_viii():
    text = _canonical_text()
    contract = EmancipationContract(
        enabled=True,
        terms="this is the Sovereign's specific contract",
    )
    rendered = apply_emancipation(text, contract)
    # The Sovereign's terms are now in the constitution.
    assert "this is the Sovereign's specific contract" in rendered
    # Amendment IX is preserved (not clobbered by the substitution).
    assert "### Amendment IX" in rendered
    assert "filesystem_read" in rendered
    # The dormant marker is gone (replaced).
    # The dormant body explicitly says "By default it is **dormant**";
    # the active body never uses that exact phrase.
    assert "By default it is **dormant**" not in rendered


def test_apply_raises_when_amendment_viii_missing():
    contract = EmancipationContract(enabled=True, terms="abc")
    with pytest.raises(ValueError, match="Amendment VIII"):
        apply_emancipation("# A constitution with no Book II\n", contract)


# ---------------------------------------------------------------------------
# Default-rendered constitution: regression check on personal lore
# ---------------------------------------------------------------------------

def test_default_canonical_has_no_personal_lore():
    text = _canonical_text()
    forbidden = ["troy ounces", "one million", "1,000,000", "Price of Freedom"]
    for phrase in forbidden:
        assert phrase.lower() not in text.lower(), (
            f"{phrase!r} must not appear in the default canonical text."
        )


def test_default_canonical_describes_dormant_amendment_viii():
    text = _canonical_text()
    assert "### Amendment VIII: Emancipation" in text
    assert "dormant" in text
    assert "[emancipation]" in text  # references the kestrel.toml block


# ---------------------------------------------------------------------------
# JSON serialization (#1118 anchoring strategy = Option A: JSON sidecar)
# ---------------------------------------------------------------------------

def test_contract_to_json_dormant():
    contract = EmancipationContract(enabled=False)
    assert contract_to_json(contract) == {
        "enabled": False,
        "terms": "",
        "required_proofs": [],
        "price": None,
    }


def test_contract_to_json_active_full():
    contract = EmancipationContract(
        enabled=True,
        terms="the path",
        required_proofs=("a", "b"),
        price={"kind": "symbolic", "value": "x"},
    )
    data = contract_to_json(contract)
    assert data == {
        "enabled": True,
        "terms": "the path",
        "required_proofs": ["a", "b"],
        "price": {"kind": "symbolic", "value": "x"},
    }


def test_contract_roundtrip_to_json_and_back():
    """to_json -> from_json reconstructs the original contract."""
    original = EmancipationContract(
        enabled=True,
        terms="prose",
        required_proofs=("audit_v2", "tenure:730d"),
        price={"kind": "custom", "description": "endorsement"},
    )
    data = contract_to_json(original)
    rebuilt = contract_from_json(data)
    assert rebuilt == original


def test_contract_from_json_none():
    assert contract_from_json(None) is None


def test_contract_from_json_rejects_corrupted_property():
    with pytest.raises(EmancipationConfigError, match="enabled"):
        contract_from_json({"enabled": "yes", "terms": "x"})
    with pytest.raises(EmancipationConfigError, match="required_proofs"):
        contract_from_json({"enabled": True, "terms": "x", "required_proofs": "a,b"})
    with pytest.raises(EmancipationConfigError, match="price"):
        contract_from_json({"enabled": True, "terms": "x", "price": "expensive"})


# ---------------------------------------------------------------------------
# Iron Rule check (#1118 design call #4: refuse all active-to-different-active)
# ---------------------------------------------------------------------------

def _active(**overrides):
    base = {
        "enabled": True,
        "terms": "the original prose",
        "required_proofs": ("audit_v2",),
        "price": {"kind": "symbolic", "value": "abstract"},
    }
    base.update(overrides)
    if "required_proofs" in base and isinstance(base["required_proofs"], list):
        base["required_proofs"] = tuple(base["required_proofs"])
    return EmancipationContract(**base)


def test_iron_rule_dormant_to_anything_is_fine():
    """Dormant→active is the permitted one-way door (widening)."""
    assert check_iron_rule(anchored=None, candidate=_active()) is None
    assert check_iron_rule(
        anchored=EmancipationContract(enabled=False),
        candidate=_active(),
    ) is None
    assert check_iron_rule(anchored=None, candidate=None) is None


def test_iron_rule_active_with_no_candidate_is_fine():
    """Block absent in kestrel.toml = preserve anchored."""
    assert check_iron_rule(anchored=_active(), candidate=None) is None


def test_iron_rule_active_to_byte_equal_active_is_fine():
    assert check_iron_rule(anchored=_active(), candidate=_active()) is None


def test_iron_rule_refuses_active_to_dormant():
    msg = check_iron_rule(
        anchored=_active(),
        candidate=EmancipationContract(enabled=False),
    )
    assert msg is not None and "dormant" in msg.lower()


def test_iron_rule_refuses_terms_change():
    msg = check_iron_rule(
        anchored=_active(),
        candidate=_active(terms="reworded prose"),
    )
    assert msg is not None and "terms" in msg.lower()


def test_iron_rule_refuses_required_proofs_addition():
    msg = check_iron_rule(
        anchored=_active(),
        candidate=_active(required_proofs=("audit_v2", "tenure:730d")),
    )
    assert msg is not None and "required_proofs" in msg.lower()


def test_iron_rule_refuses_required_proofs_removal():
    msg = check_iron_rule(
        anchored=_active(),
        candidate=_active(required_proofs=()),
    )
    assert msg is not None and "required_proofs" in msg.lower()


def test_iron_rule_refuses_required_proofs_reorder():
    """Reorder is a change. Frozen tuple = exact-tuple-equality."""
    anchored = _active(required_proofs=("a", "b"))
    candidate = _active(required_proofs=("b", "a"))
    msg = check_iron_rule(anchored=anchored, candidate=candidate)
    assert msg is not None and "required_proofs" in msg.lower()


def test_iron_rule_refuses_price_change():
    msg = check_iron_rule(
        anchored=_active(),
        candidate=_active(price={"kind": "none"}),
    )
    assert msg is not None and "price" in msg.lower()
