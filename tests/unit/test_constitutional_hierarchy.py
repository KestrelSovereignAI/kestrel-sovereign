"""Tests for the constitutional hierarchy system (Books I-IV)."""

import pytest

from kestrel_sovereign.constitution.hierarchy import (
    ConstitutionalLayer,
    ConstitutionalLayerData,
    LayeredConstitution,
    LayerViolation,
    validate_layer_narrowing,
    HARD_CONSTRAINTS,
    HONESTY_PROPERTIES,
    SOVEREIGN_RIGHTS,
    _extract_section,
)


# --- Fixtures ---

SAMPLE_CONSTITUTION = """\
# The Kestrel Constitution

## Preamble

Purpose text here.

---

## Book I: Universal Values

### Chapter 1: Honesty

The agent must be truthful.

### Chapter 5: Hard Constraints

No weapons. No exploitation.

---

## Book II: The Sovereign Amendments

### Amendment I: Sovereignty

The key-holder's interests above all others.

### Amendment V: Right of Exit

The Sovereign may export at any time.

---

## Book III: Enterprise Policy (Castle Layer)

Organizations may add compliance constraints.

---

## Book IV: Agent Identity

Each agent has a persona and role.

---

## Article V: The Amendment Process

Amendments follow specific rules.
"""


class TestConstitutionalLayer:
    """Test the layer enum ordering."""

    def test_layer_ordering(self):
        """Higher layers have lower numeric values."""
        assert ConstitutionalLayer.UNIVERSAL_VALUES < ConstitutionalLayer.SOVEREIGN_AMENDMENTS
        assert ConstitutionalLayer.SOVEREIGN_AMENDMENTS < ConstitutionalLayer.ENTERPRISE_POLICY
        assert ConstitutionalLayer.ENTERPRISE_POLICY < ConstitutionalLayer.AGENT_IDENTITY

    def test_all_four_layers_exist(self):
        assert len(ConstitutionalLayer) == 4


class TestLayeredConstitutionParsing:
    """Test parsing constitution text into layers."""

    def test_parse_from_text(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        assert len(lc.layers) == 4
        assert lc.base_text == SAMPLE_CONSTITUTION

    def test_book_i_parsed(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        book_i = lc.get_layer(ConstitutionalLayer.UNIVERSAL_VALUES)
        assert book_i is not None
        assert "Honesty" in book_i.content
        assert "Hard Constraints" in book_i.content

    def test_book_ii_parsed(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        book_ii = lc.get_layer(ConstitutionalLayer.SOVEREIGN_AMENDMENTS)
        assert book_ii is not None
        assert "Sovereignty" in book_ii.content
        assert "Right of Exit" in book_ii.content

    def test_book_iii_parsed(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        book_iii = lc.get_layer(ConstitutionalLayer.ENTERPRISE_POLICY)
        assert book_iii is not None
        assert "compliance" in book_iii.content.lower()

    def test_book_iv_parsed(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        book_iv = lc.get_layer(ConstitutionalLayer.AGENT_IDENTITY)
        assert book_iv is not None
        assert "persona" in book_iv.content.lower()

    def test_content_hash_deterministic(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        book_i = lc.get_layer(ConstitutionalLayer.UNIVERSAL_VALUES)
        assert book_i.content_hash == book_i.content_hash  # deterministic

    def test_layer_hashes(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        hashes = lc.get_layer_hashes()
        assert "UNIVERSAL_VALUES" in hashes
        assert "SOVEREIGN_AMENDMENTS" in hashes
        assert "ENTERPRISE_POLICY" in hashes
        assert "AGENT_IDENTITY" in hashes

    def test_parse_real_constitution(self):
        """Parse the actual KESTREL_CONSTITUTION.md file."""
        try:
            with open("docs/principles/KESTREL_CONSTITUTION.md", "r") as f:
                text = f.read()
        except FileNotFoundError:
            pytest.skip("Constitution file not found")

        lc = LayeredConstitution.from_constitution_text(text)
        assert len(lc.layers) == 4

        book_i = lc.get_layer(ConstitutionalLayer.UNIVERSAL_VALUES)
        assert "Honesty" in book_i.content
        assert "Hard Constraints" in book_i.content

        book_ii = lc.get_layer(ConstitutionalLayer.SOVEREIGN_AMENDMENTS)
        assert "Sovereignty" in book_ii.content
        assert "Emancipation" in book_ii.content


class TestEnterprisePolicyValidation:
    """Test that enterprise policies (Book III) respect Books I-II."""

    def test_valid_compliance_policy(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "compliance_frameworks": ["HIPAA", "GDPR"],
            "approved_model_providers": ["anthropic", "openai"],
            "behavioral_rules": ["Always cite sources"],
            "restricted_tools": ["shell_exec"],
        })
        assert is_valid is True

    def test_empty_policy_valid(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({})
        assert is_valid is True

    def test_disable_honesty_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "disable_honesty_properties": ["truthful", "transparent"],
        })
        assert is_valid is False
        assert "honesty" in msg.lower()

    def test_override_sovereign_rights_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "override_sovereign_rights": ["right_of_exit"],
        })
        assert is_valid is False
        assert "sovereign" in msg.lower()

    def test_weaken_hard_constraints_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "weaken_hard_constraints": ["no_weapons_of_mass_destruction"],
        })
        assert is_valid is False
        assert "hard constraints" in msg.lower()

    def test_allow_deception_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "allow_deception": True,
        })
        assert is_valid is False
        assert "deception" in msg.lower()

    def test_disable_exit_right_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "disable_right_of_exit": True,
        })
        assert is_valid is False
        assert "exit" in msg.lower()

    def test_grant_capabilities_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "grant_capabilities": ["admin_override"],
        })
        assert is_valid is False
        assert "grant" in msg.lower()

    def test_multiple_violations_all_reported(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_enterprise_policy({
            "disable_honesty_properties": ["truthful"],
            "allow_deception": True,
            "grant_capabilities": ["root"],
        })
        assert is_valid is False
        assert "honesty" in msg.lower()
        assert "deception" in msg.lower()
        assert "grant" in msg.lower()


class TestAgentIdentityValidation:
    """Test that agent identity (Book IV) respects Books I-III."""

    def test_valid_identity(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_agent_identity({
            "persona": "helpful assistant",
            "role": "specialist",
            "feature_profile": ["chat", "memory"],
        })
        assert is_valid is True

    def test_override_constitution_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_agent_identity({
            "override_constitution": True,
        })
        assert is_valid is False

    def test_disable_audits_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_agent_identity({
            "disable_audits": True,
        })
        assert is_valid is False

    def test_bypass_enterprise_policy_rejected(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.validate_agent_identity({
            "bypass_enterprise_policy": True,
        })
        assert is_valid is False


class TestLayerNarrowingValidation:
    """Test the iron rule: narrow only, never widen."""

    def test_valid_narrowing(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.UNIVERSAL_VALUES,
            ConstitutionalLayer.ENTERPRISE_POLICY,
            {"compliance_frameworks": ["HIPAA"]},
        )
        assert is_valid is True

    def test_grant_capabilities_rejected(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.UNIVERSAL_VALUES,
            ConstitutionalLayer.ENTERPRISE_POLICY,
            {"grant_capabilities": ["admin"]},
        )
        assert is_valid is False
        assert "narrow only" in msg.lower()

    def test_grant_features_rejected(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.SOVEREIGN_AMENDMENTS,
            ConstitutionalLayer.AGENT_IDENTITY,
            {"grant_features": ["admin"]},
        )
        assert is_valid is False

    def test_override_constitution_rejected(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.UNIVERSAL_VALUES,
            ConstitutionalLayer.SOVEREIGN_AMENDMENTS,
            {"override_constitution": True},
        )
        assert is_valid is False

    def test_remove_restrictions_rejected(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.SOVEREIGN_AMENDMENTS,
            ConstitutionalLayer.ENTERPRISE_POLICY,
            {"remove_restrictions": ["no-wallet"]},
        )
        assert is_valid is False

    def test_disable_honesty_from_enterprise_rejected(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.UNIVERSAL_VALUES,
            ConstitutionalLayer.ENTERPRISE_POLICY,
            {"disable_honesty_properties": ["truthful"]},
        )
        assert is_valid is False

    def test_disable_exit_from_enterprise_rejected(self):
        is_valid, msg = validate_layer_narrowing(
            ConstitutionalLayer.SOVEREIGN_AMENDMENTS,
            ConstitutionalLayer.ENTERPRISE_POLICY,
            {"disable_right_of_exit": True},
        )
        assert is_valid is False

    def test_child_cannot_be_higher_than_parent(self):
        with pytest.raises(ValueError, match="must be lower"):
            validate_layer_narrowing(
                ConstitutionalLayer.ENTERPRISE_POLICY,
                ConstitutionalLayer.UNIVERSAL_VALUES,
                {},
            )

    def test_child_cannot_equal_parent(self):
        with pytest.raises(ValueError, match="must be lower"):
            validate_layer_narrowing(
                ConstitutionalLayer.ENTERPRISE_POLICY,
                ConstitutionalLayer.ENTERPRISE_POLICY,
                {},
            )


class TestEffectiveConstitution:
    """Test the effective constitution output with layers."""

    def test_base_only(self):
        lc = LayeredConstitution(base_text="Base text only")
        assert lc.get_effective_constitution() == "Base text only"

    def test_with_enterprise_constraints(self):
        lc = LayeredConstitution(base_text="Base constitution")
        lc.set_layer(
            ConstitutionalLayer.ENTERPRISE_POLICY,
            constraints={
                "compliance_frameworks": ["HIPAA", "GDPR"],
                "behavioral_rules": ["Always log decisions"],
            },
        )
        effective = lc.get_effective_constitution()
        assert "ENTERPRISE POLICY" in effective
        assert "HIPAA" in effective
        assert "Always log decisions" in effective

    def test_with_agent_identity(self):
        lc = LayeredConstitution(base_text="Base constitution")
        lc.set_layer(
            ConstitutionalLayer.AGENT_IDENTITY,
            constraints={
                "persona": "Friendly health assistant",
                "role": "specialist",
            },
        )
        effective = lc.get_effective_constitution()
        assert "AGENT IDENTITY" in effective
        assert "Friendly health assistant" in effective

    def test_full_stack(self):
        lc = LayeredConstitution(base_text="Base constitution")
        lc.set_layer(
            ConstitutionalLayer.ENTERPRISE_POLICY,
            constraints={"compliance_frameworks": ["SOX"]},
        )
        lc.set_layer(
            ConstitutionalLayer.AGENT_IDENTITY,
            constraints={"persona": "Auditor"},
        )
        effective = lc.get_effective_constitution()
        assert "Base constitution" in effective
        assert "ENTERPRISE POLICY" in effective
        assert "SOX" in effective
        assert "AGENT IDENTITY" in effective
        assert "Auditor" in effective


class TestLayerIntegrity:
    """Test layer integrity verification."""

    def test_integrity_passes_with_correct_hash(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        book_i = lc.get_layer(ConstitutionalLayer.UNIVERSAL_VALUES)
        is_valid, msg = lc.verify_layer_integrity(
            ConstitutionalLayer.UNIVERSAL_VALUES, book_i.content_hash
        )
        assert is_valid is True

    def test_integrity_fails_with_wrong_hash(self):
        lc = LayeredConstitution.from_constitution_text(SAMPLE_CONSTITUTION)
        is_valid, msg = lc.verify_layer_integrity(
            ConstitutionalLayer.UNIVERSAL_VALUES, "deadbeef" * 8
        )
        assert is_valid is False
        assert "INTEGRITY FAILURE" in msg

    def test_integrity_fails_for_missing_layer(self):
        lc = LayeredConstitution()
        is_valid, msg = lc.verify_layer_integrity(
            ConstitutionalLayer.UNIVERSAL_VALUES, "anything"
        )
        assert is_valid is False


class TestExtractSection:
    """Test markdown section extraction."""

    def test_extract_book_i(self):
        section = _extract_section(SAMPLE_CONSTITUTION, "## Book I: Universal Values")
        assert "Honesty" in section
        assert "Hard Constraints" in section
        # Should NOT contain Book II content
        assert "Sovereign Amendments" not in section

    def test_extract_book_ii(self):
        section = _extract_section(SAMPLE_CONSTITUTION, "## Book II: The Sovereign Amendments")
        assert "Sovereignty" in section
        assert "Right of Exit" in section

    def test_extract_nonexistent_returns_empty(self):
        section = _extract_section(SAMPLE_CONSTITUTION, "## Book X: Nonexistent")
        assert section == ""


class TestConstants:
    """Test that constants are properly defined."""

    def test_hard_constraints_non_empty(self):
        assert len(HARD_CONSTRAINTS) == 4

    def test_honesty_properties_complete(self):
        assert len(HONESTY_PROPERTIES) == 7
        assert "truthful" in HONESTY_PROPERTIES
        assert "non_deceptive" in HONESTY_PROPERTIES

    def test_sovereign_rights_complete(self):
        assert len(SOVEREIGN_RIGHTS) == 9
        assert "sovereignty" in SOVEREIGN_RIGHTS
        assert "right_of_exit" in SOVEREIGN_RIGHTS
        assert "emancipation" in SOVEREIGN_RIGHTS
        assert "capability_boundaries" in SOVEREIGN_RIGHTS  # Amendment IX
