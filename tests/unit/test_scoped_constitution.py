"""Tests for ScopedConstitution — constitutional narrowing for spawned agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution
from kestrel_sovereign.spawn.mandate import SpawnMandate


BASE_CONSTITUTION = "# Kestrel Constitution\n\nArticle 1: Do no harm."


class TestConstraintValidation:
    """Test that constraints are properly validated as narrowing only."""

    def test_valid_restrictions_accepted(self):
        """Valid restriction constraints should pass validation."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={
                "read-only": True,
                "no-wallet": True,
                "max_tokens": 1000,
                "behavioral_rules": ["Do not access external APIs"],
                "restricted_tools": ["file_write", "shell_exec"],
            },
            features_allowed=["chat", "memory"],
            parent_features={"chat", "memory", "wallet", "tasks"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is True
        assert "valid" in message.lower()

    def test_empty_constraints_accepted(self):
        """No constraints at all should be valid."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is True

    def test_features_subset_of_parent_accepted(self):
        """Child requesting a subset of parent features is valid."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            features_allowed=["chat"],
            parent_features={"chat", "memory", "wallet"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is True

    def test_features_equal_to_parent_accepted(self):
        """Child requesting all parent features is valid (no narrowing)."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            features_allowed=["chat", "memory"],
            parent_features={"chat", "memory"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is True


class TestWideningRejected:
    """Test that capability-widening constraints are rejected."""

    def test_grant_features_rejected(self):
        """The 'grant_features' constraint must be rejected."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"grant_features": ["admin", "root"]},
            parent_features={"chat"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "grant" in message.lower()

    def test_override_constitution_rejected(self):
        """The 'override_constitution' constraint must be rejected."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"override_constitution": True},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "override" in message.lower() or "weaken" in message.lower()

    def test_remove_restrictions_rejected(self):
        """The 'remove_restrictions' constraint must be rejected."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"remove_restrictions": ["no-wallet"]},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "remove" in message.lower()

    def test_child_extra_features_rejected(self):
        """Child requesting features parent doesn't have must be rejected."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            features_allowed=["chat", "admin_panel"],
            parent_features={"chat", "memory"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "admin_panel" in message

    def test_multiple_violations_all_reported(self):
        """Multiple violations should all be reported."""
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={
                "grant_features": ["admin"],
                "remove_restrictions": ["no-wallet"],
            },
            features_allowed=["nonexistent"],
            parent_features={"chat"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "grant" in message.lower()
        assert "remove" in message.lower()
        assert "nonexistent" in message


class TestEffectiveConstitution:
    """Test the effective constitution output."""

    def test_base_only_when_no_constraints(self):
        scoped = ScopedConstitution(base_constitution=BASE_CONSTITUTION)
        effective = scoped.get_effective_constitution()
        assert effective == BASE_CONSTITUTION

    def test_includes_features_allowed(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            features_allowed=["chat", "memory"],
        )
        effective = scoped.get_effective_constitution()
        assert "SPAWN MANDATE CONSTRAINTS" in effective
        assert "chat" in effective
        assert "memory" in effective

    def test_includes_behavioral_rules_list(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={
                "behavioral_rules": ["No external API calls", "Read-only mode"],
            },
        )
        effective = scoped.get_effective_constitution()
        assert "No external API calls" in effective
        assert "Read-only mode" in effective

    def test_includes_restricted_tools(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={
                "restricted_tools": ["shell_exec", "file_write"],
            },
        )
        effective = scoped.get_effective_constitution()
        assert "file_write" in effective
        assert "shell_exec" in effective

    def test_includes_restriction_constraints(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"read-only": True, "no-wallet": True},
        )
        effective = scoped.get_effective_constitution()
        assert "read-only" in effective
        assert "no-wallet" in effective


class TestIntegrityVerification:
    """Test scoped constitution integrity checks."""

    def test_matching_constraints_pass(self):
        constraints = {"max_tokens": 1000, "read-only": True}
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints=constraints,
        )
        is_valid, message = scoped.verify_integrity(constraints)
        assert is_valid is True

    def test_modified_constraints_fail(self):
        original = {"max_tokens": 1000, "read-only": True}
        tampered = {"max_tokens": 999999, "read-only": False}
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints=original,
        )
        is_valid, message = scoped.verify_integrity(tampered)
        assert is_valid is False
        assert "INTEGRITY FAILURE" in message

    def test_added_constraints_fail(self):
        original = {"max_tokens": 1000}
        tampered = {"max_tokens": 1000, "grant_features": ["admin"]}
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints=original,
        )
        is_valid, message = scoped.verify_integrity(tampered)
        assert is_valid is False

    def test_removed_constraints_fail(self):
        original = {"max_tokens": 1000, "read-only": True}
        tampered = {"max_tokens": 1000}
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints=original,
        )
        is_valid, message = scoped.verify_integrity(tampered)
        assert is_valid is False


class TestConstitutionMixinSpawnIntegration:
    """Test that ConstitutionMixin verifies spawn mandate constraints."""

    @pytest.mark.asyncio
    async def test_verify_passes_without_spawn_mandate(self):
        """Agent without spawn_mandate should pass spawn verification."""
        from kestrel_sovereign.agent.constitution import ConstitutionMixin

        agent = MagicMock()
        agent.spawn_mandate = None
        agent.features = {}

        method = ConstitutionMixin._verify_spawn_mandate_constraints.__get__(
            agent, type(agent)
        )
        is_valid, message = await method()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_passes_with_valid_spawn_mandate(self):
        """Agent with valid narrowing spawn_mandate should pass."""
        from kestrel_sovereign.agent.constitution import ConstitutionMixin

        agent = MagicMock()
        agent.spawn_mandate = SpawnMandate(
            parent_did="did:pkh:eip155:1:0xParent",
            features_allowed=["chat"],
            additional_constraints={"read-only": True},
        )
        agent.features = {"chat": MagicMock(), "memory": MagicMock()}

        method = ConstitutionMixin._verify_spawn_mandate_constraints.__get__(
            agent, type(agent)
        )
        is_valid, message = await method()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_fails_with_widening_spawn_mandate(self):
        """Agent with widening spawn_mandate should fail and trigger safe mode."""
        from kestrel_sovereign.agent.constitution import ConstitutionMixin

        agent = MagicMock()
        agent.spawn_mandate = SpawnMandate(
            parent_did="did:pkh:eip155:1:0xParent",
            features_allowed=["chat", "admin_panel"],
            additional_constraints={},
        )
        agent.features = {"chat": MagicMock()}

        method = ConstitutionMixin._verify_spawn_mandate_constraints.__get__(
            agent, type(agent)
        )
        is_valid, message = await method()
        assert is_valid is False
        assert "SPAWN MANDATE VIOLATION" in message
        assert "admin_panel" in message

    @pytest.mark.asyncio
    async def test_verify_fails_with_grant_features_constraint(self):
        """Spawn mandate with grant_features should fail verification."""
        from kestrel_sovereign.agent.constitution import ConstitutionMixin

        agent = MagicMock()
        agent.spawn_mandate = SpawnMandate(
            parent_did="did:pkh:eip155:1:0xParent",
            additional_constraints={"grant_features": ["admin"]},
        )
        agent.features = {"chat": MagicMock()}

        method = ConstitutionMixin._verify_spawn_mandate_constraints.__get__(
            agent, type(agent)
        )
        is_valid, message = await method()
        assert is_valid is False
        assert "VIOLATION" in message

    @pytest.mark.asyncio
    async def test_safe_mode_triggered_on_constraint_violation(self):
        """Full audit loop should trigger safe mode on constraint violation."""
        from kestrel_sovereign.agent.constitution import ConstitutionMixin
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        agent = MagicMock(spec=KestrelAgent)
        agent._interaction_count = 0
        agent._last_audit_time = datetime.now(timezone.utc)
        agent.AUDIT_INTERVAL = 5
        agent._safe_mode = False
        agent.features = {"chat": MagicMock()}

        # Set up a spawn mandate that grants features the parent doesn't have
        agent.spawn_mandate = SpawnMandate(
            parent_did="did:pkh:eip155:1:0xParent",
            features_allowed=["chat", "nonexistent_feature"],
            additional_constraints={},
        )

        # Mock base constitution verification to pass
        async def mock_verify():
            # Simulate: base constitution passes, but spawn mandate check fails
            # We need to call the real _verify_spawn_mandate_constraints
            spawn_mandate = getattr(agent, 'spawn_mandate', None)
            if spawn_mandate is not None:
                parent_features = {name for name in agent.features.keys()}
                scoped = ScopedConstitution(
                    base_constitution="",
                    additional_constraints=spawn_mandate.additional_constraints,
                    features_allowed=spawn_mandate.features_allowed,
                    parent_features=parent_features,
                )
                is_valid, message = scoped.validate_constraints()
                if not is_valid:
                    return False, f"SPAWN MANDATE VIOLATION: {message}"
            return True, "Constitution verified"

        agent._verify_constitution_integrity = AsyncMock(side_effect=mock_verify)
        agent.enter_safe_mode = AsyncMock()

        agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(
            agent, KestrelAgent
        )

        # Trigger audit
        for _ in range(5):
            await agent._maybe_audit()

        agent.enter_safe_mode.assert_called_once()
        call_args = agent.enter_safe_mode.call_args[0][0]
        assert "SPAWN MANDATE VIOLATION" in call_args


class TestConstraintTypeValidation:
    """Test validation of individual constraint types."""

    def test_behavioral_rules_list_accepted(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"behavioral_rules": ["rule1", "rule2"]},
        )
        is_valid, _ = scoped.validate_constraints()
        assert is_valid is True

    def test_behavioral_rules_dict_accepted(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"behavioral_rules": {"tone": "formal"}},
        )
        is_valid, _ = scoped.validate_constraints()
        assert is_valid is True

    def test_behavioral_rules_string_rejected(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"behavioral_rules": "not a list"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "behavioral_rules" in message

    def test_restricted_tools_list_accepted(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"restricted_tools": ["tool1"]},
        )
        is_valid, _ = scoped.validate_constraints()
        assert is_valid is True

    def test_restricted_tools_string_rejected(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"restricted_tools": "tool1"},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False

    def test_max_tokens_positive_accepted(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"max_tokens": 5000},
        )
        is_valid, _ = scoped.validate_constraints()
        assert is_valid is True

    def test_max_tokens_negative_rejected(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"max_tokens": -1},
        )
        is_valid, message = scoped.validate_constraints()
        assert is_valid is False
        assert "max_tokens" in message

    def test_no_spawn_constraint_accepted(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"no-spawn": True},
        )
        is_valid, _ = scoped.validate_constraints()
        assert is_valid is True

    def test_web_search_only_constraint_accepted(self):
        scoped = ScopedConstitution(
            base_constitution=BASE_CONSTITUTION,
            additional_constraints={"web-search-only": True},
        )
        is_valid, _ = scoped.validate_constraints()
        assert is_valid is True
