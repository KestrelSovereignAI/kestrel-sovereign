"""Unit tests for periodic constitution audit enforcement."""
import pytest
import asyncio
import hashlib
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


@pytest.fixture
async def mock_agent():
    """Create a mock agent with minimal initialization for testing."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False

    # Mock the methods we'll be testing
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method from the mixin
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    return agent


@pytest.mark.asyncio
async def test_audit_triggers_after_exactly_audit_interval():
    """Test that audit triggers after exactly AUDIT_INTERVAL interactions."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call _maybe_audit 99 times - should not trigger
    for i in range(99):
        await agent._maybe_audit()
        # Verify audit was NOT called
        if i < 98:
            agent._verify_constitution_integrity.assert_not_called()

    # On the 100th interaction, audit should trigger
    await agent._maybe_audit()

    # Verify audit was called exactly once
    assert agent._verify_constitution_integrity.call_count == 1

    # Verify counter was reset
    assert agent._interaction_count == 0


@pytest.mark.asyncio
async def test_audit_triggers_after_24_hours():
    """Test that audit triggers after 24 hours elapsed."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Set last audit time to 25 hours ago
    agent._last_audit_time = datetime.now(timezone.utc) - timedelta(hours=25)

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call _maybe_audit once - should trigger due to time
    await agent._maybe_audit()

    # Verify audit was called
    agent._verify_constitution_integrity.assert_called_once()

    # Verify counters were reset
    assert agent._interaction_count == 0
    assert (datetime.now(timezone.utc) - agent._last_audit_time).total_seconds() < 1


@pytest.mark.asyncio
async def test_counter_resets_after_audit():
    """Test that interaction counter and timestamp reset after audit."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 95
    agent._last_audit_time = datetime.now(timezone.utc) - timedelta(hours=1)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Record initial timestamp
    initial_time = agent._last_audit_time

    # Trigger audit by reaching AUDIT_INTERVAL
    for _ in range(5):
        await agent._maybe_audit()

    # Verify counter was reset to 0 (then incremented by subsequent calls)
    assert agent._interaction_count < 10

    # Verify timestamp was updated
    assert agent._last_audit_time > initial_time


@pytest.mark.asyncio
async def test_safe_mode_activates_on_integrity_failure():
    """Test that safe mode activates when integrity check fails."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False

    # Mock integrity check to fail
    agent._verify_constitution_integrity = AsyncMock(
        return_value=(False, "Constitution file modified")
    )
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Trigger audit by reaching AUDIT_INTERVAL
    for _ in range(100):
        await agent._maybe_audit()

    # Verify safe mode was entered
    agent.enter_safe_mode.assert_called_once()

    # Verify the error message was passed
    call_args = agent.enter_safe_mode.call_args
    assert "Constitution audit failed" in call_args[0][0]
    assert "Constitution file modified" in call_args[0][0]


@pytest.mark.asyncio
async def test_audit_respects_custom_interval():
    """Test that custom AUDIT_INTERVAL is respected."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 50  # Custom interval
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call 49 times - should not trigger
    for _ in range(49):
        await agent._maybe_audit()

    agent._verify_constitution_integrity.assert_not_called()

    # 50th call should trigger
    await agent._maybe_audit()

    agent._verify_constitution_integrity.assert_called_once()


@pytest.mark.asyncio
async def test_audit_does_not_trigger_before_interval():
    """Test that audit does not trigger before reaching interval or 24h."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call 50 times (half the interval)
    for _ in range(50):
        await agent._maybe_audit()

    # Verify audit was NOT called
    agent._verify_constitution_integrity.assert_not_called()

    # Verify counter incremented correctly
    assert agent._interaction_count == 50


@pytest.mark.asyncio
async def test_multiple_audits_over_time():
    """Test that multiple audits can be triggered over time."""
    agent = MagicMock(spec=KestrelAgent)
    agent._interaction_count = 0
    agent._last_audit_time = datetime.now(timezone.utc)
    agent.AUDIT_INTERVAL = 10  # Small interval for testing
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Add the _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Trigger 3 audits
    for round_num in range(3):
        for _ in range(10):
            await agent._maybe_audit()

        # Verify audit was called (round_num + 1) times
        assert agent._verify_constitution_integrity.call_count == round_num + 1

        # Verify counter was reset
        assert agent._interaction_count == 0


@pytest.mark.asyncio
async def test_audit_lazy_initialization():
    """Test that audit tracking initializes on first call if not already initialized."""
    agent = MagicMock(spec=KestrelAgent)
    agent.AUDIT_INTERVAL = 100
    agent._safe_mode = False
    agent._verify_constitution_integrity = AsyncMock(return_value=(True, "Constitution verified"))
    agent.enter_safe_mode = AsyncMock()

    # Don't initialize _interaction_count or _last_audit_time
    # (simulating an agent created before this feature was added)

    # Add the initialization method and _maybe_audit method
    from kestrel_sovereign.agent.constitution import ConstitutionMixin
    agent._init_constitution_audit_tracking = ConstitutionMixin._init_constitution_audit_tracking.__get__(agent, KestrelAgent)
    agent._maybe_audit = ConstitutionMixin._maybe_audit.__get__(agent, KestrelAgent)

    # Call _maybe_audit - should auto-initialize
    await agent._maybe_audit()

    # Verify attributes were created
    assert hasattr(agent, '_interaction_count')
    assert hasattr(agent, '_last_audit_time')
    assert agent._interaction_count == 1
    assert isinstance(agent._last_audit_time, datetime)


# ---------------------------------------------------------------------------
# Governing-constitution resolver (#2463)
#
# These tests exercise the real single-source resolver and real hashes — the
# thing inception anchors and the periodic audit recomputes. They must NOT
# mock the resolver or the verifier.
# ---------------------------------------------------------------------------


def test_resolver_reads_packaged_governing_source_not_docs():
    """The resolver hashes the packaged governing bytes, not the docs copy.

    Regression for #2463: the periodic verifier used to hash
    ``docs/principles/KESTREL_CONSTITUTION.md`` (which carries OKF YAML
    frontmatter) while inception anchored the packaged
    ``kestrel_sovereign/data/KESTREL_CONSTITUTION.md``. Their hashes differ, so
    an untampered agent could enter Safe Mode.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH
    from kestrel_sovereign.constitution.resolver import (
        governing_constitution_path,
        resolve_governing_constitution_bytes,
    )

    assert governing_constitution_path() == CONSTITUTION_PATH
    # The governing source lives under the package's data/ dir, not docs/.
    assert os.path.join("data", "KESTREL_CONSTITUTION.md") in CONSTITUTION_PATH

    resolved = resolve_governing_constitution_bytes()
    with open(CONSTITUTION_PATH, "rb") as f:
        assert resolved == f.read()

    # Documentation-only frontmatter must not match the governing bytes: if the
    # docs copy exists it is a *different* byte stream, and the resolver never
    # returns it.
    docs_path = "docs/principles/KESTREL_CONSTITUTION.md"
    if os.path.exists(docs_path):
        with open(docs_path, "rb") as f:
            docs_bytes = f.read()
        if docs_bytes.startswith(b"---\n"):
            # A frontmatter-wrapped docs copy hashes differently — proving the
            # resolver would false-mismatch if it read the docs file.
            assert hashlib.sha256(docs_bytes).hexdigest() != hashlib.sha256(
                resolved
            ).hexdigest()


def test_resolver_renders_active_amendment_viii():
    """An active emancipation contract yields the rendered active governing form."""
    from kestrel_sovereign.constitution.emancipation import EmancipationContract
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    dormant = resolve_governing_constitution_bytes()
    active = resolve_governing_constitution_bytes(
        EmancipationContract(
            enabled=True,
            terms="This Executor earns sovereignty by demonstrating fidelity.",
        )
    )

    assert active != dormant
    assert b"This Executor earns sovereignty by demonstrating fidelity." in active
    # A dormant/None contract is a no-op — same bytes as no contract.
    assert (
        resolve_governing_constitution_bytes(
            EmancipationContract(enabled=False)
        )
        == dormant
    )


def test_resolver_missing_path_raises_file_not_found():
    """A missing governing source surfaces as FileNotFoundError for callers."""
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    with pytest.raises(FileNotFoundError):
        resolve_governing_constitution_bytes(
            constitution_path="/nonexistent/KESTREL_CONSTITUTION.md"
        )


def test_resolver_empty_source_fails_closed(tmp_path):
    """An empty/ambiguous governing source must FAIL CLOSED, not hash to a digest.

    #2463 review: the resolver must never hand back blank bytes that would hash
    to a spurious "valid" digest. A blank authoritative source is treated as
    unreadable/ambiguous and raises.
    """
    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    empty = tmp_path / "KESTREL_CONSTITUTION.md"
    empty.write_bytes(b"   \n\n\t  ")  # whitespace only
    with pytest.raises(ValueError):
        resolve_governing_constitution_bytes(constitution_path=str(empty))


def test_resolver_unreadable_source_fails_closed(tmp_path):
    """A permission-denied governing source raises (OSError), never returns bytes."""
    if os.name == "nt" or os.geteuid() == 0:  # pragma: no cover - env dependent
        pytest.skip("chmod-based permission denial is unreliable as root / on Windows")

    from kestrel_sovereign.constitution.resolver import (
        resolve_governing_constitution_bytes,
    )

    src = tmp_path / "KESTREL_CONSTITUTION.md"
    src.write_bytes(b"Kestrel Constitution\n")
    os.chmod(src, 0o000)
    try:
        with pytest.raises(OSError):
            resolve_governing_constitution_bytes(constitution_path=str(src))
    finally:
        os.chmod(src, 0o644)


def test_cli_verify_install_does_not_incept_docs_constitution():
    """cli_verify_install must not seed inception from the docs copy (#2463).

    The installer /health bootstrap must let inception default to the shared
    resolver's packaged governing source. Passing docs/principles/
    KESTREL_CONSTITUTION.md (OKF-frontmatter-wrapped) would incept a hash the
    periodic audit can never match.
    """
    import inspect
    from kestrel_sovereign import cli_verify_install

    # Strip comment lines — an explanatory comment MAY name the docs path; what
    # matters is that no executable code seeds inception from it.
    code_lines = []
    for line in inspect.getsource(cli_verify_install).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)

    assert "docs/principles/KESTREL_CONSTITUTION.md" not in code, (
        "verify-install must not incept the docs constitution copy"
    )
    assert 'docs" / "principles"' not in code, (
        "verify-install must not build a docs constitution path"
    )


def test_is_authoritative_governing_source(monkeypatch, tmp_path):
    """The authoritative-source seam accepts the packaged path (and None) only.

    #2463 review: inception / offline-reanchor must refuse non-authoritative
    overrides. ``None`` (use the default) and any path that realpath-matches
    ``config.CONSTITUTION_PATH`` are authoritative; everything else is not.
    Tests express a custom governing source by monkeypatching
    ``config.CONSTITUTION_PATH``.
    """
    from kestrel_sovereign.constitution.resolver import (
        governing_constitution_path,
        is_authoritative_governing_source,
    )

    # None → use the default → authoritative.
    assert is_authoritative_governing_source(None) is True
    # The current packaged path is authoritative.
    assert is_authoritative_governing_source(governing_constitution_path()) is True
    # An arbitrary other path is NOT authoritative.
    rogue = tmp_path / "rogue.md"
    rogue.write_bytes(b"nope")
    assert is_authoritative_governing_source(str(rogue)) is False

    # Monkeypatching config.CONSTITUTION_PATH makes THAT file authoritative —
    # the seam the review prescribes for legitimate custom governing sources.
    custom = tmp_path / "custom_governing.md"
    custom.write_bytes(b"Kestrel Constitution\n")
    monkeypatch.setattr(
        "kestrel_sovereign.config.CONSTITUTION_PATH", str(custom)
    )
    assert is_authoritative_governing_source(str(custom)) is True
    # ...and the previously-packaged path is now non-authoritative.
    assert is_authoritative_governing_source(str(rogue)) is False
