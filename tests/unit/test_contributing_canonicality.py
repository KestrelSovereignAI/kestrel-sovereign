"""Guard the contributor guide's public-repository mirror."""

from scripts.sync_public_repo_files import (
    PUBLIC_CONTRIBUTING,
    PUBLIC_NOTICE,
    contributing_copy_is_current,
)


def test_public_contributing_copy_matches_canonical_guide():
    """A public release cannot silently carry a different contribution guide."""
    assert contributing_copy_is_current()


def test_public_contributing_copy_has_public_provenance():
    """The generated artifact must not claim the root guide's authority."""
    assert PUBLIC_NOTICE in PUBLIC_CONTRIBUTING.read_text()
