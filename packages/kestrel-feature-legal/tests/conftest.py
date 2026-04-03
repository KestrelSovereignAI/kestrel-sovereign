"""Shared pytest configuration for kestrel-feature-legal tests."""

# When run from within the kestrel-sovereign repo, fixtures are already
# registered via the pytest11 entry point.  Only load the plugin when
# running standalone (i.e. the entry point is not installed).
try:
    import kestrel_sovereign.testing.fixtures  # noqa: F401
except ImportError:
    pass
