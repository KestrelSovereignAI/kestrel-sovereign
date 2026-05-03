"""Kestrel setup wizard package.

Submodules are imported directly (``from kestrel_sovereign.setup.wizard
import run_wizard``). This package init is intentionally empty to avoid
a circular import: ``doctor`` imports ``setup.env_file``, which used to
trigger this init, which used to eagerly load ``wizard`` → ``steps`` →
``verify`` → ``doctor``.
"""
