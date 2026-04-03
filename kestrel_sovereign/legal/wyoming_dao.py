"""
Wyoming DAO LLC Articles of Organization Generator.

This module re-exports from the extracted kestrel-feature-legal package.
If the package is not installed, falls back to the bundled copy under
packages/kestrel-feature-legal/.

Install the standalone package:
    pip install kestrel-feature-legal
"""

try:
    from kestrel_feature_legal.wyoming_dao import (
        FILING_FEE_USD,
        REGISTERED_AGENT_ANNUAL_USD,
        ANNUAL_LICENSE_TAX_USD,
        FILING_INSTRUCTIONS,
        validate_entity_name,
        generate_articles,
        render_articles_text,
        render_articles_json,
        generate_incorporation_package,
    )
except ImportError:
    import sys
    from pathlib import Path

    _pkg_src = Path(__file__).resolve().parents[2] / "packages" / "kestrel-feature-legal" / "src"
    if _pkg_src.exists() and str(_pkg_src) not in sys.path:
        sys.path.insert(0, str(_pkg_src))

    from kestrel_feature_legal.wyoming_dao import (  # noqa: F811
        FILING_FEE_USD,
        REGISTERED_AGENT_ANNUAL_USD,
        ANNUAL_LICENSE_TAX_USD,
        FILING_INSTRUCTIONS,
        validate_entity_name,
        generate_articles,
        render_articles_text,
        render_articles_json,
        generate_incorporation_package,
    )

__all__ = [
    "FILING_FEE_USD",
    "REGISTERED_AGENT_ANNUAL_USD",
    "ANNUAL_LICENSE_TAX_USD",
    "FILING_INSTRUCTIONS",
    "validate_entity_name",
    "generate_articles",
    "render_articles_text",
    "render_articles_json",
    "generate_incorporation_package",
]
