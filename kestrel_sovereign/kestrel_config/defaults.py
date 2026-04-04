"""
Default configuration values for Kestrel.

Re-exports from kestrel_sdk.config.defaults for backward compatibility.
Feature packages should import from kestrel_sdk.config.defaults directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.config.defaults import *  # noqa: F401,F403
from kestrel_sdk.config.defaults import (  # noqa: F401
    get_ollama_url,
    get_ipfs_api_url,
    get_mcp_gateway_url,
    get_lotus_rpc_url,
    get_lighthouse_api_url,
    get_openrouter_api_base,
    get_lighthouse_gateway_url,
    get_storacha_gateway_url,
    get_sovereign_ipfs_url,
    get_xai_api_url,
    get_groq_api_url,
    is_development,
    is_production,
    agents_enabled,
)
