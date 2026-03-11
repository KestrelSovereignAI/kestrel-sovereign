"""
Default configuration values for Kestrel.

All values can be overridden via environment variables.
This module provides a single source of truth for default configuration.
"""

import os
from typing import Optional


# =============================================================================
# Service URLs
# =============================================================================

def get_ollama_url() -> str:
    """Get Ollama service URL."""
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_ipfs_api_url() -> str:
    """Get IPFS API URL."""
    return os.getenv("IPFS_API_URL", "http://localhost:8889")


def get_mcp_gateway_url() -> str:
    """Get MCP Gateway URL."""
    return os.getenv("MCP_GATEWAY_URL", "http://localhost:9000/sse")


def get_lotus_rpc_url() -> str:
    """Get Lotus RPC URL for Filecoin."""
    return os.getenv("LOTUS_RPC_URL", "http://localhost:1234/rpc/v0")


def get_lighthouse_api_url() -> str:
    """Get Lighthouse API URL."""
    return os.getenv("LIGHTHOUSE_API_URL", "https://node.lighthouse.storage")


def get_openrouter_api_base() -> str:
    """Get OpenRouter API base URL."""
    return os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


def get_lighthouse_gateway_url() -> str:
    """Get Lighthouse IPFS gateway URL."""
    return os.getenv("LIGHTHOUSE_GATEWAY_URL", "https://gateway.lighthouse.storage/ipfs")


def get_storacha_gateway_url() -> str:
    """Get Storacha (web3.storage) IPFS gateway URL."""
    return os.getenv("STORACHA_GATEWAY_URL", "https://w3s.link/ipfs")


def get_xai_api_url() -> str:
    """Get xAI API base URL."""
    return os.getenv("XAI_API_URL", "https://api.x.ai/v1")


def get_groq_api_url() -> str:
    """Get Groq API base URL."""
    return os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")


# =============================================================================
# Service Configuration
# =============================================================================

# Kestrel agent
KESTREL_PORT = int(os.getenv("KESTREL_PORT", "8888"))
KESTREL_HOST = os.getenv("KESTREL_HOST", "0.0.0.0")

# Platform server (for multi-tenant deployments)
PLATFORM_PORT = int(os.getenv("PLATFORM_PORT", "7777"))
PLATFORM_HOST = os.getenv("PLATFORM_HOST", "0.0.0.0")

# Database
SQLITE_TIMEOUT = int(os.getenv("SQLITE_TIMEOUT", "30"))
POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "10"))


# =============================================================================
# Feature Flags
# =============================================================================

def is_development() -> bool:
    """Check if running in development mode."""
    return os.getenv("KESTREL_ENV", "production") == "development"


def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv("KESTREL_ENV", "production") == "production"


def agents_enabled() -> bool:
    """Check if Kestrel agents are enabled."""
    return os.getenv("ENABLE_AGENTS", "true").lower() == "true"
