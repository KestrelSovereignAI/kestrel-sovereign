"""
Shared test utilities for Kestrel and kestrel test suites.

This module provides:
- ResourceRegistry: Global crash-safe resource tracking
- CostTracker: Cloud resource cost estimation
- Cleanup hooks for pytest
- LLM credential checks for test skip conditions
"""
import os
import shutil
import subprocess

from .resource_registry import registry, TrackedResource, ResourceRegistry
from .cost_tracker import cost_tracker, CostTracker, ResourceUsage


def _check_claude_max_available() -> bool:
    """
    Check if Claude Max subscription is available via the Claude CLI.
    
    Returns True if:
    - claude CLI is installed
    - User is logged in (claude setup-token has been run)
    """
    if not shutil.which("claude"):
        return False
    
    try:
        # Check if claude CLI responds (indicates it's installed and configured)
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def has_llm_credentials() -> bool:
    """
    Check if any valid LLM provider credentials are available.
    
    Checks for:
    - ANTHROPIC_API_KEY: Standard Anthropic API key
    - ANTHROPIC_AUTH_TOKEN: Claude Max subscription OAuth token
    - OPENAI_API_KEY: OpenAI API key
    - GOOGLE_API_KEY: Google/Gemini API key
    - OPENROUTER_API_KEY: OpenRouter API key
    - Claude CLI: Claude Max subscription via `claude setup-token`
    
    Returns:
        True if at least one valid credential is found, False otherwise.
    """
    credential_vars = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",  # Claude Max subscription OAuth
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    
    # Check environment variables
    if any(os.environ.get(var, "").strip() for var in credential_vars):
        return True
    
    # Check Claude Max CLI availability
    if _check_claude_max_available():
        return True
    
    return False


def no_llm_credentials() -> bool:
    """
    Inverse of has_llm_credentials() for use in pytest.mark.skipif.
    
    Returns:
        True if NO valid LLM credentials are found (test should be skipped).
    """
    return not has_llm_credentials()


__all__ = [
    'registry',
    'TrackedResource',
    'ResourceRegistry',
    'cost_tracker',
    'CostTracker',
    'ResourceUsage',
    'has_llm_credentials',
    'no_llm_credentials',
]
