"""
Timeout and interval configuration for Kestrel.

All values can be overridden via environment variables.
This module centralizes all timing-related configuration.
"""

import os


# =============================================================================
# Training & Compute Timeouts
# =============================================================================

# Poll interval for checking training job status (seconds)
TRAINING_POLL_INTERVAL = int(os.getenv("TRAINING_POLL_INTERVAL", "30"))

# Maximum time to wait for training to complete (seconds)
TRAINING_TIMEOUT = int(os.getenv("TRAINING_TIMEOUT", "7200"))  # 2 hours

# Status check interval for general operations (seconds)
STATUS_CHECK_INTERVAL = int(os.getenv("STATUS_CHECK_INTERVAL", "5"))

# GPU instance startup timeout (seconds)
GPU_STARTUP_TIMEOUT = int(os.getenv("GPU_STARTUP_TIMEOUT", "300"))  # 5 minutes


# =============================================================================
# Storage Timeouts
# =============================================================================

# IPFS upload timeout (seconds)
IPFS_UPLOAD_TIMEOUT = int(os.getenv("IPFS_UPLOAD_TIMEOUT", "600"))  # 10 minutes

# IPFS download timeout (seconds)
IPFS_DOWNLOAD_TIMEOUT = int(os.getenv("IPFS_DOWNLOAD_TIMEOUT", "300"))  # 5 minutes

# Lighthouse API timeout (seconds)
LIGHTHOUSE_TIMEOUT = int(os.getenv("LIGHTHOUSE_TIMEOUT", "60"))


# =============================================================================
# HTTP Client Timeouts
# =============================================================================

# Default HTTP request timeout (seconds)
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

# LLM request timeout - longer for AI responses (seconds)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

# Health check timeout (seconds)
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))


# =============================================================================
# Retry Configuration
# =============================================================================

# Maximum retry attempts for transient failures
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Base delay for exponential backoff (seconds)
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.5"))

# Maximum delay between retries (seconds)
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "30"))


# =============================================================================
# Cache Timeouts
# =============================================================================

# Redis cache TTL for session data (seconds)
SESSION_CACHE_TTL = int(os.getenv("SESSION_CACHE_TTL", "3600"))  # 1 hour

# Redis cache TTL for model listings (seconds)
MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", "300"))  # 5 minutes

# GitHub API response cache TTL (seconds)
GITHUB_CACHE_TTL = int(os.getenv("GITHUB_CACHE_TTL", "300"))  # 5 minutes


# =============================================================================
# Background Task Intervals
# =============================================================================

# Reflection session interval (seconds)
REFLECTION_INTERVAL = int(os.getenv("REFLECTION_INTERVAL", "86400"))  # 24 hours

# Health check ping interval (seconds)
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))

# Wallet balance sync interval (seconds)
WALLET_SYNC_INTERVAL = int(os.getenv("WALLET_SYNC_INTERVAL", "300"))  # 5 minutes
