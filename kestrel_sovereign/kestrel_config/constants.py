"""
Kestrel Configuration Constants.

Centralized location for all magic numbers and configuration values
used throughout the Kestrel codebase.
"""

# =============================================================================
# NETWORK TIMEOUTS (in seconds)
# =============================================================================

# Connection and basic operations
REDIS_CONNECT_TIMEOUT = 2  # Fail fast if Redis is unavailable
HTTP_TIMEOUT_SHORT = 5  # Quick HTTP checks
HTTP_TIMEOUT_DEFAULT = 30  # Standard HTTP operations
HTTP_TIMEOUT_MEDIUM = 60  # Medium-length operations (file downloads, API calls)
HTTP_TIMEOUT_LONG = 180  # Long operations (3 minutes)
HTTP_TIMEOUT_DOWNLOAD = 300  # Large file downloads (5 minutes - IPFS)
HTTP_TIMEOUT_MODEL_PULL = 600  # Model pulling operations (10 minutes)
HTTP_TIMEOUT_GENERATION = 180  # Image/content generation (3 minutes)
HTTP_TIMEOUT_UPLOAD = 120  # File upload operations (2 minutes)
HTTP_TIMEOUT_QUICK = 10  # Very quick health checks
HTTP_TIMEOUT_VERY_SHORT = 5  # Ultra-quick checks (same as HTTP_TIMEOUT_SHORT for clarity)

# Worker and task management
TASK_WORKER_STOP_TIMEOUT = 10  # Graceful shutdown timeout for task workers
SHUTDOWN_TIMEOUT = 5  # General service shutdown timeout
CLIENT_CLOSE_TIMEOUT = 2  # Timeout for closing HTTP/async clients

# SSH and remote operations
SSH_COMMAND_TIMEOUT_SHORT = 10  # Quick SSH commands
SSH_COMMAND_TIMEOUT_DEFAULT = 30  # Standard SSH commands
SSH_COMMAND_TIMEOUT_MEDIUM = 60  # Medium SSH operations
SSH_COMMAND_TIMEOUT_SETUP = 120  # Environment setup via SSH (2 minutes)
SSH_COMMAND_TIMEOUT_LONG = 300  # Long SSH operations (5 minutes)
SSH_COMMAND_TIMEOUT_GENERATION = 180  # Image generation via SSH (3 minutes)

# Pod/instance wait timeouts
POD_READY_TIMEOUT = 300  # Wait for pod to be ready (5 minutes)
BACKEND_URL_TIMEOUT = 120  # Wait for backend URL (2 minutes)
BACKEND_URL_TIMEOUT_SHORT = 60  # Short wait for backend URL (1 minute)

# =============================================================================
# POLLING AND RETRY INTERVALS (in seconds)
# =============================================================================

POLL_INTERVAL_FAST = 1.0  # Fast polling (every second)
POLL_INTERVAL_MEDIUM = 5  # Medium polling (every 5 seconds)
POLL_INTERVAL_DEFAULT = 30  # Default polling interval
POLL_INTERVAL_MINUTE = 60  # Poll every minute
REFLECTION_INTERVAL = 86400  # Daily reflection (24 hours)

# Specific service polling intervals
VASTAI_POLL_INTERVAL_SECONDS = 30  # Vast.ai training status polling
REPLICATE_POLL_INTERVAL_SECONDS = 30  # Replicate training status polling
TRAINING_POLL_INTERVAL_SECONDS = 30  # Generic training protocol polling
RUNPOD_STATUS_POLL_INTERVAL = 5  # RunPod pod status polling
RUNPOD_URL_POLL_INTERVAL = 5  # RunPod backend URL polling
GCP_OPERATION_POLL_INTERVAL = 5  # GCP compute operation polling
MCP_CONNECTION_RETRY_DELAY = 0.5  # MCP container connection retry delay

# =============================================================================
# CACHE AND SESSION TTLs (in seconds)
# =============================================================================

SESSION_CACHE_TTL = 3600  # Session cache lifetime (1 hour)
LLM_CACHE_TTL_SECONDS = 300  # Model discovery cache lifetime (5 minutes)
STORAGE_CACHE_TTL_SECONDS = 60  # Storage info cache lifetime (1 minute)

# =============================================================================
# LLM RETRY CONFIGURATION
# =============================================================================

LLM_RETRY_MAX_DELAY_SECONDS = 60.0  # Maximum delay between retries

# =============================================================================
# SSE AND STREAMING
# =============================================================================

SSE_PING_INTERVAL_SECONDS = 15  # Server-sent events ping interval
MAX_SSE_CONNECTIONS_PER_CLIENT = 5  # Maximum concurrent SSE connections per client IP

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

SESSION_GAP_MINUTES = 30  # Gap between messages to start a new session

# =============================================================================
# SOVEREIGNTY
# =============================================================================

MAX_SOVEREIGNTY_PREVIEW_SIZE = 10000  # Max file preview size in bytes

# =============================================================================
# CLAUDE MAX
# =============================================================================

CLAUDE_MAX_TIMEOUT_SECONDS = 120  # Request timeout for Claude Max adapter

# =============================================================================
# TRAINING CONSTANTS
# =============================================================================

# Training timeouts
TRAINING_TIMEOUT = 600  # Training operation timeout (10 minutes)
TRAINING_GENERATION_TIMEOUT = 900  # Image generation timeout (15 minutes)
TRAINING_TIMEOUT_EXTENDED = 1800  # Extended training timeout (30 minutes)
TRAINING_TIMEOUT_LORA = 7200  # LoRA training timeout (2 hours)
TRAINING_POLL_INTERVAL = 30  # Poll for training status every 30 seconds
TRAINING_POLL_INTERVAL_FAST = 10  # Fast polling for generation status

# Training parameters
DEFAULT_TRAINING_STEPS = 1000  # Default number of training steps
DEFAULT_IMAGE_SIZE = 1024  # Default image dimensions for training
DEFAULT_TRAINING_BATCH_SIZE = 1  # Default batch size for training

# =============================================================================
# RATE LIMITS (requests per time period)
# =============================================================================

REGISTRATION_RATE_LIMIT = 20  # Registrations per hour
LOGIN_RATE_LIMIT = 10  # Login attempts per 5 minutes

# =============================================================================
# MCP SESSION MANAGEMENT
# =============================================================================

SESSION_CONNECT_TIMEOUT_SHORT = 10  # Quick MCP session connection
SESSION_CONNECT_TIMEOUT_DEFAULT = 15  # Standard MCP session connection
MCP_MAX_CONNECTION_ATTEMPTS = 10  # Maximum container connection attempts

# =============================================================================
# APPROVAL AND USER INTERACTION TIMEOUTS
# =============================================================================

APPROVAL_TIMEOUT_DEFAULT = 300  # Default user approval timeout (5 minutes)
APPROVAL_TIMEOUT_SHORT = 60  # Short approval timeout (1 minute)

# =============================================================================
# SUBPROCESS AND DOCKER TIMEOUTS
# =============================================================================

SUBPROCESS_TIMEOUT_SHORT = 5  # Quick subprocess checks (docker info, etc.)

# =============================================================================
# FILE STORAGE LIMITS
# =============================================================================

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB maximum file size for store_file()

# =============================================================================
# QUERY AND DISPLAY LIMITS
# =============================================================================

MAX_CONVERSATION_HISTORY_LIMIT = 10000  # Maximum conversation history items to retrieve
DEFAULT_OBSERVABILITY_LIMIT = 1000  # Default limit for observability events
MAX_PINNED_ITEMS_DISPLAY = 20  # Maximum pinned IPFS items to display

# =============================================================================
# IMAGE AND MEDIA
# =============================================================================

# Image generation and processing timeouts align with HTTP_TIMEOUT_* above

# =============================================================================
# USAGE NOTES
# =============================================================================
# When adding new constants:
# 1. Use SCREAMING_SNAKE_CASE naming
# 2. Group related constants together
# 3. Add inline comments explaining the value
# 4. Use seconds for all time-based values
# 5. Document units in variable name or comment
