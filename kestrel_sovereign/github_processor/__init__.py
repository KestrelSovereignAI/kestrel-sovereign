"""GitHub Ticket Processor - Autonomous issue processing with Claude Agent SDK."""

from .config import GitHubProcessorConfig
from .models import IssueContext, ProcessingResult, ProcessingStatus
from .orchestrator import (
    Orchestrator,
    OrchestrationResult,
    RepoConfig,
    RepoRelationship,
    create_multi_repo_orchestrator,
)
from .ticket_processor import TicketProcessor

__all__ = [
    # Config
    "GitHubProcessorConfig",
    # Models
    "IssueContext",
    "ProcessingResult",
    "ProcessingStatus",
    # Processor
    "TicketProcessor",
    # Orchestrator
    "Orchestrator",
    "OrchestrationResult",
    "RepoConfig",
    "RepoRelationship",
    "create_multi_repo_orchestrator",
]
