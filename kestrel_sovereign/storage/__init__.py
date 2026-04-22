#!/usr/bin/env python3
"""
Storage module for Kestrel.

Provides async storage interfaces:
- AsyncStorage: Fully async interface (recommended for all code)
- AsyncDatabase, AsyncFileStore, etc.: Individual async components

The codebase is fully async - no sync storage interfaces.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Import encryption utilities (shared)
from .encryption import (
    get_fernet,
    encrypt_bytes,
    decrypt_bytes,
    encrypt_string,
    decrypt_string,
    remove_enc_flag,
    DecryptionError,
)

# Import async storage components
from .async_storage import AsyncStorage
from .async_database import AsyncDatabase
from .async_file_store import AsyncFileStore
from .async_conversation_store import AsyncConversationStore
from .async_graph_store import AsyncGraphStore, GraphNode, Edge
from .async_rag_store import AsyncRAGStore
from .bm25_index import BM25Index, AsyncBM25Index, BM25_AVAILABLE

# Import privacy wrapper
from .privacy_wrapper import (
    PrivacyEnforcingStorage,
    PrivacyViolationError,
    PrivacyPolicy,
    wrap_storage_with_privacy,
)

# Import sovereignty types
from .sovereign_adapter import (
    SovereignStorageAdapter,
    RootManifest,
    ShardMetadata,
    AssetDescriptor,
    AssetMetadata,
    AssetCollector,
)

# Import CAR builder/reader
from .car_builder import CARBuilder, CARReader

# Import memory system components
from .memory_models import (
    MemoryMetadata,
    TemporalPattern,
    MemoryEpisode,
    EmotionalCategory,
)
from .emotional_tagger import EmotionalTagger, analyze_message
from .temporal_analyzer import TemporalAnalyzer
from .associative_linker import AssociativeLinker, LinkedConcept
from .memory_retriever import MemoryRetriever, calculate_decay
from .memory_consolidator import MemoryConsolidator
from .memory_system import MemorySystem

def get_default_agent_data_dir():
    """Returns the default agent data directory."""
    return os.environ.get("AGENT_DATA_DIR", "agent_data")


# Backward compatibility alias
Storage = AsyncStorage

__all__ = [
    # Core async storage
    "AsyncStorage",
    "AsyncDatabase",
    "AsyncFileStore",
    "AsyncConversationStore",
    "AsyncGraphStore",
    "AsyncRAGStore",
    # BM25 search
    "BM25Index",
    "AsyncBM25Index",
    "BM25_AVAILABLE",
    # Graph types
    "GraphNode",
    "Edge",
    # Privacy
    "PrivacyEnforcingStorage",
    "PrivacyViolationError",
    "PrivacyPolicy",
    "wrap_storage_with_privacy",
    # Memory system
    "MemoryMetadata",
    "TemporalPattern",
    "MemoryEpisode",
    "EmotionalCategory",
    "EmotionalTagger",
    "analyze_message",
    "TemporalAnalyzer",
    "AssociativeLinker",
    "LinkedConcept",
    "MemoryRetriever",
    "calculate_decay",
    "MemoryConsolidator",
    "MemorySystem",
    # Encryption utilities
    "get_fernet",
    "encrypt_bytes",
    "decrypt_bytes",
    "encrypt_string",
    "decrypt_string",
    "remove_enc_flag",
    "DecryptionError",
    # Utilities
    "get_default_agent_data_dir",
    # Sovereignty
    "SovereignStorageAdapter",
    "RootManifest",
    "ShardMetadata",
    "AssetDescriptor",
    "AssetMetadata",
    "AssetCollector",
    "CARBuilder",
    "CARReader",
    # Backward compatibility
    "Storage",
]
