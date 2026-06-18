---
type: Architecture Spec
title: User Lifecycle Management Architecture
description: This document defines the complete user lifecycle in Kestrel, including
  account creation, data management, archival (cryo storage), and deletion with proper
  cascade handling.
resource: /docs/architecture/USER_LIFECYCLE_MANAGEMENT.md
tags:
- docs
- architecture
- architecture-spec
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: architecture
canonical: false
generated: false
privacy: public
---

# User Lifecycle Management Architecture

## Overview

This document defines the complete user lifecycle in Kestrel, including account creation, data management, archival (cryo storage), and deletion with proper cascade handling.

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER ENTITY                                     │
│                                                                             │
│  users                                                                       │
│    ├── companions (user_id) ──────────────────────────────────────────────┐│
│    │     ├── messages (companion_id)                                      ││
│    │     ├── memories (companion_id)                                      ││
│    │     ├── conversation_threads (companion_id)                          ││
│    │     ├── companion_wallets (companion_id)                             ││
│    │     │     ├── wallet_transactions (wallet_id)                        ││
│    │     │     └── fund_reservations (wallet_id)                          ││
│    │     ├── usage_events (companion_id)                                  ││
│    │     ├── agent_feedback (companion_id)                                ││
│    │     ├── tool_usage (companion_id)                                    ││
│    │     ├── web_searches (companion_id)                                  ││
│    │     ├── companion_service_keys (companion_id)                        ││
│    │     ├── trusted_circle_members (companion_id)                        ││
│    │     ├── trust_documents (companion_id)                               ││
│    │     ├── wellness_check_config (companion_id)                         ││
│    │     ├── wellness_checks (companion_id)                               ││
│    │     ├── a2u_notifications (companion_id)                             ││
│    │     └── funding_sources (companion_id)                               ││
│    │                                                                        │
│    ├── user_sessions (user_id)                                             │
│    ├── rate_limits (user_id)                                               │
│    ├── intake_responses (user_id)                                          │
│    └── usage_events (user_id)                                              │
│                                                                             │
│  Secondary References (via beneficiary_id, creator_id, funder_user_id):    │
│    ├── companions.beneficiary_id → users.id                                │
│    ├── companions.creator_id → users.id                                    │
│    ├── trusted_circle_members.added_by → users.id                          │
│    ├── wellness_check_config.incapacity_acknowledged_by → users.id         │
│    └── funding_sources.funder_user_id → users.id                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## User Lifecycle States

```
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ TRIAL    │───▶│ ACTIVE  │───▶│ INACTIVE │───▶│ ARCHIVED │───▶│ DELETED │
│(anonymous)│    │(account)│    │(dormant) │    │(cryo)    │    │(purged) │
└──────────┘    └─────────┘    └──────────┘    └──────────┘    └─────────┘
     │               │               │               │               │
     │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼
  30 min          Normal         90 days       7 year         Permanent
  auto-delete     operations     no login      retention      removal
```

### State Definitions

| State | Description | Retention | Actions Allowed |
|-------|-------------|-----------|-----------------|
| **TRIAL** | Anonymous user with device fingerprint | 30 minutes or 25 messages | Chat only, no save |
| **ACTIVE** | Full account with email/password | Indefinite while active | All features |
| **INACTIVE** | No login for 90+ days | 1 year before archive prompt | All features on return |
| **ARCHIVED** | User-requested or auto-archived | 7 years (legal compliance) | View-only, reactivate |
| **DELETED** | Permanently purged | None | N/A |

## Deletion Strategies

### 1. Hard Delete (CASCADE)
Complete removal of user and all related data. Used for:
- Test accounts
- Trial users (after expiry)
- GDPR/CCPA deletion requests (after cryo archive period)

```sql
-- Migration: Add proper CASCADE constraints
ALTER TABLE companions
    DROP CONSTRAINT companions_user_id_fkey,
    ADD CONSTRAINT companions_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
```

### 2. Soft Delete (Deactivation)
Mark as inactive but retain data. Used for:
- User-requested account deletion (with retention period)
- Dormant accounts
- Suspended accounts

```sql
-- Add soft delete columns
ALTER TABLE users ADD COLUMN IF NOT EXISTS
    deleted_at TIMESTAMPTZ,
    deletion_reason VARCHAR(50),
    deletion_requested_by VARCHAR(20);  -- 'user', 'admin', 'system'
```

### 3. Cryo Storage (Archive)
Export user data to cold storage before deletion. Used for:
- Legal compliance (7-year retention)
- User data portability (GDPR right)
- Disaster recovery

## Cryo Storage Architecture

**Existing Implementation**: The Kestrel framework already has cryostasis support via `TieredStorageManager` and `LighthouseProvider`. User archival should leverage this existing infrastructure rather than creating a parallel system.

### Integration with TieredStorageManager

```python
# Use existing cryostasis infrastructure
from storage.tiered_manager import TieredStorageManager, PrivacyMode

async def archive_user_to_cryo(
    user_id: UUID,
    manager: TieredStorageManager,
) -> StorageResult:
    """Archive user data using existing cryostasis infrastructure."""

    # Serialize user state (all companions, messages, memories, wallets)
    state_snapshot = await serialize_user_state(user_id)

    # Use existing cryostasis method
    return await manager.initiate_cryostasis(
        agent_id=f"user:{user_id}",
        state_snapshot=state_snapshot,
        metadata={
            "type": "user_archive",
            "user_id": str(user_id),
            "archived_at": datetime.utcnow().isoformat(),
        }
    )
```

### Storage Tiers (from existing system)

| Tier | Provider | Purpose |
|------|----------|---------|
| `BROWSER` | IndexedDB | Trial users, local-only |
| `LOCAL` | SQLite/FS | Default persistence |
| `CLOUD_HOT` | Lighthouse | Active cloud storage |
| `CLOUD_COLD` | Filecoin | Cryostasis archives |

### Archive Package Structure

Stored via Lighthouse → Filecoin deal:

```
user_archive_{user_id}_{timestamp}.tar.gz  → Lighthouse → Filecoin CID
├── manifest.json           # Archive metadata
├── user.json              # User profile (sanitized)
├── companions/
│   ├── {companion_id}/
│   │   ├── companion.json     # Companion config
│   │   ├── messages.jsonl     # All messages (encrypted)
│   │   ├── memories.jsonl     # All memories (encrypted)
│   │   ├── threads.json       # Conversation threads
│   │   └── lora/
│   │       └── {lora_files}   # LoRA model artifacts
│   └── ...
├── wallet_history.jsonl    # Financial transactions
├── usage_history.jsonl     # Usage events
└── checksums.sha256        # Integrity verification
```

### Cost Model (from DA-07-cryostasis.md, updated Feb 2026)

**Lighthouse Perpetual (pay once, stored forever via endowment pool):**

| Storage Size | Cost (one-time) | Duration |
|--------------|-----------------|----------|
| 10 MB | ~$0.04 | Forever |
| 100 MB | ~$0.40 | Forever |
| 1 GB | ~$4.00 | Forever |

**$5.00 cryostasis trigger**: Enough for perpetual archival of typical agent state.

### Archive Manifest Schema

```json
{
  "version": "1.0",
  "archived_at": "2025-01-03T12:00:00Z",
  "user_id": "uuid",
  "user_email_hash": "sha256_hash",  // For lookup without PII
  "reason": "user_request|gdpr|dormant|admin",
  "retention_until": "2032-01-03T12:00:00Z",
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_id": "lighthouse_key_id",
    "envelope_encrypted": true
  },
  "storage": {
    "provider": "lighthouse",
    "cid": "bafy...",
    "filecoin_deal_id": "deal_id"
  },
  "statistics": {
    "companions_count": 5,
    "messages_count": 15000,
    "memories_count": 500,
    "lora_models_count": 3,
    "total_size_bytes": 52428800
  }
}
```

## Cascade Delete Order

When deleting a user, tables must be deleted in this order to respect FK constraints:

```python
# FK-aware deletion order (children before parents)
USER_DELETION_ORDER = [
    # Tier 4: Deepest nested (no dependents)
    "wallet_transactions",
    "fund_reservations",
    "a2u_notifications",
    "wellness_checks",

    # Tier 3: Second level nesting
    "companion_wallets",
    "wellness_check_config",
    "trust_documents",
    "trusted_circle_members",
    "service_key_usage",
    "companion_service_keys",

    # Tier 2: Direct companion children
    "messages",
    "memories",
    "conversation_threads",
    "agent_feedback",
    "tool_usage",
    "web_searches",
    "usage_events",
    "funding_sources",

    # Tier 1: Direct user children
    "companions",
    "user_sessions",
    "rate_limits",
    "intake_responses",

    # Tier 0: User itself
    "users",
]
```

## API Endpoints

### User Deletion Request
```
POST /api/admin/users/{user_id}/delete
{
    "reason": "gdpr_request|user_request|admin_action|test_cleanup",
    "archive_first": true,
    "archive_storage": "gcs|ipfs|filecoin",
    "skip_retention": false,  // Admin only, for test accounts
    "force": false  // Skip confirmation, admin only
}
```

### Archive Status
```
GET /api/admin/users/{user_id}/archive
{
    "status": "not_archived|pending|completed|failed",
    "archive_url": "gs://bucket/path",
    "archived_at": "2025-01-03T12:00:00Z",
    "retention_until": "2032-01-03T12:00:00Z",
    "can_restore": true
}
```

### Restore from Archive
```
POST /api/admin/archives/{archive_id}/restore
{
    "new_email": "optional@email.com",  // If original email reused
    "skip_companions": ["uuid1", "uuid2"],  // Partial restore
    "dry_run": false
}
```

## Migration Plan

### Phase 1: Add Soft Delete Support
```sql
-- 015_user_lifecycle.sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS
    deleted_at TIMESTAMPTZ,
    deletion_reason VARCHAR(50) CHECK (deletion_reason IN (
        'user_request', 'gdpr_request', 'admin_action',
        'dormant', 'trial_expired', 'test_cleanup'
    )),
    deletion_requested_by VARCHAR(20) CHECK (deletion_requested_by IN (
        'user', 'admin', 'system', 'automated'
    )),
    deletion_scheduled_for TIMESTAMPTZ,
    archive_id UUID;

CREATE INDEX idx_users_deleted ON users(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX idx_users_deletion_scheduled ON users(deletion_scheduled_for)
    WHERE deletion_scheduled_for IS NOT NULL;
```

### Phase 2: Add CASCADE Constraints
```sql
-- Update all FK constraints to proper CASCADE behavior
-- See full migration in kestrel/migrations/015_user_lifecycle.sql
```

### Phase 3: Create Archive Tables
```sql
-- Archive metadata tracking
CREATE TABLE IF NOT EXISTS user_archives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,  -- Original user ID (may not exist anymore)
    user_email_hash VARCHAR(64) NOT NULL,  -- SHA256 for lookup

    -- Archive details
    reason VARCHAR(50) NOT NULL,
    requested_by VARCHAR(20) NOT NULL,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Storage
    storage_provider VARCHAR(20) NOT NULL,
    storage_path TEXT NOT NULL,
    storage_cid VARCHAR(100),  -- IPFS/Filecoin CID
    encryption_key_id VARCHAR(100),

    -- Retention
    retention_until TIMESTAMPTZ NOT NULL,
    auto_delete_after_retention BOOLEAN DEFAULT TRUE,

    -- Statistics
    companions_count INTEGER,
    messages_count INTEGER,
    total_size_bytes BIGINT,

    -- Status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'completed', 'failed', 'restored', 'expired'
    )),
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_archives_user ON user_archives(user_id);
CREATE INDEX idx_archives_email ON user_archives(user_email_hash);
CREATE INDEX idx_archives_retention ON user_archives(retention_until)
    WHERE status = 'completed';
```

## Service Implementation

### UserLifecycleService

```python
class UserLifecycleService:
    """Manages user lifecycle: creation, archival, deletion."""

    async def request_deletion(
        self,
        user_id: UUID,
        reason: DeletionReason,
        requested_by: str,
        archive_first: bool = True,
        skip_retention: bool = False,
    ) -> DeletionRequest:
        """
        Request user deletion with optional archival.

        For GDPR: archive_first=True, retention=7 years
        For test cleanup: archive_first=False, skip_retention=True
        """

    async def archive_user(
        self,
        user_id: UUID,
        storage_provider: StorageProvider,
        encryption_key_id: str,
    ) -> ArchiveResult:
        """Export all user data to cryo storage."""

    async def hard_delete_user(
        self,
        user_id: UUID,
        verify_archived: bool = True,
    ) -> DeleteResult:
        """Permanently delete user and all cascading data."""

    async def restore_from_archive(
        self,
        archive_id: UUID,
        new_email: Optional[str] = None,
    ) -> RestoreResult:
        """Restore user from cryo archive."""
```

## Cleanup Script Integration

The existing `cleanup_test_companions.py` should be enhanced to:

1. **Add `--users` flag** for user cleanup mode
2. **Respect cascade order** defined above
3. **Support cryo archive** before deletion for non-test users
4. **Track protected users** (like `nurses@YOUR_DOMAIN.com`, `platform@YOUR_DOMAIN.com`)

```python
# Protected users (never delete)
PROTECTED_USERS = {
    "nurses@YOUR_DOMAIN.com",      # Real user with LoRA companions
    "platform@YOUR_DOMAIN.com",    # Admin account
    "admin@YOUR_DOMAIN.com",       # System admin
}

# Test user patterns (safe to delete without archive)
TEST_USER_PATTERNS = [
    r".*@test\.kestrel\.ai$",
    r".*@test\.example\.com$",
    r"test_.*@.*",
    r".*_gw\d+_.*@.*",  # pytest-xdist
    r".*_master_.*@.*",  # pytest master
]
```

## Retention Policy Summary

| Data Type | Active User | Archived User | After Deletion |
|-----------|-------------|---------------|----------------|
| Profile | Indefinite | 7 years | Purged |
| Companions | Indefinite | 7 years (encrypted) | Purged |
| Messages | Indefinite | 7 years (encrypted) | Purged |
| LoRA Models | Indefinite | 7 years (IPFS) | IPFS pins removed |
| Financial Records | Indefinite | 7 years (legal) | Purged |
| Usage Logs | 2 years | 7 years | Purged |
| Session Tokens | Until expiry | Purged immediately | N/A |

## Security Considerations

1. **Encryption at Rest**: All archived data encrypted with AES-256-GCM
2. **Key Management**: Keys stored in Cloud KMS, rotated annually
3. **Access Control**: Archive access requires admin + audit log
4. **Integrity**: SHA-256 checksums for all archived files
5. **Audit Trail**: All deletion/archive operations logged immutably

## Related Documents

- [Database Schema](../../kestrel/database_schema.md)
- [Privacy Modes](./PRIVACY_MODES.md)
- [Vending Machine Funding](../plans/VENDING_MACHINE_FUNDING.md)
- [Eldercare Trusted Circle](../../kestrel/migrations/009_eldercare_trusted_circle.sql)
