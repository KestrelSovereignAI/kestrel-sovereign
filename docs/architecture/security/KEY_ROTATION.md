# Key Rotation Mechanism

## Overview

Key rotation allows the master encryption key (`KESTREL_DATA_KEY`) to be changed without losing access to encrypted data. This is critical for:

1. **Key Compromise Recovery** - If the current key is leaked, rotate to a new one
2. **Security Hygiene** - Periodic key rotation reduces exposure window
3. **Agent Migration** - Transfer agent to new infrastructure with new keys

## Design Principles

1. **Never lose data** - All encrypted content remains accessible during rotation
2. **Atomic transition** - Rotation either completes fully or not at all
3. **Audit trail** - Every rotation is logged and verifiable
4. **Backward compatible** - Old content decrypted, new content encrypted with new key

## Key Rotation Process

### Phase 1: Prepare
```
OLD_KEY -> still active
NEW_KEY -> generated/received but not yet primary
```

1. Generate or receive new key
2. Store new key hash in database (not the key itself)
3. Create rotation record with timestamp

### Phase 2: Re-encrypt
```
For each encrypted record:
  1. Decrypt with OLD_KEY
  2. Re-encrypt with NEW_KEY
  3. Update record
  4. Mark as rotated
```

Records processed:
- Conversation messages
- File store blobs
- Knowledge graph encrypted properties
- Backup artifacts (re-encrypt manifest)

### Phase 3: Finalize
```
OLD_KEY -> retired (kept for emergency recovery)
NEW_KEY -> promoted to primary
```

1. Update active key reference
2. Archive old key hash (for forensics, not storage)
3. Test decryption with new key
4. Complete rotation record

## Implementation

### CLI Command
```bash
# Interactive rotation
python -m security.key_rotation rotate --new-key-file /path/to/new_key

# Automated rotation (requires existing key)
python -m security.key_rotation rotate \
  --old-key-file /run/secrets/kestrel_data_key \
  --new-key-file /run/secrets/kestrel_data_key_new

# Check rotation status
python -m security.key_rotation status

# Emergency recovery (if rotation failed mid-way)
python -m security.key_rotation recover --key-file /path/to/backup_key
```

### Agent Command
```
!rotate-key --new-key-file /path/to/new_key
!key-status
```

### Database Schema

```sql
-- Track key rotations
CREATE TABLE IF NOT EXISTS key_rotations (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    old_key_hash TEXT NOT NULL,
    new_key_hash TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'in_progress', 'completed', 'failed', 'rolled_back'
    records_processed INTEGER DEFAULT 0,
    records_total INTEGER DEFAULT 0,
    error_message TEXT
);

-- Track which records have been rotated (for resumable rotation)
CREATE TABLE IF NOT EXISTS rotation_progress (
    rotation_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    rotated_at TEXT NOT NULL,
    PRIMARY KEY (rotation_id, table_name, record_id)
);
```

## Error Handling

### Failure Scenarios

1. **Power loss during rotation**
   - Rotation table tracks progress
   - Resume from last successful record
   - Idempotent operations

2. **New key is invalid**
   - Validate new key before starting
   - Test encryption/decryption cycle

3. **Decryption fails for some records**
   - Log failed records
   - Continue with others
   - Report summary at end
   - Option to retry with different key

4. **Out of disk space**
   - Rotation is in-place, not copy
   - Minimal space overhead

### Recovery Commands

```bash
# List in-progress rotations
python -m security.key_rotation list --status in_progress

# Resume interrupted rotation
python -m security.key_rotation resume --rotation-id <id> --key-file /path/to/key

# Rollback failed rotation (if possible)
python -m security.key_rotation rollback --rotation-id <id>
```

## Security Considerations

1. **Key storage** - New key should be provided via file, not command line
2. **Old key archival** - Hash stored for forensics, actual key should be securely deleted
3. **Audit logging** - Every rotation step logged
4. **Access control** - Only sovereign user can initiate rotation

## Future Enhancements

1. **Hardware Security Module (HSM) support** - Store keys in HSM
2. **Automatic rotation schedule** - Rotate keys on a schedule
3. **Multi-key support** - Different keys for different data types
4. **Key escrow** - Split key into shares (Shamir's Secret Sharing)

## Usage Example

```python
from security.key_rotation import KeyRotationService

async def rotate_key():
    service = KeyRotationService(storage)

    # Start rotation
    rotation_id = await service.start_rotation(
        new_key_file="/run/secrets/new_key"
    )

    # Monitor progress
    while True:
        status = await service.get_status(rotation_id)
        print(f"Progress: {status.records_processed}/{status.records_total}")
        if status.is_complete:
            break
        await asyncio.sleep(1)

    if status.status == "completed":
        print("Rotation complete!")
    else:
        print(f"Rotation failed: {status.error_message}")
```

## Migration Path

For existing agents without rotation support:

1. Export agent to IPFS (creates backup)
2. Stop agent
3. Update key in secrets file
4. Run rotation command
5. Verify decryption works
6. Restart agent

This ensures no data loss even if rotation mechanism itself fails.
