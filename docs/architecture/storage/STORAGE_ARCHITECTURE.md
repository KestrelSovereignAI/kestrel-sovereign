# Kestrel Storage Architecture - Multi-Tier Design

**Last Updated:** December 8, 2025
**Status:** 🟢 Partially Implemented

---

## Related Documentation

- **[HUMAN_MEMORY_SYSTEM.md](HUMAN_MEMORY_SYSTEM.md)** - Human-like memory with emotional tagging, temporal patterns, and forgetting curves

---

## Architecture Overview

Kestrel uses a **multi-tier storage architecture** based on deployment context and privacy requirements:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KESTREL STORAGE TIERS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ TIER 1: Cloud/Server (Kestrel)                           │    │
│  │ - PostgreSQL (multi-tenant, ACID compliant)            │    │
│  │ - Redis (caching, sessions)                            │    │
│  │ - Used by: Kestrel FastAPI server                        │    │
│  │ - Location: Cloud Run, Cloud SQL                       │    │
│  │ - Concurrency: ✅ Full support (connection pooling)    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ TIER 2: Local/Agent (Kestrel Core)                     │    │
│  │ - SQLite (single-user, file-based)                     │    │
│  │ - Used by: KestrelAgent local instances                │    │
│  │ - Location: agent_data/*.db files                      │    │
│  │ - Concurrency: ⚠️  LIMITED (single writer)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ TIER 3: Browser/Mobile (Future)                        │    │
│  │ - IndexedDB (browser, ~50MB-1GB per origin)            │    │
│  │ - SQLite WASM (browser, limited)                       │    │
│  │ - SQLite Native (mobile - iOS/Android)                 │    │
│  │ - Used by: Future web/mobile clients                   │    │
│  │ - Location: Client device only                         │    │
│  │ - Concurrency: ✅ Single-threaded (browser/app thread) │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ TIER 4: Ephemeral (In-Memory)                          │    │
│  │ - Python dict/list (EPHEMERAL privacy mode)            │    │
│  │ - Used by: Off-the-record conversations                │    │
│  │ - Location: Process memory only                        │    │
│  │ - Concurrency: N/A (single process)                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: Cloud/Server Storage (Kestrel) ✅

### Technology Stack
- **Primary:** PostgreSQL 15+ with pgvector extension
- **Cache:** Redis 7+
- **Connection Pooling:** asyncpg with connection pool
- **Migrations:** Alembic (planned)

### Current Implementation
```python
# kestrel/server.py
DATABASE_URL = os.getenv("DATABASE_URL")  # postgresql://...
db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=10, max_size=100)
```

### Deployment
- **Local Dev:** Docker Compose (ports 5433, 6380)
- **Cloud:** Google Cloud SQL + Memorystore Redis
- **Isolation:** Multi-tenant (user_id foreign keys)

### Concurrency
✅ **Fully supported** - PostgreSQL handles concurrent writes natively

### Status
🟢 **PRODUCTION READY** - No CW-003 issues

---

## Tier 2: Local/Agent Storage (Kestrel Core) ⚠️

### Technology Stack
- **Primary:** SQLite 3.x
- **File Storage:** BLOBs in SQLite
- **Location:** `agent_data/{agent_id}.db`

### Current Implementation
```python
# storage/database.py
self.conn = sqlite3.connect(db_path, check_same_thread=False)  # ⚠️ ISSUE
```

### Problems (CW-003)
- `check_same_thread=False` bypasses SQLite's thread safety
- Race conditions on concurrent writes
- "Database is locked" errors under load
- Not suitable for multi-threaded FastAPI servers

### Use Cases
- ✅ **Single-user CLI agent** - Works fine (single thread)
- ⚠️ **FastAPI server with agents** - CAN cause corruption
- ❌ **Multi-instance deployment** - Not supported

### Solution Options

#### Option 1: Write Serialization (Quick Fix) ⭐
**Best for:** Kestrel Core local agents

```python
import threading

class Database:
    def __init__(self, db_path: str):
        self._write_lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def execute_write(self, query: str, params: tuple):
        """All writes must use this method"""
        with self._write_lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
            return cursor
```

**Pros:**
- Simple, low-risk change
- Preserves SQLite for local agents
- No new dependencies
- Works for single-process agents

**Cons:**
- Still limited to single-process
- Doesn't scale to multiple agent instances
- Lock contention under high load

#### Option 2: PostgreSQL for Kestrel Core (Future)
**Best for:** Multi-agent deployments

```python
# Use PostgreSQL for local agents too
DATABASE_URL = os.getenv("KESTREL_DB_URL", "sqlite:///agent_data/agent.db")

if DATABASE_URL.startswith("postgresql://"):
    # Use asyncpg
    db_pool = await asyncpg.create_pool(DATABASE_URL)
else:
    # Use SQLite with write lock
    db = Database(DATABASE_URL)
```

**Pros:**
- Proper concurrency
- Production-ready
- Consistent with Kestrel

**Cons:**
- Requires PostgreSQL installation
- More complex setup for local dev
- Overkill for single-user agents

#### Option 3: Keep SQLite, Document Limitations
**Best for:** Current state, defer to future

```python
# storage/database.py
class Database:
    """
    WARNING: This SQLite implementation is designed for SINGLE-THREADED use only.
    Do NOT use in multi-threaded environments (FastAPI, async workers, etc.).

    For multi-tenant/concurrent access, use PostgreSQL (see STORAGE_ARCHITECTURE.md).
    """
    def __init__(self, db_path: str):
        if db_path == ":memory:":
            # In-memory database - thread-safe for testing
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            # File-based - single thread only
            self.conn = sqlite3.connect(db_path, check_same_thread=True)  # ✅ FIXED
```

---

## Tier 3: Browser/Mobile Storage (Future) 🔮

### Browser (IndexedDB) - RECOMMENDED
**Best fit for sovereign data in browsers**

```javascript
// Future: Kestrel web client
const db = await idb.openDB('kestrel-agent', 1, {
  upgrade(db) {
    db.createObjectStore('conversations', { keyPath: 'id' });
    db.createObjectStore('files', { keyPath: 'content_hash' });
    db.createObjectStore('graph_nodes', { keyPath: 'node_id' });
  }
});

// Store conversation (sovereign, never leaves browser)
await db.put('conversations', {
  id: messageId,
  role: 'user',
  content: 'My private message',
  timestamp: new Date()
});
```

**Pros:**
- Native browser API (no dependencies)
- Large storage (50MB-1GB per origin)
- Async, non-blocking
- Supports blobs, indexes
- Data sovereignty (never leaves device)

**Cons:**
- JavaScript only
- User can clear browser data
- No server-side access

### Browser (SQLite WASM) - Alternative
**For advanced use cases**

```javascript
// Using sql.js or wa-sqlite
import initSqlJs from 'sql.js';

const SQL = await initSqlJs();
const db = new SQL.Database();

db.run(`CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  role TEXT,
  content TEXT
)`);
```

**Pros:**
- Familiar SQL syntax
- Can import/export .db files
- Compatible with Kestrel Core schema

**Cons:**
- Large bundle size (~1MB)
- Slower than IndexedDB
- Still limited by browser storage

### Mobile (SQLite Native) - React Native / Flutter
**For native mobile apps**

```dart
// Flutter example
import 'package:sqflite/sqflite.dart';

final db = await openDatabase(
  'kestrel_agent.db',
  version: 1,
  onCreate: (db, version) async {
    await db.execute('''
      CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        role TEXT,
        content TEXT
      )
    ''');
  },
);
```

**Pros:**
- Native performance
- Large storage (GB+)
- Full SQL support
- Data sovereignty (device-only)

**Cons:**
- Requires native app
- Platform-specific code

---

## Tier 4: Ephemeral Storage (In-Memory) ✅

### Current Implementation
```python
# features/privacy.py
class PrivacyAgent:
    def __init__(self, ...):
        self.ephemeral_session: Optional[EphemeralSession] = None

# ephemeral_session.py
class EphemeralSession:
    def __init__(self):
        self.messages: List[Dict] = []  # In-memory only
```

### Use Cases
- EPHEMERAL privacy mode
- Off-the-record conversations
- Temporary testing

### Guarantees
- ✅ Nothing persisted to disk
- ⚠️ Cleared on Python garbage collection (not cryptographically erased)
- ⚠️ Visible in memory dumps

### Future Enhancement
```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import secrets

class SecureEphemeralSession:
    def __init__(self):
        # Use mlock to prevent swapping to disk
        self._secure_buffer = secrets.token_bytes(1024 * 1024)  # 1MB buffer

    def clear(self):
        # Overwrite memory before deletion (DOD 5220.22-M)
        for i in range(len(self._secure_buffer)):
            self._secure_buffer = secrets.token_bytes(len(self._secure_buffer))
```

---

## Decision Matrix: Which Storage Tier to Use?

| Use Case | Storage Tier | Technology | Concurrency | Sovereign |
|----------|--------------|------------|-------------|-----------|
| **Kestrel Cloud Service** | Tier 1 (Cloud) | PostgreSQL + Redis | ✅ Full | ❌ Server-side |
| **Local Kestrel Agent (CLI)** | Tier 2 (Local) | SQLite (single-thread) | ⚠️ Limited | ✅ Local file |
| **Kestrel in FastAPI** | Tier 1 (Cloud) | PostgreSQL | ✅ Full | ⚠️ Depends on deployment |
| **Browser Client (Future)** | Tier 3 (Browser) | IndexedDB | ✅ Single-tab | ✅ Browser-only |
| **Mobile App (Future)** | Tier 3 (Mobile) | SQLite Native | ✅ Single-app | ✅ Device-only |
| **EPHEMERAL Mode** | Tier 4 (Memory) | Python dict | N/A | ✅ Memory-only |

---

## Recommendations

### Immediate (Week 1)
1. ✅ **Kestrel:** Keep PostgreSQL (no changes needed)
2. ⚠️ **Kestrel Core:** Add write serialization lock for SQLite
3. 📝 **Document:** Clarify SQLite is single-threaded only

### Short-term (Month 1)
4. 🔧 **Kestrel Core:** Support PostgreSQL as optional backend
5. 🧪 **Test:** Concurrent access tests for both SQLite + PostgreSQL
6. 🔐 **Ephemeral:** Implement secure memory erasure

### Long-term (Quarter 1)
7. 🌐 **Browser Client:** Implement IndexedDB storage layer
8. 📱 **Mobile Apps:** Implement native SQLite for React Native/Flutter
9. 🔄 **Sync:** Optional sync between browser ↔ cloud (user-controlled)

---

## Code Organization

```
storage/
├── __init__.py           # High-level Storage facade
├── database.py           # SQLite implementation (Tier 2)
├── postgres_adapter.py   # PostgreSQL adapter (future)
├── indexeddb_client.js   # Browser IndexedDB (future)
└── mobile_sqlite.dart    # Mobile native SQLite (future)

kestrel/
├── server.py            # Uses PostgreSQL (Tier 1) ✅
├── database_pool.py     # asyncpg connection pool
└── models.py            # SQLAlchemy models

ephemeral_session.py     # In-memory storage (Tier 4) ✅
```

---

## CW-003 Resolution

**Problem:** SQLite with `check_same_thread=False` in multi-threaded contexts

**Solution:**
- **Kestrel:** ✅ Already using PostgreSQL (no issue)
- **Kestrel Core:** ⚠️ Add write lock OR change to `check_same_thread=True`
- **Future Clients:** Use IndexedDB (browser) or native SQLite (mobile)

**Status:** 🟡 Partially resolved (Kestrel safe, Kestrel Core needs fix)

---

*Last Updated: November 20, 2025*
