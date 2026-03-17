# API Endpoint Matrix

## App-level routes

- `GET /`
- `GET /api/auth/key`
- `GET /health`
- `GET /health/detailed`
- `POST /webhooks/stripe/crypto`

## Router families

- `auth_oauth`
  - `/auth/login`
  - `/auth/callback`
  - `/auth/logout`
  - `/auth/me`
- `agent`
  - `/agent/invoke`
  - `/agent/stream`
  - `/agent/stop`
  - `/agent/info`
  - `/agent/privacy-mode`
  - `/agent/notifications`
  - `/agent/notifications/sse`
  - `/agent/context-status`
  - `/agent/reflection/status`
  - `/agent/tasks`
  - `/agent/heartbeat/status`
  - `/agent/heartbeat/trigger`
- `conversations`
  - `/api/sessions`
  - `/api/conversations`
  - `/api/conversations/{session_id}`
  - `/api/conversations/new`
  - `/api/conversations/messages/{message_id}`
  - `/api/conversations/{session_id}/transcript`
- `memories`
  - `/api/memories`
  - `/api/memories/{node_id}`
  - `/api/identity-chain`
- `sovereignty`
  - `/api/storage/stats`
  - `/api/sovereignty/exports`
  - `/api/sovereignty/export`
  - `/api/sovereignty/import`
  - `/api/sovereignty/files`
  - `/api/sovereignty/files/{filename}`
  - `/api/sovereignty/files/{filename}/preview`
- `database`
  - `/api/db/tables`
  - `/api/db/tables/{table_name}`
- `models`
  - `/api/agents`
  - `/api/identity`
  - `/api/constitution`
  - `/api/ipfs/status`
  - `/api/wallet`
  - `/api/keys`
  - `/api/models`
  - `/api/model/current`
  - `/api/model/set`
  - `/v1/models`
  - `/v1/chat/completions`
- `commands`
  - `/api/commands`
- `files`
  - `/api/files/{content_hash}`
- `security`
  - `/api/security/permissions/tree`
  - `/api/security/permissions`
  - `/api/security/permissions/feature`
  - `/api/security/pending`
  - `/api/security/approve`
  - `/api/security/audit`
  - `/api/security/cancel/{request_id}`
  - `/api/security/cancel-all`
  - `/api/security/reset-session`
- `observability`
  - `/api/observability/events`
  - `/api/observability/summary`
- `saved_items`
  - `/api/saved-items`
  - `/api/saved-items/stats`
  - `/api/saved-items/schemas`
  - `/api/saved-items/tags`
  - `/api/saved-items/by-tag/{tag}`
  - `/api/saved-items/by-schema/{schema_id}`
  - `/api/saved-items/{item_id}`
  - `/api/saved-items/structured`
  - `/api/saved-items/search`
  - `/api/saved-items/{item_id}/pin`

## Reconciliation note

The legacy catalog undercounted the live API surface and omitted the OAuth route family entirely.
