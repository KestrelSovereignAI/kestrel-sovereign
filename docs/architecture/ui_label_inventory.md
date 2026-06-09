# UI Label Inventory

Catalog of every user-facing string in the Kestrel Sovereign main console, classified for the theme + i18n system (epic #986). Produced for sub-issue #987.

## Scope

**In scope**

- [`kestrel_sovereign/static/index.html`](../../kestrel_sovereign/static/index.html) — main console UI (~80 themable labels in the `<body>` from line 1182).
- [`kestrel_sovereign/static/js/`](../../kestrel_sovereign/static/js/) — strings injected into the DOM via `textContent` / `innerHTML`, plus user-facing `alert` / `confirm` / `prompt` strings. Sampled here; exhaustive enumeration is folded into #988 as authors populate locale files.

**Out of scope**

- `voice-test.html` — auxiliary surface, no theme-relevant labels.

**Note on inline CSS:** `index.html` has a ~1,180-line inline `<style>` block (lines 9–1181). Not in scope for this epic, but flagged as a separate housekeeping follow-up.

## Classifications

Every label is tagged with one of four classes:

| Class | Varies by theme? | Varies by locale? | Example |
|---|---|---|---|
| `theme` | yes | yes | "MultiAgent" → "Mews" → "Multi-Agent" |
| `locale` | no | yes | "Loading..." → "Cargando..." |
| `mech` | no | no | Numbers, glyphs, brand glyphs, ISO codes |
| `data` | n/a | n/a | Replaced at runtime from API/state (e.g. agent name) |

**Implication for theme files:** only `theme` rows need entries in `themes/<name>/<locale>.toml`. Everything else is either a single-source-of-truth string (`locale`, lives in a translation file regardless of theme) or not a label at all (`mech`, `data`).

For the MVP shipping `en` only, `locale` rows will live alongside `theme` rows in the same files, since the schema is `themes/<theme>/<locale>.toml`. The split between theme-variant and locale-only becomes meaningful once a second locale lands.

## Summary

| Source | theme | locale | mech | data |
|---|---:|---:|---:|---:|
| `index.html` body | ~38 | ~42 | ~15 | ~6 |
| JS modules (sampled) | ~4 | ~70+ | — | many |
| **Total themable keys (theme-variant)** | **~42** | | | |

JS counts are sampled because most JS strings are operational status/error messages that are pure `locale` (no theme variation) — exhaustive enumeration is not load-bearing for the schema design and naturally gets done during #988 as authors fill in locale files.

---

## index.html — Body labels (lines 1182–2009)

### Navigation

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1185 | Kestrel (alt) | brand_logo_alt | mech | Logo alt text |
| 1187 | Kestrel | brand_name | data | h1#nav-agent-name; replaced by agent name from API |
| 1190 | Identity | tab_identity | theme | |
| 1191 | Chat | tab_chat | theme | |
| 1192 | Constitution | tab_constitution | theme | |
| 1193 | Memories | tab_memories | theme | |
| 1195 | Tasks | tab_tasks | theme | |
| 1207 | Sovereignty | tab_sovereignty | theme | |
| 1208 | Resources | tab_resources | theme | |
| 1209 | Metrics | tab_metrics | theme | |
| 1210 | Spawn | tab_spawn | theme | |
| 1211 | Features | tab_features | theme | |
| 1213 | Security | tab_security | theme | |

### Sidebars

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1236 | Agents | sidebar_agents | theme | The "MultiAgent" rename target — already labeled "Agents" on screen, but this is the key the falconry theme would surface as "Mews" |
| 1237 | Collapse (title) | btn_collapse | locale | |
| 1240 | Loading agents... | loading_agents | locale | |
| 1248 | Conversations | sidebar_conversations | theme | |
| 1249 | New Conversation (title) | btn_new_conversation | locale | |
| 1250 | Show Trash (title) | btn_show_trash | locale | |
| 1254 | Select an agent to view conversations | empty_conversations | locale | |
| 1257 | Loading trash… | loading_trash | locale | Note: this row is the *only* place in `index.html` that uses a Unicode ellipsis (`…`); all other "Loading X" / "Vendor..." / "Thinking..." strings use ASCII three dots. Source inconsistency — see open question below. |

### Identity panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1266 | Loading identity... | loading_identity | locale | |

### Chat panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1287 | History | chat_history_title | theme | |
| 1295 | Close sidebar (title) | btn_close_sidebar | locale | |
| 1302 | + New Conversation | btn_new_conversation_long | locale | |
| 1304 | Loading history... | loading_history | locale | |
| 1322 | Chat History (title) | btn_chat_history | locale | |
| 1323 | New Chat (title) | btn_new_chat | locale | |
| 1331 | Select vendor (title) | model_selector_vendor_title | locale | |
| 1332 | Vendor... | model_selector_vendor_placeholder | locale | |
| 1334 | Select route (auth/endpoint) (title) | model_selector_route_title | locale | |
| 1335 | Route... | model_selector_route_placeholder | locale | |
| 1337 | Select model (title) | model_selector_model_title | locale | |
| 1338 | Model... | model_selector_model_placeholder | locale | |
| 1350 | Hello! I am your Kestrel AI agent, bound by the Kestrel Constitution... | chat_welcome_message | theme | Heaviest theme-variant content; falconry/plain rewrite the voice |
| 1360 | Thinking... | chat_thinking | theme | falconry: "Hunting..."? plain: "Thinking..." |
| 1361 | Stop (title) | btn_stop_title | locale | |
| 1370 | Stop | btn_stop | locale | |
| 1375 | Ask me anything or use !commands... | chat_input_placeholder | theme | |
| 1376 | Send (title) | btn_send | locale | Visible button content is `&#8679;` (⇧ glyph), not the word "Send"; only the `title` attribute carries the word | |

### Constitution panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1397 | Kestrel Constitution | constitution_title | theme | |
| 1401 | Loading constitution... | loading_constitution | locale | |

### Memories panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1411 | Knowledge Graph | memories_title | theme | falconry candidate: "The Mews Library" |
| 1413 | All Types | memories_filter_all | locale | |
| 1414 | Agent | memories_filter_agent | locale | |
| 1415 | Documents | memories_filter_documents | locale | |
| 1416 | Memories | memories_filter_memories | locale | |
| 1417 | Backups | memories_filter_backups | locale | |
| 1421 | Loading memories... | loading_memories | locale | |

### Tasks panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1430 | Tasks & Activity | tasks_title | theme | |
| 1432 | Refresh | btn_refresh | locale | (recurs across panels) |
| 1442 | Background Tasks | tasks_view_tasks | theme | |
| 1447 | Activity Log | tasks_view_activity | theme | |
| 1454 | A2A background tasks (image generation, training, etc.) | tasks_description | locale | |
| 1465 | All | filter_all | locale | (recurs) |
| 1466 | Working | tasks_filter_working | locale | |
| 1467 | Completed | tasks_filter_completed | locale | |
| 1468 | Failed | tasks_filter_failed | locale | |
| 1472 | Loading tasks... | loading_tasks | locale | |
| 1479 | Real-time log of tool calls, LLM invocations, and feature executions. | activity_description | locale | |
| 1482 | Loading activity... | loading_activity | locale | |

### Sovereignty panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1492 | Data Sovereignty | sovereignty_title | theme | |
| 1494 | Export to IPFS | btn_export_ipfs | locale | |
| 1495 | Import from CID | btn_import_cid | locale | |
| 1499 | Your data is your own. Export to IPFS/Filecoin for true ownership and portability. | sovereignty_tagline | theme | |
| 1501 | Export History | sovereignty_export_history | theme | |
| 1503 | Loading exports... | loading_exports | locale | |
| 1509 | Local Cache | sovereignty_local_cache | theme | |
| 1510 | Browse Local Files | btn_browse_local_files | locale | |
| 1513 | Browse cached backup files stored locally on your device. | sovereignty_local_cache_description | locale | |
| 1517 | Loading files... | loading_files | locale | |
| 1525 | Database Explorer | sovereignty_db_explorer | theme | |
| 1526 | Browse Database | btn_browse_database | locale | |
| 1529 | Read-only view of agent database tables and contents. | sovereignty_db_description | locale | |
| 1533 | Loading database... | loading_database | locale | |
| 1541 | IPFS Network | sovereignty_ipfs_network | theme | |
| 1542 | Check Status | btn_check_status | locale | |
| 1545 | Check connectivity to local IPFS node and public gateways. | sovereignty_ipfs_description | locale | |
| 1549 | Checking IPFS... | loading_ipfs | locale | |

### Resources panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1559 | Agent Resources | resources_title | theme | falconry candidate: "Equipage" |
| 1561 | API keys, wallet balances, and usage tracking. Keys are resolved in priority order: Agent → User → Platform. | resources_description | locale | |
| 1575 | Active Key Source: | resources_active_key_label | locale | |
| 1583 | Loading... | loading_generic | locale | |
| 1589 | Agent Keys | resources_agent_keys_title | theme | |
| 1589 | (this companion only) | resources_agent_keys_subtitle | locale | |
| 1591 | + Add Key | btn_add_key | locale | |
| 1592 | Refresh | btn_refresh | locale | |
| 1596 | Loading keys... | loading_keys | locale | |
| 1603 | Your Keys | resources_user_keys_title | theme | |
| 1603 | (BYOK - shared across companions) | resources_user_keys_subtitle | locale | |
| 1606 | Unlock | btn_unlock | locale | |
| 1610 | Keys encrypted with your passphrase. No wallet charges when using your own keys. | resources_byok_description | locale | |
| 1620 | Platform Access | resources_platform_title | theme | |
| 1620 | (pay-per-use) | resources_platform_subtitle | locale | |
| 1623 | Uses your wallet balance + platform margin. Fallback when no personal keys available. | resources_platform_description | locale | |
| 1626 | Loading platform access... | loading_platform | locale | |
| 1633 | Wallet | resources_wallet_title | theme | |
| 1637 | Loading wallet... | loading_wallet | locale | |
| 1643 | Usage (Last 30 Days) | resources_usage_title | theme | |
| 1645 | Loading usage... | loading_usage | locale | |

### Metrics panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1655 | Metrics Dashboard | metrics_title | theme | |
| 1658 | Auto-refresh | label_auto_refresh | locale | (recurs) |
| 1668 | Off | refresh_off | locale | |
| 1669–1671 | 10s / 30s / 60s | refresh_10s / refresh_30s / refresh_60s | mech | Time codes; same in every locale |
| 1682 | ↻ Refresh | btn_refresh_with_glyph | locale | |
| 1693 | Loading metrics... | loading_metrics | locale | |
| 1710 | Event Timeline | metrics_event_timeline | theme | |
| 1723 | Tool Duration (avg ms) | metrics_tool_duration | theme | |
| 1744 | Event Distribution | metrics_event_distribution | theme | |
| 1757 | Recent Errors | metrics_recent_errors | theme | |
| 1759 | No errors recorded | metrics_no_errors | locale | |

### Spawn panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1770 | Spawn Manager | spawn_title | theme | falconry candidate: "Hatchery" |
| 1783 | Off | (reuses refresh_off) | locale | |
| 1784–1786 | 5s / 10s / 30s | refresh_5s / refresh_10s / refresh_30s | mech | |
| 1797 | ↻ Refresh | btn_refresh_with_glyph | locale | |
| 1803 | Active Children | spawn_active_children | theme | falconry: "Active Eyases"? |
| 1806 | Click refresh or switch to this tab to load spawn data | spawn_empty_state | locale | |
| 1825 | Delegation Chain | spawn_delegation_chain | theme | |
| 1827 | No delegation chain | spawn_no_delegation | locale | |
| 1838 | Budget Allocation | spawn_budget_allocation | theme | |
| 1852 | Spawn History | spawn_history_title | theme | |
| 1854 | No spawn events recorded | spawn_no_history | locale | |

### Features panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1864 | Feature Store | features_title | theme | |
| 1868 | ↻ Refresh | btn_refresh_with_glyph | locale | |
| 1871 | Browse, install, and manage feature packages for your agent. | features_description | locale | |
| 1876 | Search features, tags, skills... | features_search_placeholder | locale | |
| 1897 | All | features_filter_all | locale | |
| 1906 | Installed | features_filter_installed | locale | |
| 1915 | Available | features_filter_available | locale | |
| 1925 | Loading features... | loading_features | locale | |

### Security panel

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 1933 | Security Permissions | security_title | theme | |
| 1935 | Control which tools can run automatically, require approval, or are blocked. | security_description | locale | |
| 1941 | Pending Approvals | security_pending_approvals | theme | |
| 1952 | No pending approvals | security_no_pending | locale | |
| 1960 | Permission Tree | security_permission_tree | theme | |
| 1964 | ↻ Refresh | btn_refresh_with_glyph | locale | |
| 1968 | Click to load permission tree... | security_load_tree_prompt | locale | |
| 1972 | Legend: | security_legend_label | locale | |
| 1972 | Allow | security_legend_allow | locale | |
| 1972 | Ask | security_legend_ask | locale | |
| 1972 | Deny | security_legend_deny | locale | |
| 1972 | Mixed | security_legend_mixed | locale | |
| 1978 | Session Controls | security_session_controls | theme | |
| 1981 | Reset Session Permissions | btn_reset_session | locale | |
| 1984 | Cancel All Pending | btn_cancel_pending | locale | |
| 1988 | Session permissions are cleared when the browser is closed. | security_session_note | locale | |
| 1995 | Audit Log | security_audit_log | theme | |
| 1999 | ↻ Refresh | btn_refresh_with_glyph | locale | |
| 2003 | Click to load audit log... | security_load_audit_prompt | locale | |

### Page-level

| Line | Current text | Proposed key | Class | Notes |
|---|---|---|---|---|
| 7 | Kestrel Sovereign Console | document_title | theme | `<title>` element |

---

## JS modules — sampled patterns

Three patterns dominate JS-injected text:

### 1. Status/error alerts (always `locale`)

Concentrated in `resources.js`, `chat.js`, `security.js`, `identity.js`. ~30 instances of `alert(...)` / `confirm(...)` / `prompt(...)`. Examples:

| File | Sample text | Proposed key | Class |
|---|---|---|---|
| resources.js | Please enter an API key | alert_api_key_required | locale |
| resources.js | Please enter a passphrase to encrypt your key | alert_passphrase_required | locale |
| resources.js | Passphrases do not match | alert_passphrase_mismatch | locale |
| resources.js | Passphrase must be at least 8 characters | alert_passphrase_short | locale |
| resources.js | Key added successfully | alert_key_added | locale |
| resources.js | Invalid passphrase | alert_invalid_passphrase | locale |
| resources.js | Are you sure you want to delete your ${provider} key? This cannot be undone. | alert_confirm_delete_key | locale (interpolated) |
| chat.js | Move this message to Trash? You can restore it from the trash view. | confirm_trash_message | locale |

(Full enumeration: pulled into the locale file by #988 author pass.)

### 2. Inline button-state toggles (mostly `locale`, some `theme`)

Buttons swap their label between idle and active states. Sample:

| File | Idle → Active | Proposed keys | Class |
|---|---|---|---|
| identity.js | Upload → Uploading... | btn_upload / btn_upload_active | locale |
| identity.js | Generate → Generating... | btn_generate / btn_generate_active | locale |
| identity.js | Restore | btn_restore | locale |
| identity.js | Delete permanently | btn_delete_permanent | locale |
| identity.js | Trash items are automatically deleted after 30 days. | trash_retention_notice | locale |

### 3. Programmatic data interpolation (`data` — not labels)

Most `textContent` / `innerHTML` assignments inject runtime values: agent names, timestamps, message bodies, model IDs. These don't need keys. Heuristic: if the assigned value is a variable (e.g. `el.textContent = agent.name`) or template literal containing only interpolations, it's `data`.

---

## Theme-variant key set (illustrative mapping)

The complete `theme`-class key set, with example mappings to the three initial themes. This is the input to #988 — author them into `themes/<theme>/en.toml`.

| Key | legacy | falconry | plain |
|---|---|---|---|
| tab_identity | Identity | Identity | Identity |
| tab_chat | Chat | Chat | Chat |
| tab_constitution | Constitution | Constitution | Constitution |
| tab_memories | Memories | Memories | Memories |
| tab_tasks | Tasks | Tasks | Tasks |
| tab_sovereignty | Sovereignty | Sovereignty | Sovereignty |
| tab_resources | Resources | Equipage | Resources |
| tab_metrics | Metrics | Metrics | Metrics |
| tab_spawn | Spawn | Hatchery | Spawn |
| tab_features | Features | Features | Features |
| tab_security | Security | Security | Security |
| sidebar_agents | MultiAgent | Mews | Multi-Agent |
| sidebar_conversations | Conversations | Conversations | Conversations |
| chat_history_title | History | History | History |
| chat_welcome_message | (current welcome) | (falconry-voiced rewrite) | (plain rewrite) |
| chat_thinking | Thinking... | Hunting... | Thinking... |
| chat_status_reasoning | Reasoning... | Pondering... | Reasoning... |
| chat_status_searching | Searching... | Scouting... | Searching... |
| chat_status_reading | Reading... | Spotting... | Reading... |
| chat_status_writing | Writing... | Scribing... | Writing... |
| chat_status_running | Running... | Stooping... | Running... |
| chat_status_pushing | Pushing... | Delivering... | Pushing... |
| chat_status_remembering | Remembering... | Roosting... | Remembering... |
| chat_status_looking | Looking... | Eyeing... | Looking... |
| chat_status_consulting | Consulting... | Calling... | Consulting... |
| chat_status_working | Working... | Hunting... | Working... |
| chat_status_revising | Revising... | Mantling... | Revising... |
| chat_input_placeholder | Ask me anything... | Speak to your bird... | Type a message... |
| constitution_title | Kestrel Constitution | Kestrel Constitution | Kestrel Constitution |
| memories_title | Knowledge Graph | The Mews Library | Memories |
| tasks_title | Tasks & Activity | Tasks & Activity | Tasks & Activity |
| tasks_view_tasks | Background Tasks | Background Tasks | Background Tasks |
| tasks_view_activity | Activity Log | Activity Log | Activity Log |
| sovereignty_title | Data Sovereignty | Data Sovereignty | Data Sovereignty |
| sovereignty_tagline | (current) | (current) | (current) |
| sovereignty_export_history | Export History | Export History | Export History |
| sovereignty_local_cache | Local Cache | Local Cache | Local Cache |
| sovereignty_db_explorer | Database Explorer | Database Explorer | Database Explorer |
| sovereignty_ipfs_network | IPFS Network | IPFS Network | IPFS Network |
| resources_title | Agent Resources | Equipage | Agent Resources |
| resources_agent_keys_title | Agent Keys | Agent Keys | Agent Keys |
| resources_user_keys_title | Your Keys | Your Keys | Your Keys |
| resources_platform_title | Platform Access | Platform Access | Platform Access |
| resources_wallet_title | Wallet | Wallet | Wallet |
| resources_usage_title | Usage (Last 30 Days) | Usage (Last 30 Days) | Usage (Last 30 Days) |
| metrics_title | Metrics Dashboard | Metrics Dashboard | Metrics Dashboard |
| metrics_event_timeline | Event Timeline | Event Timeline | Event Timeline |
| metrics_tool_duration | Tool Duration (avg ms) | Tool Duration (avg ms) | Tool Duration (avg ms) |
| metrics_event_distribution | Event Distribution | Event Distribution | Event Distribution |
| metrics_recent_errors | Recent Errors | Recent Errors | Recent Errors |
| spawn_title | Spawn Manager | Hatchery | Spawn Manager |
| spawn_active_children | Active Children | Active Eyases | Active Children |
| spawn_delegation_chain | Delegation Chain | Delegation Chain | Delegation Chain |
| spawn_budget_allocation | Budget Allocation | Budget Allocation | Budget Allocation |
| spawn_history_title | Spawn History | Spawn History | Spawn History |
| features_title | Feature Store | Feature Store | Feature Store |
| security_title | Security Permissions | Security Permissions | Security Permissions |
| security_pending_approvals | Pending Approvals | Pending Approvals | Pending Approvals |
| security_permission_tree | Permission Tree | Permission Tree | Permission Tree |
| security_session_controls | Session Controls | Session Controls | Session Controls |
| security_audit_log | Audit Log | Audit Log | Audit Log |
| document_title | Kestrel Sovereign Console | Kestrel Sovereign Console | Kestrel Sovereign Console |

**Observation:** many `theme` keys end up identical across all three themes. That's fine — it means the rename target is concentrated on a smaller set (~10 keys do real work: `sidebar_agents`, `tab_resources`, `tab_spawn`, `memories_title`, `resources_title`, `spawn_title`, `spawn_active_children`, `chat_thinking`, `chat_input_placeholder`, `chat_welcome_message`). The rest go through the system for the i18n machinery, not for theme variation.

This is a useful finding: the theme system carries low semantic load per key but is a clean pattern for the keys that *do* matter, and it's identical to what the i18n layer needs anyway.

---

## Open questions for #988 author

1. **Welcome message** — `chat_welcome_message` is a paragraph-length string. Should the falconry voice be a full rewrite or a tweak? Affects whether we're authoring a couple sentences per theme or just swapping a few nouns.
2. **Recurring keys** — `btn_refresh`, `btn_refresh_with_glyph`, `label_auto_refresh`, `loading_X` patterns recur across panels. Worth namespacing (`common.btn_refresh`) or flat is fine?
3. **Ellipsis inconsistency in source** — `index.html` uses ASCII three dots (`...`) in 27 places (all `Loading...`, `Thinking...`, `Vendor...`, `Search...`, `Click to load...` etc.) and Unicode `…` in exactly one place (`Loading trash…` at line 1257). Locale files should normalize on one form (recommend ASCII `...` to match the dominant pattern); the source HTML inconsistency itself should be cleaned up as a small follow-up so theme/locale resolution is byte-stable.
4. **Subtitle pattern** — `(this companion only)`, `(BYOK - shared across companions)`, `(pay-per-use)` are inline modifiers. Treated as separate keys here; alternative is to fold into the parent label.

These don't block the schema design but should be answered before locale files get authored beyond English.
