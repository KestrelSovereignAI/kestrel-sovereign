// ============================================================================
// Core panel definitions (issue #2145, epic #2038 ticket 06 north star)
// ============================================================================
//
// The Spawn feature panel already flows through the `ui-ext` panel registry
// (`registerPanel`). This module completes the direction for the CORE panels
// (identity, constitution, memories, tasks, sovereignty, resources, metrics,
// features, security, approvals): each is described here as a registry
// contribution — a stable `panelId`, its nav `label`/`labelKey`, a `gate`
// derived from the historical `PANEL_CAPABILITIES` map, and the panel body
// markup that used to live statically in `index.html`.
//
// This module is deliberately PURE and dependency-light: it imports nothing
// heavy (no per-panel loader modules, no chat.js). It only carries data + gate
// predicates + body templates, so the gate contract is unit-testable in
// isolation and so the embeddable `mountPanels` host (`mount-panels.js`) can
// build a full nav + panel bodies WITHOUT `index.html` markup existing.
//
// The body markup is moved VERBATIM from `index.html`'s `#panel-<id>`
// `.panel-content` so the SAME code path serves standalone `index.html` (where
// the registry adopts the in-place body and `buildCorePanelBody` is a no-op
// because the body already exists) and an embedder (where the registry creates
// an empty container and `buildCorePanelBody` fills it). `data-label-key`
// attributes are preserved so the i18n layer re-hydrates the injected DOM.
//
// Gate semantics mirror the retired `PANEL_CAPABILITIES` behavior exactly:
//   - a single required capability (identity, memory, tasks, …), or
//   - "any of" for composite panels (resources → keys OR wallet;
//     security → audit OR permissions).
// Gating flows through the live `api.hasCapability` so an embedder's
// `KESTREL_UI_CONFIG.capabilities` opt-outs work unchanged.
// ============================================================================

// --- body templates (moved verbatim from index.html) -----------------------

const IDENTITY_BODY = `
    <div class="identity-card" id="identity-card">
        <div class="loading" data-label-key="loading_identity">Loading identity...</div>
    </div>
    <div id="genesis-audit"></div>
    <div id="identity-danger-zone"></div>
`;

const CONSTITUTION_BODY = `
    <div class="constitution-viewer" id="constitution-viewer">
        <div class="constitution-header">
            <h2 data-label-key="constitution_title">Kestrel Constitution</h2>
            <span class="text-mono text-xs text-tertiary" id="constitution-hash"></span>
        </div>
        <div class="constitution-content" id="constitution-content">
            <div class="loading" data-label-key="loading_constitution">Loading constitution...</div>
        </div>
    </div>
`;

const MEMORIES_BODY = `
    <div class="row-between mb-3">
        <h2 class="m-0" data-label-key="memories_title">Knowledge Graph</h2>
        <select id="memory-filter" class="btn btn-secondary" style="padding: 0.5rem;">
            <option value="" data-label-key="memories_filter_all">All Types</option>
            <option value="agent" data-label-key="memories_filter_agent">Agent</option>
            <option value="document" data-label-key="memories_filter_documents">Documents</option>
            <option value="memory" data-label-key="memories_filter_memories">Memories</option>
            <option value="backup_artifact" data-label-key="memories_filter_backups">Backups</option>
        </select>
    </div>
    <div class="memory-list" id="memory-list">
        <div class="loading" data-label-key="loading_memories">Loading memories...</div>
    </div>
`;

const TASKS_BODY = `
    <div class="row-between mb-3">
        <h2 class="m-0" data-label-key="tasks_title">Tasks & Activity</h2>
        <div class="row-flex">
            <button class="btn btn-secondary" id="btn-refresh-tasks"><span class="ki ki-refresh"></span> <span data-label-key="btn_refresh">Refresh</span></button>
        </div>
    </div>

    <!-- View Toggle -->
    <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
        <button class="btn btn-secondary active" id="btn-view-tasks" style="
            flex: 1;
            padding: 0.5rem;
            font-size: 0.875rem;
        "><span class="ki ki-clipboard"></span> <span data-label-key="tasks_view_tasks">Background Tasks</span></button>
        <button class="btn btn-secondary" id="btn-view-activity" style="
            flex: 1;
            padding: 0.5rem;
            font-size: 0.875rem;
        "><span class="ki ki-chart-bar"></span> <span data-label-key="tasks_view_activity">Activity Log</span></button>
    </div>

    <!-- Tasks Container -->
    <div id="tasks-container">
        <div class="row-between mb-2">
            <p class="text-muted m-0 text-md" data-label-key="tasks_description">
                A2A background tasks (image generation, training, etc.)
            </p>
            <select id="task-filter" style="
                padding: 0.375rem 0.75rem;
                background: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 6px;
                color: var(--text-primary);
                cursor: pointer;
                font-size: 0.875rem;
            ">
                <option value="" data-label-key="filter_all">All</option>
                <option value="working" data-label-key="tasks_filter_working">Working</option>
                <option value="completed" data-label-key="tasks_filter_completed">Completed</option>
                <option value="failed" data-label-key="tasks_filter_failed">Failed</option>
            </select>
        </div>
        <div class="task-list" id="task-list">
            <div class="loading" data-label-key="loading_tasks">Loading tasks...</div>
        </div>
    </div>

    <!-- Activity Log Container (hidden by default) -->
    <div id="activity-container" style="display: none;">
        <p class="text-muted mb-2 text-md" data-label-key="activity_description">
            Real-time log of tool calls, LLM invocations, and feature executions.
        </p>
        <div class="activity-list" id="activity-list">
            <div class="loading" data-label-key="loading_activity">Loading activity...</div>
        </div>
    </div>
`;

const SOVEREIGNTY_BODY = `
    <div class="row-between mb-3">
        <h2 class="m-0" data-label-key="sovereignty_title">Data Sovereignty</h2>
        <div class="row-flex">
            <button class="btn btn-primary" id="btn-export-ipfs" data-label-key="btn_export_ipfs">Export to IPFS</button>
            <button class="btn btn-secondary" id="btn-import" data-label-key="btn_import_cid">Import from CID</button>
        </div>
    </div>
    <p class="text-muted mb-4" data-label-key="sovereignty_tagline">
        Your data is your own. Export to IPFS/Filecoin for true ownership and portability.
    </p>
    <h3 class="mb-3" data-label-key="sovereignty_export_history">Export History</h3>
    <div class="export-list" id="export-list">
        <div class="loading" data-label-key="loading_exports">Loading exports...</div>
    </div>

    <!-- Display (theme + locale picker, #991) -->
    <div class="mt-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0" data-label-key="sovereignty_display">Display</h3>
        </div>
        <p class="text-muted text-md mb-3" data-label-key="sovereignty_display_description">
            Choose a UI theme and language. Stored locally on this device.
        </p>
        <div style="display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;">
            <label class="text-md text-muted row-center">
                <span data-label-key="theme_picker_label">Theme</span>
                <select id="theme-picker-theme" style="
                    padding: 0.375rem 0.75rem;
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    color: var(--text-primary);
                    cursor: pointer;
                    font-size: 0.875rem;
                ">
                    <!-- options injected by theme_picker.js -->
                </select>
            </label>
            <label class="text-md text-muted row-center">
                <span data-label-key="locale_picker_label">Language</span>
                <select id="theme-picker-locale" style="
                    padding: 0.375rem 0.75rem;
                    background: var(--bg-secondary);
                    border: 1px solid var(--border-color);
                    border-radius: 6px;
                    color: var(--text-primary);
                    cursor: pointer;
                    font-size: 0.875rem;
                ">
                    <option value="en" selected>English</option>
                </select>
            </label>
            <span class="text-tertiary text-xs" id="theme-picker-status"></span>
        </div>
    </div>

    <!-- Local File Browser (Session 3) -->
    <div class="mt-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0" data-label-key="sovereignty_local_cache">Local Cache</h3>
            <button class="btn btn-secondary" id="toggle-file-browser" onclick="toggleFileBrowser()"><span class="ki ki-folder"></span> <span data-label-key="btn_browse_local_files">Browse Local Files</span></button>
        </div>
        <p class="text-muted text-md mb-3" data-label-key="sovereignty_local_cache_description">
            Browse cached backup files stored locally on your device.
        </p>
        <div id="file-browser-section" style="display: none;">
            <div id="file-browser-container">
                <div class="loading" data-label-key="loading_files">Loading files...</div>
            </div>
        </div>
    </div>

    <!-- Database Explorer (Session 5) -->
    <div class="mt-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0" data-label-key="sovereignty_db_explorer">Database Explorer</h3>
            <button class="btn btn-secondary" id="toggle-db-explorer"><span class="ki ki-cabinet"></span> <span data-label-key="btn_browse_database">Browse Database</span></button>
        </div>
        <p class="text-muted text-md mb-3" data-label-key="sovereignty_db_description">
            Read-only view of agent database tables and contents.
        </p>
        <div id="db-explorer-section" style="display: none;">
            <div id="db-explorer-container">
                <div class="loading" data-label-key="loading_database">Loading database...</div>
            </div>
        </div>
    </div>

    <!-- IPFS Node Status (Session 5) -->
    <div class="mt-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0" data-label-key="sovereignty_ipfs_network">IPFS Network</h3>
            <button class="btn btn-secondary" id="toggle-ipfs-status" onclick="toggleIpfsStatus()"><span class="ki ki-globe"></span> <span data-label-key="btn_check_status">Check Status</span></button>
        </div>
        <p class="text-muted text-md mb-3" data-label-key="sovereignty_ipfs_description">
            Check connectivity to local IPFS node and public gateways.
        </p>
        <div id="ipfs-status-section" style="display: none;">
            <div id="ipfs-status-container">
                <div class="loading" data-label-key="loading_ipfs">Checking IPFS...</div>
            </div>
        </div>
    </div>
`;

const RESOURCES_BODY = `
    <h2 style="margin: 0 0 0.5rem 0;" data-label-key="resources_title">Agent Resources</h2>
    <p class="text-muted mb-4" data-label-key="resources_description">
        API keys, wallet balances, and usage tracking. Keys are resolved in priority order: Agent → User → Platform.
    </p>

    <!-- Active Key Source Indicator -->
    <div id="active-key-source" style="
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <span class="text-md text-muted" data-label-key="resources_active_key_label">Active Key Source:</span>
        <span id="key-source-badge" style="
            background: var(--accent-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 500;
        ">Loading...</span>
    </div>

    <!-- Agent Keys Section -->
    <div class="mb-5">
        <div class="row-between mb-3">
            <h3 class="m-0"><span class="ki ki-robot"></span> <span data-label-key="resources_agent_keys_title">Agent Keys</span> <span class="heading-subtitle" data-label-key="resources_agent_keys_subtitle">(this companion only)</span></h3>
            <div class="row-flex">
                <button class="btn btn-primary" onclick="showAddAgentKeyModal()" data-label-key="btn_add_key">+ Add Key</button>
                <button class="btn btn-secondary" onclick="refreshAgentKeys()" data-label-key="btn_refresh">Refresh</button>
            </div>
        </div>
        <div id="agent-keys-list">
            <div class="loading" data-label-key="loading_keys">Loading keys...</div>
        </div>
    </div>

    <!-- User BYOK Section -->
    <div class="mb-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0"><span class="ki ki-user"></span> <span data-label-key="resources_user_keys_title">Your Keys</span> <span class="heading-subtitle" data-label-key="resources_user_keys_subtitle">(BYOK - shared across companions)</span></h3>
            <div class="row-flex">
                <button class="btn btn-primary" onclick="showAddUserKeyModal()" data-label-key="btn_add_key">+ Add Key</button>
                <button class="btn btn-secondary" id="unlock-byok-btn" onclick="showUnlockByokModal()" style="display: none;"><span class="ki ki-lock-open"></span> <span data-label-key="btn_unlock">Unlock</span></button>
            </div>
        </div>
        <div class="callout">
            <span class="ki ki-lock"></span> <span data-label-key="resources_byok_description">Keys encrypted with your passphrase. No wallet charges when using your own keys.</span>
        </div>
        <div id="user-keys-list">
            <div class="loading" data-label-key="loading_keys">Loading keys...</div>
        </div>
    </div>

    <!-- Platform Access Section -->
    <div class="mb-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0"><span class="ki ki-building"></span> <span data-label-key="resources_platform_title">Platform Access</span> <span class="heading-subtitle" data-label-key="resources_platform_subtitle">(pay-per-use)</span></h3>
        </div>
        <div class="callout">
            <span class="ki ki-credit-card"></span> <span data-label-key="resources_platform_description">Uses your wallet balance + platform margin. Fallback when no personal keys available.</span>
        </div>
        <div id="platform-access-list">
            <div class="loading" data-label-key="loading_platform">Loading platform access...</div>
        </div>
    </div>

    <!-- Wallet Section -->
    <div class="mb-5 section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0"><span class="ki ki-wallet"></span> <span data-label-key="resources_wallet_title">Wallet</span></h3>
            <button class="btn btn-secondary" onclick="refreshWallet()" data-label-key="btn_refresh">Refresh</button>
        </div>
        <div id="wallet-details">
            <div class="loading" data-label-key="loading_wallet">Loading wallet...</div>
        </div>
    </div>

    <!-- Usage Section -->
    <div class="section-divider">
        <h3 style="margin: 0 0 1rem 0;"><span class="ki ki-chart-bar"></span> <span data-label-key="resources_usage_title">Usage (Last 30 Days)</span></h3>
        <div id="usage-details">
            <div class="loading" data-label-key="loading_usage">Loading usage...</div>
        </div>
    </div>
`;

const METRICS_BODY = `
    <div class="row-between mb-4">
        <h2 class="m-0" data-label-key="metrics_title">Metrics Dashboard</h2>
        <div class="row-center row-gap-lg">
            <label style="font-size: 0.8rem; color: var(--text-secondary);">
                Auto-refresh
                <select id="metrics-refresh-interval" style="
                    margin-left: 0.25rem;
                    padding: 0.2rem 0.4rem;
                    background: var(--bg-tertiary);
                    color: var(--text-primary);
                    border: 1px solid var(--border-color);
                    border-radius: 4px;
                    font-size: 0.8rem;
                ">
                    <option value="0">Off</option>
                    <option value="10">10s</option>
                    <option value="30" selected>30s</option>
                    <option value="60">60s</option>
                </select>
            </label>
            <button id="btn-refresh-metrics" style="
                padding: 0.375rem 0.75rem;
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                color: var(--text-primary);
                cursor: pointer;
                font-size: 0.85rem;
            "><span class="ki ki-refresh"></span> Refresh</button>
        </div>
    </div>

    <!-- KPI Cards -->
    <div id="metrics-kpi-cards" style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    ">
        <div class="loading" data-label-key="loading_metrics">Loading metrics...</div>
    </div>

    <!-- Charts Row -->
    <div style="
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <!-- Event Timeline -->
        <div style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        ">
            <h3 class="card-heading" data-label-key="metrics_event_timeline">Event Timeline</h3>
            <div class="chart-canvas-wrap">
                <canvas id="metrics-timeline-chart"></canvas>
            </div>
        </div>

        <!-- Tool Duration -->
        <div style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        ">
            <h3 class="card-heading" data-label-key="metrics_tool_duration">Tool Duration (avg ms)</h3>
            <div class="chart-canvas-wrap">
                <canvas id="metrics-duration-chart"></canvas>
            </div>
        </div>
    </div>

    <!-- Second Charts Row -->
    <div style="
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    ">
        <!-- Event Distribution -->
        <div style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        ">
            <h3 class="card-heading" data-label-key="metrics_event_distribution">Event Distribution</h3>
            <div class="chart-canvas-wrap">
                <canvas id="metrics-distribution-chart"></canvas>
            </div>
        </div>

        <!-- Recent Errors -->
        <div style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        ">
            <h3 class="card-heading" data-label-key="metrics_recent_errors">Recent Errors</h3>
            <div class="scroll-y-md" id="metrics-errors-list">
                <p class="text-muted text-sm" data-label-key="metrics_no_errors">No errors recorded</p>
            </div>
        </div>
    </div>
`;

const FEATURES_BODY = `
    <div class="row-between mb-3">
        <h2 class="m-0" data-label-key="features_title">Feature Store</h2>
        <button onclick="FeatureStore.reload()" class="btn btn-secondary" style="
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
        "><span class="ki ki-refresh"></span> Refresh</button>
    </div>
    <p class="text-muted mb-3 text-md" data-label-key="features_description">
        Browse, install, and manage feature packages for your agent.
    </p>

    <!-- Search and Filter Bar -->
    <div style="display: flex; gap: 0.75rem; margin-bottom: 1.25rem; flex-wrap: wrap; align-items: center;">
        <input type="text" id="feature-search" data-label-attr-placeholder="features_search_placeholder" placeholder="Search features, tags, skills..."
            style="
                flex: 1;
                min-width: 200px;
                padding: 0.5rem 0.75rem;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                background: var(--bg-primary);
                color: var(--text-primary);
                font-size: 0.875rem;
                outline: none;
            ">
        <div id="feature-store-filters" style="display: flex; gap: 0.25rem;">
            <button class="feature-filter-btn active" data-filter="all" style="
                padding: 0.375rem 0.75rem;
                border: 1px solid var(--border-color);
                border-radius: 6px;
                background: var(--bg-tertiary);
                color: var(--text-primary);
                font-size: 0.8rem;
                cursor: pointer;
            " data-label-key="features_filter_all">All</button>
            <button class="feature-filter-btn" data-filter="installed" style="
                padding: 0.375rem 0.75rem;
                border: 1px solid var(--border-color);
                border-radius: 6px;
                background: var(--bg-primary);
                color: var(--text-secondary);
                font-size: 0.8rem;
                cursor: pointer;
            " data-label-key="features_filter_installed">Installed</button>
            <button class="feature-filter-btn" data-filter="available" style="
                padding: 0.375rem 0.75rem;
                border: 1px solid var(--border-color);
                border-radius: 6px;
                background: var(--bg-primary);
                color: var(--text-secondary);
                font-size: 0.8rem;
                cursor: pointer;
            " data-label-key="features_filter_available">Available</button>
        </div>
    </div>

    <!-- Feature Card Grid -->
    <div id="feature-grid" style="
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1rem;
    ">
        <div class="loading" data-label-key="loading_features">Loading features...</div>
    </div>
`;

const APPROVALS_BODY = `
    <div class="row-between mb-3">
        <h2 style="margin: 0;" data-label-key="approvals_title">Approval Queue</h2>
        <button id="btn-refresh-approvals" class="btn btn-secondary text-sm">Refresh</button>
    </div>
    <p class="text-muted mb-4" data-label-key="approvals_description">Pending agent actions awaiting your decision. Review the full command, then Approve, Reject, or Approve-and-remember (adds a scoped, revocable auto-approve rule).</p>
    <div id="approvals-list">
        <div class="loading">Loading…</div>
    </div>

    <h3 style="margin: 2rem 0 0.75rem 0;">Remembered auto-approve rules</h3>
    <p class="text-muted mb-3 text-sm">
        Sovereign-curated. Each is scoped to an agent + repo and revocable here.
    </p>
    <div id="approvals-rules">
        <div class="loading">Loading…</div>
    </div>

    <h3 style="margin: 2rem 0 0.75rem 0;">Recent auto-approved invocations</h3>
    <p class="text-muted mb-3 text-sm">
        The immutable "no silent runs" record: command, agent DID, time, exit code.
    </p>
    <div id="approvals-audit">
        <div class="loading">Loading…</div>
    </div>
`;

const SECURITY_BODY = `
    <h2 style="margin: 0 0 0.5rem 0;" data-label-key="security_title">Security Permissions</h2>
    <p class="text-muted mb-4" data-label-key="security_description">
        Control which tools can run automatically, require approval, or are blocked.
    </p>

    <!-- Pending Approvals Section -->
    <div class="mb-5">
        <h3 style="margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
            <span class="ki ki-hourglass"></span> <span data-label-key="security_pending_approvals">Pending Approvals</span>
            <span id="pending-count-label" style="
                background: var(--bg-tertiary);
                padding: 0.125rem 0.5rem;
                border-radius: 10px;
                font-size: 0.75rem;
                font-weight: normal;
            ">0</span>
        </h3>
        <div id="pending-approvals">
            <p class="empty-state" style="color: var(--text-secondary); padding: 0.5rem 0;" data-label-key="security_no_pending">
                No pending approvals
            </p>
        </div>
    </div>

    <!-- Permission Tree Section -->
    <div class="mb-5">
        <div class="row-between mb-3">
            <h3 class="m-0"><span class="ki ki-lock-key"></span> <span data-label-key="security_permission_tree">Permission Tree</span></h3>
            <button onclick="Security.loadPermissionTree()" class="btn btn-secondary" style="
                padding: 0.25rem 0.75rem;
                font-size: 0.8rem;
            "><span class="ki ki-refresh"></span> Refresh</button>
        </div>
        <div id="permission-tree">
            <p class="empty-state text-muted" data-label-key="security_load_tree_prompt">
                Click to load permission tree...
            </p>
        </div>
        <p style="color: var(--text-tertiary); font-size: 0.75rem; margin-top: 0.75rem;">
            <span data-label-key="security_legend_label">Legend:</span> <span class="ki ki-check-box"></span> <span data-label-key="security_legend_allow">Allow</span> | <span class="ki ki-shield"></span> <span data-label-key="security_legend_auto">Auto</span> | <span class="ki ki-empty-box"></span> <span data-label-key="security_legend_ask">Ask</span> | <span class="ki ki-x-box"></span> <span data-label-key="security_legend_deny">Deny</span> | <span class="ki ki-half-circle"></span> <span data-label-key="security_legend_mixed">Mixed</span>
        </p>
        <p style="color: var(--warning); font-size: 0.75rem; margin-top: 0.5rem;" data-label-key="security_auto_warning">
            Auto skips human approval only when constitutional, honesty, and security checks do not flag the call.
        </p>
    </div>

    <!-- Session Controls Section -->
    <div class="mb-5 section-divider">
        <h3 style="margin: 0 0 1rem 0;"><span class="ki ki-gear"></span> <span data-label-key="security_session_controls">Session Controls</span></h3>
        <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
            <button id="security-auto-mode-btn" onclick="Security.toggleGlobalAutoMode()" class="btn btn-secondary text-sm security-auto-mode-toggle" data-label-key="btn_auto_mode">
                Auto Mode: Off
            </button>
            <button onclick="Security.resetSession()" class="btn btn-secondary text-sm" data-label-key="btn_reset_session">
                Reset Session Permissions
            </button>
            <button onclick="Security.cancelAllPending()" class="btn btn-danger text-sm" data-label-key="btn_cancel_pending">
                Cancel All Pending
            </button>
        </div>
        <p style="color: var(--text-tertiary); font-size: 0.75rem; margin-top: 0.5rem;" data-label-key="security_session_note">
            Session permissions are cleared when the browser is closed.
        </p>
    </div>

    <!-- Audit Log Section -->
    <div class="section-divider">
        <div class="row-between mb-3">
            <h3 class="m-0"><span class="ki ki-clipboard"></span> <span data-label-key="security_audit_log">Audit Log</span></h3>
            <button onclick="Security.loadAuditLog()" class="btn btn-secondary" style="
                padding: 0.25rem 0.75rem;
                font-size: 0.8rem;
            "><span class="ki ki-refresh"></span> Refresh</button>
        </div>
        <div id="security-audit-log">
            <p class="empty-state text-muted" data-label-key="security_load_audit_prompt">
                Click to load audit log...
            </p>
        </div>
    </div>
`;

// --- panel descriptors ------------------------------------------------------
//
// `order` is the nav position among core panels (mirrors index.html, chat
// excluded — chat has its own `mount()` and is not a registry panel). `gate`
// receives the live API so an embedder's capability opt-outs apply unchanged.
//
// `before` names the NEXT core panel's `panelId` so `_syncNav` re-inserts a
// rebuilt tab at the correct anchor. Without it, a core capability that toggles
// off→on at runtime (#2041 — a feature enabled/disabled without reload) rebuilds
// its tab via `insertBefore(built, null)`, appending it to the end of the nav
// strip and silently reordering the standalone console. The last entry
// (`approvals`) omits `before` — it belongs at the end.

export const CORE_PANEL_DEFS = [
    {
        panelId: 'identity',
        label: 'Identity',
        labelKey: 'tab_identity',
        gate: (api) => api.hasCapability('identity'),
        before: 'constitution',
        bodyHtml: IDENTITY_BODY,
    },
    {
        panelId: 'constitution',
        label: 'Constitution',
        labelKey: 'tab_constitution',
        gate: (api) => api.hasCapability('constitution'),
        before: 'memories',
        bodyHtml: CONSTITUTION_BODY,
    },
    {
        panelId: 'memories',
        label: 'Memories',
        labelKey: 'tab_memories',
        gate: (api) => api.hasCapability('memory'),
        before: 'tasks',
        bodyHtml: MEMORIES_BODY,
    },
    {
        panelId: 'tasks',
        label: 'Tasks',
        labelKey: 'tab_tasks',
        gate: (api) => api.hasCapability('tasks'),
        before: 'sovereignty',
        bodyHtml: TASKS_BODY,
    },
    {
        panelId: 'sovereignty',
        label: 'Sovereignty',
        labelKey: 'tab_sovereignty',
        gate: (api) => api.hasCapability('sovereignty'),
        before: 'resources',
        bodyHtml: SOVEREIGNTY_BODY,
    },
    {
        panelId: 'resources',
        label: 'Resources',
        labelKey: 'tab_resources',
        // Composite gate: shown when ANY sub-capability is on (keys OR wallet),
        // matching the retired PANEL_CAPABILITIES `resources: ['keys','wallet']`.
        gate: (api) => api.hasCapability('keys') || api.hasCapability('wallet'),
        before: 'metrics',
        bodyHtml: RESOURCES_BODY,
    },
    {
        panelId: 'metrics',
        label: 'Metrics',
        labelKey: 'tab_metrics',
        gate: (api) => api.hasCapability('metrics'),
        before: 'features',
        bodyHtml: METRICS_BODY,
    },
    {
        panelId: 'features',
        label: 'Features',
        labelKey: 'tab_features',
        gate: (api) => api.hasCapability('featureStore'),
        before: 'security',
        bodyHtml: FEATURES_BODY,
    },
    {
        panelId: 'security',
        label: 'Security',
        labelKey: 'tab_security',
        icon: 'ki ki-lock',
        // Shown when either the audit log OR permission tree is available,
        // matching the retired PANEL_CAPABILITIES `security: ['audit','permissions']`.
        gate: (api) => api.hasCapability('audit') || api.hasCapability('permissions'),
        before: 'approvals',
        bodyHtml: SECURITY_BODY,
    },
    {
        panelId: 'approvals',
        label: 'Approvals',
        labelKey: 'tab_approvals',
        icon: 'ki ki-check',
        gate: (api) => api.hasCapability('permissions'),
        // No `before` — approvals is the last core tab.
        bodyHtml: APPROVALS_BODY,
    },
];

/**
 * Fill a registry-created (empty) panel body with this panel's markup. No-op
 * when the body already has content — the standalone `index.html` case, where
 * the registry adopts the in-place `#panel-<id>` body and this must NOT clobber
 * it. Returns true when it injected markup (the embed case), false otherwise.
 *
 * @param {HTMLElement} bodyEl  - the `.panel-content` wrapper (or the panel root)
 * @param {object} def          - a CORE_PANEL_DEFS entry
 */
export function buildCorePanelBody(bodyEl, def) {
    if (!bodyEl || !def || !def.bodyHtml) return false;
    if (bodyEl.children && bodyEl.children.length > 0) return false;
    bodyEl.innerHTML = def.bodyHtml;
    // Re-translate the freshly-injected DOM into the active locale; inline text
    // is only the English fallback. No-op when the theme layer isn't present.
    try {
        const theme = (typeof window !== 'undefined') && window.KestrelTheme;
        if (theme && typeof theme._hydrate === 'function') {
            theme._hydrate(theme.getCurrentLabels());
        }
    } catch (_) { /* labels stay at their inline fallback */ }
    return true;
}

export default CORE_PANEL_DEFS;
