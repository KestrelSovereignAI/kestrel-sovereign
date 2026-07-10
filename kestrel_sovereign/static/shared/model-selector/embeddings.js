/**
 * Embeddings section of the unified Model-settings popover (#2264).
 *
 * Reads/writes the embedding-settings API (#2263):
 *   GET  /api/embedding/settings  → {embedding_route, resolved_route,
 *                                    embedding_model, embedding_dim,
 *                                    kestrel_embedding_dim}
 *   POST /api/embedding/settings  ← {embedding_route: "<vendor>:<route>"|"none"|null}
 *
 * Default state is "Auto — follow chat provider" (embedding_route == null).
 * The operator can expand to an explicit provider+route, chosen from the
 * embedding-CAPABLE routes only (openai:api, ollama:local, vertex/google…),
 * or turn embeddings off deliberately with "Off — keyword search only"
 * (embedding_route == "none", #2287). A dimension readout is always shown; a
 * warning appears when the resolved ``embedding_dim`` differs from the
 * deployment's ``kestrel_embedding_dim`` (the size the stored vectors were
 * written at) — because switching is NOT free: existing memories fall back to
 * keyword search until re-embedded.
 */

const DIM_MISMATCH_MESSAGE =
    'existing memories will use keyword search until re-embedded';

// #2287 — first-class "embeddings off" sentinel. Distinct from ``null``
// (auto/follow-chat): "none" is a deliberate operator choice, not a default.
const EMBEDDING_OFF = 'none';

/** Map an ``embedding_route`` value onto a UI mode. */
function embeddingModeForRoute(route) {
    if (route === EMBEDDING_OFF) return 'off';
    if (route) return 'explicit';
    return 'auto';
}

class EmbeddingSelector {
    /**
     * @param {Object} options
     * @param {string} [options.settingsEndpoint='/api/embedding/settings']
     * @param {string} options.modeSelectId   Auto / explicit toggle <select>.
     * @param {string} options.routeSelectId  Explicit provider:route <select>.
     * @param {string} [options.dimReadoutId] Element for the dimension readout.
     * @param {string} [options.warningId]    Element for the mismatch warning.
     * @param {string} [options.reindexButtonId] Button that re-embeds stale
     *        memories (#2336). Shown only when ``stale_rows > 0``; disabled
     *        when embeddings are off/unresolved (nothing to re-embed to).
     * @param {string} [options.reindexStatusId] Element for reindex progress text.
     * @param {string} [options.reindexEndpoint='/api/embedding/reindex']
     * @param {Function} [options.confirm] Confirmation prompt (msg) => bool.
     * @param {Function} [options.getEmbeddingRoutes] Returns the list of
     *        embedding-capable routes: ``[{vendor, route}, …]``. Typically
     *        derived from ``/api/models`` ``routes`` filtered by
     *        ``supports_embeddings``.
     * @param {Function} [options.getAuthHeader] Returns an auth-header object.
     * @param {Function} [options.onChange] Called after a successful write with
     *        the fresh settings object.
     */
    constructor(options = {}) {
        this.settingsEndpoint = options.settingsEndpoint || '/api/embedding/settings';
        this.modeSelect = options.modeSelectId ? document.getElementById(options.modeSelectId) : null;
        this.routeSelect = options.routeSelectId ? document.getElementById(options.routeSelectId) : null;
        this.dimReadout = options.dimReadoutId ? document.getElementById(options.dimReadoutId) : null;
        this.warningEl = options.warningId ? document.getElementById(options.warningId) : null;
        // #2290 — element that renders a shared local/cloud embedding space as
        // ONE entry ("qwen3-embedding-0.6b — local + cloud") instead of two
        // routes. Optional; absent in older popovers.
        this.sharedSpaceEl = options.sharedSpaceId ? document.getElementById(options.sharedSpaceId) : null;
        // #2336 — actionable re-embed control + progress readout.
        this.reindexButton = options.reindexButtonId ? document.getElementById(options.reindexButtonId) : null;
        this.reindexStatus = options.reindexStatusId ? document.getElementById(options.reindexStatusId) : null;
        this.reindexEndpoint = options.reindexEndpoint || '/api/embedding/reindex';
        this.confirm = options.confirm || ((msg) => (typeof window !== 'undefined' && window.confirm ? window.confirm(msg) : true));
        this.getEmbeddingRoutes = options.getEmbeddingRoutes || (() => []);
        this.getAuthHeader = options.getAuthHeader || (() => ({}));
        this.onChange = options.onChange || (() => {});
        // True while a reindex job is running — guards the button re-entrancy.
        this._reindexing = false;

        this.settings = null;
        // 'auto' == follow chat provider (embedding_route null); 'explicit' ==
        // a pinned provider:route; 'off' == embeddings deliberately disabled
        // (embedding_route "none", #2287).
        this.mode = 'auto';
    }

    async init() {
        this._bindEvents();
        await this.load();
    }

    _bindEvents() {
        if (this.modeSelect) {
            this.modeSelect.addEventListener('change', () => this._handleModeChange());
        }
        if (this.routeSelect) {
            this.routeSelect.addEventListener('change', () => this._handleRouteChange());
        }
        if (this.reindexButton) {
            this.reindexButton.addEventListener('click', () => this._handleReindexClick());
        }
    }

    /** Fetch the current settings and render. */
    async load() {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const response = await fetch(this.settingsEndpoint, { headers });
            if (!response.ok) return;
            this.settings = await response.json();
        } catch (e) {
            return;
        }
        this.mode = embeddingModeForRoute(this.settings && this.settings.embedding_route);
        this._render();
    }

    _render() {
        this._renderRoutes();
        this._syncModeUI();
        this._renderDimension();
        this._renderSharedSpace();
        this._renderReindex();
    }

    /**
     * Render the "Re-embed N memories" control (#2336). The dimension-mismatch
     * warning already tells the operator their memories fell back to keyword
     * search; this turns that into an action. Shown only when the backend
     * reports ``stale_rows > 0``. Disabled (with an explanatory title) when
     * embeddings are off or no provider resolves — there is nothing to
     * re-embed to. While a job runs, the button is disabled and its label
     * reflects progress via ``reindexStatus``.
     */
    _renderReindex() {
        if (!this.reindexButton) return;
        const s = this.settings || {};
        const stale = s.stale_rows;
        const hasStale = typeof stale === 'number' && stale > 0;
        const resolvable = this.mode !== 'off' && s.embedding_dim != null;

        if (!hasStale) {
            this.reindexButton.style.display = 'none';
            this.reindexButton.disabled = false;
            if (this.reindexStatus && !this._reindexing) {
                this.reindexStatus.textContent = '';
                this.reindexStatus.style.display = 'none';
            }
            return;
        }

        this.reindexButton.style.display = '';
        const noun = stale === 1 ? 'memory' : 'memories';
        this.reindexButton.textContent = `Re-embed ${stale} ${noun}`;
        // Disable while unresolvable (off/no provider) or a job is in flight.
        this.reindexButton.disabled = !resolvable || this._reindexing;
        this.reindexButton.title = resolvable
            ? `Re-embed ${stale} stale ${noun} into the current embedding provider.`
            : 'No embedding provider resolves — nothing to re-embed to.';
    }

    _setReindexStatus(text) {
        if (!this.reindexStatus) return;
        this.reindexStatus.textContent = text || '';
        this.reindexStatus.style.display = text ? '' : 'none';
    }

    /**
     * Dry-run → confirm → execute → progress → refresh (#2336).
     * A dry-run first fetches authoritative counts; on confirm it kicks off
     * the (possibly backgrounded) job and polls until it finishes, then
     * reloads settings so ``stale_rows`` and the warning refresh.
     */
    async _handleReindexClick() {
        if (this._reindexing) return;
        const dry = await this._postReindex({ dry_run: true });
        if (!dry) {
            this._setReindexStatus('Re-embed failed — could not read counts.');
            return;
        }
        const n = dry.total_stale || 0;
        if (!n) {
            await this.load();
            return;
        }
        const noun = n === 1 ? 'memory' : 'memories';
        if (!this.confirm(`Re-embed ${n} ${noun} into the current embedding provider? Stored vectors will be rewritten.`)) {
            return;
        }

        this._reindexing = true;
        this._renderReindex();
        this._setReindexStatus(`Re-embedding ${n} ${noun}…`);
        try {
            const started = await this._postReindex({ dry_run: false });
            if (!started) {
                this._setReindexStatus('Re-embed failed to start.');
                return;
            }
            let job = started;
            if (started.job_id && started.status === 'running') {
                job = await this._pollReindex(started.job_id);
            }
            if (job && job.status === 'error') {
                this._setReindexStatus(`Re-embed failed: ${job.error || 'unknown error'}`);
            } else {
                const done = (job && job.total_reembedded) || 0;
                this._setReindexStatus(`Re-embedded ${done} ${done === 1 ? 'memory' : 'memories'}.`);
            }
        } finally {
            this._reindexing = false;
            // Reload authoritative settings (stale_rows/warning refresh).
            await this.load();
        }
    }

    /** POST the reindex body and return the parsed JSON, or null on failure. */
    async _postReindex(body) {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const response = await fetch(this.reindexEndpoint, {
                method: 'POST',
                headers,
                body: JSON.stringify(body),
            });
            if (!response.ok) return null;
            return await response.json();
        } catch (e) {
            return null;
        }
    }

    /** Poll a background reindex job until it reaches a terminal state. */
    async _pollReindex(jobId, { intervalMs = 1000, maxPolls = 3600 } = {}) {
        const url = `${this.reindexEndpoint}/${encodeURIComponent(jobId)}`;
        for (let i = 0; i < maxPolls; i++) {
            await new Promise(r => setTimeout(r, intervalMs));
            let job;
            try {
                const headers = { ...(await this.getAuthHeader()) };
                const response = await fetch(url, { headers });
                if (!response.ok) return null;
                job = await response.json();
            } catch (e) {
                return null;
            }
            if (job && typeof job.total_reembedded === 'number') {
                this._setReindexStatus(`Re-embedded ${job.total_reembedded}/${job.total_stale}…`);
            }
            if (job && (job.status === 'done' || job.status === 'error')) {
                return job;
            }
        }
        return null;
    }

    /**
     * Render a declared shared local/cloud embedding space (#2290) as a single
     * entry — "qwen3-embedding-0.6b — local + cloud" — with a verified/unverified
     * marker driven by the parity probe. When no pin covers the resolved route,
     * the element is hidden. Never claims a space is shared before the probe
     * passes: an unverified pin reads "parity unverified".
     */
    _renderSharedSpace() {
        if (!this.sharedSpaceEl) return;
        const space = this.settings && this.settings.shared_space;
        if (!space) {
            this.sharedSpaceEl.style.display = 'none';
            this.sharedSpaceEl.textContent = '';
            return;
        }
        const memberCount = (space.members || []).length;
        // "local + cloud" reads better than "2 routes" for the common pairing.
        const scope = memberCount === 2 ? 'local + cloud' : `${memberCount} routes`;
        const status = space.verified ? 'shared' : 'parity unverified';
        this.sharedSpaceEl.style.display = '';
        this.sharedSpaceEl.textContent = `${space.model} — ${scope} (${status})`;
    }

    /** Populate the explicit route <select> from embedding-capable routes. */
    _renderRoutes() {
        if (!this.routeSelect) return;
        const routes = this.getEmbeddingRoutes() || [];
        const configured = this.settings && this.settings.embedding_route;
        this.routeSelect.innerHTML = routes.map(r => {
            const id = `${r.vendor}:${r.route}`;
            const label = r.label || id;
            return `<option value="${id}">${label}</option>`;
        }).join('');
        if (configured && routes.some(r => `${r.vendor}:${r.route}` === configured)) {
            this.routeSelect.value = configured;
        } else if (routes.length > 0) {
            this.routeSelect.value = `${routes[0].vendor}:${routes[0].route}`;
        }
    }

    /** Show/hide the explicit route <select> according to the current mode. */
    _syncModeUI() {
        if (this.modeSelect) this.modeSelect.value = this.mode;
        if (this.routeSelect) {
            this.routeSelect.style.display = this.mode === 'explicit' ? '' : 'none';
        }
    }

    /**
     * Render the dimension readout and, on mismatch, the keyword-fallback
     * warning. Never pretends switching is free (#2264).
     */
    _renderDimension() {
        const s = this.settings || {};
        const dim = s.embedding_dim;
        const deploymentDim = s.kestrel_embedding_dim;
        const model = s.embedding_model;

        if (this.dimReadout) {
            if (dim) {
                const modelPart = model ? `${model} · ` : '';
                this.dimReadout.textContent = `${modelPart}${dim} dimensions`;
            } else if (this.mode === 'off') {
                // Deliberate off (#2287) — a choice, not degradation.
                this.dimReadout.textContent = 'Embeddings off — keyword search only';
            } else {
                // No embedding-capable resolution — keyword search only.
                this.dimReadout.textContent = 'No embedding provider — keyword search only';
            }
        }

        if (this.warningEl) {
            const mismatch = dim != null && deploymentDim != null && dim !== deploymentDim;
            if (mismatch) {
                this.warningEl.style.display = '';
                this.warningEl.textContent =
                    `Embedding dimension ${dim} ≠ stored ${deploymentDim}: ${DIM_MISMATCH_MESSAGE}.`;
            } else {
                this.warningEl.style.display = 'none';
                this.warningEl.textContent = '';
            }
        }
    }

    _handleModeChange() {
        const next = this.modeSelect.value;
        if (next === 'auto') {
            this.mode = 'auto';
            this._syncModeUI();
            // Auto == clear the pin.
            this._commit(null);
        } else if (next === 'off') {
            this.mode = 'off';
            this._syncModeUI();
            // Off == the deliberate "none" sentinel (#2287).
            this._commit(EMBEDDING_OFF);
        } else {
            this.mode = 'explicit';
            this._syncModeUI();
            // Committing the currently-shown route makes the explicit choice real.
            if (this.routeSelect && this.routeSelect.value) {
                this._commit(this.routeSelect.value);
            }
        }
    }

    _handleRouteChange() {
        if (this.mode !== 'explicit') return;
        this._commit(this.routeSelect.value);
    }

    /**
     * POST the new ``embedding_route`` (or null for auto), then re-read the
     * resolved settings so the dimension readout/warning reflect reality.
     * @param {string|null} route
     * @returns {Promise<boolean>} success
     */
    async _commit(route) {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const response = await fetch(this.settingsEndpoint, {
                method: 'POST',
                headers,
                body: JSON.stringify({ embedding_route: route || null }),
            });
            if (!response.ok) return false;
            const data = await response.json();
            // The POST response echoes the resolved settings (success + fields).
            this.settings = {
                embedding_route: data.embedding_route,
                resolved_route: data.resolved_route,
                embedding_model: data.embedding_model,
                embedding_dim: data.embedding_dim,
                kestrel_embedding_dim: data.kestrel_embedding_dim,
                shared_space: data.shared_space,
            };
            this.mode = embeddingModeForRoute(this.settings.embedding_route);
            this._render();
            this.onChange(this.settings);
            return true;
        } catch (e) {
            return false;
        }
    }
}

// Export for ES modules / node --test
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EmbeddingSelector,
        DIM_MISMATCH_MESSAGE,
        EMBEDDING_OFF,
        embeddingModeForRoute,
    };
}

// Export globally for script-tag usage
if (typeof window !== 'undefined') {
    window.EmbeddingSelector = EmbeddingSelector;
}
