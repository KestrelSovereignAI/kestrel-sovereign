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
        this.getEmbeddingRoutes = options.getEmbeddingRoutes || (() => []);
        this.getAuthHeader = options.getAuthHeader || (() => ({}));
        this.onChange = options.onChange || (() => {});

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
