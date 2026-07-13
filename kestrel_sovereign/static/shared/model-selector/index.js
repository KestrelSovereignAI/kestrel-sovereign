/**
 * Shared Two-Dropdown Model Selector Component
 *
 * Groups models by **vendor** (see /api/models response shape:
 * {by_vendor, routes, featured, all, default}). An optional route selector is
 * shown when a vendor has more than one route configured (e.g. anthropic:api
 * vs anthropic:plan). Featured models sort first with a ★ prefix.
 *
 * Historical note: the component and its chat.js consumer still use
 * "provider" as a variable name in many places — that name is the vendor
 * semantically. Renaming everything in one pass is deferred; the API
 * contract is vendor/route/model.
 */

const VENDOR_NAMES = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'ollama': 'Ollama (Local)',
    'openrouter': 'OpenRouter',
    'vertex_ai': 'Google Vertex AI',
    'google': 'Google Gemini',
    'xai': 'xAI',
    'groq': 'Groq',
    'together': 'Together AI',
    'mistral': 'Mistral',
    'deepseek': 'DeepSeek',
    'runpod': 'RunPod',
    'llama_cpp': 'llama.cpp (Local)',
};
// Back-compat alias for any caller reading the old name.
const PROVIDER_NAMES = VENDOR_NAMES;

// Catalogs are descriptive and shared by every selector instance targeting the
// same agent endpoint. Cache the parsed response for the server's five-minute
// discovery TTL and coalesce concurrent loads, while /api/model/current remains
// an uncached server-authoritative sync on every agent switch.
const MODEL_CATALOG_CACHE_TTL_MS = 5 * 60 * 1000;
const _modelCatalogCache = new Map();

async function fetchModelCatalog(endpoint, headers) {
    const now = Date.now();
    const cached = _modelCatalogCache.get(endpoint);
    if (cached?.data && cached.expiresAt > now) return cached.data;
    if (cached?.promise) return cached.promise;

    const promise = (async () => {
        const response = await fetch(endpoint, { headers });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })();
    _modelCatalogCache.set(endpoint, { promise });
    try {
        const data = await promise;
        _modelCatalogCache.set(endpoint, {
            data,
            expiresAt: Date.now() + MODEL_CATALOG_CACHE_TTL_MS,
        });
        return data;
    } catch (error) {
        if (_modelCatalogCache.get(endpoint)?.promise === promise) {
            _modelCatalogCache.delete(endpoint);
        }
        throw error;
    }
}

class ModelSelector {
    /**
     * Create a two-dropdown model selector
     * @param {Object} options - Configuration options
     * @param {string} options.providerSelectId - ID of the provider select element
     * @param {string} options.modelSelectId - ID of the model select element
     * @param {string} [options.apiEndpoint='/api/models'] - API endpoint for models
     * @param {string} [options.currentModelEndpoint='/api/model/current'] - API endpoint for current model
     * @param {string} [options.storagePrefix='kestrel'] - localStorage key prefix
     * @param {Function} [options.onModelChange] - Callback when model changes (provider, model) => void
     * @param {Function} [options.getAuthHeader] - Function returning auth header object
     * @param {Function} [options.isCurrent] - False when this selector was superseded
     * @param {boolean} [options.sendCommandOnChange=true] - Whether to trigger onModelChange on selection
     */
    constructor(options = {}) {
        this.providerSelect = document.getElementById(options.providerSelectId);
        this.modelSelect = document.getElementById(options.modelSelectId);
        // Optional route selector — appears when a vendor has >1 configured route.
        this.routeSelect = options.routeSelectId
            ? document.getElementById(options.routeSelectId)
            : null;
        // Optional meta-provider "Upstream" facet. Appears only when the
        // selected route's catalog carries ``underlying_provider`` values
        // (openrouter, and any future aggregator). It FILTERS the model combo
        // by upstream vendor — a DISPLAY filter only, never an OpenRouter
        // provider-preferences routing pin. See #2264.
        this.upstreamSelect = options.upstreamSelectId
            ? document.getElementById(options.upstreamSelectId)
            : null;
        this.apiEndpoint = options.apiEndpoint || '/api/models';
        // Use 'in' check to allow explicit null (disables server sync)
        this.currentModelEndpoint = 'currentModelEndpoint' in options
            ? options.currentModelEndpoint
            : '/api/model/current';
        this.storagePrefix = options.storagePrefix || 'kestrel';
        this.onModelChange = options.onModelChange || (() => {});
        this.getAuthHeader = options.getAuthHeader || (() => ({}));
        this.isCurrent = options.isCurrent || (() => true);
        this.sendCommandOnChange = options.sendCommandOnChange !== false;
        this._eventHandlers = null;

        this.allModelsData = null;
        this.selectedProvider = '';
        this.selectedModel = '';
        this.selectedRoute = '';
        // Whether the active model is auto-resolved (no pinned mandate) rather
        // than an explicit operator choice. Seeded from ``/api/model/current``'s
        // ``is_auto`` flag and cleared on any explicit pick so the header button
        // can distinguish "Auto — currently <model>" from a chosen model
        // (#2419). Not persisted — it mirrors server truth, re-read on sync.
        this._isAuto = false;
        // Meta-provider upstream filter. Sentinel ``'All'`` = no filter. This is
        // pure display state (never sent to the server) so it is kept in memory
        // and restored from localStorage like the other picks. See #2264.
        this.selectedUpstream = 'All';
        // Route-scoped model cache (#2262/#2264). When the operator picks a
        // route whose model list must come from THAT route's own discovery
        // (e.g. anthropic:plan vs anthropic:api, or an OpenRouter route), the
        // component re-fetches ``/api/models?vendor=&route=`` and stashes the
        // result here keyed by ``"<vendor>::<route>"``. ``_currentModelList``
        // prefers this over the vendor-wide ``by_vendor`` bucket.
        this._routeModels = null;
        this.isInitialLoad = true;
        // The dropdown defaults to the featured set (a clean handful of current
        // models per vendor) with an explicit "Show all" expander. Flips to true
        // when the operator expands; reset per vendor switch. (#2015)
        this.showAllModels = false;

        // Model IDs the user must not be able to pick from the dropdown — e.g.
        // the OpenAI Realtime model, which is owned by the mic button. Rendered
        // as <option disabled> with a 🎙 prefix. See kestrel-sovereign#1371.
        this._unpickableModels = new Set();
        // Pin state. Non-null while a feature holds an exclusive claim that has
        // pinned the selector to a specific model. Stores the prior selection so
        // we can restore on release. Intentionally NOT persisted to localStorage:
        // it is transient UI ownership, not a user choice. See `pinToModel`.
        this._pinnedSelection = null;
        // Re-entrancy guard: `pinToModel` calls `_populateModels`, which calls
        // back into the claim's `onRefresh` — set this while applying a pin so
        // that re-assert can't recurse. See `_populateModels`.
        this._applyingPin = false;

        // Generic single-holder claim/release/refresh negotiation for this
        // shared widget (kestrel-sovereign#2047). A feature (today: voice) that
        // needs to temporarily seize the selector acquires a claim here. The
        // registry is widget-agnostic; the claim's callbacks drive the actual
        // pin via `pinToModel`/`unpinSelection`.
        this.claims = new WidgetClaimRegistry(this);

        this._loadState();
    }

    /**
     * Load saved state from localStorage
     */
    _loadState() {
        this.selectedProvider = localStorage.getItem(`${this.storagePrefix}_selected_provider`) || '';
        this.selectedModel = localStorage.getItem(`${this.storagePrefix}_selected_model`) || '';
        this.selectedRoute = localStorage.getItem(`${this.storagePrefix}_selected_route`) || '';
        this.selectedUpstream = localStorage.getItem(`${this.storagePrefix}_selected_upstream`) || 'All';
    }

    /**
     * Save state to localStorage
     */
    _saveState() {
        if (this.selectedProvider) {
            localStorage.setItem(`${this.storagePrefix}_selected_provider`, this.selectedProvider);
        }
        if (this.selectedModel) {
            localStorage.setItem(`${this.storagePrefix}_selected_model`, this.selectedModel);
        }
        if (this.selectedRoute) {
            localStorage.setItem(`${this.storagePrefix}_selected_route`, this.selectedRoute);
        } else {
            localStorage.removeItem(`${this.storagePrefix}_selected_route`);
        }
        if (this.selectedUpstream && this.selectedUpstream !== 'All') {
            localStorage.setItem(`${this.storagePrefix}_selected_upstream`, this.selectedUpstream);
        } else {
            localStorage.removeItem(`${this.storagePrefix}_selected_upstream`);
        }
    }

    /**
     * Initialize the component - load models and bind events
     */
    async init() {
        const models = await this.loadModels();
        if (!models || !this.isCurrent()) return;
        this._bindEvents();
        await this.syncWithServer();
        if (this.isCurrent()) {
            this.isInitialLoad = false;
        } else {
            this.destroy();
        }
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        if (this._eventHandlers) return;
        this._eventHandlers = {
            provider: () => this._handleProviderChange(),
            model: () => this._handleModelChange(),
            route: () => this._handleRouteChange(),
            upstream: () => this._handleUpstreamChange(),
        };
        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', this._eventHandlers.provider);
        }
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', this._eventHandlers.model);
        }
        if (this.routeSelect) {
            this.routeSelect.addEventListener('change', this._eventHandlers.route);
        }
        if (this.upstreamSelect) {
            this.upstreamSelect.addEventListener('change', this._eventHandlers.upstream);
        }
    }

    /** Remove handlers before a switch replaces this selector instance. */
    destroy() {
        if (!this._eventHandlers) return;
        this.providerSelect?.removeEventListener?.('change', this._eventHandlers.provider);
        this.modelSelect?.removeEventListener?.('change', this._eventHandlers.model);
        this.routeSelect?.removeEventListener?.('change', this._eventHandlers.route);
        this.upstreamSelect?.removeEventListener?.('change', this._eventHandlers.upstream);
        this._eventHandlers = null;
    }

    /**
     * Load models from API
     */
    async loadModels() {
        if (!this.providerSelect || !this.modelSelect) {
            console.warn('ModelSelector: Provider or model select element not found');
            return null;
        }

        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader())
            };
            if (!this.isCurrent()) return null;

            const data = await fetchModelCatalog(this.apiEndpoint, headers);
            if (!this.isCurrent()) return null;
            this.allModelsData = data;
            this._populateProviders();

            return this.allModelsData;
        } catch (e) {
            if (!this.isCurrent()) return null;
            console.error('ModelSelector: Failed to load models:', e);
            this.providerSelect.innerHTML = '<option value="">Error loading</option>';
            return null;
        }
    }

    /** Evict this agent's descriptive catalog after a settings mutation. */
    invalidateCatalog() {
        _modelCatalogCache.delete(this.apiEndpoint);
    }

    /**
     * Resolve the server's seed selection (vendor / route / model) from the
     * /api/models response. Prefers the canonical ``default`` model id; falls
     * back to the first priority-ordered route entry. Returns null when the
     * payload lacks the data needed to seed.
     */
    _serverDefaultSelection() {
        const data = this.allModelsData;
        if (!data) return null;
        const buckets = data.by_vendor || {};
        const routes = data.routes || [];

        const defaultId = data.default;
        if (defaultId) {
            for (const [vendor, models] of Object.entries(buckets)) {
                if (models.some(m => m.id === defaultId)) {
                    const r = routes.find(x => x.vendor === vendor && x.model === defaultId)
                        || routes.find(x => x.vendor === vendor);
                    return { vendor, model: defaultId, route: r?.route || null };
                }
            }
        }
        // Priority-ordered routes are how the server signals fallback intent.
        for (const r of routes) {
            if (r.vendor && buckets[r.vendor]) {
                const model = (r.model && buckets[r.vendor].some(m => m.id === r.model))
                    ? r.model
                    : null;
                return { vendor: r.vendor, model, route: r.route || null };
            }
        }
        return null;
    }

    /**
     * Populate vendor dropdown
     */
    _populateProviders() {
        const buckets = this.allModelsData?.by_vendor;
        if (!buckets) return;

        const vendors = Object.keys(buckets).sort((a, b) => {
            const nameA = VENDOR_NAMES[a] || a;
            const nameB = VENDOR_NAMES[b] || b;
            return nameA.localeCompare(nameB);
        });

        this.providerSelect.innerHTML = vendors.map(v => {
            const displayName = VENDOR_NAMES[v] || v.charAt(0).toUpperCase() + v.slice(1);
            const count = buckets[v]?.length || 0;
            return `<option value="${v}">${displayName} (${count})</option>`;
        }).join('');

        // Seed order: localStorage > server default > alphabetical.
        // Alphabetical alone is wrong — it favors "anthropic" or "gpt-3.5-turbo"
        // simply because they sort first, ignoring the server's actual routing.
        if (this.selectedProvider && vendors.includes(this.selectedProvider)) {
            this.providerSelect.value = this.selectedProvider;
        } else {
            const seed = this._serverDefaultSelection();
            if (seed && vendors.includes(seed.vendor)) {
                this.providerSelect.value = seed.vendor;
                this.selectedProvider = seed.vendor;
                if (!this.selectedRoute && seed.route) this.selectedRoute = seed.route;
                if (!this.selectedModel && seed.model) this.selectedModel = seed.model;
            } else if (vendors.length > 0) {
                this.providerSelect.value = vendors[0];
                this.selectedProvider = vendors[0];
            }
        }

        this._populateRoutes();
        this._populateModels();
    }

    /**
     * Populate the route selector for the currently-selected vendor.
     *
     * Hidden when the vendor has <=1 configured route (99% case). Visible
     * with a real selector when a vendor has multiple routes — e.g.
     * anthropic:api (metered API key) vs anthropic:plan (Claude Max OAuth).
     * Discovery-driven: reads the `routes` array from /api/models.
     */
    _populateRoutes() {
        if (!this.routeSelect) return;
        const vendor = this.providerSelect?.value;
        const routes = (this.allModelsData?.routes || [])
            .filter(r => r.vendor === vendor)
            .map(r => r.route)
            .filter(Boolean);

        if (routes.length <= 1) {
            this.routeSelect.style.display = 'none';
            this.routeSelect.innerHTML = '';
            this.selectedRoute = '';
            return;
        }

        this.routeSelect.style.display = '';
        this.routeSelect.innerHTML = routes.map(r => {
            const label = r.charAt(0).toUpperCase() + r.slice(1);
            return `<option value="${r}">${label}</option>`;
        }).join('');

        if (this.selectedRoute && routes.includes(this.selectedRoute)) {
            this.routeSelect.value = this.selectedRoute;
        } else {
            // Default to the first route for this vendor (priority-ordered by the server).
            this.routeSelect.value = routes[0];
            this.selectedRoute = routes[0];
        }
    }

    /**
     * Resolve the model list backing the model combo for the current
     * (vendor, route). Prefers a route-scoped catalog fetched from
     * ``/api/models?vendor=&route=`` (#2262 route-scoped discovery) so a plan
     * route never offers api-only models; falls back to the vendor-wide
     * ``by_vendor`` bucket when no route-scoped catalog has been fetched.
     */
    _currentModelList() {
        const vendor = this.providerSelect?.value;
        if (!vendor) return [];
        const route = this.selectedRoute || '';
        if (this._routeModels && this._routeModels.key === `${vendor}::${route}`) {
            return this._routeModels.models || [];
        }
        return this.allModelsData?.by_vendor?.[vendor] || [];
    }

    /**
     * Re-fetch the model catalog scoped to the current (vendor, route) so the
     * model combo reflects THAT route's own discovery (#2262). Best-effort:
     * on any failure the combo keeps its vendor-wide list. Awaitable so callers
     * (and tests) can sequence a repopulation; also fired from the route/vendor
     * change handlers.
     */
    async _refreshRouteScopedModels() {
        const vendor = this.providerSelect?.value;
        if (!vendor || !this.apiEndpoint) return;
        const route = this.selectedRoute || '';
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const sep = this.apiEndpoint.includes('?') ? '&' : '?';
            let url = `${this.apiEndpoint}${sep}vendor=${encodeURIComponent(vendor)}`;
            if (route) url += `&route=${encodeURIComponent(route)}`;
            const response = await fetch(url, { headers });
            if (!response.ok) return;
            const data = await response.json();
            const models = (data.by_vendor && data.by_vendor[vendor]) || data.all || [];
            this._routeModels = { key: `${vendor}::${route}`, models };
            this._populateModels();
            // The repopulate may have COERCED the selection (previous model
            // absent from this route's catalog → first valid route model).
            // The change handlers committed the PRE-fetch selection, so
            // without a re-commit here the UI shows the coerced model while
            // the server keeps the stale one (codex P2 on #2275).
            // _maybeCommit diffs against _lastSyncedSelection, so this is a
            // no-op whenever the selection survived the route switch.
            this._maybeCommit();
        } catch (e) {
            // Keep the vendor-wide list on failure — never blank the combo.
        }
    }

    /**
     * Populate the meta-provider "Upstream" facet from the current model
     * list's distinct ``underlying_provider`` values. Hidden (and the filter
     * reset to ``'All'``) when the catalog carries no upstream values. This is
     * a DISPLAY filter over the model combo only — it is never persisted to
     * the server and must not become a routing pin. See #2264.
     */
    _populateUpstream(models) {
        if (!this.upstreamSelect) return;
        const upstreams = [...new Set(
            (models || []).map(m => m.underlying_provider).filter(Boolean)
        )].sort((a, b) => a.localeCompare(b));

        if (upstreams.length === 0) {
            this.upstreamSelect.style.display = 'none';
            this.upstreamSelect.innerHTML = '';
            this.selectedUpstream = 'All';
            return;
        }

        this.upstreamSelect.style.display = '';
        const values = ['All', ...upstreams];
        this.upstreamSelect.innerHTML = values.map(v => {
            const label = v === 'All' ? 'All upstreams' : v;
            return `<option value="${v}">${label}</option>`;
        }).join('');

        if (!values.includes(this.selectedUpstream)) this.selectedUpstream = 'All';
        this.upstreamSelect.value = this.selectedUpstream;
    }

    /** Apply the current upstream display filter to a model list. */
    _filterByUpstream(models) {
        if (!this.selectedUpstream || this.selectedUpstream === 'All') return models;
        return models.filter(m => m.underlying_provider === this.selectedUpstream);
    }

    /**
     * Populate model dropdown based on selected vendor
     */
    _populateModels() {
        const vendor = this.providerSelect?.value;
        const buckets = this.allModelsData?.by_vendor;
        if (!vendor || !buckets) return;

        const fullList = this._currentModelList();
        // Populate the upstream facet from the UNFILTERED list so switching
        // away from a specific upstream can always get back to "All".
        this._populateUpstream(fullList);
        const all = [...this._filterByUpstream(fullList)];

        if (all.length === 0) {
            this.modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }

        // Featured-first, but PRESERVE server order within each group. The
        // server ranks each vendor bucket by recency (newest ``created_at``
        // first), so the first featured entry is the best default — re-sorting
        // alphabetically here is exactly what used to float ``gpt-3.5-turbo`` to
        // the top of OpenAI. Stable sort on the featured flag only. (#2015)
        const ordered = [...all].sort((a, b) =>
            (a.is_featured === b.is_featured) ? 0 : (b.is_featured ? 1 : -1));

        const featured = ordered.filter(m => m.is_featured);
        const hasFeatured = featured.length > 0;
        // Default to the featured set; if a vendor has none, fall back to all so
        // the dropdown is never empty.
        const showAll = this.showAllModels || !hasFeatured;
        let visible = showAll ? ordered : featured.slice();

        // Keep the current selection visible even when it's not featured (e.g. a
        // deprecated/older model the operator deliberately picked).
        if (this.selectedModel && !visible.some(m => m.id === this.selectedModel)) {
            const sel = ordered.find(m => m.id === this.selectedModel);
            if (sel) visible = [...visible, sel];
        }

        // Build model options.  Unpickable models render as <option disabled>
        // with a 🎙 prefix so the operator can see they exist but cannot
        // select them by hand (the mic button drives them).  Selecting the
        // option programmatically still works — see pinToModel().
        const optionsHtml = visible.map(m => {
            const isUnpickable = this._unpickableModels.has(m.id);
            const star = m.is_featured ? '★ ' : '';
            const glyph = isUnpickable ? '🎙 ' : '';
            const displayName = m.display_name || m.id;
            const disabled = isUnpickable ? ' disabled' : '';
            return `<option value="${m.id}"${disabled}>${star}${glyph}${displayName}</option>`;
        }).join('');

        // Expander / collapser. These are sentinel <option>s intercepted in
        // _handleModelChange — picking one re-renders without committing a model.
        let toggleHtml = '';
        if (!showAll && all.length > featured.length) {
            toggleHtml = `<option value="__show_all__">⋯ Show all ${all.length} models…</option>`;
        } else if (this.showAllModels && hasFeatured) {
            toggleHtml = `<option value="__show_featured__">▴ Show featured only</option>`;
        }
        this.modelSelect.innerHTML = optionsHtml + toggleHtml;

        // Seed order: saved model > server default (when visible) > first
        // visible. The first visible entry is the top-ranked featured model, so
        // this is the sane default — never an alphabetical accident. (#2015)
        if (this.selectedModel && visible.some(m => m.id === this.selectedModel)) {
            this.modelSelect.value = this.selectedModel;
        } else if (visible.length > 0) {
            const defaultId = this.allModelsData?.default;
            if (defaultId && visible.some(m => m.id === defaultId)) {
                this.modelSelect.value = defaultId;
                this.selectedModel = defaultId;
            } else {
                this.modelSelect.value = visible[0].id;
                this.selectedModel = visible[0].id;
            }
        }

        // If a feature holds an exclusive claim, the option rebuild above may
        // have wiped its pinned/injected option — let it re-assert. Guarded by
        // `_applyingPin` so the re-assert (which itself rebuilds options via
        // pinToModel → _populateModels) cannot recurse. (#2047)
        if (!this._applyingPin && this.claims && this.claims.isHeld()) {
            this.claims.refresh();
        }
    }

    /**
     * Commit gate: fire ``onModelChange`` when and only when the dropdowns
     * have settled on a complete ``(vendor, route?, model)`` triple that
     * differs from the last state the server confirmed.
     *
     * This replaces the "vendor change never commits" rule that missed the
     * case where a vendor has a single route + single model (llama_cpp/Kimi,
     * ollama/llama3.2). In that case the vendor pick IS unambiguous and
     * should commit. In the ambiguous case (vendor has many models, auto-
     * selection picks the first), a subsequent model click will commit the
     * real choice — the intermediate commit is cheap and correct on reload.
     */
    _maybeCommit() {
        if (this.isInitialLoad) return;
        const vendor = this.selectedProvider;
        const model = this.selectedModel;
        if (!vendor || !model) return;

        // Route normalization: when a vendor has exactly one route, a null
        // selectedRoute and that route's literal name are equivalent — the
        // server will pick the only route either way. Without this, diffing
        // a synced {route: "plan"} against a freshly-populated {route: null}
        // would fire a redundant commit for every same-vendor repick.
        const routeForVendor = (v) => (this.allModelsData?.routes || [])
            .filter(r => r.vendor === v)
            .map(r => r.route);
        const canon = (r, v) => {
            if (r) return r;
            const rs = routeForVendor(v);
            return rs.length === 1 ? rs[0] : null;
        };

        const route = canon(this.selectedRoute || null, vendor);
        const last = this._lastSyncedSelection || {};
        const lastRoute = canon(last.route || null, last.vendor);
        if (last.vendor === vendor && last.model === model && lastRoute === route) {
            return;  // state matches server — no POST
        }
        this._lastSyncedSelection = { vendor, model, route };
        // An explicit operator pick pins the model — no longer auto (#2419).
        this._isAuto = false;
        this.invalidateCatalog();
        this.onModelChange(vendor, model, this.isInitialLoad, route);
    }

    _handleProviderChange() {
        if (!this.isCurrent()) return;
        this.selectedProvider = this.providerSelect.value;
        this.selectedRoute = '';
        // Each vendor switch starts collapsed to the featured set, drops any
        // stale route-scoped catalog, and clears the upstream display filter.
        this.showAllModels = false;
        this._routeModels = null;
        this.selectedUpstream = 'All';
        this._saveState();
        this._populateRoutes();
        this._populateModels();
        this.selectedModel = this.modelSelect.value;
        this._saveState();
        this._maybeCommit();
        // Refine the model combo from the newly-selected route's own discovery.
        this._refreshRouteScopedModels();
    }

    _handleModelChange() {
        if (!this.isCurrent()) return;
        const picked = this.modelSelect.value;
        // Sentinel options toggle the featured/all view rather than selecting a
        // model. Re-render and stop — no commit, no state change. (#2015)
        if (picked === '__show_all__' || picked === '__show_featured__') {
            this.showAllModels = (picked === '__show_all__');
            this._populateModels();
            this.modelSelect.value = this.selectedModel || '';
            return;
        }
        this.selectedModel = picked;
        this._saveState();
        this._maybeCommit();
    }

    _handleRouteChange() {
        if (!this.isCurrent()) return;
        this.selectedRoute = this.routeSelect.value;
        // A route change repopulates the model combo from THAT route's
        // discovery (#2262). Clear the previous route's cache + upstream filter,
        // repopulate immediately from what we have, then refine via re-fetch.
        this._routeModels = null;
        this.selectedUpstream = 'All';
        this.showAllModels = false;
        this._saveState();
        // ``_populateModels`` reseeds ``selectedModel`` when the current pick is
        // absent from the new list and leaves it untouched otherwise (including
        // when there is no catalog to populate from yet), so the model is
        // preserved across a route change.
        this._populateModels();
        this._saveState();
        this._maybeCommit();
        this._refreshRouteScopedModels();
    }

    /**
     * Upstream (meta-provider) facet change — re-filter the model combo. This
     * is a DISPLAY filter only: it never commits a model change and is never
     * sent to the server as an OpenRouter provider-preferences pin (#2264).
     */
    _handleUpstreamChange() {
        if (!this.isCurrent()) return;
        this.selectedUpstream = this.upstreamSelect.value || 'All';
        this._saveState();
        this._populateModels();
    }

    /**
     * Sync with server's current model
     */
    async syncWithServer() {
        // Skip if no endpoint configured
        if (!this.currentModelEndpoint) return;

        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader())
            };
            if (!this.isCurrent()) return;

            const response = await fetch(this.currentModelEndpoint, { headers });
            if (!response.ok) return;

            const data = await response.json();
            if (!this.isCurrent()) return;
            if (!data.model) return;

            // Auto-resolution flag (#2419): the server tells us whether the
            // active model is a pinned mandate or an auto default. Capture it
            // so getSummary() can label auto-drift on the header button.
            this._isAuto = !!data.is_auto;

            // Canonical shape: /api/model/current returns
            //   {vendor, route, model_name, model: "<vendor>[:<route>]/<model_name>"}
            const bareModel = data.model_name || data.model.split('/').pop();
            const targetVendor = data.vendor;
            const targetRoute = data.route || '';

            const buckets = this.allModelsData?.by_vendor;
            if (!buckets) return;

            const search = targetVendor && buckets[targetVendor]
                ? [[targetVendor, buckets[targetVendor]]]
                : Object.entries(buckets);

            for (const [vendor, models] of search) {
                if (models.some(m => m.id === bareModel)) {
                    const vendorChanged = this.providerSelect.value !== vendor;
                    this.providerSelect.value = vendor;
                    this.selectedProvider = vendor;
                    // Set the selection BEFORE (re)populating so the collapsed
                    // featured-only view still renders the current model when it
                    // is non-featured (e.g. an operator already on a deprecated
                    // gpt-4). Otherwise modelSelect.value targets an option that
                    // was never rendered and the dropdown shows blank. (#2015)
                    this.selectedModel = bareModel;
                    if (vendorChanged) this._populateRoutes();
                    this._populateModels();
                    this.modelSelect.value = bareModel;
                    if (this.routeSelect && targetRoute) {
                        const opts = Array.from(this.routeSelect.options).map(o => o.value);
                        if (opts.includes(targetRoute)) {
                            this.routeSelect.value = targetRoute;
                            this.selectedRoute = targetRoute;
                        }
                    }
                    this._saveState();
                    // Record the server's confirmed state so _maybeCommit has
                    // something to diff user-driven changes against.
                    this._lastSyncedSelection = {
                        vendor: this.selectedProvider,
                        model: this.selectedModel,
                        route: this.selectedRoute || null,
                    };
                    break;
                }
            }
        } catch (e) {
            // Silently fail - endpoint might not exist
        }
    }

    /**
     * Update UI from agent response containing MODEL_CHANGED marker
     * @param {string} content - Response content to check for MODEL_CHANGED
     */
    checkForModelChange(content) {
        if (!content?.includes('MODEL_CHANGED:')) return false;

        // The payload can arrive as plain JSON (``{"vendor":"..."}``) or as
        // JSON-stringified escapes (``{\"vendor\":\"...\"}``) when an upstream
        // layer double-encodes the agent message. Detect the escape-shape by
        // looking at the first few chars after the marker and unescape the
        // whole content before extraction — otherwise the string-aware
        // extractor treats escaped quotes as string body and never finds the
        // closing brace.
        let effective = content;
        const markerIdx = content.indexOf('MODEL_CHANGED:');
        const probeIdx = markerIdx + 'MODEL_CHANGED:'.length;
        const probe = content.slice(probeIdx, probeIdx + 3);
        if (probe.startsWith('{\\"') || probe.startsWith('{\\\'')) {
            effective = content
                .replace(/\\"/g, '"')
                .replace(/\\n/g, '\n')
                .replace(/\\\\/g, '\\');
        }

        try {
            const rawJson = this._extractModelChangedPayload(effective);
            if (!rawJson) {
                return false;
            }
            let syncData;
            try {
                syncData = JSON.parse(rawJson);
            } catch (firstErr) {
                // Last-chance unescape if the heuristic above didn't catch it.
                const unescaped = rawJson
                    .replace(/\\"/g, '"')
                    .replace(/\\n/g, '\n')
                    .replace(/\\\\/g, '\\');
                syncData = JSON.parse(unescaped);
            }

            // MODEL_CHANGED payload (new shape): {vendor, route, model, model_name}
            const vendor = syncData.vendor || syncData.provider;
            const route = syncData.route || '';
            const bareModel = syncData.model_name
                || (syncData.model && syncData.model.includes('/')
                    ? syncData.model.split('/').pop()
                    : syncData.model);

            if (vendor && bareModel) {
                // The marker confirms a server-side mutation. Do not let a
                // later agent switch reuse descriptive defaults/capabilities
                // captured before that change.
                this.invalidateCatalog();
                // Set the selection BEFORE (re)populating so a non-featured
                // target (e.g. an agent that switched to a deprecated model)
                // is still rendered under the collapsed featured-only view —
                // otherwise modelSelect.value would target a missing option and
                // show blank. (#2015)
                const vendorChanged = this.providerSelect && this.providerSelect.value !== vendor;
                if (this.providerSelect && vendorChanged) {
                    this.providerSelect.value = vendor;
                    this.selectedProvider = vendor;
                }
                this.selectedModel = bareModel;
                if (this.providerSelect && vendorChanged) this._populateRoutes();
                if (this.modelSelect) {
                    this._populateModels();
                    this.modelSelect.value = bareModel;
                }
                if (this.routeSelect && route) {
                    const opts = Array.from(this.routeSelect.options).map(o => o.value);
                    if (opts.includes(route)) {
                        this.routeSelect.value = route;
                        this.selectedRoute = route;
                    }
                }
                this._saveState();
                // This path reflects a CONFIRMED server-side change (agent
                // emitted MODEL_CHANGED in its chat response). Record it as
                // lastSynced so the next user interaction diffs correctly.
                this._lastSyncedSelection = {
                    vendor: this.selectedProvider,
                    model: this.selectedModel,
                    route: this.selectedRoute || null,
                };
                // A confirmed model change reflects a concrete selection, not
                // an auto default (#2419).
                this._isAuto = false;
                return true;
            }
        } catch (e) {
            console.warn('ModelSelector: Failed to parse MODEL_CHANGED:', e);
        }

        return false;
    }

    /**
     * Extract the JSON object that follows a MODEL_CHANGED marker.
     * Allows normal response text before and after the marker payload.
     * @param {string} content
     * @returns {string|null}
     */
    _extractModelChangedPayload(content) {
        const marker = 'MODEL_CHANGED:';
        const markerIndex = content.indexOf(marker);
        if (markerIndex === -1) return null;

        const jsonStart = content.indexOf('{', markerIndex + marker.length);
        if (jsonStart === -1) return null;

        let depth = 0;
        let inString = false;
        let escapeNext = false;

        for (let i = jsonStart; i < content.length; i++) {
            const char = content[i];

            if (escapeNext) {
                escapeNext = false;
                continue;
            }

            if (char === '\\' && inString) {
                escapeNext = true;
                continue;
            }

            if (char === '"') {
                inString = !inString;
                continue;
            }

            if (inString) {
                continue;
            }

            if (char === '{') {
                depth += 1;
            } else if (char === '}') {
                depth -= 1;
                if (depth === 0) {
                    return content.slice(jsonStart, i + 1);
                }
            }
        }

        return null;
    }

    /**
     * Get current selection.
     *
     * Returns {vendor, route, model}. The ``provider`` alias is retained
     * for callers that still pass it through as a bare string — prefer
     * ``vendor`` in new code.
     */
    getSelection() {
        return {
            vendor: this.selectedProvider,
            route: this.selectedRoute || null,
            model: this.selectedModel,
            provider: this.selectedProvider,  // alias for back-compat
            // Display-only upstream filter (meta-provider facet). Never a
            // routing pin; surfaced so callers can reflect the visible filter.
            upstream: this.selectedUpstream && this.selectedUpstream !== 'All'
                ? this.selectedUpstream
                : null,
        };
    }

    /**
     * Build the at-a-glance label for the model-settings header button (#2419).
     *
     * Reads the same resolved state the popover drives — vendor / route / model
     * plus the auto-resolution flag — so the button never disagrees with the
     * panel. When the preference is auto the label is ``"Auto — currently
     * <model>"`` so an unchosen model change reads as auto-drift rather than a
     * setting someone changed. Otherwise it is ``"<model> · <Route>"`` (route
     * suffix omitted when the vendor has a single/unnamed route).
     *
     * Returns ``{ isAuto, vendor, route, model, displayName, label }``; ``label``
     * is ``''`` when no model has resolved yet (caller falls back to a static
     * "Model settings").
     */
    getSummary() {
        const vendor = this.selectedProvider || '';
        const route = this.selectedRoute || '';
        const model = this.selectedModel || '';
        if (!model) {
            return { isAuto: this._isAuto, vendor, route, model: '', displayName: '', label: '' };
        }
        // Prefer the human display name from the active catalog; fall back to
        // the bare model id when discovery hasn't surfaced a friendly name.
        const found = (this._currentModelList() || []).find(m => m.id === model);
        const displayName = (found && found.display_name) || model;
        const routeLabel = route ? route.charAt(0).toUpperCase() + route.slice(1) : '';
        let label;
        if (this._isAuto) {
            label = `Auto — currently ${displayName}`;
        } else {
            label = routeLabel ? `${displayName} · ${routeLabel}` : displayName;
        }
        return { isAuto: this._isAuto, vendor, route, model, displayName, label };
    }

    /**
     * Update the current-model endpoint dynamically (e.g., when companion changes)
     * @param {string|null} url - New endpoint URL, or null to disable sync
     */
    setCurrentModelEndpoint(url) {
        this.currentModelEndpoint = url;
    }

    /**
     * Set the active model on the server via POST
     * @param {string} endpoint - POST endpoint URL
     * @param {string} provider - Provider name
     * @param {string} model - Model ID
     * @returns {Promise<boolean>} - Whether the request succeeded
     */
    async setModelOnServer(endpoint, vendor, model, route) {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader())
            };
            const body = { vendor, model };
            if (route) body.route = route;
            const response = await fetch(endpoint, {
                method: 'POST',
                headers,
                body: JSON.stringify(body)
            });
            if (response.ok) this.invalidateCatalog();
            return response.ok;
        } catch (e) {
            console.warn('ModelSelector: Failed to set model on server:', e);
            return false;
        }
    }

    /**
     * Set selection programmatically
     * @param {string} provider - Provider name
     * @param {string} model - Model ID
     * @param {boolean} [triggerCallback=false] - Whether to trigger onModelChange
     */
    setSelection(provider, model, triggerCallback = false) {
        if (provider && this.providerSelect) {
            this.providerSelect.value = provider;
            this.selectedProvider = provider;
            this._populateModels();
        }

        if (model && this.modelSelect) {
            this.modelSelect.value = model;
            this.selectedModel = model;
        }

        this._saveState();

        if (triggerCallback) {
            this.invalidateCatalog();
            this.onModelChange(this.selectedProvider, this.selectedModel);
        }
    }

    /**
     * Declare which model IDs are not user-pickable.  They render in the
     * dropdown as <option disabled> with a 🎙 prefix; programmatic selection
     * via lockToVoiceModel() still works.  Typical caller: chat.js, after
     * resolving the voice route's ``voice_model``.
     *
     * @param {Iterable<string>} modelIds  Replaces any prior set.
     */
    setUnpickableModels(modelIds) {
        this._unpickableModels = new Set(modelIds || []);
        // Re-render so any visible options pick up the disabled state.
        if (this.providerSelect?.value && this.allModelsData) {
            this._populateModels();
        }
    }

    /**
     * Pin the selector to a specific model for the lifetime of a feature's
     * exclusive claim (the generic mechanism behind voice's realtime takeover —
     * #2047).  Captures the prior selection on the first pin, switches the
     * visible vendor + model to ``target``, and disables every select
     * (provider / route / model) until ``unpinSelection`` runs.  No localStorage
     * write — the pin is transient UI state.
     *
     * Safe to call again while already pinned: a re-pin re-asserts the same
     * target without clobbering the captured prior selection.  This is the
     * ``onRefresh`` path — after ``_populateModels`` rebuilds the option list,
     * the held claim re-asserts so the pinned option survives the rebuild.
     *
     * If the target model isn't present in any vendor bucket (e.g. discovery
     * hasn't surfaced it yet), a transient <option> is injected so the value
     * can still be displayed; that option is removed on release.
     *
     * @param {Object} target
     * @param {string} target.vendor   Vendor key the model lives under.
     * @param {string} target.model    Model id (e.g. ``gpt-realtime-2``).
     * @param {string} [target.route]  Optional route id.
     * @param {string} [target.label]  Option text for the injected marker
     *           (defaults to the model id; voice passes ``🎙 <model>``).
     * @param {string} [reason]        Human-readable tooltip explaining the pin.
     */
    pinToModel(target, reason) {
        const { vendor, model, route, label } = target || {};
        if (!model) return;

        // Capture prior selection only on the FIRST pin; a re-assert (onRefresh)
        // must keep the original prior so unpin restores the user's pre-pin pick.
        if (!this._pinnedSelection) {
            this._pinnedSelection = {
                priorProvider: this.selectedProvider,
                priorModel: this.selectedModel,
                priorRoute: this.selectedRoute,
                injectedOption: false,
            };
        }

        // Re-entrancy guard so the _populateModels call below doesn't recurse
        // back through the claim's onRefresh into pinToModel.
        this._applyingPin = true;
        try {
            // Flip the visible vendor first so _populateModels rebuilds the bucket.
            if (vendor && this.providerSelect) {
                this.providerSelect.value = vendor;
                this.selectedProvider = vendor;
                this._populateModels();
            }
            if (route && this.routeSelect) {
                this.routeSelect.value = route;
                this.selectedRoute = route;
            }
            // If the target model isn't in the rebuilt option list, inject a
            // transient marker option.  Tagged so we can find + remove it.
            if (this.modelSelect) {
                const existing = Array.from(this.modelSelect.options).some(o => o.value === model);
                if (!existing) {
                    const opt = document.createElement('option');
                    opt.value = model;
                    opt.textContent = label || model;
                    opt.dataset.claimInjected = 'true';
                    opt.disabled = true;  // unpickable, but value-assignable
                    this.modelSelect.appendChild(opt);
                    this._pinnedSelection.injectedOption = true;
                }
                this.modelSelect.value = model;
                this.selectedModel = model;
            }

            // Disable every select.  ``disabled`` keeps the value visible while
            // blocking user interaction; aria-disabled + title make the reason
            // discoverable.
            for (const el of [this.providerSelect, this.routeSelect, this.modelSelect]) {
                if (!el) continue;
                el.disabled = true;
                el.setAttribute('aria-disabled', 'true');
                if (reason) el.title = reason;
            }
        } finally {
            this._applyingPin = false;
        }
    }

    /**
     * Release the pin and restore the captured prior selection.
     * Idempotent — safe to call from every exit path (close, fatal error,
     * page unload, agent switch) without bookkeeping.
     */
    unpinSelection() {
        if (!this._pinnedSelection) return;
        const { priorProvider, priorModel, priorRoute, injectedOption } = this._pinnedSelection;
        this._pinnedSelection = null;

        // Re-enable selects + clear the pin tooltip.
        for (const el of [this.providerSelect, this.routeSelect, this.modelSelect]) {
            if (!el) continue;
            el.disabled = false;
            el.removeAttribute('aria-disabled');
            el.removeAttribute('title');
        }

        // Restore prior selection.  setSelection writes localStorage; the pin
        // never wrote anything, so this just restores the pre-pin value.
        if (priorProvider && this.providerSelect) {
            this.providerSelect.value = priorProvider;
            this.selectedProvider = priorProvider;
            this._populateModels();
        }
        if (priorRoute !== undefined && this.routeSelect) {
            this.routeSelect.value = priorRoute || '';
            this.selectedRoute = priorRoute || '';
        }
        if (priorModel && this.modelSelect) {
            this.modelSelect.value = priorModel;
            this.selectedModel = priorModel;
        }

        // Drop any transient option we injected so re-pins re-inject cleanly and
        // the dropdown doesn't grow stale entries.
        if (injectedOption && this.modelSelect) {
            for (const opt of Array.from(this.modelSelect.options)) {
                if (opt.dataset && opt.dataset.claimInjected === 'true') {
                    opt.remove();
                }
            }
        }
    }

    /** True while the selector is pinned by an exclusive claim. */
    isPinned() {
        return this._pinnedSelection !== null;
    }
}

/**
 * Generic single-holder claim/release/refresh negotiation for a shared widget
 * (kestrel-sovereign#2047).
 *
 * Some core widgets are occasionally seized by a feature for the lifetime of a
 * session — the canonical case is voice taking over the model selector during a
 * realtime session ("while my session is live, this control is mine, then I give
 * it back"). That is a negotiation, not a mount point, so it gets this small
 * contract instead of being forced into the slot model.
 *
 * Single-holder: while a claim is held a second `acquire` by a DIFFERENT holder
 * is rejected (returns false); re-`acquire` by the SAME holder is an idempotent
 * success. The claim is released automatically when the claiming feature's
 * capability drops (forward the UI bus `capabilities:changed` payload to
 * `onCapabilitiesChanged`).
 *
 * The registry is deliberately widget-agnostic — `widget` is opaque and simply
 * handed back to each callback. Any shared widget can own one of these; only
 * the model selector does today.
 */
class WidgetClaimRegistry {
    /** @param {object} widget - the shared widget this registry guards. */
    constructor(widget) {
        this._widget = widget;
        /** @type {{id: string, capability?: string, onAcquire?: Function, onRelease?: Function, onRefresh?: Function} | null} */
        this._claim = null;
    }

    /**
     * Seize the widget for ``claimId``.
     *
     * @param {string} claimId  Stable id identifying the holder.
     * @param {Object} [spec]
     * @param {string}   [spec.capability]  Capability whose loss auto-releases
     *           this claim (see {@link onCapabilitiesChanged}).
     * @param {(widget: any) => void} [spec.onAcquire]  Run on successful acquire.
     * @param {(widget: any) => void} [spec.onRelease]  Run on release.
     * @param {(widget: any) => void} [spec.onRefresh]  Run when the widget asks
     *           the holder to re-assert (e.g. after it rebuilds its options).
     * @returns {boolean} true if acquired (or already held by this id); false if
     *           held by a different id (single-holder reject).
     */
    acquire(claimId, spec = {}) {
        if (!claimId) return false;
        if (this._claim) {
            // Single-holder: same holder re-acquiring is an idempotent success;
            // a different holder is rejected (the stated safe default — #2047).
            return this._claim.id === claimId;
        }
        this._claim = { id: claimId, ...spec };
        this._run('onAcquire');
        return true;
    }

    /**
     * Relinquish the claim. Idempotent: a no-op when nothing is held, or when
     * ``claimId`` is supplied and does not match the current holder.
     *
     * @param {string} [claimId] When given, only releases if it matches.
     */
    release(claimId) {
        if (!this._claim) return;
        if (claimId && this._claim.id !== claimId) return;
        const claim = this._claim;
        this._claim = null;
        this._invoke(claim, 'onRelease');
    }

    /** Ask the current holder to re-assert its claim. No-op when unheld. */
    refresh() {
        this._run('onRefresh');
    }

    /** True while any claim is held. */
    isHeld() {
        return this._claim !== null;
    }

    /** The id of the current holder, or null. */
    heldBy() {
        return this._claim ? this._claim.id : null;
    }

    /** True when ``claimId`` is the current holder. */
    has(claimId) {
        return this._claim !== null && this._claim.id === claimId;
    }

    /**
     * React to a UI-bus ``capabilities:changed`` payload: if the current holder
     * declared a ``capability`` and it is now absent/false in the payload's
     * capability map, auto-release. This is how a claim is relinquished when the
     * claiming feature is disabled at runtime (#2047 / ticket 03). Generic — the
     * registry doesn't know which capability any particular feature uses.
     *
     * @param {{capabilities?: Record<string, boolean>} | null} payload
     */
    onCapabilitiesChanged(payload) {
        const claim = this._claim;
        if (!claim || !claim.capability) return;
        const caps = payload && payload.capabilities;
        // Only act when we actually have a capability map to judge against; a
        // bare/empty payload can't prove the capability is gone.
        if (!caps || typeof caps !== 'object') return;
        if (!caps[claim.capability]) this.release(claim.id);
    }

    /** Invoke a callback on the CURRENT claim (used by acquire/refresh). */
    _run(hook) {
        if (this._claim) this._invoke(this._claim, hook);
    }

    /** Invoke ``hook`` on ``claim``, isolating throws so one bad holder can't wedge the widget. */
    _invoke(claim, hook) {
        const fn = claim[hook];
        if (typeof fn !== 'function') return;
        try {
            fn(this._widget);
        } catch (err) {
            console.error(`[widget-claims] ${hook} for "${claim.id}" threw:`, err);
        }
    }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ModelSelector, WidgetClaimRegistry, VENDOR_NAMES, PROVIDER_NAMES };
}

// Export globally for script tag usage
window.SharedModelSelector = ModelSelector;
window.WidgetClaimRegistry = WidgetClaimRegistry;
window.VENDOR_NAMES = VENDOR_NAMES;
window.PROVIDER_NAMES = PROVIDER_NAMES;
