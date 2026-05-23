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
     * @param {boolean} [options.sendCommandOnChange=true] - Whether to trigger onModelChange on selection
     */
    constructor(options = {}) {
        this.providerSelect = document.getElementById(options.providerSelectId);
        this.modelSelect = document.getElementById(options.modelSelectId);
        // Optional route selector — appears when a vendor has >1 configured route.
        this.routeSelect = options.routeSelectId
            ? document.getElementById(options.routeSelectId)
            : null;
        this.apiEndpoint = options.apiEndpoint || '/api/models';
        // Use 'in' check to allow explicit null (disables server sync)
        this.currentModelEndpoint = 'currentModelEndpoint' in options
            ? options.currentModelEndpoint
            : '/api/model/current';
        this.storagePrefix = options.storagePrefix || 'kestrel';
        this.onModelChange = options.onModelChange || (() => {});
        this.getAuthHeader = options.getAuthHeader || (() => ({}));
        this.sendCommandOnChange = options.sendCommandOnChange !== false;

        this.allModelsData = null;
        this.selectedProvider = '';
        this.selectedModel = '';
        this.selectedRoute = '';
        this.isInitialLoad = true;

        // Model IDs the user must not be able to pick from the dropdown — e.g.
        // the OpenAI Realtime model, which is owned by the mic button. Rendered
        // as <option disabled> with a 🎙 prefix. See kestrel-sovereign#1371.
        this._unpickableModels = new Set();
        // Voice-owned lock state. Non-null while the mic button has taken the
        // selector. Stores prior selection so we can restore on every voice
        // exit path. Lock state is intentionally NOT persisted to localStorage:
        // it is transient UI ownership, not a user choice.
        this._voiceLock = null;

        this._loadState();
    }

    /**
     * Load saved state from localStorage
     */
    _loadState() {
        this.selectedProvider = localStorage.getItem(`${this.storagePrefix}_selected_provider`) || '';
        this.selectedModel = localStorage.getItem(`${this.storagePrefix}_selected_model`) || '';
        this.selectedRoute = localStorage.getItem(`${this.storagePrefix}_selected_route`) || '';
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
    }

    /**
     * Initialize the component - load models and bind events
     */
    async init() {
        await this.loadModels();
        this._bindEvents();
        await this.syncWithServer();
        this.isInitialLoad = false;
    }

    /**
     * Bind event listeners
     */
    _bindEvents() {
        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', () => this._handleProviderChange());
        }
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', () => this._handleModelChange());
        }
        if (this.routeSelect) {
            this.routeSelect.addEventListener('change', () => this._handleRouteChange());
        }
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

            const response = await fetch(this.apiEndpoint, { headers });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            this.allModelsData = await response.json();
            this._populateProviders();

            return this.allModelsData;
        } catch (e) {
            console.error('ModelSelector: Failed to load models:', e);
            this.providerSelect.innerHTML = '<option value="">Error loading</option>';
            return null;
        }
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
     * Populate model dropdown based on selected vendor
     */
    _populateModels() {
        const vendor = this.providerSelect?.value;
        const buckets = this.allModelsData?.by_vendor;
        if (!vendor || !buckets) return;

        const models = [...(buckets[vendor] || [])];

        if (models.length === 0) {
            this.modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }

        // Sort: featured first, then alphabetically by display name
        models.sort((a, b) => {
            if (a.is_featured !== b.is_featured) return b.is_featured ? 1 : -1;
            return (a.display_name || a.id).localeCompare(b.display_name || b.id);
        });

        // Build model options.  Unpickable models render as <option disabled>
        // with a 🎙 prefix so the operator can see they exist but cannot
        // select them by hand (the mic button drives them).  Selecting the
        // option programmatically still works — see lockToVoiceModel().
        this.modelSelect.innerHTML = models.map(m => {
            const isUnpickable = this._unpickableModels.has(m.id);
            const star = m.is_featured ? '★ ' : '';
            const glyph = isUnpickable ? '🎙 ' : '';
            const displayName = m.display_name || m.id;
            const disabled = isUnpickable ? ' disabled' : '';
            return `<option value="${m.id}"${disabled}>${star}${glyph}${displayName}</option>`;
        }).join('');

        // Seed order: saved model > server default > alphabetical first.
        if (this.selectedModel && models.some(m => m.id === this.selectedModel)) {
            this.modelSelect.value = this.selectedModel;
        } else if (models.length > 0) {
            const defaultId = this.allModelsData?.default;
            if (defaultId && models.some(m => m.id === defaultId)) {
                this.modelSelect.value = defaultId;
                this.selectedModel = defaultId;
            } else {
                this.modelSelect.value = models[0].id;
                this.selectedModel = models[0].id;
            }
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
        this.onModelChange(vendor, model, this.isInitialLoad, route);
    }

    _handleProviderChange() {
        this.selectedProvider = this.providerSelect.value;
        this.selectedRoute = '';
        this._saveState();
        this._populateRoutes();
        this._populateModels();
        this.selectedModel = this.modelSelect.value;
        this._saveState();
        this._maybeCommit();
    }

    _handleModelChange() {
        this.selectedModel = this.modelSelect.value;
        this._saveState();
        this._maybeCommit();
    }

    _handleRouteChange() {
        this.selectedRoute = this.routeSelect.value;
        this._saveState();
        this._maybeCommit();
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

            const response = await fetch(this.currentModelEndpoint, { headers });
            if (!response.ok) return;

            const data = await response.json();
            if (!data.model) return;

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
                    if (this.providerSelect.value !== vendor) {
                        this.providerSelect.value = vendor;
                        this.selectedProvider = vendor;
                        this._populateRoutes();
                        this._populateModels();
                    }
                    this.modelSelect.value = bareModel;
                    this.selectedModel = bareModel;
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
                if (this.providerSelect && this.providerSelect.value !== vendor) {
                    this.providerSelect.value = vendor;
                    this.selectedProvider = vendor;
                    this._populateRoutes();
                    this._populateModels();
                }
                if (this.modelSelect) {
                    this.modelSelect.value = bareModel;
                    this.selectedModel = bareModel;
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
        };
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
     * Voice took ownership of the selector for the lifetime of one mic
     * engagement.  Captures the prior selection, switches the visible
     * vendor + model to the given Realtime model, and disables every select
     * (provider / route / model) until ``unlockToPrior`` runs.  No localStorage
     * write — the lock is transient UI state.
     *
     * If the target model isn't present in any vendor bucket (e.g. discovery
     * hasn't surfaced it yet), a transient <option> is injected so the value
     * can still be displayed; that option is removed on unlock.
     *
     * @param {Object} target
     * @param {string} target.vendor   Vendor key the Realtime model lives under.
     * @param {string} target.model    Model id (e.g. ``gpt-realtime-2``).
     * @param {string} [target.route]  Optional route id.
     * @param {string} [reason]        Human-readable tooltip explaining the lock.
     */
    lockToVoiceModel(target, reason) {
        if (this._voiceLock) return;  // re-entrant: keep the first capture
        const { vendor, model, route } = target || {};
        if (!model) return;

        this._voiceLock = {
            priorProvider: this.selectedProvider,
            priorModel: this.selectedModel,
            priorRoute: this.selectedRoute,
            injectedOption: false,
        };

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
                opt.textContent = `🎙 ${model}`;
                opt.dataset.voiceInjected = 'true';
                opt.disabled = true;  // unpickable, but value-assignable
                this.modelSelect.appendChild(opt);
                this._voiceLock.injectedOption = true;
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
    }

    /**
     * Release the voice lock and restore the captured prior selection.
     * Idempotent — safe to call from every exit path (close, fatal error,
     * page unload, agent switch) without bookkeeping.
     */
    unlockToPrior() {
        if (!this._voiceLock) return;
        const { priorProvider, priorModel, priorRoute, injectedOption } = this._voiceLock;
        this._voiceLock = null;

        // Re-enable selects + clear the lock tooltip.
        for (const el of [this.providerSelect, this.routeSelect, this.modelSelect]) {
            if (!el) continue;
            el.disabled = false;
            el.removeAttribute('aria-disabled');
            el.removeAttribute('title');
        }

        // Restore prior selection.  setSelection writes localStorage; voice
        // never wrote anything, so this just restores the pre-engage value.
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

        // Drop any transient option we injected so re-engagements re-inject
        // cleanly and the dropdown doesn't grow stale entries.
        if (injectedOption && this.modelSelect) {
            for (const opt of Array.from(this.modelSelect.options)) {
                if (opt.dataset && opt.dataset.voiceInjected === 'true') {
                    opt.remove();
                }
            }
        }
    }

    /** True while the voice lock is engaged. */
    isVoiceLocked() {
        return this._voiceLock !== null;
    }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ModelSelector, VENDOR_NAMES, PROVIDER_NAMES };
}

// Export globally for script tag usage
window.SharedModelSelector = ModelSelector;
window.VENDOR_NAMES = VENDOR_NAMES;
window.PROVIDER_NAMES = PROVIDER_NAMES;
