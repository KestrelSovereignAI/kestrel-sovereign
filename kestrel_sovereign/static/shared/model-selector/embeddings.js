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
     * @param {Function} [options.invalidateModelCatalog] Evicts the owning
     *        agent's descriptive model catalog after a successful mutation.
     * @param {Function} [options.isCurrent] False when an agent switch has
     *        superseded this selector instance.
     */
    constructor(options = {}) {
        this.settingsEndpoint = options.settingsEndpoint || '/api/embedding/settings';
        // #2338/#2337 — the dynamically-discovered embedding catalog (per-route
        // models + the computed "Universal" shared-space options). Read once on
        // load; drives the featured Universal option and the per-route model
        // picker so the operator never hand-edits TOML.
        this.modelsEndpoint = options.modelsEndpoint || '/api/embedding/models';
        // #2337 — per-route embedding_model pin (the runtime equivalent of the
        // TOML embedding_model/embedding_dim keys), with a probe-on-save.
        this.routeModelEndpoint = options.routeModelEndpoint || '/api/embedding/route-model';
        // #2290 — the shared-space parity probe used by guided Universal setup.
        this.verifyEndpoint = options.verifyEndpoint || '/api/embedding/space/verify';
        this.modeSelect = options.modeSelectId ? document.getElementById(options.modeSelectId) : null;
        this.routeSelect = options.routeSelectId ? document.getElementById(options.routeSelectId) : null;
        // #2337 — featured "Universal — <model> (local + cloud, one search
        // space)" option, pinned at the top and rendered even when not yet
        // configured ("needs setup"). Clicking it runs guided setup.
        this.universalEl = options.universalId ? document.getElementById(options.universalId) : null;
        // #2337 — per-route embedding-model <select>, shown alongside the route
        // picker in explicit mode. Populated from the discovered catalog.
        this.modelSelect = options.modelSelectId ? document.getElementById(options.modelSelectId) : null;
        // #2337 — status readout for guided Universal setup (fails loudly).
        this.setupStatus = options.setupStatusId ? document.getElementById(options.setupStatusId) : null;
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
        this.invalidateModelCatalog = options.invalidateModelCatalog || (() => {});
        this.isCurrent = options.isCurrent || (() => true);
        this._eventHandlers = null;
        // True while a reindex job is running — guards the button re-entrancy.
        this._reindexing = false;

        this.settings = null;
        // The discovered embedding catalog ({all, by_vendor,
        // shared_space_candidates, universal}); null until loaded / on failure.
        this.catalog = null;
        // Guards guided-setup re-entrancy.
        this._settingUp = false;
        // 'auto' == follow chat provider (embedding_route null); 'explicit' ==
        // a pinned provider:route; 'off' == embeddings deliberately disabled
        // (embedding_route "none", #2287).
        this.mode = 'auto';
    }

    async init() {
        await this.load();
        if (this.isCurrent()) this._bindEvents();
    }

    _bindEvents() {
        if (this._eventHandlers) return;
        this._eventHandlers = [];
        // The popover DOM (mode/route selects, re-embed button, …) persists,
        // but chat.js rebuilds a fresh EmbeddingSelector on every agent switch
        // (loadModels()). Naively re-adding listeners each time stacks handlers
        // on the same element, so ONE click fires N POSTs — and because each
        // stale instance still targets ITS agent's endpoints, the same click
        // hits multiple agents (the Emma+Nellie double-fire in #2420). Bind
        // through _bindListener, which displaces any handler a prior instance
        // registered on the same element before adding our own.
        this._bindListener(this.modeSelect, 'change', () => this._handleModeChange());
        this._bindListener(this.routeSelect, 'change', () => this._handleRouteChange());
        this._bindListener(this.reindexButton, 'click', () => this._handleReindexClick());
        this._bindListener(this.universalEl, 'click', () => this._handleUniversalClick());
        this._bindListener(this.modelSelect, 'change', () => this._handleModelChange());
    }

    /**
     * Add a listener idempotently across EmbeddingSelector rebuilds (#2420).
     * The element outlives the instance, so before binding we remove whatever
     * handler a previous instance stashed on the element for this event type.
     */
    _bindListener(element, type, handler) {
        if (!element || typeof element.addEventListener !== 'function') return;
        const key = `_kestrelEmbedHandler_${type}`;
        const prior = element[key];
        if (prior && typeof element.removeEventListener === 'function') {
            element.removeEventListener(type, prior);
        }
        element.addEventListener(type, handler);
        element[key] = handler;
        this._eventHandlers.push({ element, type, handler, key });
    }

    /** Remove handlers before a switch replaces this selector instance. */
    destroy() {
        if (!this._eventHandlers) return;
        for (const { element, type, handler, key } of this._eventHandlers) {
            element.removeEventListener?.(type, handler);
            if (element[key] === handler) delete element[key];
        }
        this._eventHandlers = null;
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
            const settings = await response.json();
            if (!this.isCurrent()) return;
            this.settings = settings;
        } catch (e) {
            return;
        }
        // Best-effort: the discovered catalog powers the featured Universal
        // option and per-route model picker. A failure here degrades to the
        // route-only UI rather than blocking the settings render.
        await this._loadCatalog();
        if (!this.isCurrent()) return;
        this.mode = embeddingModeForRoute(this.settings && this.settings.embedding_route);
        this._render();
    }

    /** Fetch the discovered embedding catalog (#2338/#2337); best-effort. */
    async _loadCatalog() {
        if (!this.modelsEndpoint) return;
        // Only worth fetching when the UI can actually use it.
        if (!this.universalEl && !this.modelSelect) return;
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const response = await fetch(this.modelsEndpoint, { headers });
            if (!response.ok) return;
            const data = await response.json();
            // Guard: only accept a real catalog shape (a settings object served
            // by a mis-pointed endpoint must not masquerade as a catalog).
            if (data && (Array.isArray(data.universal) || Array.isArray(data.all))) {
                this.catalog = data;
            }
        } catch (e) {
            /* keep the route-only UI */
        }
    }

    _render() {
        this._renderUniversal();
        this._renderRoutes();
        this._renderModelPicker();
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
        // #2420 — a provider whose resolved dim can't write into the column is
        // exactly the state the reindex endpoint 409s on. An enabled button that
        // can only ever 409 is a lie: fold ``dim_write_blocked`` into resolvable
        // so the button's enablement mirrors the server's writability, and show
        // the same reason the endpoint would return.
        const writeBlocked = s.dim_write_blocked === true;
        const resolvable = this.mode !== 'off' && s.embedding_dim != null && !writeBlocked;

        if (!hasStale) {
            this.reindexButton.style.display = 'none';
            this.reindexButton.disabled = false;
            if (this.reindexStatus && !this._reindexing) {
                // No actionable rows. If the remaining stale rows are all
                // unembeddable (no recoverable text), explain them instead of
                // going silent — otherwise the operator is left wondering why a
                // dim-mismatch warning has no re-embed action (#2426).
                const unembeddable = s.unembeddable_rows || 0;
                if (unembeddable) {
                    this.reindexStatus.textContent =
                        `${unembeddable} ${unembeddable === 1 ? 'row has' : 'rows have'} no embeddable text — nothing to re-embed.`;
                    this.reindexStatus.style.display = '';
                } else {
                    this.reindexStatus.textContent = '';
                    this.reindexStatus.style.display = 'none';
                }
            }
            return;
        }

        this.reindexButton.style.display = '';
        const noun = stale === 1 ? 'memory' : 'memories';
        this.reindexButton.textContent = `Re-embed ${stale} ${noun}`;
        // Disable while unresolvable (off / no provider / write-blocked) or a job
        // is in flight.
        this.reindexButton.disabled = !resolvable || this._reindexing;
        if (resolvable) {
            this.reindexButton.title =
                `Re-embed ${stale} stale ${noun} into the current embedding provider.`;
        } else if (writeBlocked) {
            const dim = s.embedding_dim;
            const columnDim = s.kestrel_embedding_dim;
            this.reindexButton.title = s.dim_write_status ||
                `Selected provider can't write (resolves ${dim}-dim, columns are ` +
                `${columnDim}-dim) — re-embedding would be refused.`;
        } else {
            this.reindexButton.title = 'No embedding provider resolves — nothing to re-embed to.';
        }
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
        if (!this.isCurrent()) return;
        if (this._reindexing) return;
        // #2420 — mark busy immediately so the click has feedback while the
        // dry-run probe runs (this also disables the button via _renderReindex),
        // and reload authoritative settings on every exit path.
        this._reindexing = true;
        this._renderReindex();
        this._setReindexStatus('Checking…');
        try {
            const dry = await this._postReindex({ dry_run: true });
            if (!dry.ok) {
                // #2420 — the backend refusal (400 dim/probe guard, 409 provider
                // can't write) carries a human ``detail`` written for exactly this
                // moment. Render it inline where the operator clicked instead of
                // silently swallowing it.
                this._setReindexStatus(dry.detail
                    ? `Re-embed unavailable: ${dry.detail}`
                    : 'Re-embed failed — could not read counts.');
                return;
            }
            // Confirm against actionable rows only — rows with no embeddable
            // text can never be cleared, so counting them here would promise
            // work the run can't do (#2426). Falls back to total_stale for
            // backends that predate the split.
            const d = dry.data || {};
            const n = (d.actionable_stale != null ? d.actionable_stale : d.total_stale) || 0;
            const unembeddable = d.unembeddable_rows || 0;
            if (!n) {
                this._setReindexStatus(unembeddable
                    ? `${unembeddable} ${unembeddable === 1 ? 'row has' : 'rows have'} no embeddable text — nothing to re-embed.`
                    : '');
                return;
            }
            const noun = n === 1 ? 'memory' : 'memories';
            if (!this.confirm(`Re-embed ${n} ${noun} into the current embedding provider? Stored vectors will be rewritten.`)) {
                this._setReindexStatus('');
                return;
            }

            this._setReindexStatus(`Re-embedding ${n} ${noun}…`);
            const started = await this._postReindex({ dry_run: false });
            if (!started.ok) {
                this._setReindexStatus(started.detail
                    ? `Re-embed refused: ${started.detail}`
                    : 'Re-embed failed to start.');
                return;
            }
            let job = started.data;
            if (job && job.job_id && job.status === 'running') {
                job = await this._pollReindex(job.job_id);
            }
            if (!job) {
                // Poll timed out / dropped before a terminal state — don't
                // report "Re-embedded 0 memories." as if it succeeded (#2360).
                this._setReindexStatus('Re-embed status unavailable — check the embedding route and retry.');
            } else if (job.status === 'error') {
                this._setReindexStatus(`Re-embed failed: ${job.error || 'unknown error'}`);
            } else if (job.status === 'partial') {
                // Some rows re-embedded, some failed/skipped — surface both the
                // count and the reason instead of a bare success line (#2360).
                const done = job.total_reembedded || 0;
                const failed = job.total_failed || 0;
                const dim = job.total_skipped_dim_mismatch || 0;
                this._setReindexStatus(
                    `Re-embedded ${done} of ${job.total_stale || 0} — ${failed} failed, ` +
                    `${dim} dimension-mismatch. ${job.error || ''}`.trim());
            } else {
                const done = job.total_reembedded || 0;
                const unembeddable = job.unembeddable_rows || 0;
                let msg = `Re-embedded ${done} ${done === 1 ? 'memory' : 'memories'}.`;
                if (unembeddable) {
                    // A corpus whose only stale rows have no recoverable text is
                    // done, not an error — explain the rows instead (#2426).
                    msg = done
                        ? `${msg} ${unembeddable} ${unembeddable === 1 ? 'row has' : 'rows have'} no embeddable text.`
                        : `${unembeddable} ${unembeddable === 1 ? 'row has' : 'rows have'} no embeddable text — nothing to re-embed.`;
                }
                this._setReindexStatus(msg);
            }
        } finally {
            this._reindexing = false;
            // Reload authoritative settings (stale_rows/warning refresh).
            await this.load();
        }
    }

    /**
     * POST the reindex body and return ``{ok, data, detail}`` (#2420). ``detail``
     * carries the server's rejection message on a non-2xx (a 400 dim/probe guard
     * or a 409 "provider can't write") so the caller can render it inline instead
     * of swallowing it — the bodies were written for the user.
     */
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
            let data = null;
            try { data = await response.json(); } catch (e) { /* no/invalid body */ }
            if (!response.ok) {
                return { ok: false, data, detail: data && data.detail };
            }
            this.invalidateModelCatalog();
            return { ok: true, data };
        } catch (e) {
            return { ok: false, detail: String(e) };
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
            // Terminal states: done / partial / error. 'partial' means some rows
            // re-embedded but some failed or hit a dimension mismatch — it is
            // finished, not still running, so stop polling and let the caller
            // surface job.error + the failed/skipped counters (#2360).
            if (job && (job.status === 'done' || job.status === 'partial' || job.status === 'error')) {
                return job;
            }
        }
        return null;
    }

    // --- Featured "Universal" option (#2337) --------------------------------

    /**
     * The featured Universal candidate: an open-weight model discovered on BOTH
     * a local and a cloud route (one searchable space in every privacy mode).
     * Prefers the candidate matching the currently-resolved shared space, else
     * the first the backend computed. Never hardcoded to qwen3 — it is whatever
     * the intersection produced. Returns null when the catalog has none.
     */
    _universalOption() {
        const options = (this.catalog && this.catalog.universal) || [];
        if (!options.length) return null;
        const space = this.settings && this.settings.shared_space;
        if (space && space.model) {
            const match = options.find(o => o.model === space.model);
            if (match) return match;
        }
        return options[0];
    }

    /** True when the resolved shared space is configured AND parity-verified. */
    _universalConfigured(option) {
        const space = this.settings && this.settings.shared_space;
        return !!(space && option && space.model === option.model && space.verified);
    }

    /**
     * Render the featured "Universal — <model> (local + cloud, one search
     * space)" option, pinned at the top. Rendered even when not yet configured,
     * marked "needs setup"; when configured + verified it reads "active". This
     * is THE recommended choice: it works in every privacy mode and keeps one
     * searchable memory space, so it leads the section rather than sitting as a
     * peer of the cloud-only routes (#2337 scope amendment).
     */
    _renderUniversal() {
        if (!this.universalEl) return;
        const option = this._universalOption();
        if (!option) {
            this.universalEl.style.display = 'none';
            this.universalEl.textContent = '';
            return;
        }
        const configured = this._universalConfigured(option);
        const state = this._settingUp
            ? 'setting up…'
            : (configured ? 'active' : 'needs setup');
        this.universalEl.style.display = '';
        this.universalEl.textContent =
            `Universal — ${option.model} (local + cloud, one search space) · ${state}`;
        this.universalEl.title =
            'Recommended: works in every privacy mode and keeps one searchable ' +
            'memory space. Local + cloud sessions share the same coordinates. ' +
            (configured ? 'Active.' : 'Click to set up.');
        if ('disabled' in this.universalEl) this.universalEl.disabled = this._settingUp;
    }

    _handleUniversalClick() {
        if (!this.isCurrent()) return;
        if (this._settingUp) return;
        const option = this._universalOption();
        if (!option) return;
        return this._setupUniversal(option);
    }

    /**
     * Guided Universal setup (#2337/#2418): ATOMIC — pin the shared model on
     * EVERY member route (each with its own slug) or NONE. If any member's
     * probe-on-save fails, every pin already applied in this run is rolled back
     * so the operator is never left in a half-applied state (some routes pinned,
     * some not), and the flow ends in ONE unmistakable state: configured (with
     * the parity result) or failed (naming the member and why). A dead/misspelled
     * or auth-rejected upstream slug is surfaced by the route-model probe (#2326);
     * a below-threshold parity shows the measured cosine as a warning rather than
     * silently claiming a shared space — never success styling around a failure.
     */
    async _setupUniversal(option) {
        if (this._settingUp) return;
        const members = (option && option.members) || [];
        if (members.length < 2) {
            this._setSetupStatus('Universal setup needs a local and a cloud member route.', 'error');
            return;
        }
        this._settingUp = true;
        this._renderUniversal();
        // Snapshot the pre-setup per-route pins so a later failure restores EXACTLY
        // what was there, not just "cleared" (#2418). A route that already carried a
        // runtime pin must be put back to that pin on rollback — clearing it would
        // mutate prior configuration, which "atomic apply" forbids.
        const priorPins = (this.settings && this.settings.route_embedding_models) || {};
        // Track the members we've successfully pinned, each with its prior state,
        // so a later failure can roll them back to that state — the atomicity
        // guarantee (#2418).
        const applied = [];
        try {
            for (const member of members) {
                this._setSetupStatus(`Configuring ${member.route} → ${member.model}…`, 'pending');
                // Capture the prior pin BEFORE overwriting it (null when the route
                // had no runtime override → rollback clears rather than restores).
                const prior = priorPins[member.route] || null;
                const res = await this._postRouteModel(member.route, member.model, option.dim);
                if (!res.ok) {
                    // Roll back every pin applied so far — pin all members or none.
                    const rolledBack = await this._rollbackUniversal(applied);
                    const rollNote = applied.length
                        ? (rolledBack
                            ? ' Rolled back the other members — no partial configuration was applied.'
                            : ' WARNING: could not fully roll back earlier members; re-run setup or clear them manually.')
                        : '';
                    this._setSetupStatus(
                        `Setup failed on ${member.route}: ` +
                        `${res.detail || 'the model could not be configured (not served / not pulled / credential invalid).'}` +
                        rollNote,
                        'error'
                    );
                    return;
                }
                applied.push({ route: member.route, prior });
            }
            this._setSetupStatus('Verifying local ↔ cloud parity…', 'pending');
            const verify = await this._postVerify();
            if (!verify.ok) {
                // The parity probe itself failed (auth / 500 / timeout / malformed
                // body) — we CANNOT claim a shared space, so this is a failure, not
                // a success. Honest + atomic (#2418): roll every pin back to its
                // pre-setup state and end in the failed state, never success styling.
                const rolledBack = await this._rollbackUniversal(applied);
                const rollNote = rolledBack
                    ? ' Rolled back all members — no configuration was applied.'
                    : ' WARNING: could not fully roll back; re-run setup or clear the members manually.';
                this._setSetupStatus(
                    'Setup failed: could not verify local ↔ cloud parity ' +
                    `(${verify.detail || 'the parity probe did not complete'}).` + rollNote,
                    'error'
                );
                return;
            }
            const parity = this._summarizeParity(verify.data);
            if (parity && !parity.passed) {
                // All members pinned, but the space is not truly shared. Honest:
                // this is a warning, not a success — the pins stand (each route
                // embeds), but Universal's shared-space promise isn't met.
                this._setSetupStatus(
                    `Parity below threshold (min cosine ${parity.minCosine}). ` +
                    'Local and cloud embeddings drift too far to share one space — ' +
                    'memories embedded on one will not match the other.',
                    'warn'
                );
            } else if (parity) {
                this._setSetupStatus(
                    `Universal active — parity verified (min cosine ${parity.minCosine}).`,
                    'ok'
                );
            } else {
                this._setSetupStatus('Universal configured.', 'ok');
            }
        } finally {
            this._settingUp = false;
            // Authoritative reload refreshes shared_space + stale_rows.
            await this.load();
        }
    }

    /**
     * Roll back a partially-applied Universal setup (#2418): restore each member
     * to its PRE-SETUP pin (best-effort). ``applied`` is a list of
     * ``{route, prior}`` where ``prior`` is the route's runtime pin before this
     * run (``null`` if it had none). A route with a prior pin is restored to
     * ``{model, dim}``; a route with no prior pin is cleared. Returns true only
     * when every rollback POST succeeded, so the caller can tell the operator
     * whether the state is clean or needs manual attention.
     */
    async _rollbackUniversal(applied) {
        let allRestored = true;
        for (const { route, prior } of applied) {
            const res = prior && prior.model
                ? await this._postRouteModel(route, prior.model, prior.dim)
                : await this._postRouteModel(route, null);
            if (!res.ok) allRestored = false;
        }
        return allRestored;
    }

    /** Collapse a /space/verify response into {passed, minCosine} or null. */
    _summarizeParity(verify) {
        if (!verify || !verify.results) return null;
        const rows = Object.values(verify.results);
        if (!rows.length) return null;
        let passed = true;
        let minCosine = null;
        for (const r of rows) {
            if (r && r.passed === false) passed = false;
            const c = r && (r.min_cosine != null ? r.min_cosine : r.minCosine);
            if (typeof c === 'number' && (minCosine === null || c < minCosine)) {
                minCosine = c;
            }
        }
        return { passed, minCosine: minCosine === null ? 'n/a' : minCosine };
    }

    /**
     * Set the guided-setup status line with an explicit ``kind`` so the UI can
     * style success vs. failure distinctly (#2418) — no success styling around a
     * failure. ``kind`` is one of ``'ok'`` / ``'error'`` / ``'warn'`` /
     * ``'pending'`` / ``''``. The kind is reflected both as a class and a
     * ``data-status`` attribute so plain CSS or the host popover can react.
     */
    _setSetupStatus(text, kind = '') {
        if (!this.setupStatus) return;
        this.setupStatus.textContent = text || '';
        this.setupStatus.style.display = text ? '' : 'none';
        const state = text ? kind : '';
        // Reflect the state for styling; keep any non-status classes intact.
        const el = this.setupStatus;
        if (el.classList && typeof el.classList.remove === 'function') {
            for (const cls of ['embed-setup-ok', 'embed-setup-error', 'embed-setup-warn', 'embed-setup-pending']) {
                el.classList.remove(cls);
            }
            if (state) el.classList.add(`embed-setup-${state}`);
        }
        if (typeof el.setAttribute === 'function' && typeof el.removeAttribute === 'function') {
            if (state) {
                el.setAttribute('data-status', state);
            } else {
                el.removeAttribute('data-status');
            }
        }
    }

    // --- Per-route embedding-model picker (#2337) ---------------------------

    /** Discovered embedding models for a given "<vendor>:<route>". */
    _modelsForRoute(route) {
        const all = (this.catalog && this.catalog.all) || [];
        return all.filter(m => m.route === route);
    }

    /**
     * Populate the per-route embedding-model <select> (#2337) for the currently
     * selected explicit route, so the operator picks WHICH model that route
     * embeds with — the runtime equivalent of the TOML ``embedding_model`` key,
     * with no TOML editing. Hidden unless an explicit route with a discovered
     * catalog is selected; a route with no discovered models offers nothing to
     * pick (free-text pinning stays a config/CLI concern).
     */
    _renderModelPicker() {
        if (!this.modelSelect) return;
        const route = this.mode === 'explicit' && this.routeSelect ? this.routeSelect.value : null;
        const models = route ? this._modelsForRoute(route) : [];
        if (this.mode !== 'explicit' || !models.length) {
            this.modelSelect.style.display = 'none';
            this.modelSelect.innerHTML = '';
            return;
        }
        const pins = (this.settings && this.settings.route_embedding_models) || {};
        const pinned = pins[route] && pins[route].model;
        const resolved = this.settings && this.settings.embedding_model;
        const selected = pinned || resolved;
        // #2417 — the column dim these vectors must fit. A model whose native
        // dim differs would break every future write; mark it BEFORE selection.
        const columnDim = this.settings && this.settings.kestrel_embedding_dim;
        this.modelSelect.innerHTML = models.map(m => {
            const label = m.display_name || m.id;
            const dim = m.native_dim ? ` · ${m.native_dim}d` : '';
            // #2417 — a model that supports Matryoshka (MRL) ``dim_options``
            // including the column dim CAN write into it (truncated to fit), so
            // it's not "needs migration" even when its native dim differs.
            const migrate = this._modelNeedsMigration(m, columnDim)
                ? ` — ${m.native_dim}-dim, needs migration`
                : '';
            return `<option value="${m.id}">${label}${dim}${migrate}</option>`;
        }).join('');
        if (selected && models.some(m => m.id === selected)) {
            this.modelSelect.value = selected;
        }
        this.modelSelect.style.display = '';
    }

    /** Commit a per-route model pin from the picker (probe-on-save, #2337). */
    async _handleModelChange() {
        if (!this.isCurrent()) return;
        if (this.mode !== 'explicit' || !this.routeSelect || !this.modelSelect) return;
        const route = this.routeSelect.value;
        const model = this.modelSelect.value;
        if (!route || !model) return;
        const models = this._modelsForRoute(route);
        const chosen = models.find(m => m.id === model);
        // #2417 — pin the dim that actually fits the column: an MRL model whose
        // ``dim_options`` covers the column dim is pinned AT the column dim (so
        // the write succeeds), otherwise its native dim.
        const columnDim = this.settings && this.settings.kestrel_embedding_dim;
        const dim = this._pinDimForModel(chosen, columnDim);
        this._setSetupStatus(`Pinning ${route} → ${model}…`, 'pending');
        const res = await this._postRouteModel(route, model, dim);
        if (!res.ok) {
            this._setSetupStatus(
                `Could not pin ${model} on ${route}: ${res.detail || 'the model may not be served upstream, or the credential may be invalid.'}`,
                'error'
            );
            return;
        }
        this._setSetupStatus('');
        // Reload so the dimension readout / shared space reflect the new pin.
        await this.load();
    }

    /**
     * POST a per-route embedding_model pin (or clear). Returns
     * ``{ok, data, detail}`` — ``detail`` carries the server's rejection
     * message (e.g. a dead-slug 400) so callers can fail loudly.
     */
    async _postRouteModel(route, model, dim) {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const body = { route, embedding_model: model || null };
            if (dim != null) body.embedding_dim = dim;
            const response = await fetch(this.routeModelEndpoint, {
                method: 'POST',
                headers,
                body: JSON.stringify(body),
            });
            let data = null;
            try { data = await response.json(); } catch (e) { /* no body */ }
            if (!response.ok) {
                return { ok: false, data, detail: data && data.detail };
            }
            return { ok: true, data };
        } catch (e) {
            return { ok: false, detail: String(e) };
        }
    }

    /**
     * POST the shared-space parity probe (#2290). Returns ``{ok, data, detail}``
     * so guided setup can tell a genuine parity RESULT apart from a probe that
     * never completed (#2418): a non-2xx response, a fetch exception, or a
     * malformed/empty body is ``ok: false`` — never silently treated as success.
     */
    async _postVerify(name) {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(await this.getAuthHeader()),
            };
            const response = await fetch(this.verifyEndpoint, {
                method: 'POST',
                headers,
                body: JSON.stringify(name ? { name } : {}),
            });
            let data = null;
            try { data = await response.json(); } catch (e) { /* no/invalid body */ }
            if (!response.ok) {
                return {
                    ok: false,
                    data,
                    detail: (data && data.detail) ||
                        `parity probe failed (HTTP ${response.status})`,
                };
            }
            if (!data || typeof data !== 'object') {
                return { ok: false, detail: 'parity probe returned no result body' };
            }
            return { ok: true, data };
        } catch (e) {
            return { ok: false, detail: String(e) };
        }
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

    /**
     * Label a non-universal route with its hidden tradeoff (#2337). A cloud-only
     * model silently degrades ``force_local_only`` / private sessions to keyword
     * search; a local-only model gives cloud sessions nothing. The UI must state
     * this, not bury it. Returns '' when the route is a member of a shared
     * (universal) space — no tradeoff to warn about — or when locality is
     * unknown.
     */
    _routeTradeoff(route) {
        const option = this._universalOption();
        const members = (option && option.members) || [];
        if (members.some(m => m.route === route.id)) return '';
        if (route.is_local === true) {
            return 'local only — cloud sessions fall back to keyword search';
        }
        if (route.is_local === false || route.is_cloud === true) {
            return 'cloud only — private/local sessions fall back to keyword search';
        }
        return '';
    }

    /**
     * True when ``candidateDim`` can't write to the ``columnDim`` column (#2417):
     * both are known and differ. A selection in this state silently pauses every
     * future memory write, so the UI must flag it BEFORE selection. Unknown dims
     * (pre-init / a route that declares none) are treated as "can't tell" → false.
     */
    _dimNeedsMigration(candidateDim, columnDim) {
        return (
            candidateDim != null &&
            columnDim != null &&
            Number(candidateDim) !== Number(columnDim)
        );
    }

    /**
     * The dim a model would be PINNED at for the current column (#2417). A model
     * that advertises Matryoshka (MRL) ``dim_options`` including the column dim
     * can be truncated to fit, so we pin THAT dim — not its native one — and the
     * write succeeds. Otherwise the native dim is the only option. ``null`` when
     * nothing is known (a route/model that declares no dims).
     */
    _pinDimForModel(model, columnDim) {
        if (!model) return null;
        const options = Array.isArray(model.dim_options) ? model.dim_options : [];
        if (columnDim != null && options.some(d => Number(d) === Number(columnDim))) {
            return Number(columnDim);
        }
        return model.native_dim != null ? model.native_dim : null;
    }

    /**
     * True when a MODEL can't write into the column (#2417): its effective pin
     * dim — native, or an MRL ``dim_options`` match for the column — still
     * differs from the column dim. An MRL-compatible model is therefore NOT
     * flagged even when its native dim differs.
     */
    _modelNeedsMigration(model, columnDim) {
        return this._dimNeedsMigration(this._pinDimForModel(model, columnDim), columnDim);
    }

    /** Populate the explicit route <select> from embedding-capable routes. */
    _renderRoutes() {
        if (!this.routeSelect) return;
        const routes = (this.getEmbeddingRoutes() || []).map(r => ({
            ...r,
            id: `${r.vendor}:${r.route}`,
        }));
        const configured = this.settings && this.settings.embedding_route;
        // #2417 — the column dim any picked route must write into.
        const columnDim = this.settings && this.settings.kestrel_embedding_dim;
        this.routeSelect.innerHTML = routes.map(r => {
            const id = r.id;
            const tradeoff = this._routeTradeoff(r);
            const base = r.label || id;
            let label = tradeoff ? `${base} — ${tradeoff}` : base;
            // A route whose resolved dim can't write into the column would break
            // every future memory write — mark it before selection, mirroring
            // the cloud-only tradeoff labels (#2337).
            if (this._dimNeedsMigration(r.embedding_dim, columnDim)) {
                label += ` — ${r.embedding_dim}-dim, needs migration`;
            }
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
        // The per-route model picker only makes sense in explicit mode; its
        // population/visibility (needs a discovered catalog) is finalized in
        // _renderModelPicker, but hide it eagerly outside explicit mode.
        if (this.modelSelect && this.mode !== 'explicit') {
            this.modelSelect.style.display = 'none';
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
                const modePart = this.mode === 'auto' ? 'Auto — ' : '';
                this.dimReadout.textContent = `${modePart}${modelPart}${dim} dimensions`;
            } else if (this.mode === 'off') {
                // Deliberate off (#2287) — a choice, not degradation.
                this.dimReadout.textContent = 'Embeddings off — keyword search only';
            } else if (this.mode === 'auto') {
                // #2418 — read as ONE causal story, not a bare fragment: why
                // there's no embedding (the chat provider can't embed) and what
                // to do about it (set up Universal or pick a provider).
                this.dimReadout.textContent =
                    'Auto — your chat provider can\'t embed → keyword search. ' +
                    'Set up Universal above, or pick a provider.';
            } else {
                // Explicit route that resolved nothing — keyword search only.
                this.dimReadout.textContent = 'No embedding provider — keyword search only';
            }
        }

        if (this.warningEl) {
            // #2417 — an agent already in the broken state (selected provider's
            // dim ≠ the column dim) has its memory writes silently paused right
            // now: every write hits the storage dim guard and persists without a
            // vector. Surface the server's ``dim_write_status`` as a first-class
            // popover state, stronger than the softer #2264 re-embed hint.
            const blocked = s.dim_write_blocked === true;
            const mismatch = dim != null && deploymentDim != null && dim !== deploymentDim;
            if (blocked) {
                this.warningEl.style.display = '';
                this.warningEl.textContent =
                    s.dim_write_status ||
                    `Selected provider cannot write — memory vectors paused ` +
                    `(resolves ${dim}-dim, columns are ${deploymentDim}-dim).`;
            } else if (mismatch) {
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
        if (!this.isCurrent()) return;
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
        if (!this.isCurrent()) return;
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
                // #2417 — carry the write-blocked status so the popover can show
                // "memory vectors paused" immediately after a change, not only
                // after a full reload.
                dim_write_blocked: data.dim_write_blocked,
                dim_write_status: data.dim_write_status,
                shared_space: data.shared_space,
                // Changing the route can create stale memories; the POST echoes
                // the authoritative count so the "Re-embed N memories" button
                // renders immediately without waiting for a full reload (#2338).
                // ``stale_rows`` counts only actionable rows; rows with no
                // embeddable text are surfaced separately (#2426).
                stale_rows: data.stale_rows,
                unembeddable_rows: data.unembeddable_rows,
            };
            this.mode = embeddingModeForRoute(this.settings.embedding_route);
            this._render();
            this.invalidateModelCatalog();
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
