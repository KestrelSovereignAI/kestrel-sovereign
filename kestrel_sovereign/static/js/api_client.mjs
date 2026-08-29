/**
 * Kestrel Sovereign Console - Testable API Client
 *
 * Issues calls to canonical Kestrel paths (e.g. /api/identity, /agent/invoke).
 * Hosts that embed Kestrel UI under a different URL shape are responsible for
 * routing those canonical paths back to the right backend (e.g. by mounting
 * Kestrel UI under a path prefix and forwarding /api/* to a per-agent
 * subprocess). See issue #863 for the rationale.
 *
 * Auth is delegated to an `authProvider` so a host can supply its own
 * (e.g. a JWT it minted) without modifying Kestrel. The default provider
 * preserves the standalone Kestrel-server behavior: try /api/auth/key,
 * fall back to /auth/me, redirect to /auth/login if both fail.
 */

const HOST_LEVEL_AGENTS_RE = /^\/api\/agents\/[^/]+\/(start|stop|status|logs)/;
// A bare ``/api/agents/{name}`` (no further path segments) is the multi-agent
// manager's own resource — create/delete of an agent (#2208), not a per-agent
// proxy target. It must NOT be host-agent-prefixed into
// ``/api/agents/{selected}/api/agents/{name}``.
const HOST_LEVEL_AGENT_RESOURCE_RE = /^\/api\/agents\/[^/]+(\?.*)?$/;

// Host-scoped surfaces (#2293): the multi-agent host serves host-feature routes
// at the host ROOT (``/api/host/…`` for the host CSRF + UI-contributions
// endpoints, ``/host/…`` for host-feature static assets). These must NEVER be
// host-agent-prefixed — they belong to the host, not the selected agent — or
// they'd be proxied into ``/api/agents/{selected}/api/host/…`` and 404.
const HOST_ROOT_RE = /^\/(api\/host|host)(\/|\?|$)/;

// HTTP methods that mutate state and therefore require a CSRF token when the
// caller is cookie-authenticated. Mirrors the server's STATE_CHANGING_METHODS
// (kestrel_sovereign/security/csrf.py). Safe methods carry no CSRF header.
const STATE_CHANGING_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

// Header the console echoes the double-submit CSRF token in. Mirrors the
// server's CSRF_HEADER_NAME.
const CSRF_HEADER_NAME = 'X-CSRF-Token';

const MAX_ERROR_TEXT_LENGTH = 1000;
const MAX_ERROR_DETAILS = 20;
const MAX_ERROR_CODE_LENGTH = 128;

function isAbortError(error) {
    return !!(error && error.name === 'AbortError');
}

function throwIfAborted(signal) {
    if (!signal?.aborted) return;
    if (signal.reason) throw signal.reason;
    const error = new Error('The operation was aborted.');
    error.name = 'AbortError';
    throw error;
}

function normalizeCorrelationId(value) {
    if (typeof value !== 'string') return null;
    const trimmed = value.trim();
    if (!trimmed || trimmed.length > 128) return null;
    return /^[A-Za-z0-9._:-]+$/.test(trimmed) ? trimmed : null;
}

function normalizeErrorCode(value) {
    if (typeof value !== 'string') return null;
    const trimmed = value.trim();
    if (!trimmed || trimmed.length > MAX_ERROR_CODE_LENGTH) return null;
    return /^[A-Za-z0-9._:-]+$/.test(trimmed) ? trimmed : null;
}

function normalizeUserText(value) {
    if (typeof value !== 'string') return '';
    // An upstream proxy may place an entire HTML error page in a nominal JSON
    // message field. Treat document/script/style markup as non-displayable,
    // just as the raw-text response path does, rather than surfacing fragments.
    if (looksLikeHtml(value)) return '';
    let text = value
        .slice(0, MAX_ERROR_TEXT_LENGTH * 4)
        .replace(/<\/?[A-Za-z][^>]*(?:>|$)/g, ' ')
        .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    // Error payloads sometimes contain exception strings from upstream SDKs.
    // Redact common credential assignments before anything becomes visible.
    text = text
        .replace(/\b(authorization)\s*[:=]\s*(?:bearer\s+)?\S+/gi, '$1: [redacted]')
        .replace(/\b(api[ _-]?key|access[ _-]?token|password|secret)\s*[:=]\s*\S+/gi, '$1: [redacted]')
        .replace(/\bbearer\s+[A-Za-z0-9._~+/-]+=*/gi, 'Bearer [redacted]')
        .replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, '[redacted]');
    return text.slice(0, MAX_ERROR_TEXT_LENGTH);
}

function looksLikeHtml(value) {
    return typeof value === 'string' && /<\s*(?:!doctype|html|head|body|script|style)\b/i.test(value);
}

function firstUserText(...values) {
    for (const value of values) {
        const text = normalizeUserText(value);
        if (text) return text;
    }
    return '';
}

function normalizeDetailItem(item) {
    if (typeof item === 'string') {
        const message = normalizeUserText(item);
        return message ? { message } : null;
    }
    if (!item || typeof item !== 'object' || Array.isArray(item)) return null;

    const rawLocation = item.location ?? item.loc;
    const rawLocationParts = Array.isArray(rawLocation)
        ? rawLocation
        : (typeof rawLocation === 'string' || typeof rawLocation === 'number'
            ? [rawLocation]
            : []);
    const locationParts = rawLocationParts
        .filter((part) => typeof part === 'string' || typeof part === 'number')
        .map((part) => normalizeUserText(String(part)))
        .filter(Boolean);
    const message = firstUserText(
        item.message,
        item.msg,
        item.reason,
        typeof item.detail === 'string' ? item.detail : '',
        typeof item.error === 'string' ? item.error : '',
        item.title,
    );
    if (!message) return null;

    const normalized = { message };
    if (locationParts.length) normalized.location = locationParts.join('.');
    const code = normalizeErrorCode(item.code) || normalizeErrorCode(item.type);
    if (code) normalized.code = code;
    return normalized;
}

function normalizeDetails(value) {
    const items = Array.isArray(value) ? value : [value];
    return items
        .slice(0, MAX_ERROR_DETAILS)
        .map(normalizeDetailItem)
        .filter(Boolean);
}

function detailDisplay(detail) {
    return detail.location ? `${detail.location}: ${detail.message}` : detail.message;
}

/**
 * Typed error returned for every non-successful HTTP response.
 *
 * `body` is retained for structured consumers such as upgrade/tier gates.
 * `message` and `details` are separately normalized for safe user display.
 */
export class ApiError extends Error {
    constructor({ status, body, code = null, message, details = [], correlationId = null }) {
        const safeCorrelationId = normalizeCorrelationId(correlationId);
        const baseMessage = normalizeUserText(message) || `HTTP ${status}`;
        super(safeCorrelationId
            ? `${baseMessage} (Reference: ${safeCorrelationId})`
            : baseMessage);
        this.name = 'ApiError';
        this.status = status;
        this.body = body;
        this.code = normalizeErrorCode(code);
        this.details = normalizeDetails(details).slice(0, MAX_ERROR_DETAILS);
        this.correlationId = safeCorrelationId;
    }
}

async function readResponseErrorBody(response) {
    if (typeof response.text === 'function') {
        try {
            const raw = await response.text();
            if (!raw || !raw.trim()) return null;
            try {
                return JSON.parse(raw);
            } catch (_) {
                return raw;
            }
        } catch (error) {
            if (isAbortError(error)) throw error;
            // Some test doubles and non-standard fetch implementations expose
            // json() but no usable text() body. Fall through to that path.
        }
    }
    if (typeof response.json === 'function') {
        try {
            return await response.json();
        } catch (error) {
            if (isAbortError(error)) throw error;
            return null;
        }
    }
    return null;
}

/** Parse a failed Fetch Response into the console's single ApiError shape. */
export async function parseResponseError(response, {
    fallbackMessage = '',
    signal = null,
} = {}) {
    throwIfAborted(signal);
    const status = Number.isInteger(response?.status) ? response.status : 0;
    const body = await readResponseErrorBody(response || {});
    // Body readers do not consistently reject with AbortError when callers
    // supply a custom abort reason. Consult the request signal after the await
    // so cancellation is never flattened into an HTTP ApiError.
    throwIfAborted(signal);
    const envelope = body && typeof body === 'object' && !Array.isArray(body)
        && body.error && typeof body.error === 'object' && !Array.isArray(body.error)
        ? body.error
        : null;

    const details = [
        ...normalizeDetails(envelope?.details),
        ...normalizeDetails(body && typeof body === 'object' && !Array.isArray(body)
            ? body.detail
            : null),
    ].filter((detail, index, all) => all.findIndex((candidate) => (
        candidate.message === detail.message
        && candidate.location === detail.location
        && candidate.code === detail.code
    )) === index).slice(0, MAX_ERROR_DETAILS);

    const objectBody = body && typeof body === 'object' && !Array.isArray(body) ? body : null;
    let message = firstUserText(
        envelope?.message,
        objectBody?.message,
        typeof objectBody?.detail === 'string' ? objectBody.detail : '',
        typeof objectBody?.error === 'string' ? objectBody.error : '',
    );

    const detailSummary = details.slice(0, 3).map(detailDisplay).join('; ');
    if (detailSummary && (!message || /^request validation failed\.?$/i.test(message))) {
        const prefix = message || (status === 422 ? 'Request validation failed.' : 'Request failed.');
        message = `${prefix} ${detailSummary}`;
    }
    if (!message && typeof body === 'string' && !looksLikeHtml(body)) {
        message = normalizeUserText(body);
    }
    message = message
        || normalizeUserText(fallbackMessage)
        || normalizeUserText(response?.statusText)
        || `HTTP ${status}`;

    let correlationId = null;
    try {
        correlationId = normalizeCorrelationId(response?.headers?.get?.('X-Correlation-ID'));
    } catch (_) { /* malformed/non-standard headers — use the envelope fallback */ }
    correlationId = correlationId || normalizeCorrelationId(envelope?.correlation_id);

    const code = normalizeErrorCode(envelope?.code)
        || normalizeErrorCode(objectBody?.code);

    return new ApiError({ status, body, code, message, details, correlationId });
}

// Canonical list of known UI capability keys (#879, #2041).
//
// Two classes live here:
//   * core/static — host-chrome + console capabilities with no backing feature
//     (chrome, chat, conversations, …). A host opts out by setting the key
//     ``false`` on ``KESTREL_UI_CONFIG.capabilities``; missing keys default to
//     ``true`` so standalone Kestrel renders unchanged.
//   * feature-backed (#2041) — voice, spawn, memory, … . These are NO LONGER
//     static defaults: their effective value is *derived* from the enabled
//     feature set, computed server-side and delivered via
//     ``KESTREL_UI_CONFIG.featureCapabilities`` (or ``GET /api/ui/capabilities``).
//     A new feature gains a working ``hasCapability(<name>)`` with zero edits
//     here. The entries below stay only as a canonical-name catalogue.
//
// Object-shaped values are supported via dot-paths (e.g. ``keys.agent``):
// any sub-key absent from the host config is treated as ``true``.
export const CAPABILITY_KEYS = Object.freeze({
    chrome: true,         // console brand/nav chrome; set false for chat-only embeds
    chat: true,
    identity: true,
    constitution: true,
    privacy: true,
    memory: true,
    tasks: true,
    sovereignty: true,
    storage: true,
    wallet: true,
    conversations: true,
    keys: true,           // object: { agent, user, platform }
    audit: true,
    permissions: true,
    multi_agent: true,        // host-level: "Other agents" sidebar (/api/agents)
    spawn: true,
    featureStore: true,
    metrics: true,
    voice: true,
});

function getRequiredDependency(name, value) {
    if (!value) {
        throw new Error(`API client requires ${name}`);
    }
    return value;
}

export function isHostLevelEndpoint(endpoint) {
    if (endpoint === '/api/agents' || endpoint.startsWith('/api/agents?')) return true;
    if (HOST_LEVEL_AGENTS_RE.test(endpoint)) return true;
    if (HOST_LEVEL_AGENT_RESOURCE_RE.test(endpoint)) return true;
    if (HOST_ROOT_RE.test(endpoint)) return true;
    if (endpoint === '/api/auth/key' || endpoint.startsWith('/api/auth/')) return true;
    if (endpoint === '/health') return true;
    return false;
}

export function applyHostAgentPrefix(endpoint, selectedHostAgent) {
    if (!selectedHostAgent || isHostLevelEndpoint(endpoint)) {
        return endpoint;
    }
    return `/api/agents/${encodeURIComponent(selectedHostAgent)}${endpoint}`;
}

// Build a host-ROOT URL for a host-scoped endpoint (#2293). Host features and
// host management routes are served by the host itself, so the path is used
// verbatim — never host-agent-prefixed. Exported for symmetry with
// applyHostAgentPrefix and for unit tests.
export function buildHostUrl(endpoint) {
    return endpoint;
}

export function createKestrelStandaloneAuthProvider({
    fetchFn,
    sessionStorage,
    location,
    logger,
} = {}) {
    const fetchImpl = getRequiredDependency('fetch', fetchFn);
    const sessionStore = getRequiredDependency('sessionStorage', sessionStorage);
    const locationRef = getRequiredDependency('location', location);
    const log = getRequiredDependency('console', logger);

    let apiKey = null;
    let oauthSession = false;
    let bootstrapDisabled = false;

    async function bootstrapApiKey() {
        try {
            const resp = await fetchImpl('/api/auth/key');
            if (resp.ok) {
                const data = await resp.json();
                apiKey = data.key;
                sessionStore.setItem('kestrel_api_key', apiKey);
                return 'ok';
            }
            if (resp.status === 401 || resp.status === 404 || resp.status === 403) {
                bootstrapDisabled = true;
                return 'disabled';
            }
            log.error('Failed to get API key:', resp.status);
            return 'error';
        } catch (error) {
            log.error('Failed to initialize authentication:', error);
            return 'error';
        }
    }

    return {
        async ensureAuthenticated() {
            const params = new URLSearchParams(locationRef.search || '');
            if (params.get('key')) {
                apiKey = params.get('key');
                sessionStore.setItem('kestrel_api_key', apiKey);
                log.log('API key set from URL parameter');
                return;
            }

            apiKey = sessionStore.getItem('kestrel_api_key');
            if (apiKey) {
                log.log('Using cached API key from sessionStorage');
                return;
            }

            const status = await bootstrapApiKey();
            if (status === 'ok') {
                log.log('API key retrieved and cached');
                return;
            }
            if (status === 'disabled') {
                log.log('API key bootstrap unavailable — checking OAuth session');
            }

            try {
                const meResp = await fetchImpl('/auth/me');
                if (meResp.ok) {
                    const user = await meResp.json();
                    oauthSession = true;
                    log.log(`Authenticated via OAuth: ${user.email}`);
                    return;
                }
            } catch (error) {
                // OAuth session check failed; fall through to redirect/no-auth.
            }

            if (!apiKey && !oauthSession) {
                if (bootstrapDisabled) {
                    log.warn('OAuth required — redirecting to login');
                    locationRef.href = '/auth/login';
                } else {
                    log.warn('No authentication available');
                }
            }
        },

        applyAuth(headers) {
            if (apiKey) {
                return { ...headers, 'X-API-Key': apiKey };
            }
            return headers;
        },

        async onUnauthorized() {
            log.warn('Authentication failed - refreshing API key...');
            sessionStore.removeItem('kestrel_api_key');
            apiKey = null;

            const status = await bootstrapApiKey();
            if (status === 'ok') {
                log.log('API key refreshed - retrying request');
                return 'refreshed';
            }

            if (oauthSession) {
                log.warn('OAuth session expired — redirecting to login');
                locationRef.href = '/auth/login';
                return 'redirected';
            }

            return 'failed';
        },

        getApiKey() {
            return apiKey;
        },
    };
}

export function createBearerTokenAuthProvider({
    getToken,
    onUnauthenticated,
    headerName = 'Authorization',
    tokenPrefix = 'Bearer ',
} = {}) {
    if (typeof getToken !== 'function') {
        throw new Error('BearerTokenAuthProvider requires getToken()');
    }

    return {
        async ensureAuthenticated() {
            const token = await getToken();
            if (!token) {
                if (typeof onUnauthenticated === 'function') {
                    await onUnauthenticated();
                }
                throw new Error('Bearer token unavailable');
            }
        },

        async applyAuth(headers) {
            const token = await getToken();
            if (!token) return headers;
            return { ...headers, [headerName]: `${tokenPrefix}${token}` };
        },

        async onUnauthorized() {
            if (typeof onUnauthenticated === 'function') {
                await onUnauthenticated();
                return 'redirected';
            }
            return 'failed';
        },
    };
}

// Resolve a dot-path against the host-supplied capabilities map.  Returns
// ``true`` when the key is absent (default-on), ``false`` when explicitly
// disabled at any segment.  Boolean leaves win immediately; object leaves
// recurse on the next segment.  Exported for unit tests.
export function resolveCapability(capabilities, path) {
    if (!path) return true;
    const segments = String(path).split('.');
    let cursor = capabilities;
    for (const seg of segments) {
        if (cursor === null || cursor === undefined) return true;
        if (typeof cursor === 'boolean') return cursor;
        if (typeof cursor !== 'object') return true;
        if (!(seg in cursor)) return true;
        cursor = cursor[seg];
    }
    if (cursor === null || cursor === undefined) return true;
    if (typeof cursor === 'boolean') return cursor;
    // Object leaf with no matching path segment — feature is enabled overall;
    // its subkeys gate sub-sections.  Treat the parent as enabled.
    if (typeof cursor === 'object') {
        // If every sub-key is explicitly false, the parent is effectively off.
        const values = Object.values(cursor);
        if (values.length > 0 && values.every((v) => v === false)) return false;
        return true;
    }
    return true;
}

// #2041: merge the server-derived feature capability map over host overrides.
//
// The merge is ASYMMETRIC, because a disabled feature has no UI assets to gate
// into:
//   * Feature-backed keys (present in ``featureCaps``): the server's *disabled*
//     verdict is authoritative. A host override may only force a capability
//     OFF; a force-TRUE on a disabled feature is ignored (with a console
//     warning). Effective = ``serverEnabled && (hostOverride !== false)``.
//   * Core/static keys (only in host overrides): host override wins, exactly as
//     before. Keys absent from both stay default-on via ``resolveCapability``.
//
// Exported for unit tests.
export function mergeCapabilities(featureCaps, hostOverrides, logger = globalThis.console) {
    const features = (featureCaps && typeof featureCaps === 'object') ? featureCaps : {};
    const overrides = (hostOverrides && typeof hostOverrides === 'object') ? hostOverrides : {};
    const merged = {};

    for (const [key, enabled] of Object.entries(features)) {
        const override = overrides[key];
        if (enabled) {
            merged[key] = override === false ? false : true;
        } else {
            if (override === true) {
                logger?.warn?.(
                    `[capabilities] ignoring host override forcing disabled feature '${key}' on — `
                    + 'a disabled feature has no UI assets to gate into',
                );
            }
            merged[key] = false;
        }
    }

    // Core/static + nested host overrides flow through untouched. Feature-backed
    // keys were already resolved above and must not be re-overwritten here.
    for (const [key, val] of Object.entries(overrides)) {
        if (key in merged) continue;
        merged[key] = val;
    }

    return merged;
}

export function createApiClient({
    fetchFn = globalThis.fetch,
    sessionStorage = globalThis.sessionStorage,
    location = globalThis.location,
    logger = globalThis.console,
    AbortControllerCtor = globalThis.AbortController,
    TextDecoderCtor = globalThis.TextDecoder,
    authProvider = null,
    capabilities = null,
    featureCapabilities = null,
} = {}) {
    const fetchImpl = getRequiredDependency('fetch', fetchFn);
    const sessionStore = getRequiredDependency('sessionStorage', sessionStorage);
    const locationRef = getRequiredDependency('location', location);
    const log = getRequiredDependency('console', logger);
    const AbortCtor = getRequiredDependency('AbortController', AbortControllerCtor);
    const DecoderCtor = getRequiredDependency('TextDecoder', TextDecoderCtor);

    const auth = authProvider || createKestrelStandaloneAuthProvider({
        fetchFn: fetchImpl,
        sessionStorage: sessionStore,
        location: locationRef,
        logger: log,
    });

    // Capabilities (#879, #2041).  ``capabilities`` carries host overrides
    // (``null``/``{}`` = opted out of nothing); ``featureCapabilities`` is the
    // server-derived feature-backed map.  The effective ``capsMap`` is the
    // merge of the two (see ``mergeCapabilities``); missing keys still default
    // to ``true`` via the resolver.  ``capsMap`` is mutable so a runtime
    // enable/disable can re-derive it without a page reload — read it only via
    // ``hasCapability``/``getCapabilities``.
    const hostOverrides = capabilities && typeof capabilities === 'object' ? capabilities : {};
    let serverFeatureCaps =
        featureCapabilities && typeof featureCapabilities === 'object' ? featureCapabilities : {};
    let capsMap = mergeCapabilities(serverFeatureCaps, hostOverrides, log);

    function emitCapabilitiesChanged() {
        // #2041: notify the (interim) bus so the registry / navigation re-gates
        // without a reload. Dispatched on globalThis so DOM and non-DOM hosts
        // can both subscribe; a no-op where CustomEvent/dispatchEvent are absent.
        try {
            if (typeof CustomEvent === 'function' && typeof globalThis.dispatchEvent === 'function') {
                globalThis.dispatchEvent(
                    new CustomEvent('capabilities:changed', { detail: { capabilities: capsMap } }),
                );
            }
        } catch (_) {
            /* non-DOM environment — nothing subscribes */
        }
    }

    const state = {
        selectedHostAgent: null,
        // Per-agent stream bookkeeping. Keyed by host-agent name (null key
        // for standalone mode). Was a single slot, which clobbered Agent A's
        // controller when a second stream started on Agent B and made the
        // Stop button target the wrong stream.
        streamAbortControllers: new Map(),
        currentStreamRequestIds: new Map(),
        // Effective session_id reported by the most recent /stream or
        // /invoke for each agent. The server resolves this via the 30-
        // min-gap heuristic when the caller passes null, so before this
        // map existed the frontend pane never learned its durable
        // session id and stayed anchored on null indefinitely.
        effectiveSessionIds: new Map(),
        // Cached double-submit CSRF token for host-scoped state-changing
        // requests (#2293). Fetched lazily from GET /api/host/csrf (which also
        // sets the matching cookie) and reused for the page's lifetime. Only
        // cookie-authenticated callers attach it; API-key/bearer callers are
        // CSRF-exempt (a browser never auto-attaches those headers).
        csrfToken: null,
    };
    const hostAgentChangeListeners = new Set();

    // Fetch (once) and cache the host CSRF token. Returns null on failure so a
    // caller degrades to sending no header (the server then 403s a genuinely
    // cookie-authed mutation, surfacing the misconfig rather than masking it).
    async function ensureCsrfToken() {
        if (state.csrfToken) return state.csrfToken;
        try {
            const data = await performRequest(buildHostUrl('/api/host/csrf'), { method: 'GET' });
            if (data && data.csrf_token) state.csrfToken = data.csrf_token;
        } catch (e) {
            log.warn?.(
                '[csrf] failed to fetch host CSRF token',
                e && e.message ? e.message : 'Request failed.',
            );
        }
        return state.csrfToken;
    }

    // Whether the active auth is a machine credential (API key / bearer). Such
    // callers are CSRF-exempt server-side, so we must NOT attach a CSRF header
    // (there may be no cookie to match, which would 403 a valid request).
    function isMachineAuthed() {
        return typeof auth.getApiKey === 'function' && !!auth.getApiKey();
    }

    async function recoverUnauthorized(response, signal = null) {
        // Parse the response before invoking mutable auth state. This retained
        // typed error is the safe fallback if refresh succeeds but rebuilding
        // credentials or starting the retry fails before a new HTTP response
        // can provide a stronger outcome.
        const unauthorizedError = await parseResponseError(response, {
            fallbackMessage: 'Authentication failed - please refresh the page',
            signal,
        });
        // The body read above is asynchronous. A user can cancel while it is
        // in progress even when the reader itself resolves normally; never
        // turn that cancellation into an auth refresh or typed 401.
        throwIfAborted(signal);
        try {
            const recovery = await auth.onUnauthorized();
            // Cancellation can happen while a refresh/redirect callback is
            // pending. Do not rebuild credentials or begin a retry afterward.
            throwIfAborted(signal);
            return {
                recovery,
                unauthorizedError,
            };
        } catch (error) {
            // Preserve cancellation even when the auth provider reports it by
            // throwing, including custom abort reasons without AbortError.name.
            throwIfAborted(signal);
            if (isAbortError(error)) throw error;
            // A failing refresh/redirect callback must not erase the server's
            // actionable 401 status, body, and correlation reference.
            throw unauthorizedError;
        }
    }

    // Single-source the fetch + auth + 401-retry pipeline so both
    // request() and requestForAgent() share it. Without this factor
    // out, an explicit-agent call would either skip auth-refresh or
    // re-implement it (drift risk). The url passed in is FINAL — the
    // caller has already applied host-agent prefixing if needed.
    async function performRequest(url, options = {}, retried = false) {
        let alreadyRetried = retried;
        let pendingUnauthorizedError = null;

        while (true) {
            let headers;
            try {
                headers = await buildHeaders(options.headers);
            } catch (error) {
                throwIfAborted(options.signal);
                if (pendingUnauthorizedError && !isAbortError(error)) {
                    throw pendingUnauthorizedError;
                }
                throw error;
            }
            // A custom AbortController reason is not necessarily named
            // AbortError. Re-check the signal after asynchronous credential
            // setup so an aborted retry never reaches fetch or becomes a 401.
            throwIfAborted(options.signal);
            if (options.body instanceof FormData) {
                delete headers['Content-Type'];
            }

            let response;
            try {
                response = await fetchImpl(url, { ...options, headers });
            } catch (error) {
                throwIfAborted(options.signal);
                if (pendingUnauthorizedError && !isAbortError(error)) {
                    throw pendingUnauthorizedError;
                }
                throw error;
            }
            // A retry response is authoritative. Any ApiError parsed below
            // must propagate instead of being replaced by the original 401.
            pendingUnauthorizedError = null;

            if (response.status === 401 && !alreadyRetried) {
                const { recovery, unauthorizedError } = await recoverUnauthorized(
                    response,
                    options.signal,
                );
                if (recovery === 'redirected') return;
                if (recovery === 'refreshed') {
                    alreadyRetried = true;
                    pendingUnauthorizedError = unauthorizedError;
                    continue;
                }

                throw unauthorizedError;
            }

            if (!response.ok) {
                throw await parseResponseError(response, { signal: options.signal });
            }
            return response.json();
        }
    }

    const client = {
        async init() {
            await auth.ensureAuthenticated();
        },

        async request(endpoint, options = {}, retried = false) {
            const url = applyHostAgentPrefix(endpoint, state.selectedHostAgent);
            return performRequest(url, options, retried);
        },

        // Like request() but pins the host-agent prefix to the explicit
        // `agent` arg instead of state.selectedHostAgent. Used when a
        // caller must reach a specific agent's endpoint while the user
        // is currently viewing a different one — e.g. clicking the
        // sidebar Stop control on Agent A while the chat pane shows
        // Agent B. The caller must pass the canonical /api/agent/...
        // path; routing is applied here so we don't double-prefix.
        async requestForAgent(endpoint, options = {}, agent, retried = false) {
            const url = applyHostAgentPrefix(endpoint, agent);
            return performRequest(url, options, retried);
        },

        // Request a host-ROOT endpoint (#2293). Unlike request(), the URL is
        // never host-agent-prefixed — it targets the host itself (host-feature
        // routes, host CSRF/UI-contributions). For state-changing methods on a
        // cookie-authenticated session it also attaches the double-submit CSRF
        // token the host enforces; API-key/bearer callers are exempt.
        async requestHost(endpoint, options = {}, retried = false) {
            const url = buildHostUrl(endpoint);
            const method = (options.method || 'GET').toUpperCase();
            if (STATE_CHANGING_METHODS.has(method) && !isMachineAuthed()) {
                const token = await ensureCsrfToken();
                if (token) {
                    options = {
                        ...options,
                        headers: { ...(options.headers || {}), [CSRF_HEADER_NAME]: token },
                    };
                }
            }
            return performRequest(url, options, retried);
        },

        health: () => client.request('/health'),
        getAgentInfo: () => client.request('/api/agent/info'),
        getAgents: () => client.request('/api/agents'),
        // Create a fresh top-level (parentless) agent (#2351) — the
        // multi-agent manager's ``POST /api/agents`` (see
        // endpoints/models.py::create_agent). Runs inception + load and returns
        // 409 on a duplicate name / 400 on a bad name; the standalone console's
        // Create Agent dialog surfaces those details inline. Host-level endpoint
        // — never host-agent-prefixed.
        createAgent: (name) => client.request('/api/agents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        }),
        // Native agent deletion (#2208) — the multi-agent manager's
        // ``DELETE /api/agents/{name}`` (see endpoints/models.py). Only meaningful
        // when the server runs in multi-agent mode; the danger-zone section gates
        // this on the ``multi_agent`` capability before ever calling it. Carries
        // the destructive opt-in header like every other destructive UI action
        // (#766) so the demo-isolation rail sees an intentional, user-initiated
        // delete. Host-level endpoint — never host-agent-prefixed.
        deleteAgent: (name) => client.request(`/api/agents/${encodeURIComponent(name)}`, {
            method: 'DELETE',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui' },
        }),
        getIdentity: () => client.request('/api/identity'),
        updateIdentity: (data) => client.request('/api/identity', {
            method: 'PATCH',
            body: JSON.stringify(data),
        }),
        uploadAvatar: (file) => {
            const formData = new FormData();
            formData.append('file', file);
            return client.request('/api/identity/avatar', {
                method: 'POST',
                body: formData,
            });
        },
        // #1662: upload a chat attachment (image or document). Returns a ref
        // {hash, mime, name, kind, size, url}; the bytes stay in the encrypted
        // store. Same multipart pattern as the avatar upload.
        uploadAttachment: (file) => {
            const formData = new FormData();
            formData.append('file', file);
            return client.request('/api/agent/attachments', {
                method: 'POST',
                body: formData,
            });
        },
        setAvatarFromUrl: (url) => client.request('/api/identity/avatar', {
            method: 'POST',
            body: JSON.stringify({ url }),
        }),
        generateAvatar: (description, numOutputs = 2) => client.request('/api/identity/avatar/generate', {
            method: 'POST',
            body: JSON.stringify({ description, num_outputs: numOutputs }),
        }),
        getConstitution: () => client.request('/api/constitution'),
        getPrivacyMode: () => client.request('/api/agent/privacy-mode'),
        // setPrivacyMode is destructive on live agents (#867 gates the
        // endpoint) — a flip into EPHEMERAL primes the leak-purge for the
        // next exit.  The UI carries the opt-in header so the rail sees an
        // intentional change; scripts must opt in explicitly.
        setPrivacyMode: (mode) => client.request('/api/agent/privacy-mode', {
            method: 'POST',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-mode-change' },
            body: JSON.stringify({ mode }),
        }),
        // Apply a privacy-mode change that setPrivacyMode staged with
        // {requires_confirmation:true} — a data-destructive downgrade (e.g.
        // PUBLIC→EPHEMERAL) the server refuses to apply until the user confirms.
        // Carries the same destructive opt-in header.
        confirmPrivacyMode: () => client.request('/api/agent/privacy-mode/confirm', {
            method: 'POST',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-mode-change' },
        }),
        // Discard a staged (requires_confirmation) privacy-mode change when the
        // user declines, so a later confirm can't apply what they cancelled.
        cancelPrivacyMode: () => client.request('/api/agent/privacy-mode/cancel', {
            method: 'POST',
        }),
        getSessions: (limit = 50) => client.request(`/api/sessions?limit=${limit}`),
        getMemories: (nodeType = null, limit = 100) => {
            let url = `/api/memories?limit=${limit}`;
            if (nodeType) url += `&node_type=${nodeType}`;
            return client.request(url);
        },
        deleteMemory: (nodeId) => client.request(`/api/memories/${encodeURIComponent(nodeId)}`, { method: 'DELETE' }),
        getMemoryDetail: (nodeId) => client.request(`/api/memories/${encodeURIComponent(nodeId)}`),
        getIdentityChain: () => client.request('/api/identity-chain'),
        getStorageStats: () => client.request('/api/storage/stats'),
        getSovereigntyExports: () => client.request('/api/sovereignty/exports'),
        exportSovereignty: (tier, encrypt) => client.request('/api/sovereignty/export', { method: 'POST', body: JSON.stringify({ tier, encrypt }) }),
        importSovereignty: (cid) => client.request('/api/sovereignty/import', { method: 'POST', body: JSON.stringify({ cid }) }),
        getSovereigntyFiles: () => client.request('/api/sovereignty/files'),
        getSovereigntyFilePreview: (filename) => client.request(`/api/sovereignty/files/${encodeURIComponent(filename)}/preview`),
        // `cursor` continues a previous page; the response's `next_cursor` is
        // the token for the one after it, and null at the end of the list
        // (#2960). Opaque — it encodes the server's ordering keys, which is not
        // a shape a caller may build or parse.
        getConversations: (decrypt = true, view = 'active', cursor = null) => client.request(
            `/api/conversations?decrypt=${decrypt}&view=${encodeURIComponent(view)}`
            + (cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''),
        ),
        // Full-text search over conversation content + titles. Same endpoint
        // as the list — `q` flips it into search mode and each returned
        // session carries match_count / match_role / match_snippet.
        searchConversations: (query, view = 'active', decrypt = true) => client.request(
            `/api/conversations?decrypt=${decrypt}&view=${encodeURIComponent(view)}&q=${encodeURIComponent(query)}`,
        ),
        getConversation: (sessionId, decrypt = true) => client.request(`/api/conversations/${encodeURIComponent(sessionId)}?decrypt=${decrypt}`),
        renameConversation: (sessionId, name) => client.request(`/api/conversations/${encodeURIComponent(sessionId)}`, {
            method: 'PATCH',
            body: JSON.stringify({ name }),
        }),
        // Recent restart_status lifecycle events for repainting the bubble
        // trail on conversation reload (#1816). Scoped to the session so a
        // restart filed from one conversation never repaints in another.
        getRestartStatusEvents: (sessionId = '', limit = 200) => client.request(
            `/api/restart/status-events?session=${encodeURIComponent(sessionId)}&limit=${encodeURIComponent(limit)}`,
        ),
        newConversation: () => client.request('/api/conversations/new', { method: 'POST' }),
        // Soft-delete moves to Trash, recoverable (#763 / #765).  The
        // X-Kestrel-Allow-Destructive header satisfies the demo-isolation
        // rail (#766) which gates every destructive endpoint behind an
        // explicit opt-in so a stray script can't wipe a live agent.
        deleteMessage: (messageId) => client.request(`/api/conversations/messages/${encodeURIComponent(messageId)}`, {
            method: 'DELETE',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui' },
        }),
        deleteConversation: (sessionId) => client.request(`/api/conversations/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui' },
        }),
        // Trash sub-view surface — list trashed items, restore one out of
        // Trash, or hard-purge with an audit reason.  Purge is irreversible
        // and always carries the destructive header.
        listTrash: (limit = 200) => client.request(`/api/trash?limit=${encodeURIComponent(limit)}`),
        // Archive sub-view surface (#2149) — non-destructive; archive/unarchive
        // simply toggle a session between the active and archived lists. The
        // archived list itself is served by getConversations(view='archived'),
        // the single listing path shared with the active view.
        archiveConversation: (sessionId) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/archive`,
            { method: 'POST' },
        ),
        unarchiveConversation: (sessionId) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/unarchive`,
            { method: 'POST' },
        ),
        restoreConversation: (sessionId) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/restore`,
            { method: 'POST' },
        ),
        purgeConversation: (sessionId, reason = 'user-initiated') => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/purge`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Kestrel-Allow-Destructive': 'user-initiated-purge',
                },
                body: JSON.stringify({ reason }),
            },
        ),
        restoreMessage: (messageId) => client.request(
            `/api/conversations/messages/${encodeURIComponent(messageId)}/restore`,
            { method: 'POST' },
        ),
        purgeMessage: (messageId, reason = 'user-initiated') => client.request(
            `/api/conversations/messages/${encodeURIComponent(messageId)}/purge`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Kestrel-Allow-Destructive': 'user-initiated-purge',
                },
                body: JSON.stringify({ reason }),
            },
        ),
        renameConversation: (sessionId, name) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}`,
            {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            },
        ),
        // Database responses can contain tenant-private data. Pin both the
        // initial request and any auth-refresh retry to the agent selected at
        // dispatch time; client.request() can recapture the current selection
        // on retry and is therefore wrong for this surface.
        getDbTables: (agent = state.selectedHostAgent) =>
            client.requestForAgent('/api/db/tables', {}, agent),
        queryDbTable: (
            table,
            limit = 50,
            offset = 0,
            search = null,
            agent = state.selectedHostAgent,
        ) => {
            let url = `/api/db/tables/${encodeURIComponent(table)}`
                + `?limit=${encodeURIComponent(limit)}&offset=${encodeURIComponent(offset)}`;
            if (search) url += `&search=${encodeURIComponent(search)}`;
            return client.requestForAgent(url, {}, agent);
        },
        getIpfsStatus: () => client.request('/api/ipfs/status'),
        getWallet: () => client.request('/api/wallet'),
        invoke: async (input, model = null, sessionId = null, provider = null) => {
            // Capture dispatchAgent BEFORE the await so the session_id
            // we record on the response is bound to the agent that
            // owned this dispatch — not whichever agent was selected
            // when the response landed.
            const dispatchAgent = state.selectedHostAgent;
            const result = await client.request('/api/agent/invoke', {
                method: 'POST',
                body: JSON.stringify({ input, model, session_id: sessionId, provider }),
            });
            if (result && typeof result === 'object' && result.session_id) {
                state.effectiveSessionIds.set(dispatchAgent, result.session_id);
            }
            return result;
        },
        // Explicit-agent invoke — mirrors stop(id, agent). sendMessage
        // dispatches a chat against a specific agent's pane; if the
        // user switches agents while sendMessage awaits this call, the
        // unprefixed `invoke()` would route to whichever agent is
        // currently selected and the response would land in the wrong
        // pane. invokeForAgent pins the URL to the captured dispatch
        // agent so the request always reaches the agent the chat was
        // sent to.
        invokeForAgent: async (input, model = null, sessionId = null, provider = null, agent) => {
            const opts = {
                method: 'POST',
                body: JSON.stringify({ input, model, session_id: sessionId, provider }),
            };
            const dispatchAgent = agent === undefined ? state.selectedHostAgent : agent;
            const result = agent !== undefined
                ? await client.requestForAgent('/api/agent/invoke', opts, agent)
                : await client.request('/api/agent/invoke', opts);
            if (result && typeof result === 'object' && result.session_id) {
                state.effectiveSessionIds.set(dispatchAgent, result.session_id);
            }
            return result;
        },
        // Two-arg overload: pass `agent` to target a specific agent's
        // /stop endpoint regardless of which agent is currently
        // selected. Without it, a sidebar "Stop Agent A" click while
        // viewing Agent B would route the stop to B's backend (because
        // request() pins to state.selectedHostAgent), aborting client-
        // side but never telling A's server to halt.
        stop: (requestId = null, agent) => {
            const opts = {
                method: 'POST',
                body: JSON.stringify(requestId ? { request_id: requestId } : {}),
            };
            return agent !== undefined
                ? client.requestForAgent('/api/agent/stop', opts, agent)
                : client.request('/api/agent/stop', opts);
        },
        getModels: (options = {}) => {
            const params = new URLSearchParams();
            if (options.featuredOnly !== undefined) params.append('featured_only', options.featuredOnly);
            if (options.category) params.append('category', options.category);
            if (options.providers) params.append('providers', options.providers.join(','));
            if (options.useCache !== undefined) params.append('use_cache', options.useCache);
            const queryString = params.toString();
            return client.request(`/api/models${queryString ? `?${queryString}` : ''}`);
        },
        getStreamAbortController(agent) {
            // Default to the current host agent so existing single-agent
            // call sites keep working unchanged.
            const key = agent === undefined ? state.selectedHostAgent : agent;
            return state.streamAbortControllers.get(key) || null;
        },
        getCurrentStreamRequestId(agent) {
            const key = agent === undefined ? state.selectedHostAgent : agent;
            return state.currentStreamRequestIds.get(key) || null;
        },
        // Effective session_id surfaced by the server's most recent
        // /stream or /invoke for this agent. Returns null until the
        // first response has been received. sendMessage reads this to
        // update pane.sessionId so each agent's pane learns its
        // durable conversation id without relying on the prior
        // pane.sessionId=null + server-side implicit-derive heuristic.
        getEffectiveSessionId(agent) {
            const key = agent === undefined ? state.selectedHostAgent : agent;
            return state.effectiveSessionIds.get(key) || null;
        },
        async *streamInvoke(input, model = null, sessionId = null, provider = null, retried = false, agent, attachments = null, requestId = null) {
            // Pin the dispatch agent. The sixth `agent` parameter lets a
            // caller (sendMessage) capture state.selectedHostAgent at
            // its own dispatch boundary and pass it through, so the user
            // switching agents between sendMessage's capture and
            // streamInvoke's first await can't drift the URL to the
            // wrong backend. The 401 retry loop also retains this value
            // instead of recapturing state.selectedHostAgent — recapturing
            // after an auth refresh could route Agent A's retry to Agent B.
            const dispatchAgent = agent === undefined ? state.selectedHostAgent : agent;

            const clientRequestId = requestId === null ? null : String(requestId);
            if (clientRequestId !== null && (
                clientRequestId.length < 1 || clientRequestId.length > 256
            )) {
                throw new Error('stream request id must be 1-256 characters');
            }
            // Invocation IDs are logical Unicode values, while HTTP headers
            // carry the server's one-pass RFC 3986 wire representation. Keep
            // those identities separate so opaque IDs survive auth retries
            // and response-header comparison without double encoding.
            const wireRequestId = clientRequestId === null
                ? null
                : encodeURIComponent(clientRequestId).replace(
                    /[!'()*]/g,
                    (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`,
                );

            // Chat allocates its cancellation address before opening the fetch.
            // Publish that address before the first await so an immediate Stop
            // cannot widen into an agent-scoped request while auth or response
            // headers are still pending.
            let activeRequestId = clientRequestId;
            if (activeRequestId !== null) {
                state.currentStreamRequestIds.set(dispatchAgent, activeRequestId);
            }

            // Build auth headers BEFORE installing the abort controller in
            // the per-agent map. If buildHeaders() throws (auth provider
            // failure, bearer-token unavailable, etc.) we must not leave a
            // stale controller behind for the next Stop click to fire on.
            const controller = new AbortCtor();
            const signal = controller.signal;
            state.streamAbortControllers.set(dispatchAgent, controller);

            try {
                let headers = await buildHeaders({
                    'Content-Type': 'application/json',
                    ...(wireRequestId !== null
                        ? { 'X-Request-ID': wireRequestId }
                        : {}),
                });
                const url = applyHostAgentPrefix('/api/agent/stream', dispatchAgent);
                const body = JSON.stringify({
                    input, model, session_id: sessionId, provider,
                    // #1662: attachment refs for this turn (uploaded
                    // separately via /api/agent/attachments). Omitted when
                    // there are none so non-attachment turns are unchanged.
                    ...(attachments && attachments.length ? { attachments } : {}),
                });
                let alreadyRetried = retried;
                let pendingUnauthorizedError = null;

                while (true) {
                    throwIfAborted(signal);

                    let response;
                    try {
                        response = await fetchImpl(url, {
                            method: 'POST',
                            headers,
                            body,
                            signal,
                        });
                    } catch (error) {
                        throwIfAborted(signal);
                        if (pendingUnauthorizedError && !isAbortError(error)) {
                            throw pendingUnauthorizedError;
                        }
                        throw error;
                    }
                    // Once the retry receives an HTTP response, that response
                    // supersedes the original 401 regardless of its status.
                    pendingUnauthorizedError = null;

                    if (response.status === 401 && !alreadyRetried) {
                        const { recovery, unauthorizedError } = await recoverUnauthorized(
                            response,
                            signal,
                        );
                        if (recovery === 'redirected') return;
                        if (recovery === 'refreshed') {
                            // Keep the original controller and dispatch target.
                            // A Stop during token refresh must abort this turn,
                            // not allow a recursive retry to resurrect it.
                            alreadyRetried = true;
                            try {
                                headers = await buildHeaders({
                                    'Content-Type': 'application/json',
                                    ...(wireRequestId !== null
                                        ? { 'X-Request-ID': wireRequestId }
                                        : {}),
                                });
                                // applyAuth may complete after Stop aborted the
                                // request with an arbitrary reason. Do not
                                // start the refreshed fetch in that state.
                                throwIfAborted(signal);
                            } catch (error) {
                                throwIfAborted(signal);
                                if (isAbortError(error)) throw error;
                                throw unauthorizedError;
                            }
                            pendingUnauthorizedError = unauthorizedError;
                            continue;
                        }

                        throw unauthorizedError;
                    }

                    if (!response.ok) {
                        throw await parseResponseError(response, { signal });
                    }
                    const responseRequestId = response.headers.get('X-Request-ID');
                    if (
                        clientRequestId !== null
                        && responseRequestId !== null
                        && responseRequestId !== wireRequestId
                    ) {
                        throw new Error('server changed the client-issued stream request id');
                    }
                    activeRequestId = clientRequestId || responseRequestId;
                    if (activeRequestId !== null) {
                        state.currentStreamRequestIds.set(
                            dispatchAgent,
                            activeRequestId,
                        );
                    }
                    // Capture the server-resolved session_id BEFORE the body
                    // streams. sendMessage reads it via getEffectiveSessionId
                    // immediately so pane.sessionId can be set on the very
                    // first turn; subsequent turns send it back as an
                    // explicit value, anchoring the pane to a durable id.
                    const headerSid = response.headers.get('X-Session-Id');
                    if (headerSid) {
                        state.effectiveSessionIds.set(dispatchAgent, headerSid);
                    }
                    const reader = response.body.getReader();
                    const decoder = new DecoderCtor();

                    try {
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            yield decoder.decode(value, { stream: true });
                        }
                    } finally {
                        reader.releaseLock();
                    }
                    return;
                }
            } catch (error) {
                // Re-throw AbortError so sendMessage can distinguish a
                // user-initiated stop from a clean stream end. The old
                // silent return swallowed the signal — sendMessage then
                // toasted "agent finished responding" on a non-visible
                // agent the user had just stopped from the sidebar.
                if (error && error.name === 'AbortError') {
                    log.log('Stream aborted by user');
                }
                throw error;
            } finally {
                // Clear only this dispatch's slot — never the *current* agent's,
                // which may belong to a different in-flight stream.
                if (state.streamAbortControllers.get(dispatchAgent) === controller) {
                    state.streamAbortControllers.delete(dispatchAgent);
                }
                if (
                    activeRequestId !== null
                    && state.currentStreamRequestIds.get(dispatchAgent)
                        === activeRequestId
                ) {
                    state.currentStreamRequestIds.delete(dispatchAgent);
                }
            }
        },
        getApiKey() {
            return typeof auth.getApiKey === 'function' ? auth.getApiKey() : null;
        },
        // Sign an arbitrary headers object with whatever the active auth
        // provider attaches. Use this from any code that has to call fetch()
        // directly instead of going through client.request() — most commonly
        // anything that needs Content-Type control (FormData) or a non-JSON
        // protocol (EventSource preflight, etc.).
        async applyAuth(headers = {}) {
            return await auth.applyAuth({ ...headers });
        },
        parseResponseError,
        setHostAgent(agentName) {
            const previousAgent = state.selectedHostAgent;
            state.selectedHostAgent = agentName;
            if (previousAgent === agentName) return;
            for (const listener of [...hostAgentChangeListeners]) {
                try {
                    listener(agentName, previousAgent);
                } catch (error) {
                    log.error?.('[api] host-agent change listener failed', error);
                }
            }
        },
        getHostAgent() {
            return state.selectedHostAgent;
        },
        onHostAgentChange(listener) {
            if (typeof listener !== 'function') {
                throw new TypeError('onHostAgentChange requires a function');
            }
            hostAgentChangeListeners.add(listener);
            return () => hostAgentChangeListeners.delete(listener);
        },
        isMultiAgentMode() {
            return state.selectedHostAgent !== null;
        },
        buildAgentUrl(path) {
            return applyHostAgentPrefix(path, state.selectedHostAgent);
        },
        // Build a URL pinned to a SPECIFIC agent rather than the currently
        // selected one. Needed when a request must target the agent that owns
        // a notification stream (e.g. a QR image), not whatever pane is mounted
        // now — the selection can change before a late SSE event is handled.
        buildAgentUrlFor(path, agentName) {
            return applyHostAgentPrefix(path, agentName || state.selectedHostAgent);
        },
        getContextStatus: (sessionId = null, { full = false } = {}) => {
            const parts = [];
            if (sessionId) parts.push(`session_id=${encodeURIComponent(sessionId)}`);
            if (full) parts.push('full=true');
            const qs = parts.length ? `?${parts.join('&')}` : '';
            return client.request(`/api/agent/context-status${qs}`);
        },
        _getState() {
            return { ...state };
        },
        // #879: capability gating.  Hosts supply ``capabilities`` on
        // ``KESTREL_UI_CONFIG`` to opt out of panels that don't apply in
        // their embed.  Default is "everything on" so standalone Kestrel and
        // pre-#879 hosts behave exactly as before.  Pass dot-paths for
        // nested checks (e.g. ``hasCapability('keys.agent')``).
        hasCapability(path) {
            return resolveCapability(capsMap, path);
        },
        getCapabilities() {
            return capsMap;
        },
        // #2041: re-derive the capability set from a fresh server-side feature
        // map (host overrides are re-applied with the same precedence), then
        // emit ``capabilities:changed`` so the UI re-gates without a reload.
        applyServerCapabilities(featureCaps, { emit = true } = {}) {
            serverFeatureCaps = featureCaps && typeof featureCaps === 'object' ? featureCaps : {};
            capsMap = mergeCapabilities(serverFeatureCaps, hostOverrides, log);
            if (emit) emitCapabilitiesChanged();
            return capsMap;
        },
        // Fetch the current feature capability map from the server and apply it.
        // Used at boot when the page render did not inject ``featureCapabilities``
        // (e.g. multi-agent host mode) and after a runtime enable/disable.
        async refreshCapabilities({ expectedAgent = state.selectedHostAgent } = {}) {
            // Pin both the initial request and an auth retry to the agent that
            // owned this refresh. A rapid switch must never let a late response
            // replace the newly-selected agent's capability map.
            const requestAgent = expectedAgent;
            try {
                const data = requestAgent
                    ? await client.requestForAgent('/api/ui/capabilities', {}, requestAgent)
                    : await client.request('/api/ui/capabilities');
                if (state.selectedHostAgent !== requestAgent) return capsMap;
                if (data && data.capabilities) {
                    this.applyServerCapabilities(data.capabilities);
                }
            } catch (e) {
                log.warn?.(
                    '[capabilities] refresh failed; keeping current set',
                    e && e.message ? e.message : 'Request failed.',
                );
            }
            return capsMap;
        },
    };

    async function buildHeaders(extraHeaders = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...extraHeaders,
        };
        return await auth.applyAuth(headers);
    }

    return client;
}
