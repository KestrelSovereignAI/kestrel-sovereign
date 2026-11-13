const HANDLE = /^[A-Za-z0-9_-]{43}$/;
const PREFIX = /^(?:|\/api\/agent-routes\/[A-Za-z0-9_-]+)$/;
const SCOPES = new Map([
    ['memory:read', 'Read and search memory'],
    ['memory:write', 'Save information to memory'],
    ['conversation:write', 'Save exchanges from this connector'],
]);

export function takeConsentRequest(browser, apiPrefix) {
    const query = new URLSearchParams(browser.location.search);
    browser.history.replaceState(null, '', browser.location.pathname);
    if (typeof apiPrefix !== 'string' || !PREFIX.test(apiPrefix)) throw Error('Invalid Console route');
    const storageKey = `kestrel.mcp-server-consent.${apiPrefix}`;
    let handle;
    if ([...query.keys()].length) {
        browser.sessionStorage.removeItem(storageKey);
        if ([...query.keys()].some(key => key !== 'request') || query.getAll('request').length !== 1) {
            throw Error('Invalid consent request');
        }
        handle = query.get('request');
        if (!HANDLE.test(handle ?? '')) throw Error('Invalid consent request');
        browser.sessionStorage.setItem(storageKey, handle);
    } else {
        handle = browser.sessionStorage.getItem(storageKey);
        if (!HANDLE.test(handle ?? '')) {
            browser.sessionStorage.removeItem(storageKey);
            throw Error('Missing consent request');
        }
    }
    return { handle, storageKey };
}

export class MCPConsentFlow {
    constructor(browser, apiPrefix, loadApi) {
        this.browser = browser;
        this.prefix = apiPrefix;
        this.loadApi = loadApi;
        this.request = takeConsentRequest(browser, apiPrefix);
        this.loading = null;
        this.details = null;
        this.api = null;
        this.decided = false;
    }

    load() {
        if (!this.loading) this.loading = this._load();
        return this.loading;
    }

    async _load() {
        this.api = await this.loadApi();
        await this.api.init();
        const details = await this.api.requestHost(
            `${this.prefix}/api/mcp/server/consent/requests/${this.request.handle}`, { method: 'GET' }, true);
        if (!details || typeof details.client_name !== 'string' || !details.client_name
            || typeof details.agent_name !== 'string' || !details.agent_name
            || typeof details.resource !== 'string' || typeof details.redirect_uri !== 'string'
            || !Number.isFinite(details.expires_at) || details.expires_at <= 0
            || !Array.isArray(details.scopes) || !details.scopes.length
            || new Set(details.scopes).size !== details.scopes.length
            || !details.scopes.every(scope => SCOPES.has(scope))) throw Error('Invalid consent description');
        const callback = new URL(details.redirect_uri);
        if (callback.protocol !== 'https:' || callback.username || callback.password || callback.hash) {
            throw Error('Invalid registered callback');
        }
        this.details = Object.freeze({ ...details, scopes: Object.freeze([...details.scopes]) });
        return this.details;
    }

    async decide(decision, scopes = []) {
        if (this.decided || !this.details || !this.api) throw Error('Consent request is unavailable');
        if (!['approve', 'deny'].includes(decision) || !Array.isArray(scopes)
            || new Set(scopes).size !== scopes.length
            || !scopes.every(scope => this.details.scopes.includes(scope))
            || (decision === 'approve' ? !scopes.length : scopes.length !== 0)) throw Error('Invalid consent selection');
        // Intent consumes the local hint before I/O. An uncertain COMMIT must
        // never become an automatic replay of an approval or denial.
        this.decided = true;
        this.browser.sessionStorage.removeItem(this.request.storageKey);
        const value = await this.api.requestHost(
            `${this.prefix}/api/mcp/server/consent/requests/${this.request.handle}/decision`,
            { method: 'POST', body: JSON.stringify({ decision, scopes: [...scopes] }) }, true);
        const separator = this.details.redirect_uri.includes('?') ? '&' : '?';
        if (typeof value?.redirect_url !== 'string'
            || !value.redirect_url.startsWith(this.details.redirect_uri + separator)) throw Error('Invalid authorization response');
        this.browser.location.assign(value.redirect_url);
    }
}

export async function runConsentPage(browser, loadApi) {
    const document = browser.document;
    const root = document.getElementById('mcp-consent');
    const status = document.getElementById('consent-status');
    const allow = document.getElementById('consent-allow');
    const deny = document.getElementById('consent-deny');
    allow.disabled = deny.disabled = true;
    try {
        const flow = new MCPConsentFlow(browser, root.dataset.apiPrefix, loadApi);
        const details = await flow.load();
        document.getElementById('consent-client').textContent = details.client_name;
        document.getElementById('consent-agent').textContent = details.agent_name;
        document.getElementById('consent-resource').textContent = details.resource;
        const list = document.getElementById('consent-scopes');
        const choices = details.scopes.map(scope => {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = true;
            input.value = scope;
            label.append(input, document.createTextNode(SCOPES.get(scope)));
            list.append(label);
            return input;
        });
        const submit = async decision => {
            const selected = decision === 'approve' ? choices.filter(input => input.checked).map(input => input.value) : [];
            if (decision === 'approve' && !selected.length) {
                status.textContent = 'Choose at least one permission, or select Don’t allow.';
                return;
            }
            allow.disabled = deny.disabled = true;
            status.textContent = 'Completing authorization…';
            try {
                await flow.decide(decision, selected);
            } catch {
                status.textContent = 'Authorization could not be confirmed. Reconnect from the connector to start a new request.';
            }
        };
        document.getElementById('consent-form').addEventListener('submit', event => {
            event.preventDefault();
            void submit('approve');
        });
        deny.addEventListener('click', () => { void submit('deny'); });
        status.textContent = 'Allow this connector to use the selected permissions?';
        allow.disabled = deny.disabled = false;
    } catch (error) {
        status.textContent = error?.status === 403
            ? 'This authorization requires the owner’s durable API key. Sign in with that key through Console, then reconnect from the connector.'
            : 'This authorization request is unavailable. Sign in through Console, then reconnect from the connector.';
    }
    const link = document.getElementById('consent-return');
    link.href = '/';
}

if (typeof window !== 'undefined') {
    void runConsentPage(window, async () => (await import('./api.js')).default);
}
