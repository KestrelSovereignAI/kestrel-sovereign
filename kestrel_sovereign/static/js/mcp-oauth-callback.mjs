// Capture once and remove secrets from browser history before imports or I/O.
export function takeCallback(browser) {
    const query = new URLSearchParams(browser.location.search);
    browser.history.replaceState(null, '', browser.location.pathname);
    const one = (key) => {
        const values = query.getAll(key);
        if (values.length !== 1 || !values[0]) throw new Error('Invalid callback');
        return values[0];
    };
    const state = one('state');
    const key = `kestrel.mcp-oauth.${state}`;
    const raw = browser.sessionStorage.getItem(key);
    browser.sessionStorage.removeItem(key);
    if (!raw || raw.length > 4096) throw new Error('Missing Console authorization');
    const hint = JSON.parse(raw);
    if (!hint || typeof hint !== 'object' || Array.isArray(hint)
        || Object.keys(hint).some(key => !['agentPrefix', 'operationId', 'returnTo'].includes(key))
        || typeof hint.operationId !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(hint.operationId)
        || typeof hint.agentPrefix !== 'string'
        || !/^(?:|\/api\/(?:agents\/[^/?#%\\]+|agent-routes\/[A-Za-z0-9_-]+))$/.test(hint.agentPrefix)
        || hint.agentPrefix.split('/').some(segment => segment === '.' || segment === '..')
        || /[\s\u0000-\u001f\u007f]/.test(hint.agentPrefix)
        || typeof hint.returnTo !== 'string' || hint.returnTo !== '/') {
        throw new Error('Invalid Console authorization');
    }
    if (query.has('code') === query.has('error')) throw new Error('Invalid callback');
    // RFC 6749 requires ignoring extensions, while parameters remain unique.
    if ([...query.keys()].some(key => query.getAll(key).length !== 1)) {
        throw new Error('Invalid callback');
    }
    const payload = { state, iss: one('iss') };
    payload[query.has('code') ? 'code' : 'error'] = one(query.has('code') ? 'code' : 'error');
    return { hint, payload };
}

export async function completeCallback(browser, loadApi) {
    const status = browser.document.getElementById('oauth-status');
    try {
        const { hint, payload } = takeCallback(browser);
        const api = await loadApi();
        await api.init();
        // A local routing hint never selects credentials. Core checks the exact
        // original caller, agent, operation, state and live privacy policy.
        await api.requestHost(`${hint.agentPrefix}/api/mcp/oauth/authorizations/${hint.operationId}/callback`, {
            method: 'POST', body: JSON.stringify(payload),
        }, true);
        status.textContent = 'Authorization response received. Return to Console to check the connection status.';
        const link = browser.document.getElementById('oauth-return');
        link.href = hint.returnTo;
        link.hidden = false;
    } catch {
        // Callback parameters and upstream errors must never reach page text.
        status.textContent = 'Authorization could not be completed. Restart from your Console tab; do not replay this callback.';
    }
}

if (typeof window !== 'undefined') {
    void completeCallback(window, async () => (await import('./api.js')).default);
}
