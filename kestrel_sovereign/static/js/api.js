import {
    createApiClient,
    createBearerTokenAuthProvider,
    createKestrelStandaloneAuthProvider,
} from './api_client.mjs';

// Hosts embedding Kestrel UI can override auth/fetch by setting
// window.KESTREL_UI_CONFIG before this module loads. The shape is:
//   {
//     authProvider?: <object implementing the provider contract>,
//     bearerToken?: { getToken, onUnauthenticated?, headerName?, tokenPrefix? },
//     fetchFn?: (url, options) => Response,
//   }
// When nothing is configured, Kestrel UI behaves exactly like the standalone
// server: bootstrap an API key from /api/auth/key, fall back to /auth/me, then
// /auth/login. See #863 for the embedding contract.
const config = (typeof globalThis !== 'undefined' && globalThis.KESTREL_UI_CONFIG) || {};

let authProvider = config.authProvider || null;
if (!authProvider && config.bearerToken) {
    authProvider = createBearerTokenAuthProvider(config.bearerToken);
}

const API = createApiClient({
    authProvider,
    fetchFn: config.fetchFn || globalThis.fetch,
});

export default API;
export {
    createApiClient,
    createBearerTokenAuthProvider,
    createKestrelStandaloneAuthProvider,
};
