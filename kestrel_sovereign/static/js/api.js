import {
    createApiClient,
    createBearerTokenAuthProvider,
    createKestrelStandaloneAuthProvider,
    CAPABILITY_KEYS,
} from './api_client.mjs';

// Hosts embedding Kestrel UI can override auth/fetch by setting
// window.KESTREL_UI_CONFIG before this module loads. The shape is:
//   {
//     authProvider?: <object implementing the provider contract>,
//     bearerToken?: { getToken, onUnauthenticated?, headerName?, tokenPrefix? },
//     fetchFn?: (url, options) => Response,
//     capabilities?: {
//       // Boolean for simple on/off — missing keys default to true
//       chrome: false,
//       chat: false,
//       multi_agent: false,
//       spawn: false,
//       featureStore: false,
//       voice: false,
//       audit: false,
//       permissions: false,
//       // Object for partial / nested support (e.g. only some key tiers apply)
//       keys: { agent: false, user: true, platform: true },
//     },
//   }
// When nothing is configured, Kestrel UI behaves exactly like the standalone
// server: bootstrap an API key from /api/auth/key, fall back to /auth/me, then
// /auth/login. See #863 for the embedding contract; #879 for capabilities.
const config = (typeof globalThis !== 'undefined' && globalThis.KESTREL_UI_CONFIG) || {};

let authProvider = config.authProvider || null;
if (!authProvider && config.bearerToken) {
    authProvider = createBearerTokenAuthProvider(config.bearerToken);
}

const API = createApiClient({
    authProvider,
    fetchFn: config.fetchFn || globalThis.fetch,
    capabilities: config.capabilities || null,
});

export default API;
export {
    createApiClient,
    createBearerTokenAuthProvider,
    createKestrelStandaloneAuthProvider,
    CAPABILITY_KEYS,
};
