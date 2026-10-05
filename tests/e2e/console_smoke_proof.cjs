// @ts-check
//
// Origin proofs for the Sovereign Console smoke (issue #2682).
//
// `kestrel demo smoke` creates a fresh isolated instance, proves where its
// server's code and database come from, and records those facts in
// `console-smoke-instance.json`. The smoke spec re-checks them here before
// it touches the instance, and checks what the browser observed against
// them. Dependency-free on purpose, like demos/shared/demo_safety.cjs, so the
// fail-closed decisions are unit-tested without a browser or a server
// (tests/frontend/console_smoke_proof.test.mjs).
const fs = require('fs');
const path = require('path');

const {
  assertIsolatedDemoEnv,
  assertOnlyDemoAgents,
  isInsideSandbox,
} = require('../../demos/shared/demo_safety.cjs');

const MANIFEST_ENV = 'KESTREL_CONSOLE_SMOKE_MANIFEST';
const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]']);

/**
 * Read the instance manifest written by `kestrel demo smoke`.
 * @param {string|undefined} [manifestPath]
 * @returns {Record<string, any>}
 */
function loadSmokeManifest(manifestPath = process.env[MANIFEST_ENV]) {
  if (!manifestPath) {
    throw new Error(
      `Refusing to run the Console smoke: ${MANIFEST_ENV} is unset. Launch it with `
      + '`kestrel demo smoke`, which creates the isolated instance and records its origin.',
    );
  }
  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  } catch (e) {
    throw new Error(`Refusing to run the Console smoke: cannot read ${manifestPath} (${e.message}).`);
  }
  if (!manifest || typeof manifest !== 'object') {
    throw new Error(`Refusing to run the Console smoke: ${manifestPath} is not an object.`);
  }
  return manifest;
}

/**
 * Prove, before any network call, that the target is the instance this run
 * created: the URL is the loopback server the runner started on a non-live
 * port, its database lies inside the fresh home, and its server runs this
 * checkout's `kestrel_sovereign`.
 *
 * @param {Record<string, any>} manifest
 * @param {{ baseUrl: string|undefined, dbPath: string|undefined, checkoutRoot: string,
 *           exists?: (p: string) => boolean }} observed
 */
function assertSmokeInstance(manifest, { baseUrl, dbPath, checkoutRoot, exists = fs.existsSync }) {
  if (!baseUrl) {
    throw new Error('Refusing to run the Console smoke: KESTREL_URL is unset.');
  }
  // Not the live port, and launched as an isolated demo-server run.
  assertIsolatedDemoEnv(baseUrl);

  // URL origin.
  const target = new URL(baseUrl);
  const recorded = new URL(String(manifest.base_url || ''), 'invalid://');
  if (target.origin !== recorded.origin) {
    throw new Error(
      `Refusing to run the Console smoke: KESTREL_URL ${target.origin} is not the `
      + `instance the runner started (${manifest.base_url}).`,
    );
  }
  if (!LOOPBACK_HOSTS.has(target.hostname)) {
    throw new Error(`Refusing to run the Console smoke: ${target.hostname} is not a loopback host.`);
  }
  if (target.port !== String(manifest.port)) {
    throw new Error(
      `Refusing to run the Console smoke: KESTREL_URL port ${target.port} is not the `
      + `recorded port ${manifest.port}.`,
    );
  }

  // Database origin.
  const home = path.resolve(String(manifest.home || ''));
  const dataDir = path.resolve(String(manifest.data_dir || ''));
  const db = path.resolve(String(manifest.db_path || ''));
  if (!dbPath || path.resolve(dbPath) !== dataDir) {
    throw new Error(
      `Refusing to run the Console smoke: KESTREL_DB_PATH ${dbPath} is not the `
      + `instance data dir ${dataDir}.`,
    );
  }
  if (!manifest.home || dataDir === home || !isInsideSandbox(home, dataDir)) {
    throw new Error(
      `Refusing to run the Console smoke: data dir ${dataDir} is not inside the `
      + `fresh instance home ${home}.`,
    );
  }
  if (db === dataDir || !isInsideSandbox(dataDir, db)) {
    throw new Error(
      `Refusing to run the Console smoke: database ${db} is not inside the data dir ${dataDir}.`,
    );
  }
  if (!exists(db)) {
    throw new Error(`Refusing to run the Console smoke: database ${db} does not exist.`);
  }
  if (typeof manifest.agent_did !== 'string' || !manifest.agent_did.startsWith('did:')) {
    throw new Error('Refusing to run the Console smoke: the manifest records no agent DID.');
  }

  // Module origin.
  const expectedOrigin = path.join(path.resolve(checkoutRoot), 'kestrel_sovereign');
  if (path.resolve(String(manifest.module_origin || '')) !== expectedOrigin) {
    throw new Error(
      `Refusing to run the Console smoke: the server runs kestrel_sovereign from `
      + `${manifest.module_origin}, not from this checkout (${expectedOrigin}).`,
    );
  }
}

/**
 * Check the /api/agents response the console received: a standalone demo
 * server serving exactly the agent this run created.
 *
 * @param {{ ok: boolean, status: number, data: any }} response
 * @param {Record<string, any>} manifest
 */
function assertServedAgentRoster(response, manifest) {
  assertOnlyDemoAgents(response);
  const { data } = response;
  if (data.mode !== 'standalone' || data.server_demo_mode !== true) {
    throw new Error(
      `The console reached a ${data.mode} server with server_demo_mode=${data.server_demo_mode}; `
      + 'the smoke instance is a standalone demo server.',
    );
  }
  const ids = data.agents.map((/** @type {any} */ a) => a.id);
  if (ids.length !== 1 || ids[0] !== manifest.agent_did) {
    throw new Error(
      `The console routed to ${JSON.stringify(ids)}, not the freshly created agent `
      + `${manifest.agent_did}.`,
    );
  }
}

/**
 * The request URLs that left the instance's origin.
 * @param {string[]} urls
 * @param {string} baseUrl
 * @returns {string[]}
 */
function foreignOriginRequests(urls, baseUrl) {
  const origin = new URL(baseUrl).origin;
  return urls.filter((url) => new URL(url).origin !== origin);
}

module.exports = {
  MANIFEST_ENV,
  loadSmokeManifest,
  assertSmokeInstance,
  assertServedAgentRoster,
  foreignOriginRequests,
};
