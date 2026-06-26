// @ts-check
//
// Safety-critical demo primitives (issue #1973). Dependency-free on purpose:
// these guard a destructive filesystem operation, so they must be unit-testable
// without pulling in @kestrel/flight or Playwright. demo_helpers.cjs re-exports
// everything here.
const fs = require('fs');
const path = require('path');

const DEMO_FLAG_VALUES = new Set(['1', 'true', 'yes']);

/**
 * True when this process is an isolated demo run launched by `kestrel demo run`
 * (which sets KESTREL_DEMO_SERVER=1).
 */
function isDemoServerEnv() {
  return DEMO_FLAG_VALUES.has(String(process.env.KESTREL_DEMO_SERVER || '').toLowerCase());
}

/**
 * Resolve, and assert, the isolated demo data directory.
 *
 * `kestrel demo run` sets KESTREL_DB_PATH to a throwaway sandbox (agent_data/demo).
 * Returns its resolved path and throws if we are not in a demo run — the single
 * source of truth for "where it is safe to delete demo databases".
 * @returns {string} resolved absolute path of the isolated demo sandbox
 */
function requireDemoSandbox() {
  if (!isDemoServerEnv()) {
    throw new Error(
      'Refusing demo DB operation: KESTREL_DEMO_SERVER is not set, so this is not '
      + 'an isolated demo run. Launch the demo via `kestrel demo run` — never raw '
      + '`npx playwright test` against a live instance.',
    );
  }
  const sandbox = process.env.KESTREL_DB_PATH;
  if (!sandbox) {
    throw new Error(
      'Refusing demo DB operation: KESTREL_DB_PATH is unset. The isolated demo '
      + 'sandbox is unknown, so no path can be proven safe to delete.',
    );
  }
  return path.resolve(sandbox);
}

/**
 * True iff `target` is the sandbox itself or a descendant of it.
 * @param {string} sandbox resolved sandbox path
 * @param {string} target resolved candidate path
 */
function isInsideSandbox(sandbox, target) {
  if (target === sandbox) return true;
  const rel = path.relative(sandbox, target);
  return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
}

/**
 * Reset the isolated demo agent's database(s) so each recording starts from a
 * clean context window. DESTRUCTIVE: deletes kestrel_prime.db files; the agent
 * re-creates an empty DB on the next startFreshSession().
 *
 * Hard-guarded against the #867/#868-class wipe: refuses unless we are in a
 * `kestrel demo run` (KESTREL_DEMO_SERVER=1) AND `dataDir` resolves *inside* the
 * isolated KESTREL_DB_PATH sandbox. It will never walk a parent of the sandbox
 * (e.g. the live `agent_data/` tree), so it cannot touch live agents' data.
 *
 * @param {{ narrate: (msg: string) => void }} narrator
 * @param {string} dataDir - the isolated demo data dir (must be within KESTREL_DB_PATH)
 */
function resetDemoAgentDatabases(narrator, dataDir) {
  const sandbox = requireDemoSandbox();
  const target = path.resolve(dataDir);
  if (!isInsideSandbox(sandbox, target)) {
    throw new Error(
      `Refusing to reset databases under ${target}: it is outside the isolated demo `
      + `sandbox (KESTREL_DB_PATH=${sandbox}). This guard exists to prevent wiping `
      + 'live agent data — see issue #1973.',
    );
  }

  const dbs = [];
  function walk(dir, depth) {
    if (depth > 3) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch { return; }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full, depth + 1);
      else if (entry.name === 'kestrel_prime.db') dbs.push(full);
    }
  }
  try {
    walk(target, 0);
    let unlinked = 0;
    for (const db of dbs) {
      try { fs.unlinkSync(db); unlinked++; } catch { /* locked or already gone */ }
    }
    narrator.narrate(`Reset ${unlinked}/${dbs.length} demo agent database(s); fresh session will recreate`);
  } catch (e) {
    narrator.narrate(`Could not reset demo databases: ${e.message}`);
  }
}

module.exports = {
  isDemoServerEnv,
  requireDemoSandbox,
  isInsideSandbox,
  resetDemoAgentDatabases,
};
