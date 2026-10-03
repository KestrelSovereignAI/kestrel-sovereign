// Origin proofs for the Sovereign Console smoke (issue #2682).
//
// The smoke spec refuses to touch a target unless it is the fresh isolated
// instance `kestrel demo smoke` created: loopback URL on the recorded,
// non-live port; database inside the fresh home; server running this
// checkout's kestrel_sovereign; and a roster of exactly the agent that run
// minted. These tests pin each refusal.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const require = createRequire(import.meta.url);
const {
  MANIFEST_ENV,
  loadSmokeManifest,
  assertSmokeInstance,
  assertServedAgentRoster,
  foreignOriginRequests,
} = require('../e2e/console_smoke_proof.cjs');

const CHECKOUT = path.resolve('/srv/checkout');
const HOME = path.resolve('/tmp/kestrel-console-smoke-x');
const DATA_DIR = path.join(HOME, 'agent_data', 'console-smoke');
const DID = 'did:web:localhost:kestrel-demo-agent-abc123';

function manifest(overrides = {}) {
  return {
    base_url: 'http://127.0.0.1:8910',
    port: 8910,
    home: HOME,
    data_dir: DATA_DIR,
    db_path: path.join(DATA_DIR, 'kestrel_prime.db'),
    agent_did: DID,
    module_origin: path.join(CHECKOUT, 'kestrel_sovereign'),
    ...overrides,
  };
}

function observed(overrides = {}) {
  return {
    baseUrl: 'http://127.0.0.1:8910',
    dbPath: DATA_DIR,
    checkoutRoot: CHECKOUT,
    exists: () => true,
    ...overrides,
  };
}

function withDemoFlag(value, fn) {
  const saved = process.env.KESTREL_DEMO_SERVER;
  if (value === undefined) delete process.env.KESTREL_DEMO_SERVER;
  else process.env.KESTREL_DEMO_SERVER = value;
  try {
    return fn();
  } finally {
    if (saved === undefined) delete process.env.KESTREL_DEMO_SERVER;
    else process.env.KESTREL_DEMO_SERVER = saved;
  }
}

test('a fresh isolated instance passes every origin proof', () => {
  withDemoFlag('1', () => assertSmokeInstance(manifest(), observed()));
});

test('refuses a raw run without the demo-server flag', () => {
  withDemoFlag(undefined, () => {
    assert.throws(() => assertSmokeInstance(manifest(), observed()), /KESTREL_DEMO_SERVER is not set/);
  });
});

test('refuses the live port even when the manifest agrees', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(
        manifest({ base_url: 'http://127.0.0.1:8888', port: 8888 }),
        observed({ baseUrl: 'http://127.0.0.1:8888' }),
      ),
      /port 8888 is the live server/,
    );
  });
});

test('refuses a URL that is not the instance the runner started', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(manifest(), observed({ baseUrl: 'http://127.0.0.1:8911' })),
      /is not the instance the runner started/,
    );
    assert.throws(
      () => assertSmokeInstance(manifest(), observed({ baseUrl: undefined })),
      /KESTREL_URL is unset/,
    );
  });
});

test('refuses a non-loopback host', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(
        manifest({ base_url: 'http://agents.example.com:8910' }),
        observed({ baseUrl: 'http://agents.example.com:8910' }),
      ),
      /not a loopback host/,
    );
  });
});

test('refuses a port the manifest did not record', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(manifest({ port: 8920 }), observed()),
      /is not the recorded port 8920/,
    );
  });
});

test('refuses a database outside the fresh instance', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(manifest(), observed({ dbPath: '/srv/live/agent_data/emma' })),
      /is not the instance data dir/,
    );
    assert.throws(
      () => assertSmokeInstance(
        manifest({ data_dir: '/srv/live/agent_data/emma', db_path: '/srv/live/agent_data/emma/kestrel_prime.db' }),
        observed({ dbPath: '/srv/live/agent_data/emma' }),
      ),
      /is not inside the fresh instance home/,
    );
    assert.throws(
      () => assertSmokeInstance(manifest({ db_path: '/srv/live/kestrel_prime.db' }), observed()),
      /is not inside the data dir/,
    );
    assert.throws(
      () => assertSmokeInstance(manifest(), observed({ exists: () => false })),
      /does not exist/,
    );
  });
});

test('refuses a manifest without an agent DID', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(manifest({ agent_did: '' }), observed()),
      /records no agent DID/,
    );
  });
});

test('refuses a server running another checkout', () => {
  withDemoFlag('1', () => {
    assert.throws(
      () => assertSmokeInstance(
        manifest({ module_origin: '/srv/primary-checkout/kestrel_sovereign' }),
        observed(),
      ),
      /not from this checkout/,
    );
  });
});

test('loadSmokeManifest refuses when the runner did not record an instance', () => {
  assert.throws(() => loadSmokeManifest(''), new RegExp(`${MANIFEST_ENV} is unset`));
  assert.throws(
    () => loadSmokeManifest(path.join(os.tmpdir(), 'no-such-console-smoke.json')),
    /cannot read/,
  );
});

test('loadSmokeManifest reads the recorded instance', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'console-smoke-proof-'));
  try {
    const file = path.join(dir, 'console-smoke-instance.json');
    fs.writeFileSync(file, JSON.stringify(manifest()));
    assert.equal(loadSmokeManifest(file).agent_did, DID);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

function roster(data, { ok = true, status = 200 } = {}) {
  return { ok, status, data };
}

test('the console must reach exactly the freshly created demo agent', () => {
  assertServedAgentRoster(
    roster({ mode: 'standalone', server_demo_mode: true, agents: [{ id: DID, is_demo: true }] }),
    manifest(),
  );
});

test('a roster with another agent, a live agent, or an error is refused', () => {
  assert.throws(
    () => assertServedAgentRoster(
      roster({ mode: 'standalone', server_demo_mode: true, agents: [{ id: 'did:web:localhost:other', is_demo: true }] }),
      manifest(),
    ),
    /not the freshly created agent/,
  );
  assert.throws(
    () => assertServedAgentRoster(
      roster({ mode: 'standalone', server_demo_mode: true, agents: [{ id: DID, name: 'Emma', is_demo: false }] }),
      manifest(),
    ),
    /non-demo agent/,
  );
  assert.throws(
    () => assertServedAgentRoster(roster({ detail: 'Unauthorized' }, { ok: false, status: 401 }), manifest()),
    /HTTP 401/,
  );
});

test('a multi-agent host or a non-demo server is not the smoke instance', () => {
  assert.throws(
    () => assertServedAgentRoster(
      roster({ mode: 'multi_agent', server_demo_mode: true, agents: [{ id: DID, is_demo: true }] }),
      manifest(),
    ),
    /standalone demo server/,
  );
  assert.throws(
    () => assertServedAgentRoster(
      roster({ mode: 'standalone', server_demo_mode: false, agents: [{ id: DID, is_demo: true }] }),
      manifest(),
    ),
    /standalone demo server/,
  );
});

test('foreignOriginRequests names every request that left the instance', () => {
  assert.deepEqual(
    foreignOriginRequests(
      [
        'http://127.0.0.1:8910/api/auth/key',
        'http://127.0.0.1:8910/api/agents',
        'http://127.0.0.1:8888/api/agents',
        'https://example.com/x',
      ],
      'http://127.0.0.1:8910',
    ),
    ['http://127.0.0.1:8888/api/agents', 'https://example.com/x'],
  );
});
