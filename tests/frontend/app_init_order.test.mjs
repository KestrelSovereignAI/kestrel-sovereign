import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/js/app.js', import.meta.url),
    'utf8',
);

test('app initializes Security after multi-agent selection has run', () => {
    const loadAgentsIndex = source.indexOf('await loadAgents();');
    const securityInitIndex = source.indexOf('Security.init();');

    assert.notEqual(loadAgentsIndex, -1, 'app init must load agents');
    assert.notEqual(securityInitIndex, -1, 'app init must initialize security');
    assert.ok(
        loadAgentsIndex < securityInitIndex,
        'Security.init() must run after loadAgents() so /api/security calls are agent-prefixed in multi_agent mode',
    );
});
