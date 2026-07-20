// #2649: database values cross an untrusted storage boundary. Exercise the
// real explorer against jsdom and prove names, columns, and cells stay text.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
    url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.location = dom.window.location;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.localStorage = dom.window.localStorage;
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
dom.window.kicon = globalThis.kicon;

const API = (await import('../../kestrel_sovereign/static/js/api.js')).default;
const { state } = await import('../../kestrel_sovereign/static/js/ui.js');
const { default: bus } = await import('../../kestrel_sovereign/static/js/ui-ext/bus.js');
const {
    loadDbTables,
    loadDbTable,
    toggleDbExplorer,
    initDatabaseExplorer,
} = await import('../../kestrel_sovereign/static/js/database.js');

function mountExplorer({ resetState = true } = {}) {
    document.body.replaceChildren();
    const toggle = document.createElement('button');
    toggle.id = 'toggle-db-explorer';
    const section = document.createElement('section');
    section.id = 'db-explorer-section';
    section.style.display = 'none';
    const container = document.createElement('div');
    container.id = 'db-explorer-container';
    section.appendChild(container);
    document.body.append(toggle, section);
    initDatabaseExplorer({ api: API, root: document });
    if (resetState) {
        state.dbTables = null;
        state.dbCurrentTable = null;
        state.dbCurrentPage = 0;
        state.dbExplorerVisible = false;
        state.dbAgent = API.getHostAgent();
    }
    return container;
}

async function flush() {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
}

test('hostile stored markup remains inert text across explorer rendering and pagination', async () => {
    const container = mountExplorer();
    const tableName = 'messages"><img id="table-xss" src=x onerror="alert(1)">';
    const columnName = 'content</th><script id="column-xss">alert(3)</script>';
    const cellValue = '<img id="cell-xss" src=x onerror="globalThis.databaseXssExecuted=true">';
    const unicodeLong = `${'a'.repeat(99)}🦅tail`;
    const calls = [];

    API.getDbTables = async (agent) => ({
        table_count: 1,
        db_size: 2048,
        db_path: '/tmp/<img id="database-xss" src=x onerror="alert(4)">.db',
        tables: [{
            name: tableName,
            row_count: 40,
            queryable: true,
            columns: [{ name: columnName }, { name: 'content' }, { name: 'unicode' }],
        }],
    });
    API.queryDbTable = async (table, limit, offset, search, agent) => {
        calls.push({ table, limit, offset, search, agent });
        return {
            table: tableName,
            columns: [columnName, 'content', 'unicode'],
            rows: [{
                [columnName]: 'safe value',
                content: cellValue,
                unicode: unicodeLong,
            }],
            total_rows: 40,
            has_more: offset + limit < 40,
        };
    };

    await loadDbTables();
    const tableButton = container.querySelector('.db-table-item');
    assert.ok(tableButton, 'queryable table has a bound control');
    assert.ok(container.textContent.includes(tableName), 'hostile table name is visible as text');
    assert.equal(tableButton.getAttribute('onclick'), null, 'table action is not inline JavaScript');
    assert.equal(tableButton.getAttribute('onmouseover'), null);
    assert.equal(tableButton.getAttribute('onmouseout'), null);

    tableButton.click();
    await flush();

    assert.deepEqual(calls[0], {
        table: tableName,
        limit: 20,
        offset: 0,
        search: null,
        agent: API.getHostAgent(),
    });
    assert.ok(container.textContent.includes(tableName), 'selected table name is text');
    assert.ok(container.textContent.includes(columnName), 'hostile column name is text');
    assert.ok(container.textContent.includes(cellValue), 'hostile stored cell is text');
    const cells = Array.from(container.querySelectorAll('tbody td'));
    assert.equal(cells[2].textContent, `${'a'.repeat(99)}🦅`);
    assert.equal(Array.from(cells[2].textContent).length, 100,
        'long Unicode values are truncated without splitting a surrogate pair');
    assert.equal(
        container.querySelectorAll('img, svg, script, [onerror], [onload], [onclick]').length,
        0,
        'database data creates no executable elements or event attributes',
    );
    assert.equal(container.textContent.includes('database-xss'), false,
        'unrendered database path metadata cannot enter the DOM');

    const next = Array.from(container.querySelectorAll('.db-pagination button'))
        .find((button) => button.textContent.includes('Next'));
    assert.ok(next);
    assert.equal(next.getAttribute('onclick'), null, 'pagination uses a bound listener');
    next.click();
    await flush();
    assert.deepEqual(calls[1], {
        table: tableName,
        limit: 20,
        offset: 20,
        search: null,
        agent: API.getHostAgent(),
    });
});

test('a mismatched response identifier cannot retarget pagination', async () => {
    API.setHostAgent('Agent A');
    const container = mountExplorer();
    const calls = [];
    API.queryDbTable = async (table, limit, offset, search, agent) => {
        calls.push({ table, limit, offset, search, agent });
        return {
            table: 'graph_nodes',
            columns: ['content'],
            rows: [{ content: 'wrong table' }],
            total_rows: 40,
            has_more: true,
        };
    };

    state.dbAgent = 'Agent A';
    state.dbExplorerVisible = true;
    state.dbTables = {
        table_count: 1,
        db_size: 0,
        tables: [{
            name: 'conversation_history',
            row_count: 40,
            queryable: true,
            columns: [{ name: 'content' }],
        }],
    };
    const viewer = document.createElement('div');
    viewer.id = 'db-table-viewer';
    container.appendChild(viewer);

    await loadDbTable('conversation_history');

    assert.deepEqual(calls, [{
        table: 'conversation_history',
        limit: 20,
        offset: 0,
        search: null,
        agent: 'Agent A',
    }]);
    assert.ok(container.textContent.includes('did not match the requested table'));
    assert.equal(container.querySelector('.db-pagination'), null);
});

test('hostile database error text cannot become markup', async () => {
    const container = mountExplorer();
    const databaseError = '<img id="database-error-xss" src=x onerror="alert(5)">';
    API.getDbTables = async () => {
        throw new Error(databaseError);
    };

    await loadDbTables();

    assert.ok(container.textContent.includes(databaseError));
    assert.equal(container.querySelector('img'), null);
    assert.equal(container.querySelector('[onerror]'), null);
});

test('agent switch invalidates cached explorer data and rejects delayed responses', async () => {
    API.setHostAgent('Agent A');
    const container = mountExplorer();
    state.dbExplorerVisible = true;
    document.getElementById('db-explorer-section').style.display = 'block';

    let resolveStaleTables;
    const staleTables = new Promise((resolve) => { resolveStaleTables = resolve; });
    let resolveStaleRows;
    const staleRows = new Promise((resolve) => { resolveStaleRows = resolve; });
    let agentATableLoads = 0;
    let agentBTableLoads = 0;

    const tablesFor = (agent) => ({
        table_count: 1,
        db_size: 1024,
        tables: [{
            name: 'conversation_history',
            row_count: agent === 'Agent A' ? 11 : 22,
            queryable: true,
            columns: [{ name: 'content' }],
        }],
    });

    API.getDbTables = async () => {
        const dispatchAgent = API.getHostAgent();
        if (dispatchAgent === 'Agent A' && ++agentATableLoads > 1) {
            return staleTables;
        }
        if (dispatchAgent === 'Agent B') agentBTableLoads += 1;
        return tablesFor(dispatchAgent);
    };
    API.queryDbTable = async (table) => {
        const dispatchAgent = API.getHostAgent();
        if (dispatchAgent === 'Agent A') return staleRows;
        return {
            table,
            columns: ['content'],
            rows: [{ content: 'Agent B private row' }],
            total_rows: 1,
            has_more: false,
        };
    };

    await loadDbTables();
    container.querySelector('.db-table-item').click();
    await flush();
    const staleTableLoad = loadDbTables();

    API.setHostAgent('Agent B');
    assert.equal(state.dbAgent, 'Agent B',
        'routing changes invalidate the old cache before the delayed switch event');
    assert.equal(container.textContent.includes('11 rows'), false);
    await flush();

    assert.equal(agentBTableLoads, 0,
        'setHostAgent invalidates synchronously but does not fetch before capability refresh');
    assert.equal(container.textContent.includes('22 rows'), false);
    assert.equal(container.textContent.includes('11 rows'), false,
        'Agent A table metadata is evicted even when both agents expose the same table');
    assert.equal(state.dbAgent, 'Agent B');

    bus.emit('agent:switch', { prev: 'Earlier Agent', next: 'Agent A' });
    await flush();
    assert.equal(agentBTableLoads, 0,
        'a stale switch event cannot release the capability barrier for Agent B');

    bus.emit('agent:switch', { prev: 'Agent A', next: 'Agent B' });
    await flush();
    assert.ok(container.textContent.includes('22 rows'));
    assert.equal(agentBTableLoads, 1,
        'the post-capability event performs the first Agent B database request');

    container.querySelector('.db-table-item').click();
    await flush();
    assert.ok(container.textContent.includes('Agent B private row'));

    resolveStaleTables(tablesFor('Agent A'));
    resolveStaleRows({
        table: 'conversation_history',
        columns: ['content'],
        rows: [{ content: 'Agent A private row' }],
        total_rows: 1,
        has_more: false,
    });
    await staleTableLoad;
    await flush();

    assert.ok(container.textContent.includes('22 rows'));
    assert.equal(container.textContent.includes('11 rows'), false);
    assert.ok(container.textContent.includes('Agent B private row'));
    assert.equal(container.textContent.includes('Agent A private row'), false,
        'a delayed row response cannot repaint the same table/page for the new agent');
});

test('post-capability switch does not load database metadata when sovereignty is disabled', async () => {
    const container = mountExplorer();
    let activeAgent = 'Agent A';
    let sovereigntyEnabled = true;
    let notifyAgentChange = () => {};
    const calls = [];
    const scopedApi = {
        getHostAgent: () => activeAgent,
        hasCapability: (name) => name === 'sovereignty' && sovereigntyEnabled,
        onHostAgentChange(listener) {
            notifyAgentChange = listener;
            return () => { notifyAgentChange = () => {}; };
        },
        async getDbTables(agent) {
            calls.push(agent);
            return {
                table_count: 1,
                db_size: 1,
                tables: [{
                    name: `${agent} private table`,
                    row_count: 1,
                    queryable: true,
                    columns: [{ name: 'content' }],
                }],
            };
        },
        async queryDbTable() {
            throw new Error('not used');
        },
    };
    initDatabaseExplorer({ api: scopedApi, root: document });

    document.getElementById('toggle-db-explorer').click();
    await flush();
    assert.deepEqual(calls, ['Agent A']);
    assert.ok(container.textContent.includes('Agent A private table'));

    activeAgent = 'Agent B';
    notifyAgentChange('Agent B', 'Agent A');
    assert.equal(container.textContent.includes('Agent A private table'), false,
        'the prior tenant is cleared at the routing boundary');

    // This is the real production order: setHostAgent first, capability result
    // second, and agent:switch only after the new map has been applied.
    sovereigntyEnabled = false;
    bus.emit('agent:switch', { prev: 'Agent A', next: 'Agent B' });
    await flush();

    assert.deepEqual(calls, ['Agent A'],
        'a capability-disabled agent never receives a database metadata request');
    assert.equal(container.textContent.includes('Agent B private table'), false);
});

test('reopening after route change waits for the post-capability switch event', async () => {
    API.setHostAgent('Agent A');
    const container = mountExplorer();
    API.getDbTables = async () => {
        const agent = API.getHostAgent();
        return {
            table_count: 1,
            db_size: 1024,
            tables: [{
                name: `${agent} table`,
                row_count: 1,
                queryable: true,
                columns: [{ name: 'content' }],
            }],
        };
    };

    await loadDbTables();
    assert.ok(container.textContent.includes('Agent A table'));

    // The agent-list component pins API routing before capabilities resolve and
    // before selectAgent emits the generic bus event. Opening in that window
    // must neither reuse Agent A's cache nor dispatch against Agent B yet.
    API.setHostAgent('Agent B');
    toggleDbExplorer();
    await flush();

    assert.equal(container.textContent.includes('Agent A table'), false);
    assert.equal(container.textContent.includes('Agent B table'), false);

    bus.emit('agent:switch', { prev: 'Agent A', next: 'Agent B' });
    await flush();
    assert.ok(container.textContent.includes('Agent B table'));
});

test('external-router agent switch invalidates rows when API routing remains null', async () => {
    API.setHostAgent(null);
    const container = mountExplorer();
    state.dbExplorerVisible = true;
    document.getElementById('db-explorer-section').style.display = 'block';
    let externallySelectedAgent = 'External A';
    let loads = 0;

    API.getDbTables = async (agent) => {
        loads += 1;
        assert.equal(agent, null, 'the embed keeps canonical API routing external');
        return {
            table_count: 1,
            db_size: -1,
            tables: [{
                name: 'conversation_history',
                row_count: 1,
                queryable: true,
                columns: [{ name: 'content' }],
            }],
            selectedByHost: externallySelectedAgent,
        };
    };
    API.queryDbTable = async (table) => ({
        table,
        columns: ['content'],
        rows: [{ content: `${externallySelectedAgent} private row` }],
        total_rows: 1,
        has_more: false,
    });

    await loadDbTables();
    container.querySelector('.db-table-item').click();
    await flush();
    assert.ok(container.textContent.includes('External A private row'));

    externallySelectedAgent = 'External B';
    bus.emit('agent:switch', { prev: 'External A', next: 'External B' });
    assert.equal(container.textContent.includes('External A private row'), false,
        'the previous tenant is evicted synchronously at the external switch boundary');
    await flush();

    assert.equal(loads, 2, 'the external switch reloads despite an unchanged null API route');
    assert.equal(container.textContent.includes('External A private row'), false);
    assert.equal(state.dbAgent, null);

    container.querySelector('.db-table-item').click();
    await flush();
    assert.ok(container.textContent.includes('External B private row'));
    assert.equal(container.textContent.includes('External A private row'), false);
});

test('legacy onclick explorer button is not toggled a second time by delegation', async () => {
    API.setHostAgent(null);
    mountExplorer();
    let loads = 0;
    API.getDbTables = async () => {
        loads += 1;
        return { tables: [], table_count: 0, db_size: -1 };
    };

    const toggle = document.getElementById('toggle-db-explorer');
    toggle.setAttribute('onclick', 'toggleDbExplorer()');
    toggle.onclick = window.toggleDbExplorer;
    toggle.click();
    await flush();

    assert.equal(state.dbExplorerVisible, true);
    assert.equal(document.getElementById('db-explorer-section').style.display, 'block');
    assert.equal(toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(loads, 1, 'one legacy click performs exactly one toggle/load');
});

test('sovereignty capability off-on remount resets explorer visibility and cache', async () => {
    API.setHostAgent('Remount Agent');
    let loads = 0;
    API.getDbTables = async () => {
        loads += 1;
        return {
            tables: [{
                name: `load-${loads}`,
                row_count: 1,
                queryable: true,
                columns: [{ name: 'content' }],
            }],
            table_count: 1,
            db_size: -1,
        };
    };

    mountExplorer();
    document.getElementById('toggle-db-explorer').click();
    await flush();
    assert.equal(state.dbExplorerVisible, true);
    assert.ok(document.body.textContent.includes('load-1'));

    // panels.js emits this before capability gating detaches the panel.
    bus.emit('panel:hidden', { panelId: 'sovereignty' });
    assert.equal(state.dbExplorerVisible, false);
    assert.equal(state.dbTables, null, 'detached-panel cache is discarded');

    mountExplorer({ resetState: false });
    document.getElementById('toggle-db-explorer').click();
    await flush();

    assert.equal(state.dbExplorerVisible, true,
        'the first click after remount opens instead of flipping stale visible state off');
    assert.equal(document.getElementById('db-explorer-section').style.display, 'block');
    assert.ok(document.body.textContent.includes('load-2'), 'the remounted panel reloads fresh data');
    assert.equal(loads, 2);
});
