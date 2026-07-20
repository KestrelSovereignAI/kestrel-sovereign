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
    toggleDbExplorer,
} = await import('../../kestrel_sovereign/static/js/database.js');

function mountExplorer() {
    document.body.replaceChildren();
    const toggle = document.createElement('button');
    toggle.id = 'toggle-db-explorer';
    const section = document.createElement('section');
    section.id = 'db-explorer-section';
    const container = document.createElement('div');
    container.id = 'db-explorer-container';
    section.appendChild(container);
    document.body.append(toggle, section);
    state.dbTables = null;
    state.dbCurrentTable = null;
    state.dbCurrentPage = 0;
    state.dbExplorerVisible = false;
    return container;
}

async function flush() {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
}

test('hostile stored markup remains inert text across explorer rendering and pagination', async () => {
    const container = mountExplorer();
    const tableName = 'messages"><img id="table-xss" src=x onerror="alert(1)">';
    const selectedName = '<svg id="selected-xss" onload="alert(2)"></svg>';
    const columnName = 'content</th><script id="column-xss">alert(3)</script>';
    const cellValue = '<img id="cell-xss" src=x onerror="globalThis.databaseXssExecuted=true">';
    const unicodeLong = '漢字🦅 café — Δοκιμή '.repeat(20);
    const calls = [];

    API.getDbTables = async () => ({
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
    API.queryDbTable = async (table, limit, offset) => {
        calls.push({ table, limit, offset });
        return {
            table: selectedName,
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

    assert.deepEqual(calls[0], { table: tableName, limit: 20, offset: 0 });
    assert.ok(container.textContent.includes(selectedName), 'selected table name is text');
    assert.ok(container.textContent.includes(columnName), 'hostile column name is text');
    assert.ok(container.textContent.includes(cellValue), 'hostile stored cell is text');
    const cells = Array.from(container.querySelectorAll('tbody td'));
    assert.equal(cells[2].textContent, unicodeLong.substring(0, 100));
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
    assert.deepEqual(calls[1], { table: selectedName, limit: 20, offset: 20 });
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
    const container = mountExplorer();
    state.dbExplorerVisible = true;
    API.setHostAgent('Agent A');

    let resolveStaleTables;
    const staleTables = new Promise((resolve) => { resolveStaleTables = resolve; });
    let resolveStaleRows;
    const staleRows = new Promise((resolve) => { resolveStaleRows = resolve; });
    let agentATableLoads = 0;

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
    bus.emit('agent:switch', { prev: 'Agent A', next: 'Agent B' });
    await flush();

    assert.ok(container.textContent.includes('22 rows'));
    assert.equal(container.textContent.includes('11 rows'), false,
        'Agent A table metadata is evicted even when both agents expose the same table');
    assert.equal(state.dbAgent, 'Agent B');

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

test('reopening binds cached table metadata to the API active agent before the switch event', async () => {
    const container = mountExplorer();
    API.setHostAgent('Agent A');
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

    // The agent-list component pins API routing before selectAgent emits the
    // generic bus event. Opening in that window must not reuse Agent A's cache.
    API.setHostAgent('Agent B');
    toggleDbExplorer();
    await flush();

    assert.ok(container.textContent.includes('Agent B table'));
    assert.equal(container.textContent.includes('Agent A table'), false);
});
