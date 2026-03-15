/**
 * Kestrel Sovereign Console - Database Module
 * Database Explorer
 */

import API from './api.js';
import { state } from './ui.js';

// ============================================================================
// Database Explorer
// ============================================================================

state.dbTables = null;
state.dbCurrentTable = null;
state.dbCurrentPage = 0;
state.dbExplorerVisible = false;

export async function loadDbTables() {
    try {
        const data = await API.getDbTables();
        state.dbTables = data;
        renderDbExplorer(data);
    } catch (e) {
        const container = document.getElementById('db-explorer-container');
        if (container) {
            container.innerHTML = `<p style="color: var(--error); padding: 1rem;">Failed to load database: ${e.message}</p>`;
        }
    }
}

function renderDbExplorer(data) {
    const container = document.getElementById('db-explorer-container');
    if (!container) return;

    if (!data.tables || data.tables.length === 0) {
        container.innerHTML = `<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No database tables found.</p>`;
        return;
    }

    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    };

    container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-bottom: 1rem;">
            <div style="background: var(--bg-tertiary); padding: 0.75rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 600;">${data.table_count}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Tables</div>
            </div>
            <div style="background: var(--bg-tertiary); padding: 0.75rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 600;">${formatSize(data.db_size)}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">DB Size</div>
            </div>
            <div style="background: var(--bg-tertiary); padding: 0.75rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 600;">${data.tables.reduce((sum, t) => sum + (t.row_count >= 0 ? t.row_count : 0), 0)}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Total Rows</div>
            </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 0.5rem; max-height: 400px; overflow-y: auto;">
            ${data.tables.map(table => `
                <div class="db-table-item"
                     data-table="${table.name}"
                     onclick="${table.queryable ? `loadDbTable('${table.name}')` : ''}"
                     style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        background: ${state.dbCurrentTable === table.name ? 'var(--accent-color)' : 'var(--bg-secondary)'};
                        color: ${state.dbCurrentTable === table.name ? 'white' : 'var(--text-primary)'};
                        border: 1px solid var(--border-color);
                        border-radius: 8px;
                        padding: 0.75rem;
                        cursor: ${table.queryable ? 'pointer' : 'default'};
                        opacity: ${table.queryable ? '1' : '0.6'};
                        transition: all 0.2s;
                     "
                     ${table.queryable ? `onmouseover="if(this.dataset.table !== '${state.dbCurrentTable}') this.style.borderColor='var(--accent-color)'" onmouseout="this.style.borderColor='var(--border-color)'"` : ''}>
                    <div>
                        <div style="font-weight: 500; font-size: 0.85rem;">
                            ${table.queryable ? '\u{1F50D}' : '\u{1F512}'} ${table.name}
                        </div>
                        <div style="font-size: 0.7rem; opacity: 0.7;">
                            ${table.columns.length} columns
                        </div>
                    </div>
                    <div style="
                        font-size: 0.75rem;
                        background: ${state.dbCurrentTable === table.name ? 'rgba(255,255,255,0.2)' : 'var(--bg-tertiary)'};
                        padding: 0.25rem 0.5rem;
                        border-radius: 4px;
                    ">${table.row_count >= 0 ? table.row_count + ' rows' : 'N/A'}</div>
                </div>
            `).join('')}
        </div>

        <div id="db-table-viewer" style="margin-top: 1rem; display: ${state.dbCurrentTable ? 'block' : 'none'};">
            <div class="loading">Select a table to view data...</div>
        </div>
    `;
}

window.loadDbTable = async function(tableName, page = 0) {
    state.dbCurrentTable = tableName;
    state.dbCurrentPage = page;

    const viewer = document.getElementById('db-table-viewer');
    if (!viewer) return;

    viewer.style.display = 'block';
    viewer.innerHTML = '<div class="loading">Loading table data...</div>';

    try {
        const limit = 20;
        const offset = page * limit;
        const data = await API.queryDbTable(tableName, limit, offset);

        renderDbTableData(data, page, limit);

        if (state.dbTables) renderDbExplorer(state.dbTables);
    } catch (e) {
        viewer.innerHTML = `<p style="color: var(--error);">Failed to load table: ${e.message}</p>`;
    }
};

function renderDbTableData(data, page, limit) {
    const viewer = document.getElementById('db-table-viewer');
    if (!viewer) return;

    const totalPages = Math.ceil(data.total_rows / limit);

    viewer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; font-size: 0.9rem;">\u{1F4CA} ${data.table} (${data.total_rows} rows)</h4>
            <div style="display: flex; gap: 0.5rem;">
                <button onclick="loadDbTable('${data.table}', ${page - 1})"
                        class="btn btn-secondary"
                        style="padding: 0.25rem 0.5rem; font-size: 0.75rem;"
                        ${page === 0 ? 'disabled' : ''}>\u{25C0} Prev</button>
                <span style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                    Page ${page + 1} / ${totalPages || 1}
                </span>
                <button onclick="loadDbTable('${data.table}', ${page + 1})"
                        class="btn btn-secondary"
                        style="padding: 0.25rem 0.5rem; font-size: 0.75rem;"
                        ${!data.has_more ? 'disabled' : ''}>Next \u{25B6}</button>
            </div>
        </div>

        <div style="overflow-x: auto; max-height: 300px; border: 1px solid var(--border-color); border-radius: 8px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.75rem;">
                <thead>
                    <tr style="background: var(--bg-tertiary); position: sticky; top: 0;">
                        ${data.columns.map(col => `
                            <th style="padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--border-color); white-space: nowrap;">
                                ${col}
                            </th>
                        `).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${data.rows.length > 0 ? data.rows.map(row => `
                        <tr style="border-bottom: 1px solid var(--border-color);">
                            ${data.columns.map(col => `
                                <td style="padding: 0.5rem; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    ${row[col] !== null && row[col] !== undefined ? String(row[col]).substring(0, 100) : '<span style="color: var(--text-tertiary);">null</span>'}
                                </td>
                            `).join('')}
                        </tr>
                    `).join('') : `
                        <tr>
                            <td colspan="${data.columns.length}" style="padding: 1rem; text-align: center; color: var(--text-secondary);">
                                No data in this table
                            </td>
                        </tr>
                    `}
                </tbody>
            </table>
        </div>
    `;
}

window.toggleDbExplorer = function() {
    const container = document.getElementById('db-explorer-section');
    const toggleBtn = document.getElementById('toggle-db-explorer');

    if (!container || !toggleBtn) return;

    state.dbExplorerVisible = !state.dbExplorerVisible;

    if (state.dbExplorerVisible) {
        container.style.display = 'block';
        toggleBtn.textContent = '\u{1F5C4}\u{FE0F} Hide Database Explorer';
        if (!state.dbTables) {
            loadDbTables();
        }
    } else {
        container.style.display = 'none';
        toggleBtn.textContent = '\u{1F5C4}\u{FE0F} Browse Database';
    }
};
