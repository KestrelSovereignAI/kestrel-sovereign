/**
 * Kestrel Sovereign Console - Files Module
 * Local File Browser functionality
 */

import API from './api.js';
import { state, Toast, Modal, formatBytes, escapeHtml } from './ui.js';

// ============================================================================
// Local File Browser
// ============================================================================

state.localFiles = null;
state.fileBrowserVisible = false;

export async function loadLocalFiles() {
    try {
        const data = await API.getSovereigntyFiles();
        state.localFiles = data;
        renderLocalFiles(data);
    } catch (e) {
        const container = document.getElementById('file-browser-container');
        if (container) {
            container.innerHTML = `<p style="color: var(--error); padding: 1rem;">Failed to load files: ${e.message}</p>`;
        }
    }
}

function renderLocalFiles(data) {
    const container = document.getElementById('file-browser-container');
    if (!container) return;

    if (!data.files || data.files.length === 0) {
        container.innerHTML = `
            <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                No local cache files found.
            </p>
        `;
        return;
    }

    const cacheFiles = data.files.filter(f => f.type === 'cache');

    container.innerHTML = `
        <div style="
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        ">
            <div style="
                background: var(--bg-tertiary);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 600;">${cacheFiles.length}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">Cache Files</div>
            </div>
            <div style="
                background: var(--bg-tertiary);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 600;">${formatBytes(data.total_size)}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">Total Size</div>
            </div>
            <div style="
                background: var(--bg-tertiary);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 1.5rem; font-weight: 600;">${data.files.filter(f => f.has_meta).length}</div>
                <div style="font-size: 0.75rem; color: var(--text-secondary);">With Metadata</div>
            </div>
        </div>

        <div style="margin-bottom: 1rem;">
            <input type="text" id="file-search" placeholder="Search files by hash..." style="
                width: 100%;
                padding: 0.75rem 1rem;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                font-size: 0.875rem;
                background: var(--bg-primary);
                color: var(--text-primary);
                outline: none;
            " oninput="filterLocalFiles(this.value)" />
        </div>

        <div id="file-list" style="display: flex; flex-direction: column; gap: 0.5rem; max-height: 400px; overflow-y: auto;">
            ${cacheFiles.map(file => renderFileItem(file)).join('')}
        </div>
    `;
}

function renderFileItem(file) {
    const truncatedHash = file.hash.length > 16 ? file.hash.slice(0, 8) + '...' + file.hash.slice(-8) : file.hash;
    const modified = new Date(file.modified * 1000).toLocaleString();

    return `
        <div class="file-item" data-hash="${file.hash}" data-filename="${file.name}" style="
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transition: border-color 0.2s;
        " onmouseover="this.style.borderColor='var(--accent-color)'" onmouseout="this.style.borderColor='var(--border-color)'">
            <span style="font-size: 1.25rem;">${file.has_meta ? '\u{1F4C4}' : '\u{1F4E6}'}</span>

            <div style="flex: 1; min-width: 0;">
                <div style="
                    font-family: var(--font-mono);
                    font-size: 0.8rem;
                    color: var(--text-primary);
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                " title="${file.hash}">${truncatedHash}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">
                    ${formatBytes(file.size)} \u2022 ${modified}
                </div>
            </div>

            <div style="display: flex; gap: 0.375rem;">
                <button onclick="previewFile('${file.name}')" class="btn btn-secondary" style="
                    padding: 0.25rem 0.5rem;
                    font-size: 0.7rem;
                " title="Preview">\u{1F441}</button>
                <button onclick="downloadFile('${file.name}')" class="btn btn-secondary" style="
                    padding: 0.25rem 0.5rem;
                    font-size: 0.7rem;
                " title="Download">\u2B07</button>
                <button onclick="copyToClipboard('${file.hash}')" class="btn btn-secondary" style="
                    padding: 0.25rem 0.5rem;
                    font-size: 0.7rem;
                " title="Copy Hash">\u{1F4CB}</button>
            </div>
        </div>
    `;
}

window.filterLocalFiles = function(query) {
    if (!state.localFiles) return;

    const fileItems = document.querySelectorAll('.file-item');
    const queryLower = query.toLowerCase();

    fileItems.forEach(item => {
        const hash = item.dataset.hash?.toLowerCase() || '';
        const filename = item.dataset.filename?.toLowerCase() || '';
        const matches = hash.includes(queryLower) || filename.includes(queryLower);
        item.style.display = matches ? 'flex' : 'none';
    });
};

window.previewFile = async function(filename) {
    try {
        const preview = await API.getSovereigntyFilePreview(filename);
        showFilePreviewModal(preview);
    } catch (e) {
        Toast.error(`Failed to preview file: ${e.message}`);
    }
};

window.downloadFile = function(filename) {
    const link = document.createElement('a');
    link.href = `/api/sovereignty/files/${encodeURIComponent(filename)}`;
    link.download = filename;
    link.click();
    Toast.success('Download started');
};

function showFilePreviewModal(preview) {
    const contentDisplay = preview.is_text
        ? (preview.content_type === 'json'
            ? `<pre style="
                margin: 0;
                padding: 1rem;
                background: var(--bg-tertiary);
                border-radius: 8px;
                overflow-x: auto;
                font-size: 0.8rem;
                max-height: 400px;
                overflow-y: auto;
            ">${escapeHtml(JSON.stringify(JSON.parse(preview.content), null, 2))}</pre>`
            : `<pre style="
                margin: 0;
                padding: 1rem;
                background: var(--bg-tertiary);
                border-radius: 8px;
                overflow-x: auto;
                font-size: 0.8rem;
                max-height: 400px;
                overflow-y: auto;
                white-space: pre-wrap;
            ">${escapeHtml(preview.content)}</pre>`)
        : `<div style="
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-family: var(--font-mono);
            font-size: 0.7rem;
            word-break: break-all;
        ">
            <p style="margin: 0 0 0.5rem 0; color: var(--text-secondary);">Binary content (hex):</p>
            ${preview.content}
        </div>`;

    Modal.show({
        title: `File Preview`,
        content: `
            <div style="margin-bottom: 1rem;">
                <div style="display: grid; gap: 0.5rem; font-size: 0.8rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary);">Filename:</span>
                        <span style="font-family: var(--font-mono);">${preview.filename}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary);">Size:</span>
                        <span>${formatBytes(preview.size)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--text-secondary);">Type:</span>
                        <span>${preview.content_type}</span>
                    </div>
                    ${preview.truncated ? `
                        <div style="
                            padding: 0.5rem;
                            background: var(--warning);
                            color: white;
                            border-radius: 4px;
                            font-size: 0.75rem;
                        ">
                            \u26A0 Content truncated (showing first ${formatBytes(preview.preview_size)} of ${formatBytes(preview.size)})
                        </div>
                    ` : ''}
                </div>
            </div>
            ${contentDisplay}
        `,
        buttons: [
            { label: 'Download', type: 'secondary', onClick: () => { window.downloadFile(preview.filename); } },
            { label: 'Close', type: 'primary', onClick: () => Modal.hide() }
        ]
    });
}

window.toggleFileBrowser = function() {
    const container = document.getElementById('file-browser-section');
    const toggleBtn = document.getElementById('toggle-file-browser');

    if (!container || !toggleBtn) return;

    state.fileBrowserVisible = !state.fileBrowserVisible;

    if (state.fileBrowserVisible) {
        container.style.display = 'block';
        toggleBtn.textContent = '\u{1F4C1} Hide Local Files';
        if (!state.localFiles) {
            loadLocalFiles();
        }
    } else {
        container.style.display = 'none';
        toggleBtn.textContent = '\u{1F4C1} Browse Local Files';
    }
};
