/**
 * Kestrel Sovereign Console - Tasks Component
 * Background task monitoring, activity log, and management UI
 */

import API from './api.js';
import { Toast } from './ui.js';

// ============================================================================
// DOM References
// ============================================================================

let taskList = null;
let taskFilter = null;
let refreshBtn = null;
let tasksBadge = null;
let activityList = null;

// Track working tasks for badge updates
let workingTasksCount = 0;

// Track which view is active: 'tasks' or 'activity'
let activeView = 'tasks';

// ============================================================================
// Initialization
// ============================================================================

export function initTasks() {
    // #879: skip wiring when the host disabled the tasks panel.
    if (!API.hasCapability('tasks')) return;
    taskList = document.getElementById('task-list');
    taskFilter = document.getElementById('task-filter');
    refreshBtn = document.getElementById('btn-refresh-tasks');
    tasksBadge = document.getElementById('tasks-badge');
    activityList = document.getElementById('activity-list');

    // Event listeners
    refreshBtn?.addEventListener('click', () => {
        if (activeView === 'tasks') {
            loadTasks();
        } else {
            loadActivityLog();
        }
    });
    taskFilter?.addEventListener('change', loadTasks);

    // Set up view toggle buttons
    setupViewToggle();

    // Set up auto-refresh while on tasks panel
    setupAutoRefresh();
}

// ============================================================================
// View Toggle (Tasks vs Activity Log)
// ============================================================================

function setupViewToggle() {
    const btnTasks = document.getElementById('btn-view-tasks');
    const btnActivity = document.getElementById('btn-view-activity');

    btnTasks?.addEventListener('click', () => switchView('tasks'));
    btnActivity?.addEventListener('click', () => switchView('activity'));
}

function switchView(view) {
    activeView = view;

    const btnTasks = document.getElementById('btn-view-tasks');
    const btnActivity = document.getElementById('btn-view-activity');
    const tasksContainer = document.getElementById('tasks-container');
    const activityContainer = document.getElementById('activity-container');

    if (view === 'tasks') {
        btnTasks?.classList.add('active');
        btnActivity?.classList.remove('active');
        if (tasksContainer) tasksContainer.style.display = 'block';
        if (activityContainer) activityContainer.style.display = 'none';
        loadTasks();
    } else {
        btnTasks?.classList.remove('active');
        btnActivity?.classList.add('active');
        if (tasksContainer) tasksContainer.style.display = 'none';
        if (activityContainer) activityContainer.style.display = 'block';
        loadActivityLog();
    }
}

// ============================================================================
// Auto-refresh
// ============================================================================

let autoRefreshInterval = null;

function setupAutoRefresh() {
    // Observe panel visibility to start/stop auto-refresh
    const tasksPanel = document.getElementById('panel-tasks');
    if (!tasksPanel) return;

    // Use MutationObserver to detect when panel becomes visible
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.attributeName === 'class') {
                const isActive = tasksPanel.classList.contains('active');
                if (isActive) {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            }
        }
    });

    observer.observe(tasksPanel, { attributes: true });

    // Also check initial state
    if (tasksPanel.classList.contains('active')) {
        startAutoRefresh();
    }
}

function startAutoRefresh() {
    if (autoRefreshInterval) return;

    // Refresh every 5 seconds while panel is visible
    autoRefreshInterval = setInterval(() => {
        if (activeView === 'tasks') {
            loadTasks();
        } else {
            loadActivityLog();
        }
    }, 5000);
    console.log('Tasks auto-refresh started');
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        console.log('Tasks auto-refresh stopped');
    }
}

// ============================================================================
// Load Tasks
// ============================================================================

// Track which tasks are expanded (persists across refreshes)
let expandedTaskIds = new Set();

export async function loadTasks() {
    // #879: deep-link defense — no /api/agent/tasks fetches when disabled.
    if (!API.hasCapability('tasks')) return;
    if (!taskList) return;

    const statusFilter = taskFilter?.value || '';

    try {
        const params = new URLSearchParams();
        if (statusFilter) {
            params.append('status', statusFilter);
        }
        params.append('limit', '50');

        const response = await API.request(`/api/agent/tasks?${params.toString()}`);

        if (response.message) {
            // TaskManager not available
            renderEmptyState(response.message);
            updateBadge(0);
            return;
        }

        const tasks = response.tasks || [];

        // Count working tasks for badge
        const newWorkingCount = tasks.filter(t => t.status === 'working').length;
        updateBadge(newWorkingCount);

        if (tasks.length === 0) {
            renderEmptyState('No tasks found');
            return;
        }

        renderTasks(tasks);

        // Restore expanded state after rendering
        restoreExpandedState();

    } catch (e) {
        console.error('Failed to load tasks:', e);
        renderEmptyState(`Error loading tasks: ${e.message}`);
    }
}

/**
 * Restore expanded state for tasks that were previously expanded
 */
function restoreExpandedState() {
    for (const taskId of expandedTaskIds) {
        const details = document.getElementById(`task-details-${taskId}`);
        const taskItem = document.querySelector(`[data-task-id="${taskId}"]`);
        const expandIcon = taskItem?.querySelector('.expand-icon');

        if (details) {
            details.style.display = 'block';
            if (expandIcon) {
                expandIcon.textContent = '▼';
            }
        }
    }
}

// ============================================================================
// Rendering
// ============================================================================

function renderEmptyState(message) {
    taskList.innerHTML = `
        <div class="empty-state" style="
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        ">
            <p>${message}</p>
        </div>
    `;
}

function renderTasks(tasks) {
    taskList.innerHTML = tasks.map(task => renderTaskItem(task)).join('');
}

function renderTaskItem(task) {
    const statusIcon = getStatusIcon(task.status);
    const statusClass = getStatusClass(task.status);
    const taskId = task.id.substring(0, 8);
    const fullTaskId = task.id;
    const agentLabel = task.agent_id ? `${task.agent_id}/${task.skill || 'task'}` : 'Unknown';

    // Format timestamps if available
    const createdAt = task.created_at ? formatTime(task.created_at) : '';
    const updatedAt = task.updated_at ? formatTime(task.updated_at) : '';

    return `
        <div class="task-item" data-task-id="${fullTaskId}" onclick="window.toggleTaskDetails('${fullTaskId}')" style="
            padding: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 0.75rem;
            background: var(--bg-secondary);
            cursor: pointer;
            transition: background 0.2s;
        " onmouseover="this.style.background='var(--bg-tertiary)'" onmouseout="this.style.background='var(--bg-secondary)'">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span class="status-icon">${statusIcon}</span>
                    <span class="task-id" style="
                        font-family: monospace;
                        font-size: 0.875rem;
                        color: var(--text-secondary);
                    ">${taskId}</span>
                    <span class="expand-icon" style="color: var(--text-tertiary); font-size: 0.75rem;">▶</span>
                </div>
                <span class="status-badge ${statusClass}" style="
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    font-weight: 500;
                    text-transform: uppercase;
                ">${task.status}</span>
            </div>
            <div class="task-agent" style="
                font-size: 0.875rem;
                color: var(--text-primary);
                margin-bottom: 0.25rem;
            ">
                ${agentLabel}
            </div>
            ${task.message ? `
                <div class="task-message-preview" style="
                    font-size: 0.8125rem;
                    color: var(--text-secondary);
                    margin-top: 0.5rem;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    max-width: 100%;
                ">
                    ${escapeHtml(truncate(task.message, 100))}
                </div>
            ` : ''}

            <!-- Expandable details section (hidden by default) -->
            <div class="task-details" id="task-details-${fullTaskId}" style="
                display: none;
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border-color);
            ">
                <div style="font-size: 0.8125rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                    <strong>Full Task ID:</strong> <code style="background: var(--bg-tertiary); padding: 0.125rem 0.25rem; border-radius: 3px;">${fullTaskId}</code>
                </div>
                ${createdAt ? `<div style="font-size: 0.75rem; color: var(--text-tertiary);">Created: ${createdAt}</div>` : ''}
                ${updatedAt ? `<div style="font-size: 0.75rem; color: var(--text-tertiary);">Updated: ${updatedAt}</div>` : ''}

                ${task.message ? `
                    <div style="margin-top: 0.75rem;">
                        <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.25rem;">Message:</div>
                        <div class="task-message-full" style="
                            font-size: 0.8125rem;
                            color: var(--text-secondary);
                            padding: 0.75rem;
                            background: var(--bg-tertiary);
                            border-radius: 4px;
                            max-height: 200px;
                            overflow-y: auto;
                            white-space: pre-wrap;
                            word-break: break-word;
                        ">${escapeHtml(task.message)}</div>
                    </div>
                ` : ''}

                ${task.context ? `
                    <div style="margin-top: 0.75rem;">
                        <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.25rem;">Context:</div>
                        <div style="
                            font-size: 0.8125rem;
                            color: var(--text-secondary);
                            padding: 0.75rem;
                            background: var(--bg-tertiary);
                            border-radius: 4px;
                            max-height: 150px;
                            overflow-y: auto;
                            white-space: pre-wrap;
                        ">${escapeHtml(task.context)}</div>
                    </div>
                ` : ''}

                ${task.result ? `
                    <div style="margin-top: 0.75rem;">
                        <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.25rem;">Result:</div>
                        <div style="
                            font-size: 0.8125rem;
                            color: var(--text-secondary);
                            padding: 0.75rem;
                            background: var(--bg-tertiary);
                            border-radius: 4px;
                            max-height: 200px;
                            overflow-y: auto;
                            white-space: pre-wrap;
                            word-break: break-word;
                        ">${escapeHtml(typeof task.result === 'object' ? JSON.stringify(task.result, null, 2) : task.result)}</div>
                    </div>
                ` : ''}

                ${task.error ? `
                    <div style="margin-top: 0.75rem;">
                        <div style="font-size: 0.75rem; color: rgb(239, 68, 68); margin-bottom: 0.25rem;">Error:</div>
                        <div style="
                            font-size: 0.8125rem;
                            color: rgb(239, 68, 68);
                            padding: 0.75rem;
                            background: rgba(239, 68, 68, 0.1);
                            border-radius: 4px;
                            white-space: pre-wrap;
                        ">${escapeHtml(task.error)}</div>
                    </div>
                ` : ''}

                ${task.artifacts_count > 0 ? `
                    <div style="margin-top: 0.75rem;">
                        <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.25rem;">
                            Artifacts: ${task.artifacts_count}
                        </div>
                        <button onclick="event.stopPropagation(); loadTaskArtifacts('${fullTaskId}')" style="
                            padding: 0.375rem 0.75rem;
                            background: var(--bg-tertiary);
                            border: 1px solid var(--border-color);
                            border-radius: 4px;
                            color: var(--text-primary);
                            cursor: pointer;
                            font-size: 0.75rem;
                        ">Load Artifacts</button>
                        <div id="artifacts-${fullTaskId}" style="margin-top: 0.5rem;"></div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

function truncate(str, maxLength) {
    if (!str) return '';
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
}

function formatTime(isoString) {
    try {
        const date = new Date(isoString);
        return date.toLocaleString();
    } catch (e) {
        return isoString;
    }
}

// Toggle task details visibility
window.toggleTaskDetails = function(taskId) {
    const details = document.getElementById(`task-details-${taskId}`);
    const taskItem = document.querySelector(`[data-task-id="${taskId}"]`);
    const expandIcon = taskItem?.querySelector('.expand-icon');

    if (details) {
        const isVisible = details.style.display !== 'none';
        details.style.display = isVisible ? 'none' : 'block';
        if (expandIcon) {
            expandIcon.textContent = isVisible ? '▶' : '▼';
        }

        // Track expanded state for persistence across refreshes
        if (isVisible) {
            expandedTaskIds.delete(taskId);
        } else {
            expandedTaskIds.add(taskId);
        }
    }
};

// Load artifacts for a task
window.loadTaskArtifacts = async function(taskId) {
    const container = document.getElementById(`artifacts-${taskId}`);
    if (!container) return;

    container.innerHTML = '<div style="color: var(--text-tertiary); font-size: 0.75rem;">Loading...</div>';

    try {
        const response = await API.request(`/api/agent/tasks/${taskId}`);
        const artifacts = response.artifacts || [];

        if (artifacts.length === 0) {
            container.innerHTML = '<div style="color: var(--text-tertiary); font-size: 0.75rem;">No artifacts found</div>';
            return;
        }

        container.innerHTML = artifacts.map((artifact, idx) => `
            <div style="
                margin-top: 0.5rem;
                padding: 0.5rem;
                background: var(--bg-tertiary);
                border-radius: 4px;
                font-size: 0.8125rem;
            ">
                <div style="font-weight: 500; margin-bottom: 0.25rem;">${artifact.name || `Artifact ${idx + 1}`}</div>
                <pre style="
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    max-height: 150px;
                    overflow-y: auto;
                    font-size: 0.75rem;
                    color: var(--text-secondary);
                ">${escapeHtml(JSON.stringify(artifact.data || artifact, null, 2))}</pre>
            </div>
        `).join('');
    } catch (e) {
        container.innerHTML = `<div style="color: rgb(239, 68, 68); font-size: 0.75rem;">Error: ${e.message}</div>`;
    }
};

function getStatusIcon(status) {
    switch (status) {
        case 'completed': return kicon('check-circle');
        case 'failed': return kicon('x-circle');
        case 'canceled': return kicon('warning');
        case 'working': return kicon('hourglass');
        case 'submitted': return kicon('inbox');
        case 'input_required': return kicon('question');
        default: return '•';
    }
}

function getStatusClass(status) {
    const styles = {
        completed: 'background: rgba(34, 197, 94, 0.2); color: rgb(34, 197, 94);',
        failed: 'background: rgba(239, 68, 68, 0.2); color: rgb(239, 68, 68);',
        canceled: 'background: rgba(245, 158, 11, 0.2); color: rgb(245, 158, 11);',
        working: 'background: rgba(59, 130, 246, 0.2); color: rgb(59, 130, 246);',
        submitted: 'background: rgba(147, 51, 234, 0.2); color: rgb(147, 51, 234);',
        input_required: 'background: rgba(236, 72, 153, 0.2); color: rgb(236, 72, 153);',
    };
    return `style="${styles[status] || ''}"`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Badge Updates
// ============================================================================

function updateBadge(workingCount) {
    if (!tasksBadge) return;

    workingTasksCount = workingCount;

    if (workingCount > 0) {
        tasksBadge.textContent = workingCount.toString();
        tasksBadge.style.display = 'inline-block';
    } else {
        tasksBadge.style.display = 'none';
    }
}

/**
 * Manually trigger a badge update (called from SSE notifications)
 */
export function incrementWorkingTasks() {
    updateBadge(workingTasksCount + 1);
}

export function decrementWorkingTasks() {
    updateBadge(Math.max(0, workingTasksCount - 1));
}

// ============================================================================
// Activity Log (Observability Events)
// ============================================================================

// Track expanded activity events
let expandedActivityIds = new Set();

export async function loadActivityLog() {
    if (!activityList) return;

    try {
        const response = await API.request('/api/observability/events?limit=50');

        if (response.error) {
            renderActivityEmpty(response.error);
            return;
        }

        const events = response.events || [];

        if (events.length === 0) {
            renderActivityEmpty('No activity recorded yet');
            return;
        }

        renderActivityLog(events);

        // Restore expanded state
        restoreActivityExpandedState();

    } catch (e) {
        console.error('Failed to load activity log:', e);
        renderActivityEmpty(`Error loading activity: ${e.message}`);
    }
}

function renderActivityEmpty(message) {
    if (!activityList) return;
    activityList.innerHTML = `
        <div class="empty-state" style="
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        ">
            <p>${message}</p>
        </div>
    `;
}

function renderActivityLog(events) {
    if (!activityList) return;
    activityList.innerHTML = events.map(event => renderActivityItem(event)).join('');
}

function renderActivityItem(event) {
    const typeIcon = getActivityTypeIcon(event.event_type);
    const statusIcon = event.success ? kicon('checkmark') : kicon('x-mark');
    const statusColor = event.success ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)';
    const timestamp = formatActivityTime(event.timestamp);
    const eventId = event.event_id;
    const shortId = eventId.substring(0, 8);

    // Extract useful info from metadata
    const toolName = event.tool_name || event.metadata?.tool_name || '';
    const durationStr = event.duration_ms ? `${event.duration_ms}ms` : '';

    // Build summary line
    let summary = toolName || event.event_type;
    if (event.metadata?.model) {
        summary += ` → ${event.metadata.model}`;
    }
    if (event.metadata?.tools_count) {
        summary += ` (${event.metadata.tools_count} tools)`;
    }

    return `
        <div class="activity-item" data-event-id="${eventId}" onclick="window.toggleActivityDetails('${eventId}')" style="
            padding: 0.75rem 1rem;
            border-left: 3px solid ${statusColor};
            background: var(--bg-secondary);
            margin-bottom: 0.5rem;
            border-radius: 0 4px 4px 0;
            cursor: pointer;
            transition: background 0.2s;
        " onmouseover="this.style.background='var(--bg-tertiary)'" onmouseout="this.style.background='var(--bg-secondary)'">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1rem;">${typeIcon}</span>
                    <span style="font-size: 0.875rem; font-weight: 500;">${escapeHtml(summary)}</span>
                    <span class="activity-expand-icon" style="color: var(--text-tertiary); font-size: 0.75rem;">▶</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; font-size: 0.75rem; color: var(--text-secondary);">
                    ${durationStr ? `<span>${durationStr}</span>` : ''}
                    <span>${timestamp}</span>
                    <span style="color: ${statusColor}; font-weight: bold;">${statusIcon}</span>
                </div>
            </div>

            <!-- Expandable details -->
            <div class="activity-details" id="activity-details-${eventId}" style="
                display: none;
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid var(--border-color);
                font-size: 0.8125rem;
            ">
                <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.25rem 0.75rem; color: var(--text-secondary);">
                    <span style="color: var(--text-tertiary);">Event ID:</span>
                    <code style="font-size: 0.75rem;">${eventId}</code>

                    <span style="color: var(--text-tertiary);">Type:</span>
                    <span>${event.event_type}</span>

                    ${toolName ? `
                        <span style="color: var(--text-tertiary);">Tool:</span>
                        <span>${escapeHtml(toolName)}</span>
                    ` : ''}

                    ${event.duration_ms ? `
                        <span style="color: var(--text-tertiary);">Duration:</span>
                        <span>${event.duration_ms}ms</span>
                    ` : ''}

                    ${event.error_message ? `
                        <span style="color: var(--text-tertiary);">Error:</span>
                        <span style="color: rgb(239, 68, 68);">${escapeHtml(event.error_message)}</span>
                    ` : ''}
                </div>

                ${Object.keys(event.metadata || {}).length > 0 ? `
                    <div style="margin-top: 0.5rem;">
                        <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.25rem;">Metadata:</div>
                        <pre style="
                            margin: 0;
                            padding: 0.5rem;
                            background: var(--bg-tertiary);
                            border-radius: 4px;
                            font-size: 0.7rem;
                            max-height: 150px;
                            overflow-y: auto;
                            white-space: pre-wrap;
                            word-break: break-word;
                        ">${escapeHtml(JSON.stringify(event.metadata, null, 2))}</pre>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

function restoreActivityExpandedState() {
    for (const eventId of expandedActivityIds) {
        const details = document.getElementById(`activity-details-${eventId}`);
        const item = document.querySelector(`[data-event-id="${eventId}"]`);
        const expandIcon = item?.querySelector('.activity-expand-icon');

        if (details) {
            details.style.display = 'block';
            if (expandIcon) expandIcon.textContent = '▼';
        }
    }
}

window.toggleActivityDetails = function(eventId) {
    const details = document.getElementById(`activity-details-${eventId}`);
    const item = document.querySelector(`[data-event-id="${eventId}"]`);
    const expandIcon = item?.querySelector('.activity-expand-icon');

    if (details) {
        const isVisible = details.style.display !== 'none';
        details.style.display = isVisible ? 'none' : 'block';
        if (expandIcon) {
            expandIcon.textContent = isVisible ? '▶' : '▼';
        }

        // Track expanded state
        if (isVisible) {
            expandedActivityIds.delete(eventId);
        } else {
            expandedActivityIds.add(eventId);
        }
    }
};

function getActivityTypeIcon(eventType) {
    switch (eventType) {
        case 'tool_call': return kicon('wrench');
        case 'tool_response': return kicon('inbox');
        case 'agent_response': return kicon('robot');
        case 'llm_call': return kicon('sparkles');
        case 'error': return kicon('x-circle');
        case 'metric': return kicon('chart-bar');
        default: return '•';
    }
}

function formatActivityTime(timestamp) {
    if (!timestamp) return '';
    try {
        const date = new Date(timestamp);
        const now = new Date();

        // If today, show time only
        if (date.toDateString() === now.toDateString()) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        // Otherwise show date and time
        return date.toLocaleString([], {
            month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    } catch (e) {
        return timestamp;
    }
}
