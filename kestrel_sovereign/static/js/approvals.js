/**
 * Kestrel Sovereign Console - Approvals Panel (epic #1290, D2)
 *
 * A persistent, reviewable approval queue that replaces typing approvals
 * on the CLI. Each pending agent action is a row with the agent name, a
 * one-line summary, the FULL command/diff, and three actions:
 *   - Approve        (once)
 *   - Reject
 *   - Approve & remember  (adds a scoped, revocable auto-approve rule)
 *
 * Also surfaces the remembered rules (with revoke) and the immutable
 * auto-approve audit feed so the Sovereign can see exactly what ran.
 */

import API from './api.js';
import { Toast } from './ui.js';

let listEl = null;
let rulesEl = null;
let auditEl = null;
let badgeEl = null;
let refreshBtn = null;

function esc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function updateBadge(n) {
    if (!badgeEl) return;
    badgeEl.textContent = String(n);
    badgeEl.style.display = n > 0 ? 'inline-block' : 'none';
}

export function initApprovals() {
    if (!API.hasCapability('permissions')) return;
    listEl = document.getElementById('approvals-list');
    rulesEl = document.getElementById('approvals-rules');
    auditEl = document.getElementById('approvals-audit');
    badgeEl = document.getElementById('approvals-pending-badge');
    refreshBtn = document.getElementById('btn-refresh-approvals');
    refreshBtn?.addEventListener('click', () => loadApprovals());
}

async function decide(approvalId, approved, scope, remember) {
    try {
        const res = await API.request('/api/security/approve', {
            method: 'POST',
            body: JSON.stringify({
                approval_id: approvalId,
                approved,
                scope: scope || 'once',
                remember: !!remember,
            }),
        });
        if (res && res.success) {
            if (res.remembered && res.remembered.skipped) {
                Toast.success(`Approved (not remembered: ${res.remembered.skipped})`);
            } else if (res.remembered) {
                Toast.success(
                    `Approved & remembered: ${res.remembered.pattern}`
                );
            } else {
                Toast.success(approved ? `Approved (${scope})` : 'Rejected');
            }
        } else {
            Toast.error('Decision failed');
        }
    } catch (e) {
        Toast.error(`Decision failed: ${e.message || e}`);
    }
    loadApprovals();
}

async function revokeRule(ruleId) {
    try {
        const res = await API.request(
            `/api/security/auto-approve/rules/${ruleId}`,
            { method: 'DELETE' }
        );
        if (res && res.success) Toast.success(`Revoked rule ${ruleId}`);
        else Toast.error('Revoke failed');
    } catch (e) {
        Toast.error(`Revoke failed: ${e.message || e}`);
    }
    loadApprovals();
}

// Exposed for inline onclick handlers in rendered rows.
window.Approvals = { decide, revokeRule, load: loadApprovals };

function renderPending(pending) {
    if (!pending.length) {
        listEl.innerHTML =
            '<p class="text-muted">No pending approvals. The Sovereign is only observing.</p>';
        return;
    }
    listEl.innerHTML = pending
        .map((r) => {
            const agent = esc(r.agent_name || 'agent');
            const summary = esc(r.action_summary || `${r.feature}.${r.tool}`);
            const preview = esc(r.command_preview || '');
            const id = esc(r.id);
            return `
            <div class="approval-row" style="border:1px solid var(--border);border-radius:8px;padding:1rem;margin-bottom:1rem;">
              <div class="row-between mb-2">
                <strong>${agent}</strong>
                <span class="text-muted text-sm">${esc(r.timestamp)}</span>
              </div>
              <div class="mb-2">${summary}</div>
              <pre style="background:var(--bg-secondary);padding:0.75rem;border-radius:6px;overflow:auto;max-height:240px;white-space:pre-wrap;">${preview}</pre>
              <div class="row-flex" style="gap:0.5rem;margin-top:0.75rem;">
                <button class="btn btn-primary text-sm" onclick="Approvals.decide('${id}',true,'once',false)">Approve</button>
                <button class="btn btn-secondary text-sm" onclick="Approvals.decide('${id}',false,'once',false)">Reject</button>
                <button class="btn btn-secondary text-sm" onclick="Approvals.decide('${id}',true,'once',true)" title="Approve and add a scoped, revocable auto-approve rule">Approve &amp; remember</button>
              </div>
            </div>`;
        })
        .join('');
}

function renderRules(rules) {
    if (!rules.length) {
        rulesEl.innerHTML =
            '<p class="text-muted text-sm">No remembered rules. Operator-seeded patterns live in kestrel.toml.</p>';
        return;
    }
    rulesEl.innerHTML = rules
        .map(
            (r) => `
        <div class="row-between" style="border:1px solid var(--border);border-radius:6px;padding:0.5rem 0.75rem;margin-bottom:0.5rem;">
          <div>
            <code>${esc(r.pattern)}</code>
            <span class="text-muted text-sm"> · ${esc(r.agent || 'any agent')} · ${esc(r.repo_scope || 'any repo')}</span>
          </div>
          <button class="btn btn-secondary text-sm" onclick="Approvals.revokeRule(${r.id})">Revoke</button>
        </div>`
        )
        .join('');
}

function renderAudit(rows) {
    if (!rows.length) {
        auditEl.innerHTML =
            '<p class="text-muted text-sm">No auto-approved invocations yet.</p>';
        return;
    }
    auditEl.innerHTML = rows
        .map((a) => {
            const ec =
                a.exit_code === null || a.exit_code === undefined
                    ? 'running…'
                    : `exit ${a.exit_code}`;
            return `
        <div style="border:1px solid var(--border);border-radius:6px;padding:0.5rem 0.75rem;margin-bottom:0.5rem;">
          <div class="text-sm"><code>${esc(a.command)}</code></div>
          <div class="text-muted text-sm">${esc(a.agent_did || '?')} · ${esc(a.created_at)} · ${esc(ec)} · ${esc(a.rule_source || '')}</div>
        </div>`;
        })
        .join('');
}

export async function loadApprovals() {
    if (!API.hasCapability('permissions')) return;
    if (!listEl) initApprovals();
    try {
        const pend = await API.request('/api/security/pending');
        const pending = (pend && pend.pending) || [];
        renderPending(pending);
        updateBadge(pending.length);
    } catch (e) {
        if (listEl) listEl.innerHTML = `<p class="text-muted">Failed to load: ${esc(e.message || e)}</p>`;
    }
    try {
        const rl = await API.request('/api/security/auto-approve/rules');
        renderRules((rl && rl.rules) || []);
    } catch (e) {
        if (rulesEl) rulesEl.innerHTML = `<p class="text-muted text-sm">Failed to load rules.</p>`;
    }
    try {
        const au = await API.request('/api/security/auto-approve/audit?limit=25');
        renderAudit((au && au.audit) || []);
    } catch (e) {
        if (auditEl) auditEl.innerHTML = `<p class="text-muted text-sm">Failed to load audit.</p>`;
    }
}
