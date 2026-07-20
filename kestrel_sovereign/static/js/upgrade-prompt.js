/**
 * Host tier-gate ("upgrade required") rendering helpers (#2232).
 *
 * A host's policy layer can tier-gate an approval scope and reject it with a
 * structured 403 whose body carries everything needed for an upsell:
 *
 *     403 { code: 'upgrade_required', action, required_tier, current_tier,
 *           message, upgrade_href }
 *
 * Kestrel renders that envelope as an upgrade prompt (the `message` plus a
 * link/button to `upgrade_href`) instead of a generic failure. Kestrel has NO
 * knowledge of billing — it renders exactly what the envelope says. Hosts
 * without gating never emit these 403s, so the standalone console is
 * unaffected.
 *
 * Reactive rendering (react to the 403) is the MVP; an optional proactive gate
 * map (`capabilities.approvalScopes`) lets the UI badge/disable gated scopes
 * up-front. See `getApprovalScopeGates`.
 */

import API from './api.js';

export const UPGRADE_REQUIRED_CODE = 'upgrade_required';

function esc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Normalize an API error into upgrade info, or null when it is not a
 * tier-gate rejection. The shared `ApiError` (api_client.mjs) retains `status`
 * and the parsed `body`, so a caller can inspect the structured envelope here.
 */
export function extractUpgradeRequired(error) {
    const body = error && error.body;
    const nested = body && typeof body === 'object'
        && body.error && typeof body.error === 'object'
        && !Array.isArray(body.error)
        ? body.error
        : null;
    // Retain host-specific top-level fields while preferring the canonical
    // nested error fields when both forms are present during migration.
    const payload = body && typeof body === 'object' && !Array.isArray(body)
        ? { ...body, ...(nested || {}) }
        : null;
    if (
        !error ||
        error.status !== 403 ||
        !payload ||
        (error.code || payload.code) !== UPGRADE_REQUIRED_CODE
    ) {
        return null;
    }
    return {
        action: payload.action || null,
        requiredTier: payload.required_tier || null,
        currentTier: payload.current_tier || null,
        message: payload.message || 'This action requires an upgrade.',
        upgradeHref: sanitizeUpgradeHref(payload.upgrade_href),
    };
}

/**
 * Allow only safe link targets for the upgrade CTA: a same-origin relative
 * path (`/upgrade.html`, but NOT scheme-relative `//host`) or an absolute
 * http(s) URL. Everything else (javascript:, data:, vbscript:, malformed)
 * returns null — renderers then omit the link but keep the message (codex
 * P2 on #2280: the href is server-provided and lands in innerHTML).
 */
export function sanitizeUpgradeHref(href) {
    if (!href || typeof href !== 'string') return null;
    const trimmed = href.trim();
    if (trimmed.startsWith('/') && !trimmed.startsWith('//')) return trimmed;
    try {
        const url = new URL(trimmed);
        if (url.protocol === 'http:' || url.protocol === 'https:') return trimmed;
    } catch (_) { /* not an absolute URL */ }
    return null;
}

/**
 * Optional proactive gate map. A host declares which approval scopes are
 * gated and at what tier via `capabilities.approvalScopes`, e.g.
 *   { session: 'premium', always: 'sovereign' }
 * Returns {} when absent (standalone console → nothing gated up-front).
 */
export function getApprovalScopeGates() {
    const caps = (API && API.getCapabilities && API.getCapabilities()) || {};
    const gates = caps.approvalScopes;
    if (!gates || typeof gates !== 'object') return {};
    const out = {};
    for (const [scope, tier] of Object.entries(gates)) {
        if (tier) out[scope] = String(tier);
    }
    return out;
}

/**
 * A small inline badge showing the tier a gated option needs. Rendered next
 * to a disabled scope label so the user SEES that session/always exist.
 */
export function tierBadgeHtml(tier) {
    if (!tier) return '';
    return `<span class="upgrade-tier-badge" style="` +
        `display: inline-block; margin-left: 0.4rem; padding: 0.05rem 0.4rem;` +
        `border-radius: 999px; font-size: 0.7rem; font-weight: 600;` +
        `background: var(--warning, #d97706); color: white; vertical-align: middle;">` +
        `${esc(tier)}</span>`;
}

/**
 * A banner rendering the upgrade message with a link/button to `upgrade_href`.
 * Used inside the approval modal — it stays open showing this on rejection.
 */
export function upgradeBannerHtml(upgrade) {
    if (!upgrade) return '';
    const link = upgrade.upgradeHref
        ? `<a href="${esc(upgrade.upgradeHref)}" target="_blank" rel="noopener noreferrer" ` +
          `class="btn btn-primary upgrade-required-link" style="` +
          `display: inline-block; margin-top: 0.5rem; padding: 0.4rem 0.9rem;` +
          `border-radius: 8px; background: var(--accent-color, #2563eb); color: white;` +
          `text-decoration: none; font-size: 0.85rem; font-weight: 600;">` +
          `Upgrade${upgrade.requiredTier ? ' to ' + esc(upgrade.requiredTier) : ''}</a>`
        : '';
    return `<div class="upgrade-required-banner" role="alert" style="` +
        `margin-top: 0.75rem; padding: 0.75rem 1rem; border-radius: 8px;` +
        `border: 1px solid var(--warning, #d97706);` +
        `background: rgba(217, 119, 6, 0.12);">` +
        `<div style="font-size: 0.875rem;">${esc(upgrade.message)}</div>` +
        `${link}</div>`;
}

/**
 * One-line HTML for a toast (used where a full banner doesn't fit, e.g. the
 * Approvals / Security panel rows). Includes an inline link when a href is
 * present. Callers must pass this only to ``Toast.showTrustedHtml``; ordinary
 * toast methods intentionally render their messages through ``textContent``.
 */
export function upgradeToastHtml(upgrade) {
    if (!upgrade) return '';
    const link = upgrade.upgradeHref
        ? ` <a href="${esc(upgrade.upgradeHref)}" target="_blank" rel="noopener noreferrer" ` +
          `style="color: white; text-decoration: underline;">` +
          `Upgrade${upgrade.requiredTier ? ' to ' + esc(upgrade.requiredTier) : ''}</a>`
        : '';
    return `${esc(upgrade.message)}${link}`;
}
