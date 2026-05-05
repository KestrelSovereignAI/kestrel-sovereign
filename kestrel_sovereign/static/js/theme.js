/**
 * UI theme + i18n label hydration (epic #986, sub-issue #990).
 *
 * Walks the DOM for [data-label-key] / [data-label-attr-*] elements and
 * replaces their text content / attribute values with labels resolved
 * from /api/ui/theme.
 *
 * Inline HTML keeps the legacy theme as fallback so JS-disabled or
 * pre-API-call rendering is unsurprising.
 *
 * Exposes window.KestrelTheme:
 *   - applyTheme(theme, locale): swap to a different (theme, locale)
 *   - getCurrentTheme(), getCurrentLocale(): read state
 *   - listAvailableThemes(): fetches /api/ui/themes
 *
 * Picker UI (#991) reads/writes localStorage and calls applyTheme().
 */
(function () {
  'use strict';

  const STORAGE_THEME_KEY = 'kestrel_ui_theme';
  const STORAGE_LOCALE_KEY = 'kestrel_ui_locale';
  const DEFAULT_THEME = 'legacy';
  const DEFAULT_LOCALE = 'en';

  let _currentTheme = DEFAULT_THEME;
  let _currentLocale = DEFAULT_LOCALE;
  let _currentLabels = {};
  let _currentFallbackKeys = [];

  function readStoredTheme() {
    try {
      return localStorage.getItem(STORAGE_THEME_KEY) || DEFAULT_THEME;
    } catch (_) {
      return DEFAULT_THEME;
    }
  }

  function readStoredLocale() {
    try {
      return localStorage.getItem(STORAGE_LOCALE_KEY) || DEFAULT_LOCALE;
    } catch (_) {
      return DEFAULT_LOCALE;
    }
  }

  function writeStoredTheme(theme) {
    try {
      localStorage.setItem(STORAGE_THEME_KEY, theme);
    } catch (_) {
      /* localStorage disabled — picker still works for the session */
    }
  }

  function writeStoredLocale(locale) {
    try {
      localStorage.setItem(STORAGE_LOCALE_KEY, locale);
    } catch (_) {
      /* localStorage disabled */
    }
  }

  /**
   * Apply a label map to the DOM. Walks every element with
   * [data-label-key] (textContent) and [data-label-attr-*] (attributes).
   * Elements whose key is missing from the map are left alone (legacy
   * inline fallback applies).
   */
  function hydrate(labels) {
    if (!labels || typeof labels !== 'object') return 0;
    let applied = 0;

    // textContent labels
    document.querySelectorAll('[data-label-key]').forEach((el) => {
      const key = el.getAttribute('data-label-key');
      if (key && Object.prototype.hasOwnProperty.call(labels, key)) {
        el.textContent = labels[key];
        applied++;
      }
    });

    // attribute labels: data-label-attr-<attr>="key"
    const ATTR_PREFIX = 'data-label-attr-';
    document.querySelectorAll('[data-label-attr-title], [data-label-attr-placeholder], [data-label-attr-alt], [data-label-attr-aria-label]').forEach((el) => {
      for (const attr of el.attributes) {
        if (!attr.name.startsWith(ATTR_PREFIX)) continue;
        const targetAttr = attr.name.slice(ATTR_PREFIX.length);
        const key = attr.value;
        if (key && Object.prototype.hasOwnProperty.call(labels, key)) {
          el.setAttribute(targetAttr, labels[key]);
          applied++;
        }
      }
    });

    return applied;
  }

  async function fetchTheme(theme, locale) {
    const params = new URLSearchParams({ theme, locale });
    const resp = await fetch('/api/ui/theme?' + params.toString(), {
      credentials: 'same-origin',
    });
    if (!resp.ok) {
      throw new Error('theme fetch failed: ' + resp.status);
    }
    return resp.json();
  }

  /**
   * Load the requested (theme, locale) and apply to the DOM.
   *
   * On error: leaves the inline legacy text in place. We deliberately
   * don't fall back to a different theme silently — the inline HTML is
   * already a working fallback.
   */
  async function applyTheme(theme, locale) {
    const t = theme || readStoredTheme();
    const l = locale || readStoredLocale();
    let bundle;
    try {
      bundle = await fetchTheme(t, l);
    } catch (err) {
      console.warn('[theme] fetch failed, keeping inline labels:', err);
      window.dispatchEvent(new CustomEvent('themeerror', {
        detail: { theme: t, locale: l, error: String(err) },
      }));
      return null;
    }
    _currentTheme = bundle.theme;
    _currentLocale = bundle.locale;
    _currentLabels = bundle.labels || {};
    _currentFallbackKeys = bundle.fallback_keys || [];

    writeStoredTheme(bundle.theme);
    writeStoredLocale(bundle.locale);

    const applied = hydrate(_currentLabels);

    if (_currentFallbackKeys.length > 0) {
      console.info(
        '[theme] %s/%s: %d keys fell back to legacy: %o',
        bundle.theme, bundle.locale,
        _currentFallbackKeys.length,
        _currentFallbackKeys.slice(0, 5),
      );
    }

    window.dispatchEvent(new CustomEvent('themechange', {
      detail: {
        theme: bundle.theme,
        locale: bundle.locale,
        applied: applied,
        fallback_keys: _currentFallbackKeys,
      },
    }));

    return bundle;
  }

  async function listAvailableThemes() {
    const resp = await fetch('/api/ui/themes', { credentials: 'same-origin' });
    if (!resp.ok) {
      throw new Error('themes list fetch failed: ' + resp.status);
    }
    const body = await resp.json();
    return body.themes || [];
  }

  window.KestrelTheme = {
    applyTheme,
    listAvailableThemes,
    getCurrentTheme: () => _currentTheme,
    getCurrentLocale: () => _currentLocale,
    getCurrentLabels: () => Object.assign({}, _currentLabels),
    getFallbackKeys: () => _currentFallbackKeys.slice(),
    // internals exposed for tests
    _hydrate: hydrate,
  };

  function init() {
    // Don't block first paint: the inline HTML already shows legacy labels.
    // Hydrate asynchronously, then any subsequent renders see the chosen theme.
    applyTheme();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
