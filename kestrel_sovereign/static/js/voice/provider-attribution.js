/**
 * Derive the honest provider label shown beside the voice catalog.
 *
 * The route response is authoritative: the active chat vendor may differ from
 * the provider that owns the selected voice catalog when Realtime is
 * unavailable and the resolver falls back to Pipeline.
 */

export function providerDisplayName(provider, providerNames = {}) {
  const raw = String(provider || '').trim();
  if (!raw) return '';
  const base = raw.replace(/_realtime$/i, '');
  return providerNames[base] || providerNames[raw] || base;
}

export function describeVoiceCatalog(
  route = {},
  providerName = '',
  requestedMode = 'auto',
  providerNames = {},
) {
  let displayProvider = providerName;
  if (route.path === 'realtime') {
    displayProvider = route.conversation_capabilities?.[providerName]?.vendor
      || providerName;
  }
  const displayName = providerDisplayName(displayProvider, providerNames);

  if (route.path === 'realtime') {
    if (!displayName) {
      return { label: 'Realtime provider', title: '', isFallback: false };
    }
    return {
      label: `${displayName} Realtime`,
      title: `${displayName} owns this realtime voice catalog and the full voice turn.`,
      isFallback: false,
    };
  }

  if (route.path === 'pipeline') {
    const isFallback = requestedMode !== 'pipeline';
    const owner = displayName || (isFallback
      ? 'An automatically selected provider'
      : 'The selected provider');
    const label = displayName
      ? `${displayName} ${isFallback ? 'fallback' : 'Pipeline'}`
      : (isFallback ? 'Pipeline fallback' : 'Pipeline');
    return {
      label,
      title: isFallback
        ? (route.reason || `${owner} supplies speech because the requested Realtime route is unavailable.`)
        : `${owner} supplies speech for the explicitly selected Pipeline path.`,
      isFallback,
    };
  }

  if (route.path === 'local') {
    if (!displayName) {
      return { label: 'Local voice', title: '', isFallback: false };
    }
    return {
      label: `${displayName} Local`,
      title: `${displayName} supplies speech on the local voice path.`,
      isFallback: false,
    };
  }

  if (!displayName) return { label: '', title: '', isFallback: false };
  return { label: displayName, title: '', isFallback: false };
}

export function applyVoiceCatalogAttribution(
  element,
  route,
  providerName,
  requestedMode = 'auto',
  providerNames = {},
) {
  const attribution = describeVoiceCatalog(
    route,
    providerName,
    requestedMode,
    providerNames,
  );
  if (!element) return attribution;

  element.textContent = attribution.label ? `· ${attribution.label}` : '';
  element.title = attribution.title;
  element.hidden = !attribution.label;
  element.classList.toggle('is-fallback', attribution.isFallback);
  return attribution;
}
