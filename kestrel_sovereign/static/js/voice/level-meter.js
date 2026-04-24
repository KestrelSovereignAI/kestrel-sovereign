/**
 * level-meter.js — Tiny visual input-level meter.
 *
 *   const destroy = mountLevelMeter(containerEl, capture.getLevel, { bars: 12 });
 *   // later…
 *   destroy();
 *
 * Reads a 0..1 peak amplitude each animation frame and renders N horizontal
 * bars lit from the left as the level rises. Pure DOM — no canvas, no deps.
 */

export function mountLevelMeter(container, getLevelFn, { bars = 12 } = {}) {
  if (!container) return () => {};
  container.innerHTML = '';
  container.classList.add('kestrel-voice-level-meter');

  const nodes = [];
  for (let i = 0; i < bars; i++) {
    const bar = document.createElement('span');
    bar.className = 'kestrel-voice-level-bar';
    bar.dataset.index = String(i);
    container.appendChild(bar);
    nodes.push(bar);
  }

  let rafId = 0;
  let running = true;

  function tick() {
    if (!running) return;
    const raw = getLevelFn();
    // Slight perceptual curve: dB-ish response feels more natural than linear.
    const level = Math.sqrt(Math.max(0, Math.min(1, raw)));
    const lit = Math.round(level * bars);
    for (let i = 0; i < bars; i++) {
      nodes[i].classList.toggle('kestrel-voice-level-bar--lit', i < lit);
    }
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);

  return () => {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    container.innerHTML = '';
    container.classList.remove('kestrel-voice-level-meter');
  };
}
