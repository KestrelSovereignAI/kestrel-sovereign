---
type: Design Note
title: Kestrel Landing Page HTML Structure
description: This document translates the landing-page copy into an implementation-ready
  HTML structure for engineering.
resource: /docs/design/launch/LANDING_PAGE_HTML_STRUCTURE.md
tags:
- docs
- design
- design-note
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Kestrel Landing Page HTML Structure

This document translates the landing-page copy into an implementation-ready HTML structure for engineering.

## Goal

Provide a simple, semantic page structure that can be implemented quickly without re-deciding the message order.

## Page Principles

- Keep the page linear and easy to scan.
- Do not overload the hero with product chrome.
- Use semantic HTML first; keep JavaScript optional.
- Design for fast implementation and easy copy revision.
- Support both the full publish-ready page and a simpler one-screen fallback.

## Recommended File Shape

```html
<main class="landing-page">
  <section class="hero">
    ...
  </section>

  <section class="reframe">
    ...
  </section>

  <section class="foundation">
    ...
  </section>

  <section class="pillars">
    ...
  </section>

  <section class="open-boundary">
    ...
  </section>

  <section class="proof">
    ...
  </section>

  <section class="audience-fit">
    ...
  </section>

  <section class="final-cta">
    ...
  </section>
</main>
```

## Suggested Semantic Structure

### 1. Hero

```html
<section class="hero" aria-labelledby="hero-title">
  <div class="container">
    <p class="eyebrow">The open foundation for Sovereign AI</p>
    <h1 id="hero-title">Own your AI.</h1>
    <p class="subheadline">Kestrel is the open foundation for Sovereign AI: identity, memory, privacy, and governance that belong to the user, not the platform.</p>
    <p class="hero-body">Most AI today is rented intelligence...</p>
    <div class="cta-row">
      <a class="btn btn-primary" href="/QUICKSTART.md">Read the quickstart</a>
      <a class="btn btn-secondary" href="https://github.com/KestrelSovereignAI/kestrel-sovereign">Inspect the repo</a>
    </div>
  </div>
</section>
```

#### Notes

- If a product screenshot is used, place it below the copy block, not above the headline.
- Keep the CTA row visible without scrolling on desktop.

### 2. Reframe

```html
<section class="reframe" aria-labelledby="reframe-title">
  <div class="container narrow">
    <h2 id="reframe-title">AI should not belong to the platform.</h2>
    <p>As AI systems become more personal and more trusted...</p>
  </div>
</section>
```

### 3. Foundation Layer Section

```html
<section class="foundation" aria-labelledby="foundation-title">
  <div class="container narrow">
    <h2 id="foundation-title">Not another wrapper. The foundation layer.</h2>
    <p>Kestrel is an open-source framework for building sovereign agents...</p>
  </div>
</section>
```

### 4. Pillars Grid

```html
<section class="pillars" aria-labelledby="pillars-title">
  <div class="container">
    <h2 id="pillars-title">The three durable layers of Sovereign AI</h2>
    <div class="pillar-grid">
      <article class="pillar-card">
        <h3>Portable identity</h3>
        <p>Every agent has a cryptographic identity...</p>
      </article>
      <article class="pillar-card">
        <h3>Persistent memory you own</h3>
        <p>Memory lives outside the model...</p>
      </article>
      <article class="pillar-card">
        <h3>Constitutional governance</h3>
        <p>Rules are enforced above the LLM...</p>
      </article>
    </div>
  </div>
</section>
```

#### Notes

- On mobile, stack the cards vertically.
- Avoid turning these into feature lists.

### 5. Open Boundary Section

```html
<section class="open-boundary" aria-labelledby="open-title">
  <div class="container narrow">
    <h2 id="open-title">What is open at launch</h2>
    <p>Kestrel open-sources the sovereign agent framework itself...</p>
  </div>
</section>
```

#### Notes

- Keep this section easy to update if open-core wording changes.
- Consider a short supporting caption or callout block instead of a table.

### 6. Proof Section

```html
<section class="proof" aria-labelledby="proof-title">
  <div class="container narrow">
    <h2 id="proof-title">This is not a thought experiment.</h2>
    <p>Kestrel is already being used in a real clinical setting...</p>
  </div>
</section>
```

#### Notes

- This is a credibility section, not a customer-story section.
- If a stat block is added later, keep it restrained.

### 7. Audience Fit Section

```html
<section class="audience-fit" aria-labelledby="audience-title">
  <div class="container narrow">
    <h2 id="audience-title">Built for developers working on durable AI systems</h2>
    <p>Kestrel is for builders creating AI systems that need continuity...</p>
  </div>
</section>
```

### 8. Final CTA Section

```html
<section class="final-cta" aria-labelledby="cta-title">
  <div class="container narrow">
    <h2 id="cta-title">Start where the trust surface becomes visible.</h2>
    <p>If AI is going to become durable infrastructure...</p>
    <div class="cta-row">
      <a class="btn btn-primary" href="/QUICKSTART.md">Read the quickstart</a>
      <a class="btn btn-secondary" href="https://github.com/KestrelSovereignAI/kestrel-sovereign">Inspect the repo</a>
      <a class="text-link" href="#open-title">See what is open</a>
    </div>
  </div>
</section>
```

## Suggested Content Mapping

- `PUBLISH_READY_LANDING_PAGE_COPY.md` should be the source of truth for final prose.
- `LANDING_PAGE_WIREFRAME.md` should be used when revisiting page order or section logic.
- `SIMPLE_LAUNCH_PAGE_ONE_SCREEN.md` should be used if the team chooses a stripped-down one-screen implementation.

## CSS Guidance

- Use a narrow max width for prose sections.
- Give the hero the most visual breathing room.
- Keep heading hierarchy strong and simple.
- Use one accent color for CTAs and one neutral palette for everything else.
- Avoid generic startup gradients unless they support the existing Kestrel visual direction.

## Responsive Guidance

- Mobile order should match desktop order exactly.
- CTAs should stack vertically on smaller screens.
- Pillar cards should become a single-column stack below tablet width.
- Keep hero copy readable before any screenshot appears.

## Minimal One-Screen Variant

If engineering chooses the one-screen launch page, the structure can collapse to this:

```html
<main class="landing-page landing-page--simple">
  <section class="hero hero--simple" aria-labelledby="hero-title">
    <div class="container narrow">
      <p class="eyebrow">The open foundation for Sovereign AI</p>
      <h1 id="hero-title">Own your AI.</h1>
      <p class="subheadline">Kestrel is the open foundation for Sovereign AI...</p>
      <p class="hero-body">Most AI today is rented intelligence...</p>
      <div class="cta-row">
        <a class="btn btn-primary" href="/QUICKSTART.md">Read the quickstart</a>
        <a class="btn btn-secondary" href="https://github.com/KestrelSovereignAI/kestrel-sovereign">Inspect the repo</a>
      </div>
    </div>
  </section>
</main>
```

## Engineering Handoff Notes

- Build the page so copy can be replaced without changing the section structure.
- Keep section IDs stable if marketing links or nav anchors get added later.
- Favor static HTML/CSS for the first implementation unless there is a strong reason to add framework behavior.
- If launch timing gets tight, ship the one-screen version first and expand later.