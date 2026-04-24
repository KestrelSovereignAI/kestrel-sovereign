# Kestrel Landing Page Wireframe

Prepared for the open-source launch on Apr 23 / May 7 planning path.

## Goal

Create a public-facing landing page that explains Kestrel in the right order:

1. emotional hook
2. category frame
3. what is actually open
4. why it matters
5. proof that it is real
6. clear next step

This draft is intentionally copy-first. It is meant to help Noel, Gabi, and engineering align before anyone overbuilds the page.

## Page Strategy

- The page should feel like an opening argument, not a product catalog.
- The first job is to make the reader care about ownership and trust in AI.
- The second job is to explain Kestrel as the open foundation for that problem.
- The third job is to give a simple next step: read the quickstart, inspect the repo, understand what is open.

## Wireframe

### 1. Hero

#### Headline

Own your AI.

#### Subheadline

Kestrel is the open foundation for Sovereign AI: identity, memory, privacy, and governance that belong to the user, not the platform.

#### Supporting copy

Most AI today is rented intelligence. The model may help you, but the identity, memory, and rules still belong to someone else. Kestrel gives developers a way to build agents whose durable layers are portable, inspectable, and governed above the model.

#### Primary CTA

Read the quickstart

#### Secondary CTA

See what is open

#### Optional eyebrow

The open foundation for Sovereign AI

#### Notes

- Keep this section visually sparse.
- Do not open with a feature grid.
- If a screenshot is used above the fold, it should support credibility, not compete with the headline.

### 2. The Reframe

#### Section title

AI should not belong to the platform.

#### Body copy

As AI systems become more personal and more trusted, the most important layers stop being the model alone. Identity, memory, privacy, and governance become the real source of continuity and trust. Today, those layers are usually trapped inside a provider. Kestrel exists to separate them from the model so they can belong to the person or organization being served.

#### Notes

- This is where the reader should understand why the problem matters now.
- Keep the tone declarative, not academic.

### 3. What Kestrel Is

#### Section title

Not another wrapper. The foundation layer.

#### Body copy

Kestrel is an open-source framework for building sovereign agents.

It is the foundation layer for AI systems that need continuity, accountability, and user ownership. It gives developers a way to run agents with portable identity, persistent memory, and constitutional governance enforced above the LLM.

#### Optional pull quote

Not rented intelligence. Owned intelligence.

### 4. The Three Pillars

#### Section title

The durable layers of Sovereign AI

#### Pillar 1

##### Portable identity

Every agent has a cryptographic identity that is not trapped inside one platform or provider.

#### Pillar 2

##### Persistent memory you own

Memory lives outside the model, remains portable across environments, and can be governed by the user or organization that depends on it.

#### Pillar 3

##### Constitutional governance

Rules are enforced above the LLM, auditable in principle, and designed to make behavior inspectable rather than implicit.

#### Notes

- Keep these cards simple.
- Do not overload them with implementation details.
- If needed, link deeper technical docs from this section rather than expanding inline.

### 5. Why It Matters

#### Section title

The trust problem is getting worse, not better.

#### Body copy

The more useful an AI system becomes, the more dangerous it is for the user to have no durable claim on its identity, memory, or behavior. If those layers belong to the platform, the user is always renting the relationship. Kestrel is built for a different future: one where those layers can remain portable, inspectable, and user-aligned even as the model changes.

### 6. What Is Open

#### Section title

What is open at launch

#### Body copy

Kestrel open-sources the sovereign agent framework itself: identity, memory, constitutional governance, and the runtime needed to run sovereign agents.

Clinical deployments and product-specific integrations are not the headline of the launch. They serve as proof that the system is real, not as the center of the message.

#### Optional supporting line

The foundation is open. Productized deployments and domain-specific integrations remain separate.

#### Notes

- This section should align with the final open-core decision language.
- If #612 changes, update this section first.

### 7. Proof

#### Section title

This is not a thought experiment.

#### Body copy

Kestrel is already being used in a real clinical setting, where continuity, trust, and governance are not optional. That proof matters because this launch should not read like speculative architecture or a feature parade. It should read like infrastructure that has already had to survive contact with reality.

#### Notes

- Keep Caprock as proof of seriousness.
- Do not let the clinical story become the page headline.

### 8. Who It Is For

#### Section title

For builders working on durable AI systems

#### Body copy

Kestrel is for developers and teams building AI systems that need continuity, accountability, and user ownership. If your agent needs to persist beyond one session, one model, or one vendor, this is the layer you build on.

### 9. Final CTA

#### Section title

Start where the trust surface becomes visible.

#### Body copy

If AI is going to become durable infrastructure, it cannot remain rented, opaque, and platform-owned. Kestrel is the open foundation for building the alternative.

#### CTA block

- Read the quickstart
- Inspect the repo
- Decide whether your AI should still belong to someone else

## Suggested Page Order Summary

1. Hero
2. Reframe
3. What Kestrel is
4. Three pillars
5. Why it matters
6. What is open
7. Proof
8. Who it is for
9. Final CTA

## Section-by-Section Copy Shortlist

Use these lines as likely keepers even if the full page changes:

- `Own your AI.`
- `Kestrel is the open foundation for Sovereign AI.`
- `Most AI today is rented intelligence.`
- `Not another wrapper. The foundation layer.`
- `The durable layers of AI should not belong to the platform.`
- `This is not a thought experiment.`

## Implementation Notes

- If engineering needs the fastest path, ship the hero, three pillars, what is open, proof, and final CTA first.
- The one-screen version in `SIMPLE_LAUNCH_PAGE_ONE_SCREEN.md` is the fallback if a full page is too heavy for the launch window.
- If the preview feedback shows confusion around `Sovereign AI`, the first revision should change the explanatory copy around the category, not the headline itself.