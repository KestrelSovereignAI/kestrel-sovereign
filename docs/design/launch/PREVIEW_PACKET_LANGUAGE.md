---
type: Design Note
title: Preview Packet Language
description: This document stores the current canonical preview-packet language for
  the Kestrel open-source launch so the team has one repo-based source of truth alongside
  the other launch a...
resource: /docs/design/launch/PREVIEW_PACKET_LANGUAGE.md
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

# Preview Packet Language

This document stores the current canonical preview-packet language for the Kestrel open-source launch so the team has one repo-based source of truth alongside the other launch assets.

## What This Is

Use this when the team needs the actual send language for the preview cohort rather than landing-page copy. This file consolidates the latest preview-packet wording developed in issue #188 and places it beside the other launch documents in `docs/design/launch/`.

## Primary Builder Preview Packet

Use this for the core trusted builder preview group.

### Outreach Message

Personalize `[Name]` and the channel line per recipient. Keep it short.

**Subject:** Kestrel preview is live — and you're seeing it before anyone else

Hi [Name],

Today's the day — Kestrel is live for the preview group, and you're one of the people I most wanted in it.

Repo: **https://github.com/KestrelSovereignAI/kestrel-sovereign**

Start with [QUICKSTART.md](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/QUICKSTART.md) — about 30 minutes to a running sovereign agent with your own cryptographic identity, persistent memory, and a constitutional governance layer enforced above the LLM. It's already deployed in a clinical study with a regional home healthcare organization, so this isn't a thought experiment — it's running in production for real patients.

**Four questions I'm hoping you'll answer afterward** (bullets are perfect, a 10-minute voice note is even better):

1. **What would make you actually use this** instead of building on a standard LLM API? And are there specific use cases where you can see this being a clear win?
2. The one sentence you would use to explain Kestrel to someone else.
3. What felt clear, what felt confusing, and what felt weak or unconvincing.
4. Anything that would have stopped you cold if you weren't doing this as a favor to me.

**Where to send feedback:** [primary channel — pick per recipient]. If something breaks during install, message me there immediately — I want to know same-day so I can fix it before the next person hits it.

**A few asks:**
- **Embargo:** please don't post publicly about Kestrel (social, blog, talks) until May 7. Sharing 1:1 with people you'd personally vouch for is fine.
- **Quotes:** if you say something useful, I may ask to quote it for the May 7 launch — named or anonymous, your call. Just flagging upfront so it's not a surprise.
- **Timing:** feedback by **May 1** lets me fold it into the public launch.

Thank you for being in this group. It means a lot.

Gabi

## Friendly Stranger Variant

Use this lighter version for smart technical people outside the AI-agent bubble when the goal is readability and install-signal, not deep product-framing critique.

### Lighter Message

Send via text, WhatsApp, or another lightweight personal channel. Personalize `[Name]`.

**Subject:** quick favor — would you take 20 min to look at something I'm building?

Hey [Name],

Hope you and the family are doing well. Quick ask — I'm launching a project called Kestrel publicly on May 7 and could really use your eyes on it before then.

It's an open-source AI tool (already running in a clinical study with a regional home healthcare organization). I'm not asking you to be an AI expert — quite the opposite. I want to know if it makes sense to a smart technical person who isn't deep in AI agents.

Repo: **https://github.com/KestrelSovereignAI/kestrel-sovereign**

Three quick things, when you have 20 min:
1. Did the README make you curious enough to try the install? If not, where did you tune out?
2. If you tried [QUICKSTART.md](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/main/QUICKSTART.md) — where did you get stuck or confused?
3. In one sentence, what do you think Kestrel does?

Bullet points are perfect, a voice memo is even better. By **May 1** if possible.

One ask: please keep it private until May 7 (no posting on social or sharing widely). Sharing 1:1 with someone you'd vouch for is fine.

Thank you — means a lot.

Gabi

## Usage Notes

- Use the primary builder packet for the core preview cohort.
- Use the friendly-stranger variant when testing outside-the-bubble readability.
- Keep the three-pillar language aligned with the current README, quickstart framing, and landing-page drafts.
- Treat this as launch copy, not long-term product documentation.

## Source Basis

This repo copy is derived from the latest preview-packet work captured in issue #188, including:
- the Apr 22 `Preview Packet v2` send-materials comment
- the Apr 22 cohort split / lighter packet comment
- the Apr 22 deadline update changing the feedback window to May 1