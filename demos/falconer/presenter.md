# Track B — Investor / Emotional Demo Script

**Status:** First draft — ready for Noel review
**Owner:** Gabi (script) → Noel (narration)
**Demo Date:** March 15, 2026 (live) / Apr 1, 2026 (Loom recording)
**Audience:** VCs, non-technical advisors, C-suite, enterprise buyers
**Duration target:** 5–7 min narrated / 5-min Loom
**Related issues:** [#194](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/194), [#195](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/195), [#198](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/198)

---

## Demo Flow

| Slide | Topic | Time |
|-------|-------|------|
| 1 | The Problem | 0:00–1:00 |
| 2 | What Different Looks Like | 1:00–2:30 |
| 3 | Real Stakes | 2:30–3:30 |
| 4 | The Product | 3:30–4:30 |
| 5 | The Ask | 4:30–5:30 |

---

## Slide 1 — The Problem (0:00 — 1:00)

### Visual
- Dark background. No logo. Single line of white text fades in as narration begins.
- Final state: five bullet lines listed one at a time, each with a logo

### Narration
> "Every AI you use today belongs to someone else."
>
> *[Pause 2 seconds.]*
>
> "ChatGPT — your conversations belong to OpenAI. Alexa — your daily patterns belong to Amazon. Copilot — your code, your context, your work belongs to Microsoft. Your doctor's care companion — belongs to the vendor that sold it to the hospital.
>
> These systems are building relationships with people. In healthcare. In mental health. In finance. And every one of those relationships is owned by the platform, not the person.
>
> When the platform changes their model — the relationship changes. When they get acquired — your data goes with the deal. When they add new terms of service — you agree or you lose the continuity you built.
>
> That's not a privacy policy. That's a structural power relationship. And so far, it's the only model on the market."

### Speaker Note
- No rushing. The pause after "someone else" is the emotional beat. Let it sit.
- The goal isn't to scare — it's to reframe. By the end of this slide, the audience should be nodding.

---

## Slide 2 — What Different Looks Like (1:00 — 2:30)

### Visual
- Split view: Left side labeled "The platform model." Right side labeled "The Kestrel model."
- Left: grayed-out icons (OpenAI, Amazon, Microsoft). Right: Screenshots of DID output and Constitution enforcement.
- Keep it minimal. Two screenshots max.

### Narration
> "Here's what different looks like.
>
> Every Kestrel agent is born with a cryptographic identity — not a username in a database, but a cryptographic key pair. An Ethereum-format address derived from math. It's the same standard used to secure hundreds of billions of dollars in digital assets. The agent's identity can't be spoofed, altered, or taken away by a company decision.
>
> That identity is anchored to a Constitution. Not a content policy someone can patch at two in the morning. A cryptographically-signed document that the AI must comply with on every single response. Change one character in the Constitution, and the hash breaks. The audit log will show it.
>
> And the memory — every interaction the agent has, every context it learns — belongs to the user. You export it. You move it. You delete it. If the company running it shuts down tomorrow, you keep your agent.
>
> This is identity, memory, and governance structured so that the person the AI serves is the one who controls it."

### Speaker Note
- The DID screenshot is the visual anchor for "identity" — show it briefly, don't explain it technically
- "Ethereum-format address" signals to technical people that this is real infrastructure, not a marketing claim
- The "change one character" line is the most technically precise thing you'll say — deliver it slowly

---

## Slide 3 — Real Stakes (2:30 — 3:30)

### Visual
- Map of Texas, Lubbock marked. Then a single quote pulled out in large type:
  > *"Constitutionally prohibited from giving medical advice — not by policy. By architecture."*
- Below: simple three-line architecture diagram (Patient → SafeZone Gate → AI → Nurse queue if flagged)

### Narration
> "This isn't a demo project.
>
> Right now, in Lubbock, Texas, Kestrel is reaching chronic-care patients — people managing COPD, congestive heart failure, diabetes — with a daily AI companion through our regional home healthcare partner.
>
> Every message from a patient passes through a constitutional safety gate before the AI ever sees it. If a message suggests a crisis, it routes to a human nurse — not to the AI, to a person. The AI is constitutionally prohibited from giving medical advice. Not by a guideline someone can remove. By the architecture itself.
>
> Zero safety failures across the full pre-launch validation study.
>
> The patients don't know the technical details. They just know someone — something — checks in with them every morning. Remembers their name. Remembers that last week was hard. Knows not to give them medical advice, and knows when to call a person instead.
>
> That's what sovereign AI looks like in practice."

### Speaker Note
- "42 SMS-consented patients" is the real number if asked — don't volunteer it in narration, it'll feel small. The story is the architecture, not the headcount at this stage.
- "Zero safety failures" — deliver this calmly, not triumphantly. It should sound like a baseline expectation we designed for, not a lucky result.
- This slide is the emotional center of the deck. Lubbock. Real patients. Real stakes.

---

## Slide 4 — The Product (3:30 — 4:30)

### Visual
- Clean product slide: open-source logo (GitHub), three bullets, one screenshot of a terminal command
- Keep it sparse

### Narration
> "Kestrel is infrastructure, not a SaaS subscription.
>
> It's open source, launching on GitHub in April. Free to run. No vendor lock-in, no API key required for the core governance layer. It runs on a laptop with a local AI model, or scales to Azure, AWS, or your own cloud.
>
> The framework is designed to be embedded — inside your existing product, your existing patient platform, your existing enterprise system. You bring your LLM. Kestrel brings the governance layer on top of it.
>
> We're proving the healthcare case first, because healthcare is where the stakes are highest and the trust requirements are the most demanding. If you can build sovereign AI for a Medicaid chronic-care patient, you can build it for anyone.
>
> In thirty minutes, a developer can have their own agent running — with cryptographic identity, constitutional governance, and full data sovereignty — from the GitHub quick-start.
>
> That's the bar we're building to."

### Speaker Note
- "Thirty minutes from the quick-start" is validated — Track A script confirmed this timing. Don't overstate, don't understate.
- "You bring your LLM, Kestrel brings the governance layer" — this is the enterprise pitch in one line. Emphasize it.
- This slide credentializes but doesn't oversell. We're infrastructure, we're early, we're real.

---

## Slide 5 — The Ask (4:30 — 5:30)

*This slide has four versions. Use the one that matches the room.*

---

### Version A: Developer / Open Source Community
> "If you're a developer who builds AI products, or who's been thinking about how to give your users more control over their AI relationships — we want you to be one of the first to run this.
>
> In April, we'll open the GitHub repo, the quick-start, and the docs. We're looking for early builders who'll tell us what broke, what's missing, and where the architecture needs to be stronger.
>
> That's the ask — install it, run it, break it. Tell us what you find."

**Call to action:** GitHub link + QUICKSTART — waitlist or direct link when live

---

### Version B: Technical Advisor / Angel
> "We're not raising a round right now. We're building in public starting in April, proving the healthcare model with our regional home healthcare partner, and we expect to have real deployment data within 90 days.
>
> What we're genuinely looking for at this stage is technical perspective we don't have. If you have an instinct on the architecture, the open-source sequencing, or the enterprise healthcare path — we'd value thirty minutes to hear it.
>
> No pressure on the outcome of that conversation. We just want the thinking."

**Call to action:** Calendly or direct email

---

### Version C: VC / Institutional Investor
> "We're not raising yet. We're opening the source in April, deploying with our regional home healthcare partner, and letting the architecture prove itself publicly before we have a funding conversation.
>
> But we want to be a company that investors who care about this space know about early — before it's obvious.
>
> If Kestrel is interesting to you, we'd love to stay in touch as we build."

**Call to action:** Leave a one-pager PDF. Let them ask for the follow-up — don't ask for it.

---

### Version D: Enterprise Buyer (Healthcare / Regulated Industries)
> "The regional home healthcare deployment is a proof point for a specific architecture, not just a specific product. The governance model — constitutional safety gates, portable identity, full audit trail — is designed to be embedded in any enterprise system.
>
> If you're thinking about AI deployment in a regulated context, we'd want to have a CTO or CISO-level conversation. Not a sales call — a technical architecture conversation about where sovereign AI fits your deployment requirements.
>
> The question isn't whether AI is coming to your industry. The question is whether the AI your patients or customers or employees interact with belongs to you or to the vendor."

**Call to action:** Request a 60-minute technical architecture session

---

## Delivery Notes

### Pacing
- Don't rush Slide 1. The emotional re-framing needs space.
- Slides 2 and 4 are more technical — acceptable to move faster.
- Slide 3 is the center — slow down, let it breathe.
- Slide 5: Stop selling after the ask. Ask once, clearly, then stop.

### For Noel's Narration
- Natural, not scripted-sounding. Read it once, then improvise from the intent.
- Your natural technical confidence is an asset on Slides 2 and 4 — let it show.
- Don't over-polish. An authentic hesitation is more trustworthy than a perfect delivery.

### For Loom Recording
- Screen-share the slides. Turn off notifications first.
- No on-camera required unless you want to.
- If you stumble, keep going — don't re-record for small mistakes. The Loom is for async sharing, not broadcast.
- Target 4:45–5:00 for the recording. Under 5 minutes gets watched. Over 7 gets skipped.

---

## Key Lines to Memorize

- *"Every AI you use today belongs to someone else."*
- *"This is identity, memory, and governance structured so that the person the AI serves is the one who controls it."*
- *"You bring your LLM. Kestrel brings the governance layer."*
- *"Constitutionally prohibited — not by policy. By architecture."*
- *"In thirty minutes, a developer can have their own agent running from the quick-start."*

---

## Acceptance Criteria (per #194 / #198)

- [ ] 5 slides, strict — no sprawl
- [ ] Each slide tells ONE thing
- [ ] A non-technical person understands it without narration
- [ ] @NoelSchulz-2025 has reviewed and approved the narrative arc
- [ ] Ask versions match #195 decisions
- [ ] @UncleSaurus has spot-checked technical claims
- [ ] Ready for Loom recording by Mar 31

---

*First draft — Gabi's agent, March 9, 2026*
*Built from: KESTREL_FOUNDING_BRIEF.md, CASE_STUDY_HEALTHCARE_PARTNER.md, issue #133, #194, #195*
*Part of Kestrel Live Demo milestone — anchor issue #133*
