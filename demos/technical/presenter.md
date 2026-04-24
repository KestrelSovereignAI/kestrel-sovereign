# Kestrel Demo — Presenter Slides Guide

**Audience:** Investor / Emotional (Noel)
**Core emotion to leave with:** TRUST — with a touch of love
**Total time:** ~12 minutes
**Screenshots:** `demos/technical/demo-output/`

---

## Opening — No Screen Yet
*Stand. Face the audience. No computer yet.*

**Speech:**
> "Most AI agents being deployed today — your bank's chatbot, your doctor's care companion, your insurance advisor — they belong to the vendor. The memory lives on their servers. The rules are set by their product team. When they change the model, the personality changes. When they sunset the product, your data disappears.
>
> And if the agent does something it shouldn't? There's no audit trail you can actually access.
>
> Kestrel is an open-source framework for building AI agents that work differently. Cryptographic identity. Self-governing principles. Data you actually own and can move. Let me show you exactly how it works."

*Sit. Open browser.*

---

## Slide 01 — `01-did-identity.png`
**On screen:** Identity panel — agent name, DID, identicon avatar

**Speech:**
> "Every Kestrel agent has its own identity — like a passport. See this identifier? `did:pkh:eip155:1:0x...` — this is a W3C standard Decentralized Identifier. It belongs to the agent, not to us, not to any platform.
>
> Unlike ChatGPT or Gemini, where your AI is a session on someone else's server — this agent has its own cryptographic identity. It's portable. If you don't like our platform, take your agent and leave. The identity goes with you."

**Key message:** *Your AI has a passport. It's not tied to any company.*

---

## Slide 02 — `02-chat-response.png`
**On screen:** Chat panel — agent's response about its constitutional principles

**Speech:**
> "I asked the agent to tell me about the principles that guide its behavior. Look at this response — it's not reciting a help page. It's citing its own Digital Bill of Rights by article. Freedom of Mind. Data Sanctity. Verifiable History. Right of Exit.
>
> This isn't prompt engineering. These rules are baked in at birth and can't be changed without breaking the agent."

**Key message:** *The agent knows its own rules. Nobody secretly changed them.*

---

## Slide 03 — `03-constitution-panel.png`
**On screen:** Constitution panel — full text with SHA-256 hash

**Speech:**
> "Here's the full constitution — The Kestrel Digital Bill of Rights. See that SHA-256 hash in the corner? That hash is verified on every single interaction. If someone tampers with the constitution — even one character — the agent enters safe mode and tells you.
>
> Every AI company is scrambling to answer 'how do you govern your AI?' Kestrel ships with the answer baked in. Immutable. Tamper-evident. Open source."

**Key message:** *If someone tries to change how it behaves, it shuts down and tells you.*

---

## Slide 04 — `04-memory-stored.png`
**On screen:** Chat panel — agent acknowledging a stored fact

**Speech:**
> "Now I'm going to show you memory. I told the agent something personal — my favorite language and a lucky number. It acknowledged and stored it.
>
> But here's the real test."

**Key message:** *The agent remembers. Now let's prove it.*

---

## Slide 05 — `05-memory-recalled.png`
**On screen:** Fresh session — agent attempting cross-session recall

**Speech:**
> "I opened a completely fresh session — zero conversation history, like a brand new tab. And I asked the same question.
>
> The memory system searched across sessions. ChatGPT remembers things about you too — but try to export that memory, move it to Claude, or even just see it. You can't. With Kestrel, the memory belongs to the user. Visible, portable, deletable."

**Key message:** *Memory that belongs to you — not to us.*

---

## Slide 06 — `06-memories-panel.png`
**On screen:** Knowledge Graph — agent identity node, constitution node, learned facts

**Speech:**
> "This is the Knowledge Graph — every structured thing the agent knows. The agent node at inception. The constitution it was born with. Facts it learned from our conversation.
>
> Every node is inspectable. Every node is deletable. Your data governance team can see exactly what the AI knows. GDPR right-to-erasure? One click per node."

**Key message:** *Everything your AI knows about you is right here. You control it.*

---

## Slide 07 — `07-privacy-normal.png`
**On screen:** Chat panel — green NORMAL privacy indicator top-right

**Speech:**
> "See this indicator in the top right? That's the privacy mode. Right now we're in NORMAL — full persistence, all features enabled. Kestrel has five privacy levels."

**Key message:** *You choose how private you want to be.*

---

## Slide 08 — `08-privacy-dropdown.png`
**On screen:** Privacy dropdown showing all 5 levels

**Speech:**
> "EPHEMERAL — nothing stored, local AI only. ISOLATED — session only, deleted when you close. ANONYMOUS — PII stripped before any write. NORMAL — full persistence. PUBLIC — can be shared and exported.
>
> These aren't settings you hope people follow. The storage engine enforces them."

**Key message:** *Five levels. Enforced by the infrastructure, not by policy.*

---

## Slide 09 — `09-privacy-ephemeral.png`
**On screen:** Red EPHEMERAL indicator active, toast confirmation

**Speech:**
> "I switched to EPHEMERAL. The indicator turned red, and the LLM provider automatically switched to local-only — Ollama running on this machine. Nothing leaves this device. Not to our servers. Not to any AI cloud. Nothing."

**Key message:** *EPHEMERAL = incognito mode that actually works.*

---

## Slide 10 — `10-privacy-ephemeral-response.png`
**On screen:** Agent response in EPHEMERAL mode

**Speech:**
> "The agent responded — but nothing was stored. No record written. No network call to a cloud AI. Generated entirely on this device. For sensitive conversations — medical, legal, financial — this is the answer regulators are looking for."

**Key message:** *Zero data leaves the device. Enforced at the code level.*

---

## Slide 11 — `11-privacy-restored.png`
**On screen:** Green NORMAL restored

**Speech:**
> "Back to NORMAL. The switch is instant. The privacy level follows the conversation, not the account."

**Key message:** *Seamless. One click.*

---

## Slide 12 — `12-sovereignty-panel.png`
**On screen:** Data Sovereignty panel

**Speech:**
> "This is the Sovereignty panel. This is where data ownership actually happens — not just as a promise, but as an action you can take right now."

**Key message:** *Ownership you can see. And use.*

---

## Slide 13 — `13-export-modal.png`
**On screen:** Export modal — Local / IPFS / Filecoin tiers

**Speech:**
> "One click opens this. Three storage tiers — keep it on device, distribute it to IPFS, or archive it on Filecoin for long-term decentralized storage. Encryption is on by default.
>
> This is the GPT-4o insurance policy. When OpenAI turned off a model and people lost their AI — they had no recourse. A Kestrel user exports to IPFS, gets a content hash, and can restore their agent on any compatible platform. The switching cost is zero. That's how you win trust."

**Key message:** *Your AI survives any platform. Including ours disappearing.*

---

## Slide 14 — `14-export-result.png`
**On screen:** Export complete with CID hash

**Speech:**
> "Export complete. That CID — content identifier — is a cryptographic receipt. It's verifiable from anywhere. The export contains the agent's identity, constitution, full memory graph, and conversation history. Everything needed to restore this agent on any Kestrel-compatible runtime.
>
> The agent doesn't live in our cloud. It lives in that hash."

**Key message:** *"The agent doesn't live in our cloud. It lives in that hash."*

---

## Slide 15 — `15-demo-final.png`
**On screen:** Final state of the demo

**Speech:**
> "With that receipt, the owner can restore their AI companion on any compatible platform. No vendor lock-in. True data ownership. That's what we mean when we say your AI belongs to you."

**Key message:** *This is AI that actually works for you.*

---

## Slide 16 — `16-security-panel.png`
**On screen:** Security panel — tool permission matrix

**Speech:**
> "One more thing. Every tool the agent can use has its own permission level — Allow, Ask, or Deny. Not set by a policy document. Enforced at the architecture level before any tool call runs."

**Key message:** *Granular control. Enforced at the code level.*

---

## Slide 17 — `17-security-deny-set.png`
**On screen:** Export tool set to DENY

**Speech:**
> "Watch this. I'm setting the export tool to DENY. One click. The agent has just lost the ability to export your data — even if it tried, even if someone instructed it to."

**Key message:** *One click. The agent can't take your data anywhere.*

---

## Slide 18 — `18-security-blocked.png`
**On screen:** Agent response — export attempted and blocked

**Speech:**
> "I asked the agent to export. It tried. The security hook intercepted the tool call before the code executed. The export didn't happen. The agent told the user it couldn't do it.
>
> That refusal is architectural, not corporate."

**Key message:** *"That refusal is architectural, not corporate."*

---

## Slide 19 — `19-security-audit.png`
**On screen:** Security panel — audit log showing denied tool call

**Speech:**
> "And right here in the audit log — every permission decision, timestamped, logged. If an auditor asks 'did your AI ever try to export sensitive data without permission?' — you show them this. The compliance guarantee is in the architecture."

**Key message:** *Proof you can show a regulator. Not a promise.*

---

## Slide 20 — `20-security-restored.png`
**On screen:** Export permission restored to Allow

**Speech:**
> "I restored the permission. Back to Allow. The whole sequence took three clicks and is fully logged."

**Key message:** *Instant. Auditable. Yours.*

---

## Closer — Step Away from Screen
*Stand. Face audience.*

**Speech:**
> "What you just saw:
>
> A cryptographic identity generated in two seconds — no authority required.
> A constitution anchored at birth and tamper-evident.
> Memory that belongs to the user — visible, portable, deletable.
> Privacy enforced by the storage layer, not by policy.
> A complete data export with a content hash you can independently verify.
> And tool-level permissions enforced before code executes — with a full audit log.
>
> Kestrel is MIT-licensed, open source, runs on any machine. In 30 minutes you can have your own agent running with all of this active.
>
> *(Pause.)*
>
> The question I'd ask in your position: what does your AI deployment look like when the vendor changes the model without telling you? When the safety guidelines get updated in a patch? When you need to prove to a regulator that the agent followed your rules on a specific date?
>
> This is the framework that makes those answers a guarantee, not a promise."

**Investor closer:**
> "Every major AI platform locks users in. Kestrel is the infrastructure for AI that users actually own. Cryptographic identity, immutable governance, portable data. When the next GPT-4o moment happens, Kestrel users won't even notice — because their data was never at risk."

---

## Phrases to Memorize

| Phrase | Use it when |
|--------|------------|
| *"That refusal is architectural, not corporate."* | Act 6 — security block |
| *"The compliance guarantee is in the architecture."* | Act 6 — audit log |
| *"The agent doesn't live in our cloud. It lives in that hash."* | Act 5 — export CID |
| *"In 30 minutes you can have your own agent running with all of this active."* | Closer |
| *"Your data was never at risk."* | Investor closer |

---

*Generated from `docs/demos/DEMO_SCRIPT.md` + latest Playwright run output*
*Screenshots: `demos/technical/demo-output/01-did-identity.png` → `20-security-restored.png`*
