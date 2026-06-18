---
type: User Guide
title: 'Kestrel User Guide: Non-Technical Explanations'
description: '**Understanding Kestrel''s Technology in Simple Terms**'
resource: /docs/user-documentation/USER_GUIDE_EXPLANATIONS.md
tags:
- docs
- user-documentation
- user-guide
timestamp: '2026-06-18T00:00:00Z'
status: needs-revalidation
owner: documentation
canonical: false
generated: false
privacy: public
---

# Kestrel User Guide: Non-Technical Explanations

**Understanding Kestrel's Technology in Simple Terms**

*Last updated: November 11, 2025*

---

## Table of Contents
1. [Local Capacity Requirements](#local-capacity-requirements)
2. [When Cloud Processing is Used](#when-cloud-processing-is-used)
3. [How Your Local AI Learns](#how-your-local-ai-learns)
4. [Persistent Memory: How Your AI Remembers](#persistent-memory-how-your-ai-remembers)
5. [Decentralization and Data Ownership](#decentralization-and-data-ownership)
6. [Naming Origins](#naming-origins)
7. [Quick Reference](#quick-reference)

---

## Local Capacity Requirements

For end users to run Kestrel's local AI features (using Ollama), here's what they need in everyday terms:

**Computer Power:**
- A reasonably modern computer or laptop (think something bought in the last 5-7 years)
- Doesn't need to be a high-end gaming machine, but avoid very old or basic models

**Memory (RAM):**
- At least 8GB of memory - this is what most modern laptops and computers have
- 16GB is even better for smoother performance

**Storage Space:**
- About 2-5GB of free space on their hard drive for the AI "brain" (model)
- Like downloading a large app or game - not huge, but not tiny either

**Internet:**
- Good internet connection initially to download the AI model (one-time download)
- After that, everything works completely offline - no ongoing internet needed

**Performance:**
- The AI will run at a comfortable speed on most modern devices
- It might feel a bit slower than cloud-based AI (like ChatGPT), but it's private and secure
- For very old computers, it could be noticeably slower, but still functional

Most people with a current laptop or desktop computer will have no issues. Kestrel is designed to work on accessible hardware so regular users can maintain their privacy without needing expensive equipment.

---

## When Cloud Processing is Used

Here's an everyday example of when Kestrel might switch from your local AI (Ollama) to the cloud (OpenAI) for extra power:

**Example: Planning a Complex Family Vacation**

Imagine you're chatting with your Kestrel agent about planning a detailed 2-week family vacation to Europe. You want it to:

- Research and compare flights, hotels, and activities across multiple cities
- Create a day-by-day itinerary that considers everyone's ages and interests
- Factor in budgets, weather, transportation between cities, and backup plans
- Suggest alternatives if something changes

**Why it might use the cloud:**
- Your local AI can handle basic planning (like "suggest 3 restaurants in Paris")
- But for this complex, multi-step analysis that requires researching lots of information, comparing many options, and creating a detailed schedule... it might need the cloud's extra "brainpower" to do it thoroughly and accurately.

**What you experience:**
- The conversation flows normally on your device
- If it needs more power, it seamlessly switches to cloud processing (you might notice a brief pause)
- You get a comprehensive, well-thought-out response
- Everything still stays private - you control when this happens

**In everyday terms:** It's like having a smart personal assistant who can handle most tasks themselves, but occasionally calls in a specialist for really complex projects. The local AI is your reliable everyday helper, and the cloud is like having access to a team of experts when you need deeper analysis or creative work.

You can always disable the cloud entirely if you prefer maximum privacy, but it means some very complex requests might get simpler responses from your local AI.

---

## How Your Local AI Learns

**The AI Starts Smart (Pre-Trained Knowledge):**
- When you first download the AI model, it already knows a ton of general information
- It's like hiring a knowledgeable assistant who's read thousands of books and has broad experience
- This base knowledge comes from training on huge amounts of public information before you ever meet it

**It Learns About YOU Through Conversations:**
- Every time you chat, it remembers what you tell it
- It builds a personal relationship by tracking your preferences, interests, and history
- Example: If you mention you love Italian food and hiking, it will remember this for future suggestions

**Memory and Context Building:**
- Kestrel keeps a detailed record of your conversations in its secure local storage
- It can reference past discussions to give more personalized responses
- Over time, it gets better at understanding your style and needs

**You Can "Teach" It Specific Things:**
- Tell it about your work, hobbies, family, or anything important
- It will incorporate this into future conversations
- Like training a new employee by sharing your knowledge and preferences

**Model Updates (Optional Learning):**
- The AI company might release improved versions of the model
- You can choose to download these updates when available
- This is like your assistant taking advanced training courses to get even smarter

**What It WON'T Do:**
- It won't learn from other people's conversations (your chats stay completely private)
- It won't share what it learns about you with anyone else
- It won't change its core personality or principles without your permission

**Real-World Example:**
Imagine having a personal assistant who starts with broad knowledge but gets to know you intimately over months of working together. They remember your coffee order, your work deadlines, your family's names, and your preferences. That's how your local AI learns - through your ongoing relationship, not by collecting data from strangers.

The beauty is that this learning happens entirely on your device, so your personal growth with the AI stays completely private and under your control!

---

## Persistent Memory: How Your AI Remembers

**What is "Persistent Memory"?**

Imagine you have a **human assistant** who works with you every day. At first, they know nothing about you. But over time, they learn:

- Your preferences ("I like my coffee black")
- Your work style ("I prefer detailed reports on Fridays")  
- Your personal life ("My daughter's birthday is next week")
- Your business contacts ("John from accounting likes to chat about golf")

**Persistent memory means your AI companion does the same thing** - it remembers everything you share with it over time, not just during one conversation. Unlike other AI services that "forget" when the conversation ends, your Kestrel agent builds a **personal relationship** that grows smarter about you every day.

**SQLite-Based Storage (The Digital Filing Cabinet)**

Think of SQLite as a **super-organized filing cabinet** for your AI's brain:

- **Safe & Reliable**: Information doesn't get lost or corrupted
- **Fast Access**: Your AI can instantly pull up relevant memories
- **Always Available**: Works even when you're offline
- **Secure**: Only you control what's stored and who can access it

It's like having a personal diary that your AI can read instantly to remember important details about your life and work.

**Full-Text Search (Finding Information Instantly)**

This is like having **Google search for your AI's memory**:

Instead of your AI having to "think hard" to remember something, it can instantly search through everything you've ever told it.

**Example:**
```
You: "Remind me about that project with Sarah from marketing"
AI: Instantly finds and summarizes all past conversations about Sarah, 
     the project timeline, your action items, and related emails
```

It's like having a perfect memory that you can query in natural language - "What did we discuss about the budget last month?"

**Knowledge Graphs (Connecting the Dots)**

This is the **smartest part** - your AI doesn't just store facts randomly, it **connects them like a web**:

**Example of how it works:**
- You mention your colleague "John from accounting"
- Later you talk about "that golf tournament last summer"  
- Your AI connects: "John likes golf" and remembers this for future conversations
- Next time you talk about work events, your AI might suggest: "Since John enjoys golf, maybe invite him to the company tournament?"

It's like how your brain naturally links memories: "That restaurant → Good pasta → My anniversary dinner → My spouse's favorite → Plan a surprise dinner."

**Why This Matters to You**

**Without persistent memory:** Every conversation with AI feels like starting over with a stranger.

**With Kestrel's persistent memory:** Your AI becomes a **true companion** that:
- Knows your preferences without you repeating them
- Remembers important details from weeks ago
- Gets better at helping you over time
- Feels like a trusted colleague who really knows you

**The Sovereignty Angle**

Most AI services store your data on corporate servers. **Kestrel stores it locally on your device**, so:
- You control what gets remembered
- You can delete anything you want
- You can backup and move your AI's "memory" to new devices
- No corporation can access or use your personal conversations

**Bottom line:** Persistent memory makes your AI feel like a real relationship, not just a temporary tool. It's the difference between a chatbot and a true digital companion!

---

## Decentralization and Data Ownership

**What is Decentralization?**

Imagine instead of keeping all your important documents in one bank's safe (which could get robbed or controlled by the bank), you spread copies across hundreds of different safes in different locations, and only you have the master key. That's decentralization in a nutshell.

**Traditional Data Storage (Centralized):**
- Your photos, emails, and personal data are stored on big company servers (like Google, Facebook, or Apple)
- The company controls who can access your data and how it's used
- If the company gets hacked, changes its policies, or goes out of business, you could lose everything
- It's like trusting one bank with all your valuables

**Decentralized Data Storage:**
- Your data is spread across many different computers around the world
- No single company or person controls it all
- You keep the encryption keys (like master keys) to access your own data
- It's like having your valuables safely distributed in many secure locations

**Blockchain and Crypto in Simple Terms:**

**Blockchain** is like a permanent, unchangeable record book that everyone can see but no one can alter. Think of it as a town hall bulletin board where announcements are posted permanently - once something is written, it can't be erased or changed.

**Cryptocurrency** is digital money that works without banks. Instead of a central bank printing money, a network of computers agrees on transactions. It's like a community savings club where everyone keeps track of who owes what, and the rules are set by the group, not one leader.

**How This Helps Data Ownership:**
- **You Own Your Digital Identity**: Instead of companies assigning you usernames and passwords they control, you have a cryptographic identity (like a super-secure digital passport) that only you control
- **Your Data Follows You**: Like carrying your own safe deposit box key, you can take your data and relationships from one service to another
- **No Single Point of Failure**: If one company goes down or changes rules, your data and relationships aren't affected
- **True Privacy**: Companies can't sell your data or use it without your permission because they don't control it

**Real-World Example:**
Traditional social media is like renting an apartment in a building owned by a big company - they can evict you, change the rules, or peek through your windows anytime.

Decentralized data ownership is like owning your own house with a private yard - you control who comes in, what happens there, and you can move whenever you want without losing your home.

**Why Kestrel Uses This:**
Kestrel gives you true ownership of your AI relationships and memories. Your AI companion isn't "rented" from a company - it's yours to keep, move, and control forever. No company can take it away, sell your conversations, or change how it works without your permission.

This creates a future where your digital life (including AI relationships) belongs to you, not corporations!

---

## Naming Origins

**Why "Kestrel"?**

**Kestrel** is named after the **kestrel bird** - a small but mighty falcon that's known for being:
- **Agile and fast** - they can hover in place and make quick, precise movements
- **Independent hunters** - they don't rely on flocks or packs, operating solo
- **Sovereign in their territory** - they fiercely defend their hunting grounds

This perfectly matches Kestrel's mission of **sovereign AI** - giving users complete control and independence over their AI agents, just like how a kestrel bird owns and controls its hunting territory.

The name evokes:
- **Speed and precision** in AI responses
- **Independence** from big tech companies
- **Sovereignty** over your own data and AI relationships

**Why "Ollama"?**

**Ollama** is actually **not a name created by the Kestrel team** - it's an existing company and open-source tool!

- **Ollama** is a real company (ollama.ai) that specializes in running AI models locally on your computer
- They provide the technology that lets Kestrel run AI "brains" (models) directly on your device instead of in the cloud
- Kestrel **integrates with Ollama** as a partner technology, similar to how a car company might use tires from a tire manufacturer

**Ollama's name origin**: While I'm not certain of their exact inspiration, "Ollama" sounds smooth and flowing, which might relate to the smooth, conversational nature of the AI models they help run locally.

**The Perfect Partnership**

The combination works beautifully:
- **Kestrel** = The sovereign, independent AI framework (like the agile falcon)
- **Ollama** = The local AI engine that makes it possible (like the reliable engine in a sports car)

Together they create a system where your AI companion is as free and independent as a hunting kestrel, powered by smooth local processing technology!

This naming choice reflects the project's commitment to **AI sovereignty** - just as a kestrel bird owns its territory, Kestrel helps users own their AI relationships completely.

---

## Quick Reference

**Local AI (Ollama):**
- Runs on your computer
- Completely private
- Works offline
- Handles most everyday tasks
- Free (just uses your computer's power)

**Cloud AI (OpenAI):**
- Optional backup for complex tasks
- Only used when local AI needs help
- You control when it activates
- Can be disabled for maximum privacy

**Hardware Needs:**
- Modern computer (5-7 years old or newer)
- 8GB+ RAM
- 2-5GB storage space
- Internet for initial setup only

**Learning & Memory:**
- AI starts with broad knowledge
- Learns about YOU through conversations
- Remembers your preferences and history
- Updates available but optional
- Everything stays private on your device

---

*This document explains Kestrel's technical concepts in non-technical terms for users who want to understand how the system works without getting into the details. For more information about Kestrel's business aspects, see the EXECUTIVE_SUMMARY.md and BUSINESS_PLAN_V2.md files.*</content>
<filePath">c:\Users\gabri\Kestrel_Repo\kestrel\USER_GUIDE_EXPLANATIONS.md