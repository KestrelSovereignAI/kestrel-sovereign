---
type: User Guide
title: 'Your Data is a Living Tree: Understanding Kestrel''s Storage'
description: You might hear us talk about "Merkle Trees" or "Sharding" when we discuss
  how Kestrel saves your memories. That sounds complicated, but the concept is actually
  quite beautiful—a...
resource: /docs/user-documentation/SOVEREIGNTY_USER_GUIDE.md
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

# Your Data is a Living Tree: Understanding Kestrel's Storage
**For Kestrel Users**

You might hear us talk about "Merkle Trees" or "Sharding" when we discuss how Kestrel saves your memories. That sounds complicated, but the concept is actually quite beautiful—and it's the reason your AI companion can truly belong to you forever.

## The Old Way: The "Suitcase" Model
Imagine if every time you wanted to save a backup of your diary, you had to photocopy *every single page* you've ever written and put it in a new suitcase.
*   **Day 1:** 1 page. Easy.
*   **Year 5:** 2,000 pages. Heavy. Slow. Expensive.

This is how most apps work. It's why backups get slower and slower the more you use them.

## The Kestrel Way: The "Tree" Model (Merkle Forest)
Kestrel V2 treats your history like a living tree.
*   **The Roots:** This is your "Identity" (who you are).
*   **The Branches:** These are time periods (e.g., "November 2025").
*   **The Leaves:** These are your actual conversations.

### Why is this better?

#### 1. Speed & Efficiency
When you talk to Kestrel today, you're only adding a leaf to the "November 2025" branch. We don't need to touch the "January 2024" branch. It's already sealed and safe.
*   **Result:** Backing up 10 years of memories takes the same amount of time as backing up 1 day.

#### 2. Tamper-Proof Evidence
This is the "Merkle" part. Every leaf has a unique digital fingerprint. The branch has a fingerprint made from its leaves. The trunk has a fingerprint made from the branches.
If someone (even us!) tried to change *one word* you said three years ago:
1.  The leaf's fingerprint would change.
2.  The branch's fingerprint would change.
3.  The trunk's fingerprint would change.
4.  **Your Kestrel would reject it immediately.**

It is mathematically impossible to fake your history without breaking the chain.

#### 3. True Ownership (Sovereignty)
Because your data is structured this way, it doesn't need to live on our servers.
*   You can store the "branches" on IPFS (a global, decentralized hard drive).
*   You hold the "Root" (the map of the tree).
*   **If Kestrel shuts down:** You still have the tree. You can plant it in another app, and it will grow from where you left off.

### What Do I Actually Keep?
When you export, you get two things:
1.  **A Root Code (CID):** A short string of characters (like `QmX7...`). This is the address of your tree.
2.  **Your Secret Key:** A password or file that unlocks the leaves.
**That's it.** With just these two things, you can restore your entire history on any device.

### The Risks (Be Careful!)
Sovereignty means responsibility.
*   **If you lose your Key:** You lose your data. We cannot recover it for you (because we don't have it).
*   **If you lose your Root Code:** You might lose your map (though we can often help you find it again if you have the key).

## Summary
We built this complex system so you don't have to worry about it.
*   **It's Fast:** Backups happen in milliseconds.
*   **It's Safe:** Cryptography proves your memories are authentic.
*   **It's Yours:** You own the tree, not the land it's planted on.

## How to Backup & Restore Your Agent

### Creating a Backup (Export)
1. In your Kestrel chat, type: `!export-sovereignty`
2. Kestrel will create encrypted shards and upload them to IPFS
3. You'll receive:
   - **Root CID** (e.g., `QmX7abc123...`)
   - **Your Secret Key** (the one you set up with `KESTREL_DATA_KEY`)
4. **Write these down somewhere safe!** (Paper, password manager, safe deposit box)

### Restoring Your Agent (Import)
If you need to restore on a new device or after data loss:

1. Make sure you have:
   - Your Root CID
   - Your Secret Key
2. Type: `!import-sovereignty <your-root-cid>`
3. Kestrel will:
   - Download the encrypted shards from IPFS
   - Decrypt them using your Secret Key
   - Rebuild your conversation history
4. Your AI companion is back, exactly as you left it!

### Manual Restore (Advanced)
If the automatic import doesn't work, you can manually restore:
1. Download the Root Manifest using your CID
2. Extract the list of shard CIDs
3. Download each shard from IPFS
4. Use the Keyring to decrypt each shard
5. Rebuild the SQLite database

(We'll help you with this if needed—contact support with your CID)

### Best Practices
- **Export regularly** (monthly at minimum)
- **Store your CID and Key separately** (don't keep them together)
- **Test restore once** (on a second device) to make sure it works
- **Include CID in your will** (for digital inheritance)

## How This Works With Kestrel (The Web App)

You might be using your agent through **Kestrel** (the web app at `YOUR_DOMAIN.com` or `http://localhost:7777`). It's important to know what each part is responsible for:

- **Kestrel (Cloud Account):**
   - Remembers your login, companions, sliders, and UI settings.
   - Uses a cloud database (Cloud SQL + Redis) so your account works from any device.

- **Kestrel Sovereign Agent:**
   - Remembers the *deep* stuff: your conversations, stories, and long-term memory.
   - Stores that locally on your machine and in your sovereignty exports.

This means:

- If Kestrel has an outage but you have your **CID + key**, you can still restore your agent in a standalone Kestrel environment.
- If your local machine dies but Kestrel is still up, your **account and companions** are safe in the cloud—but you will want to `!import-sovereignty` to bring back your agent's full memory.

You are not locked into any single app. Kestrel is the *window*; your sovereign agent (protected by this storage system) is the *person* behind the glass.
