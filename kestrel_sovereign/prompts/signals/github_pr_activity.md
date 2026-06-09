[GITHUB_PR_ACTIVITY] GitHub PR/issue `{payload[repo]}#{payload[number]}` changed. This wake fired from the periodic `github_pr_watch` poll, NOT from a user prompt — decide whether the change needs a follow-up action and act, or acknowledge silently.

What changed (categories): `{payload[changed]}`.

**Current state:**

  * state: `{payload[state]}` (merged: `{payload[merged]}`)
  * comments: `{payload[comments]}` — review comments: `{payload[review_comments]}`
  * checks: `{payload[checks_status]}`
  * last updated: `{payload[updated_at]}`

Link: {payload[html_url]}

**Change categories explained:**

  * `state` — the PR/issue opened/closed transition. If it closed without merging, decide whether re-opening or a follow-up issue is warranted.
  * `merge` — the PR merged. Close the loop: summarize, file any follow-up, or do nothing if the merge already finished the work.
  * `comments` — new (review) comments arrived. Read them; a reviewer may be asking for a revision. If a change is requested, dispatch the revision (e.g. `talon iterate --pr`) or reply.
  * `checks` — CI/check status changed. If it turned red, diagnose and decide whether to push a fix. If green on a PR awaiting merge, it may be ready.

Only the categories listed in this watch's `triggers` woke you; a bare timestamp bump does not. If nothing actionable is needed, acknowledge and move on — you do not have to act on every poll.

source={source}
arrived_at={arrived_at}
urgency={urgency}

payload={payload}
