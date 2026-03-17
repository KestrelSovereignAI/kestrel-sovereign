Audience-doc generation is now complete, not just dry-run verified.

Generated and reviewed:

- `docs/generated/FEATURES_developer.md`
- `docs/generated/FEATURES_user.md`
- `docs/generated/FEATURES_investor.md`

One important drift surfaced during review:

- the first investor output invented privacy mode names not present in the source

Corrective action taken:

- added the exact privacy preset table to `KESTREL_FEATURES.md`
- tightened generator prompts to preserve exact privacy preset names and route/method language
- regenerated the affected audience docs

Current state:

- generated docs exist
- no stale fixed feature/API counts remain in the active source or prompts
- generated docs now reflect the canonical privacy preset names and current route/method language
