# Historical mirror and doctrine receipts

`KESTREL_CONSTITUTION.md` preserves the complete first-edition documentation
mirror, including its original metadata. The public reader bookmark at
`docs/principles/KESTREL_CONSTITUTION.md` links to the
[released second edition](https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/constitution-v2.0.0/kestrel_sovereign/data/constitutions/v2/KESTREL_CONSTITUTION.md).
Neither file changes an agent's adopted governing constitution.

Verification, idempotent anchoring and the dispatcher's live-hash check first
hash the literal selections. Only an existing receipt may prove one historical
view. The selected file is opened once, following links; its content and
`os.fstat` identity come from that same descriptor. Substitution requires its
nonzero `(st_dev, st_ino)` to equal the independently opened stock bookmark's
identity and its content to match one of the pinned wrapper representations:

- LF: `b539c29f8950c1ed41a8f300ffce90e71b8ed09bf39f3e756e0e2ab07f787a0d`
- CRLF: `2dfc0fd4607339a42b304cd5d42535916a7f56d6b274091b7b70eeb87816db00`

Symlinks and hardlinks can identify that object. Separate copies cannot.
No path spelling or canonicalization establishes identity. Unavailable or zero
identity, a failed open/read/fstat, a replacement between opens, or a missing
archive leaves the literal hash available to the normal drift check.

The historical view reads bytes from this fixed archive location under each
selected file's original basename. Archive bytes are not independently pinned:
an operator's customized original moved here byte for byte is proved by their
existing **whole bundle receipt**. Only an exact whole-digest match accepts the
candidate; every other outcome returns the literal snapshot. Selection order,
duplicates, missing-file skips and bootstrap contributions are unchanged.
The default's historical slot first inspects the old bookmark: an older or
customized checkout's own mirror can still be proved by that same receipt.

Successful interpretation logs `Doctrine bundle historical reader mapping v1
verified`, with logical name, configured path and physical byte source. Snapshot
paths keep the configured selections; stored configuration, receipt file lists,
hashes and timestamps are never rewritten. New anchors use literal selections.
