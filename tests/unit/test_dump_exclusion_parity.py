"""A database dump must be excluded from every channel that ships the tree.

This test exists because fixing one channel at a time did not converge. Review
found the same defect three rounds running, each time in a channel the previous
fix had not considered:

1. a dump in the repo root could be committed          -> .gitignore
2. ...and `COPY . .` baked it into the image           -> .dockerignore
3. ...and `gcloud builds submit .` uploaded it to GCS  -> .gcloudignore

Each fix was correct and each left the exposure open somewhere else, because
the lists are independent: Docker does not read .gitignore, and gcloud consults
.gcloudignore *before* Docker sees .dockerignore. Hiding a dump from
``git status`` without excluding it from the other two is actively worse than
doing nothing, since the operator stops seeing the file that is still shipping.

So the invariant is asserted rather than remembered. Adding a fourth channel
means adding it here, and the test says what to do.

Why this matters concretely: ``docs/deployment/README.md`` instructs operators
to write a pre-migration dump into the repository root
(``pg_dump --file=./kestrel-pre-scheduler-v2.dump``), every deployment
Dockerfile does ``COPY . .``, and ``cli_docker_build.py`` submits the tree to
Cloud Build. A cluster dump carries conversation rows and role password hashes.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every file that decides whether a path leaves this repository.
#: A dump must be excluded by ALL of them, not merely by whichever one the
#: last reviewer happened to look at.
EXCLUSION_CHANNELS = (
    ".gitignore",      # committed to the repo / visible in `git status`
    ".dockerignore",   # copied into an image by `COPY . .`
    ".gcloudignore",   # uploaded to Cloud Build source storage
)

#: The dump shapes. ``pg_dump --format=custom`` writes ``.dump``; plain-text
#: dumps are conventionally ``*-dump.sql`` here; ``.sql.gz`` covers the
#: compressed form.
DUMP_PATTERNS = ("*.dump", "*-dump.sql", "*.sql.gz")


def _patterns(channel: str) -> set:
    """Non-comment, non-blank entries of an ignore file."""
    text = (REPO_ROOT / channel).read_text()
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


@pytest.mark.parametrize("channel", EXCLUSION_CHANNELS)
@pytest.mark.parametrize("pattern", DUMP_PATTERNS)
def test_every_channel_excludes_every_dump_shape(channel, pattern):
    assert pattern in _patterns(channel), (
        f"{channel} does not exclude {pattern!r}. A database dump left in the "
        f"repository root would still leave the machine through this channel, "
        f"carrying conversation rows and role password hashes. Add {pattern!r} "
        f"to {channel} — and note that excluding it from only some of "
        f"{list(EXCLUSION_CHANNELS)} is worse than excluding it from none, "
        "because .gitignore alone hides the file from `git status` while it "
        "keeps shipping."
    )


def _is_dump_shaped(pattern: str) -> bool:
    """Whether an ignore rule looks like it is about a database dump.

    Deliberately decided by SHAPE, not by membership of :data:`DUMP_PATTERNS`.
    An earlier version of the test below filtered to the known list first,
    which made it vacuous: the parameterized test above already requires every
    known pattern in every channel, so comparing only known patterns could
    never fail. A new suffix added to one channel — the actual drift this
    guards — was invisible.

    ``*.sqlite``/``*.sqlite3``/``*.sqlite-wal`` are live database files rather
    than dumps and are excluded on their own terms elsewhere; ``endswith``
    rather than a substring test keeps them out of this set.
    """
    p = pattern.lower()
    return "dump" in p or p.endswith(".sql") or p.endswith(".sql.gz")


def test_the_channels_agree_exactly():
    """No channel may carry a dump rule the others lack.

    Drift in this direction is the quiet failure: a reviewer sees the pattern
    present in the file they are reading and concludes the class is handled.
    Whoever adds a new dump suffix must add it everywhere, and this fails until
    they do — including for suffixes nobody has thought of yet.
    """
    per_channel = {
        channel: {p for p in _patterns(channel) if _is_dump_shaped(p)}
        for channel in EXCLUSION_CHANNELS
    }
    distinct = set(map(frozenset, per_channel.values()))
    assert len(distinct) == 1, (
        "dump rules have drifted between exclusion channels — a dump matching "
        "the extra rule leaves the machine through the channels that lack it: "
        + "; ".join(f"{c} has {sorted(p)}" for c, p in per_channel.items())
    )
