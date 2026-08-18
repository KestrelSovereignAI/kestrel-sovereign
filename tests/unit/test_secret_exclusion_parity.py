"""A secret must be excluded from every channel that ships the tree.

This test exists because fixing one channel at a time did not converge. Review
found the same class of defect three rounds running, each time in a channel the
previous fix had not considered:

1. a dump in the repo root could be committed          -> .gitignore
2. ...and `COPY . .` baked it into the image           -> .dockerignore
3. ...and `gcloud builds submit .` uploaded it to GCS  -> .gcloudignore

Each fix was correct and each left the exposure open somewhere else, because
the lists are independent: Docker does not read .gitignore, and gcloud consults
.gcloudignore *before* Docker sees .dockerignore. Hiding a file from
``git status`` without excluding it from the other two is actively worse than
doing nothing, since the operator stops seeing the file that is still shipping.

Two things this test asserts, and one it deliberately does not:

* It asserts **paths**, not pattern strings. The previous version compared the
  literal rules across the three files, which is a proxy — and a measurably
  wrong one, see the dialect note below. The channels are also entitled to
  reach the same exclusion by different means: .gitignore excludes ``*.json``
  wholesale and re-includes the handful it wants, while the other two name the
  credential shapes. Comparing rule text would have demanded redundant patterns
  in .gitignore to satisfy the test rather than to close a hole.

* It asserts a **shipped** set as well. A channel that excluded everything would
  satisfy the secret half trivially; requiring representative source files to
  survive is what stops the fix being "ignore more".

* It does not assert that the three files are identical. They are not, and
  should not be: .dockerignore excludes ``tests/`` and .gitignore does not.

Why it matters concretely: ``docs/deployment/README.md`` instructs operators to
write a pre-migration dump into the repository root, every deployment Dockerfile
does ``COPY . .``, and ``cli_docker_build.py`` submits the tree to Cloud Build,
whose source uploads persist in a GCS bucket with its own lifecycle and IAM —
outliving the image that was built from them.

The `**/` dialect note
----------------------

``.gitignore`` and ``.gcloudignore`` use gitignore syntax, where a pattern
containing no slash matches at every depth. ``.dockerignore`` does not: its
patterns are Go ``filepath.Match`` against the whole context-relative path, and
``*`` does not cross ``/``. Measured against a real ``docker build`` with a
context holding ``root.pem`` and ``sub/nested.pem`` and a ``.dockerignore`` of
``*.pem``::

    root.pem        excluded
    sub/nested.pem  SHIPPED

The same held for directory rules: a bare ``secrets/`` left ``sub/secrets/a`` in
the context. So identical rule text meant *different coverage* in the two
dialects, which is exactly why the old text-comparison test could pass over a
hole. Prefixing ``**/`` makes a rule mean "at any depth" in both — verified in
the same way, with every secret path excluded and every keeper surviving.

That is what :func:`test_dockerignore_secret_rules_match_at_any_depth` pins: the
path assertions below evaluate every channel under gitignore semantics, which is
only a faithful model of ``.dockerignore`` while its secret rules keep the
``**/`` spelling.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every file that decides whether a path leaves this repository.
#: A secret must be excluded by ALL of them, not merely by whichever one the
#: last reviewer happened to look at.
EXCLUSION_CHANNELS = (
    ".gitignore",      # committed to the repo / visible in `git status`
    ".dockerignore",   # copied into an image by `COPY . .`
    ".gcloudignore",   # uploaded to Cloud Build source storage
)

#: Representative paths that must not leave the machine through any channel.
#: Each is listed at the repository root *and* nested, because the nested form
#: is the one a bare pattern silently misses in Docker's dialect.
SECRET_PATHS = (
    # Environment files: the local .env carries live provider keys.
    ".env",
    ".env.local",
    "sub/.env",
    "sub/deep/.env.production",
    # Secret bundles and the trust root.
    ".secrets",
    "secrets/token.txt",
    "sub/secrets/token.txt",
    ".kestrel/sovereign/root.key",
    "sub/.kestrel/sovereign/root.key",
    # Agent signing keys and the at-rest encryption key cache.
    "credentials/kestrel-agent.private-key.pem",
    "sub/credentials/agent.pem",
    "storage_cache/key_deadbeef.key",
    "sub/storage_cache/key_deadbeef.key",
    "server.pem",
    "deploy/tls/server.pem",
    "root.key",
    "deploy/tls/server.key",
    # Cloud credential shapes.
    "my-credentials.json",
    "deploy/my-credentials.json",
    "sa-service-account.json",
    "deploy/gcp-service-account.json",
    "deploy/signing-key.json",
    "deploy/token.private.json",
    # Database dumps: conversation rows and role password hashes.
    "kestrel-pre-scheduler-v2.dump",
    "backups/nightly.dump",
    "backups/cluster-dump.sql",
    "backups/cluster.sql.gz",
)

#: Representative paths that must reach every channel. Without these, "exclude
#: everything" would pass the half of this test that matters most.
SHIPPED_PATHS = (
    "README.md",
    "pyproject.toml",
    "kestrel_sovereign/main.py",
    "kestrel_sovereign/data/semantic/profiles/default.json",
    "package.json",
)


@pytest.fixture(scope="session")
def channel_repos(tmp_path_factory):
    """One scratch git repo per channel, that channel's rules as its .gitignore.

    ``git check-ignore`` is the reference implementation of the syntax all three
    files are written in, so the rules are evaluated rather than pattern-matched
    by hand. The repo is empty and the paths need not exist — the query is about
    the rules, not about this working tree, which also keeps the result
    independent of whatever untracked files a developer happens to have.
    """
    repos = {}
    for channel in EXCLUSION_CHANNELS:
        repo = tmp_path_factory.mktemp(channel.lstrip("."))
        subprocess.run(
            ["git", "init", "-q", "."], cwd=repo, check=True,
            capture_output=True,
        )
        (repo / ".gitignore").write_text((REPO_ROOT / channel).read_text())
        repos[channel] = repo
    return repos


def _matching_rule(repo: Path, path: str):
    """The rule that excludes ``path``, or ``None`` if nothing does.

    ``git check-ignore -v`` prints ``<source>:<line>:<pattern>\\t<pathname>``.

    Its exit code cannot be used as the answer here: under ``-v`` it reports 0
    whenever *some* pattern matched, including a negation, so a re-included path
    such as ``package.json`` would read as excluded. (Without ``-v`` the same
    query exits 1 for it, which is the authoritative answer.) The last matching
    pattern decides, and a leading ``!`` on it means the path survives — so that
    is what is checked, and the rule text stays available for the message.
    """
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", "--", path],
        cwd=repo, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    rule = proc.stdout.partition("\t")[0].split(":", 2)[2]
    return None if rule.startswith("!") else rule


@pytest.mark.parametrize("channel", EXCLUSION_CHANNELS)
@pytest.mark.parametrize("path", SECRET_PATHS)
def test_every_channel_excludes_every_secret(channel, path, channel_repos):
    assert _matching_rule(channel_repos[channel], path) is not None, (
        f"{channel} does not exclude {path!r}. That file would leave the "
        f"machine through this channel — committed, baked into an image, or "
        f"uploaded to Cloud Build source storage, where it outlives the image "
        f"and has its own IAM. Add a rule to {channel}, and add it to all of "
        f"{list(EXCLUSION_CHANNELS)}: excluding it from only some is worse "
        "than excluding it from none, because .gitignore alone hides the file "
        "from `git status` while it keeps shipping. Write the rule `**/`-"
        "prefixed so it means the same thing in Docker's dialect."
    )


@pytest.mark.parametrize("channel", EXCLUSION_CHANNELS)
@pytest.mark.parametrize("path", SHIPPED_PATHS)
def test_no_channel_excludes_source_it_must_ship(channel, path, channel_repos):
    rule = _matching_rule(channel_repos[channel], path)
    assert rule is None, (
        f"{channel} excludes {path!r} via {rule!r}, but that path has to reach "
        "the build. An exclusion broad enough to catch source is not a fix; "
        "narrow the rule rather than re-including the file downstream."
    )


@pytest.mark.parametrize("path", [p for p in SECRET_PATHS if "/" in p])
def test_dockerignore_secret_rules_match_at_any_depth(path, channel_repos):
    """Every nested secret must be caught by a ``**/``-prefixed rule.

    The assertions above evaluate .dockerignore under gitignore semantics. That
    is only a faithful model while its rules are spelled so the two dialects
    agree. Docker's ``*`` does not cross ``/``, so a rule like ``*.pem`` covers
    the context root alone — it would satisfy every test above for
    ``deploy/tls/server.pem`` (git matches it at any depth) while Docker shipped
    the file into the image. Measured, not inferred; the module docstring
    records the build the numbers came from.
    """
    rule = _matching_rule(channel_repos[".dockerignore"], path)
    assert rule is not None and rule.startswith("**/"), (
        f".dockerignore matches {path!r} with {rule!r}, which Docker applies to "
        "the context root only — `*` does not cross `/` in its dialect, so the "
        "nested file ships into the image even though this rule looks like it "
        f"covers it. Spell it `**/{(rule or '').lstrip('/')}`."
    )
