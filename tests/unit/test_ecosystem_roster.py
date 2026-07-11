"""Roster-expansion tests for scheduled stale-work scans (#2269)."""

from kestrel_sovereign.signals.sources.ecosystem_roster import (
    RosterSpec,
    expand_roster,
    is_tekspear_repo,
    is_wildcard,
    parse_roster_spec,
    wildcard_prefix,
)


UNIVERSE = {
    "KestrelSovereignAI/kestrel-sovereign",
    "KestrelSovereignAI/kestrel-feature-github",
    "KestrelSovereignAI/kestrel-feature-voice",
    "KestrelSovereignAI/tekspear-core",
    "OtherOrg/unrelated",
}


class TestWildcardHelpers:
    def test_is_wildcard(self):
        assert is_wildcard("KestrelSovereignAI/kestrel-feature-*")
        assert not is_wildcard("KestrelSovereignAI/kestrel-sovereign")

    def test_wildcard_prefix_strips_star(self):
        assert (
            wildcard_prefix("KestrelSovereignAI/kestrel-feature-*")
            == "KestrelSovereignAI/kestrel-feature-"
        )

    def test_is_tekspear_repo(self):
        assert is_tekspear_repo("KestrelSovereignAI/tekspear-core")
        assert is_tekspear_repo("Foo/Tekspear-Bar")
        assert not is_tekspear_repo("KestrelSovereignAI/kestrel-sovereign")


class TestParseRosterSpec:
    def test_wildcard_in_repos_routes_to_prefix_not_allowlist(self):
        spec = parse_roster_spec(repos=["KestrelSovereignAI/kestrel-feature-*"])
        assert spec.allowlist == ()
        assert spec.prefixes == ("KestrelSovereignAI/kestrel-feature-",)

    def test_explicit_repo_stays_in_allowlist(self):
        spec = parse_roster_spec(repos=["KestrelSovereignAI/kestrel-sovereign"])
        assert spec.allowlist == ("KestrelSovereignAI/kestrel-sovereign",)
        assert spec.prefixes == ()

    def test_org_allowlist_prefix_and_exclude_parsed(self):
        spec = parse_roster_spec(
            org="KestrelSovereignAI",
            repos=["A/b"],
            repo_prefix=["KestrelSovereignAI/kestrel-"],
            exclude_repos=["A/skip"],
        )
        assert spec.orgs == ("KestrelSovereignAI",)
        assert spec.allowlist == ("A/b",)
        assert spec.prefixes == ("KestrelSovereignAI/kestrel-",)
        assert spec.exclude == ("A/skip",)

    def test_legacy_single_repo_wildcard_becomes_prefix(self):
        spec = parse_roster_spec(repo="KestrelSovereignAI/kestrel-feature-*")
        assert spec.prefixes == ("KestrelSovereignAI/kestrel-feature-",)
        assert spec.allowlist == ()

    def test_empty_spec(self):
        assert parse_roster_spec().is_empty


class TestExpandRoster:
    def test_prefix_expands_against_universe(self):
        spec = parse_roster_spec(repos=["KestrelSovereignAI/kestrel-feature-*"])
        out = expand_roster(spec, accessible_repos=UNIVERSE)
        assert set(out.repos) == {
            "KestrelSovereignAI/kestrel-feature-github",
            "KestrelSovereignAI/kestrel-feature-voice",
        }
        assert out.failures == ()

    def test_org_expands_and_excludes_tekspear(self):
        spec = parse_roster_spec(org="KestrelSovereignAI")
        out = expand_roster(spec, accessible_repos=UNIVERSE)
        assert "KestrelSovereignAI/tekspear-core" not in out.repos
        assert "KestrelSovereignAI/tekspear-core" in out.excluded
        assert "OtherOrg/unrelated" not in out.repos

    def test_explicit_inaccessible_repo_is_failure(self):
        spec = parse_roster_spec(repos=["KestrelSovereignAI/ghost"])
        out = expand_roster(spec, accessible_repos=UNIVERSE)
        assert out.repos == ()
        assert out.failures == ({"repo": "KestrelSovereignAI/ghost", "reason": "inaccessible"},)

    def test_explicit_exclude_drops_repo(self):
        spec = parse_roster_spec(
            org="KestrelSovereignAI",
            exclude_repos=["KestrelSovereignAI/kestrel-feature-voice"],
        )
        out = expand_roster(spec, accessible_repos=UNIVERSE)
        assert "KestrelSovereignAI/kestrel-feature-voice" not in out.repos
        assert "KestrelSovereignAI/kestrel-feature-voice" in out.excluded

    def test_discovery_failure_reports_all_targets(self):
        spec = parse_roster_spec(
            org="KestrelSovereignAI",
            repos=["KestrelSovereignAI/kestrel-sovereign"],
            repo_prefix=["KestrelSovereignAI/kestrel-feature-"],
        )
        out = expand_roster(
            spec, accessible_repos=set(), discovery_error="503 no token"
        )
        assert out.repos == ()
        reasons = {f["reason"] for f in out.failures}
        assert reasons == {"503 no token"}
        # Every requested target surfaced as a failure — none silently skipped.
        assert len(out.failures) == 3

    def test_prefix_matching_nothing_is_failure(self):
        spec = parse_roster_spec(repo_prefix=["KestrelSovereignAI/nomatch-"])
        out = expand_roster(spec, accessible_repos=UNIVERSE)
        assert out.repos == ()
        assert out.failures[0]["pattern"] == "KestrelSovereignAI/nomatch-*"

    def test_tekspear_never_included_even_if_named_explicitly(self):
        spec = parse_roster_spec(repos=["KestrelSovereignAI/tekspear-core"])
        out = expand_roster(spec, accessible_repos=UNIVERSE)
        assert out.repos == ()
        assert "KestrelSovereignAI/tekspear-core" in out.excluded
