"""Shell exit tokens — #658.

New-user friction: the only exit command was `!quit`, which is not
visible anywhere at startup. Users who typed `/exit`, `exit`, `quit`,
or `q` got their input sent to the LLM as a message. These tests lock
in (a) the alias set and (b) the startup hint, so a future cleanup
doesn't silently narrow the accepted vocabulary.
"""
from kestrel_sovereign.cli import _SHELL_EXIT_TOKENS, _SHELL_EXIT_HINT


class TestExitTokenSet:
    def test_contains_all_documented_aliases(self):
        """If we're going to document five ways to exit, we'd better
        actually accept all five."""
        for token in ("!quit", "/exit", "exit", "quit", "q"):
            assert token in _SHELL_EXIT_TOKENS, (
                f"{token!r} is advertised in the startup hint but not in "
                f"the accepted set — users who try it will have their "
                f"input sent to the LLM. See #658."
            )

    def test_tokens_are_lowercase(self):
        """Shell loops match against ``user_input.strip().lower()`` so
        uppercase variants (``EXIT``, ``QUIT``) work too. The set MUST
        store only lowercase values for that comparison to be effective."""
        for token in _SHELL_EXIT_TOKENS:
            assert token == token.lower(), (
                f"Exit token {token!r} has uppercase characters — "
                f"case-insensitive matching won't find it."
            )

    def test_hint_mentions_at_least_one_token(self):
        """The startup hint must actually name an exit command so the
        user has somewhere to start. This test is deliberately loose —
        any one token counts — because hint wording may evolve."""
        assert any(token in _SHELL_EXIT_HINT for token in _SHELL_EXIT_TOKENS), (
            "Startup hint mentions no exit tokens — what are we telling "
            "users to type?"
        )

    def test_common_chat_text_is_not_accidentally_an_exit_token(self):
        """Sanity check: make sure we didn't add a generic word like
        ``done`` or ``ok`` that a user might say conversationally."""
        conversational = {"done", "ok", "okay", "yes", "no", "hi", "bye"}
        assert _SHELL_EXIT_TOKENS.isdisjoint(conversational), (
            "An exit token overlaps with conversational text — typing "
            "this in chat would drop the user out of the shell mid-thought."
        )
