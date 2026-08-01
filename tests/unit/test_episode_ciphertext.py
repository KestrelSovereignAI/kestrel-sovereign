"""Episode synthesis must never tokenize the at-rest envelope (#2850).

`MemoryConsolidator` reads `conversation_history` with its own SQL, so
`content` arrives as the stored AEAD envelope. Before this fix it was used as
text, producing real episode titles like:

    Discussion of ksav2, aykk3eiacyxxkmq-lkng1jamqa20vlijzpzphnz8cclrrw4zhqjk...

That is silent and compounding: relevance recall ranks over garbage, embeddings
are computed from base64, and every downstream consumer degrades with no error.
"""

from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator


ENVELOPE = "KSAv2:AbxU07F7GkhHybe7Ig_5Es4-q3tnZc-jVg"
PLAINTEXT = "we agreed to migrate the scheduler to durable leases"
ENC_META = {"enc": True, "key_version": 1}


def _consolidator(store=None):
    c = MemoryConsolidator.__new__(MemoryConsolidator)
    c.agent_id = "did:test:agent"
    c._conversation_store = store
    return c


def _store_that_decrypts():
    store = MagicMock()
    store.decrypt_stored_content.side_effect = (
        lambda content, meta: PLAINTEXT if content == ENVELOPE else content
    )
    return store


class TestRowPlaintext:
    def test_encrypted_row_is_decrypted_through_the_store(self):
        c = _consolidator(_store_that_decrypts())
        assert c._row_plaintext(ENVELOPE, ENC_META) == PLAINTEXT

    def test_plaintext_row_passes_through(self):
        c = _consolidator(_store_that_decrypts())
        assert c._row_plaintext("already clear", None) == "already clear"

    def test_undecryptable_row_is_skipped_not_summarized(self):
        """Fail closed: a dropped message shows in counts, a ciphertext topic doesn't."""
        store = MagicMock()
        store.decrypt_stored_content.side_effect = RuntimeError("no key")
        assert _consolidator(store)._row_plaintext(ENVELOPE, ENC_META) is None

    def test_envelope_is_refused_when_no_store_is_wired(self):
        """The guard that would have caught the original bug."""
        assert _consolidator(None)._row_plaintext(ENVELOPE, ENC_META) is None

    def test_guard_recognises_the_sdk_envelope_prefix(self):
        from kestrel_sdk.security.aead import KSA_V2_PREFIX

        assert MemoryConsolidator._looks_like_ciphertext(
            KSA_V2_PREFIX.decode() + "whatever"
        )
        assert not MemoryConsolidator._looks_like_ciphertext(PLAINTEXT)


class TestTopicsNeverContainCiphertext:
    def test_topics_come_from_plaintext(self):
        messages = [
            {"role": "user", "content": PLAINTEXT, "metadata": {}},
            {"role": "assistant", "content": "durable leases it is", "metadata": {}},
        ]
        topics = MemoryConsolidator._extract_episode_topics(messages, limit=5)
        assert "scheduler" in topics
        assert "ksav2" not in topics

    def test_envelope_tokens_would_have_dominated(self):
        """Characterises the bug: raw envelopes tokenize into plausible 'topics'."""
        messages = [{"role": "user", "content": ENVELOPE, "metadata": {}}]
        topics = MemoryConsolidator._extract_episode_topics(messages, limit=5)
        # This is what the old pipeline persisted — proof the guard is needed
        # upstream, since the extractor itself cannot tell ciphertext from prose.
        assert "ksav2" in topics


@pytest.mark.parametrize("bad", ["KSAv2:abc", "KSAv2:"])
def test_title_input_is_gated_before_synthesis(bad):
    """No envelope may reach title/summary/affect synthesis."""
    assert _consolidator(None)._row_plaintext(bad, {"enc": True}) is None


# ---------------------------------------------------------------------------
# Legacy Fernet rows (codex review, #2850)
# ---------------------------------------------------------------------------
# The SDK also supports pre-KSAv2 Fernet ciphertext, which starts with
# "gAAAA..." and carries no KSAv2 envelope. Those rows are still marked
# `enc: true`, so a prefix-only guard would wave them through and tokenize
# Fernet ciphertext into episode titles — the same bug, different envelope.
# The metadata flag is authoritative, not the prefix.

FERNET = "gAAAAABn1QhKZ3rV9nU2mYhP0sT7cQxL4wE8dR6vN1pA"


def test_legacy_fernet_row_is_skipped_when_no_store_is_wired():
    assert _consolidator(None)._row_plaintext(FERNET, ENC_META) is None


def test_legacy_fernet_row_is_decrypted_when_a_store_is_wired():
    store = MagicMock()
    store.decrypt_stored_content.side_effect = lambda content, meta: PLAINTEXT
    assert _consolidator(store)._row_plaintext(FERNET, ENC_META) == PLAINTEXT


def test_unencrypted_row_still_passes_without_a_store():
    """The flag gates the skip — plaintext rows must not be collateral."""
    assert _consolidator(None)._row_plaintext("plain text", {}) == "plain text"
    assert _consolidator(None)._row_plaintext("plain text", None) == "plain text"


def test_plaintext_beginning_with_the_marker_survives_decryption():
    """Authenticated plaintext may legitimately start with 'KSAv2:'.

    Someone discussing the envelope format, or pasting a token, must not have
    their message silently dropped from episode synthesis (codex review r2).
    """
    store = MagicMock()
    store.decrypt_stored_content.side_effect = (
        lambda content, meta: "KSAv2: is the envelope prefix we use"
    )
    got = _consolidator(store)._row_plaintext(ENVELOPE, ENC_META)
    assert got == "KSAv2: is the envelope prefix we use"


# ---------------------------------------------------------------------------
# codex review r3 — existing corrupt episodes, and a real-decrypt gate
# ---------------------------------------------------------------------------

class TestExistingCorruptEpisodesAreInvalidated:
    """Forward-only would leave the 8 damaged episodes served forever.

    `_covered_message_ids` treats them as covering their source messages, so
    consolidation skips those messages as already-consolidated and the
    ciphertext title, summary, arc and embedding keep being served.
    """

    def test_ciphertext_derived_title_is_detected(self):
        c = _consolidator(None)
        assert c._is_ciphertext_derived(
            "Discussion of ksav2, aykk3eiacyxxkmq-lkng1jamqa20vlijz, oa-gadalxj", None
        )

    def test_ciphertext_derived_summary_is_detected(self):
        c = _consolidator(None)
        assert c._is_ciphertext_derived("A conversation", "Topics: ksav2, adlb.")

    def test_healthy_episode_is_not_flagged(self):
        c = _consolidator(None)
        assert not c._is_ciphertext_derived(
            "Discussion of scheduler, leases, migration",
            "A conversation with 12 messages. Topics: scheduler, leases.",
        )

    def test_prose_mentioning_the_format_in_passing_is_not_a_topic_term(self):
        """The signature is the magic surviving as a TOPIC, not any mention."""
        c = _consolidator(None)
        assert not c._is_ciphertext_derived("Discussion of encryption, storage", None)


class TestBackstopRequiresARealDecrypt:
    def test_wired_store_does_not_exempt_a_row_with_no_enc_flag(self):
        """A no-op decrypt must not disable the backstop."""
        store = MagicMock()
        store.decrypt_stored_content.side_effect = lambda content, meta: content
        # metadata lost / malformed -> decrypt is a no-op, envelope survives
        assert _consolidator(store)._row_plaintext(ENVELOPE, {}) is None

    def test_authenticated_decrypt_still_exempts_marker_prefixed_plaintext(self):
        store = MagicMock()
        store.decrypt_stored_content.side_effect = (
            lambda content, meta: "KSAv2: is the prefix we use"
        )
        assert (
            _consolidator(store)._row_plaintext(ENVELOPE, ENC_META)
            == "KSAv2: is the prefix we use"
        )


@pytest.mark.asyncio
class TestCoveredIdsReleasesAndPurges:
    """Drive `_covered_message_ids`, not just the detector.

    A mutation proved the detector tests alone are insufficient: disabling the
    invalidation branch left them all green while corrupt episodes kept
    claiming their source messages.
    """

    def _consolidator_with_rows(self, rows):
        from unittest.mock import AsyncMock

        c = _consolidator(None)
        c._db = MagicMock()
        c._db.fetchall = AsyncMock(return_value=rows)
        c._db.execute = AsyncMock()
        return c

    async def test_corrupt_episode_does_not_cover_its_messages(self):
        c = self._consolidator_with_rows([
            ('["11","12"]', "ep-corrupt", "Discussion of ksav2, adlb, fldy9", None),
            ('["21"]', "ep-healthy", "Discussion of scheduler, leases", None),
        ])
        covered = await c._covered_message_ids()
        assert covered == {"21"}, covered
        assert "11" not in covered, "corrupt episode still claims its messages"

    async def test_corrupt_episode_is_deleted_so_it_can_be_rebuilt(self):
        c = self._consolidator_with_rows([
            ('["11"]', "ep-corrupt", "Discussion of ksav2, adlb", None),
        ])
        await c._covered_message_ids()
        c._db.execute.assert_awaited()
        sql, params = c._db.execute.await_args.args
        assert "DELETE FROM memory_episodes" in sql
        assert "ep-corrupt" in params

    async def test_healthy_episodes_are_never_deleted(self):
        c = self._consolidator_with_rows([
            ('["21"]', "ep-healthy", "Discussion of scheduler, leases", None),
        ])
        covered = await c._covered_message_ids()
        assert covered == {"21"}
        c._db.execute.assert_not_awaited()
