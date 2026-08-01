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
