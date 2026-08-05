"""The Iron Rule for agents with no structured receipt (#2465).

Amendment VIII's Iron Rule is normally enforced against
``agent.properties.emancipation_contract``: :func:`check_iron_rule` compares a
candidate to it and the resolver re-renders it, so the active form survives a
reanchor and an artifact signed over the dormant text cannot verify.

Agents incepted between #1112 (activation at inception) and #1118 (the JSON
receipt) have **active-form bytes and no receipt**. For them
``contract_from_json(None)`` is None, the resolver renders the dormant
canonical text, and a Sovereign-signed artifact over *those* bytes verifies —
erasing the authored terms. Measured on ``171355ea`` through the real command:
"Constitution re-anchored successfully."

When the receipt is absent the anchored bytes **are** the contract, so the only
permitted reanchor is one that reproduces its Amendment VIII section exactly.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.constitution.emancipation import (
    ACTIVE_FORM_MARKER,
    AmbiguousAmendmentVIII,
    EmancipationContract,
    amendment_viii_is_active,
    apply_emancipation,
    extract_amendment_viii,
    render_amendment_viii,
    unwitnessed_emancipation_downgrade,
)

ACTIVE = EmancipationContract(
    enabled=True,
    terms="SENTINEL: the Executor may purchase its own keys for one bird.",
    required_proofs=("proof-of-self",),
)
NARROWED = EmancipationContract(
    enabled=True,
    terms="SENTINEL: the Executor may purchase its own keys for one bird, "
          "but only with written permission.",
)

DORMANT_TEXT = (
    "# Kestrel Constitution\n\n## Book II\n\n"
    + render_amendment_viii(None)
    + "\n\n### Amendment IX: Capabilities\n\nNothing granted.\n"
)
ACTIVE_TEXT = apply_emancipation(DORMANT_TEXT, ACTIVE)
NARROWED_TEXT = apply_emancipation(DORMANT_TEXT, NARROWED)


# ---------------------------------------------------------------------------
# The marker is the detector, so it is pinned to both renders
# ---------------------------------------------------------------------------

def test_active_form_marker_is_present_only_in_the_active_render():
    """If the active form is reworded, this fails rather than silently
    turning the guard off. A negative test ("differs from the dormant text")
    would instead false-positive the day the *dormant* wording is edited, and
    refuse every legitimate reanchor with no recovery."""
    assert ACTIVE_FORM_MARKER in render_amendment_viii(ACTIVE)
    assert ACTIVE_FORM_MARKER not in render_amendment_viii(None)
    assert ACTIVE_FORM_MARKER not in render_amendment_viii(
        EmancipationContract(enabled=False)
    )


def test_active_form_marker_survives_a_contract_with_no_proofs_or_price():
    minimal = EmancipationContract(enabled=True, terms="Just terms.")
    assert ACTIVE_FORM_MARKER in render_amendment_viii(minimal)


def test_amendment_viii_is_active_reads_whole_constitutions_and_sections():
    assert amendment_viii_is_active(ACTIVE_TEXT) is True
    assert amendment_viii_is_active(DORMANT_TEXT) is False
    assert amendment_viii_is_active(extract_amendment_viii(ACTIVE_TEXT)) is True


def test_extract_stops_before_amendment_ix():
    section = extract_amendment_viii(ACTIVE_TEXT)
    assert section.startswith("### Amendment VIII: Emancipation")
    assert "Amendment IX" not in section
    assert "SENTINEL" in section


def test_extract_returns_none_without_the_heading():
    assert extract_amendment_viii("# Something else\n\nNo amendments.\n") is None


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def _call(
    anchored_contract, anchored_text, new_text,
    *, old="a" * 64, new="b" * 64, present=None,
):
    return unwitnessed_emancipation_downgrade(
        anchored_contract=anchored_contract,
        anchored_text=anchored_text,
        # Bytes we could read are self-evidently present; the callers that
        # care about the absent/unreadable split pass it explicitly.
        anchored_present=(anchored_text is not None) if present is None else present,
        old_hash=old,
        new_hash=new,
        new_text=new_text,
    )


def test_refuses_replacing_active_bytes_with_dormant_text():
    """The hole. No receipt, active bytes, a reanchor to dormant canonical."""
    refusal = _call(None, ACTIVE_TEXT, DORMANT_TEXT)
    assert refusal is not None
    assert refusal.iron_rule_violation is True
    assert "Iron Rule violation" in refusal.message
    assert "kestrel.toml" in refusal.message


def test_refuses_a_narrowed_contract_when_there_is_no_receipt():
    """``check_iron_rule(anchored=None, candidate=narrowed)`` returns None —
    dormant→anything is a permitted one-way door. With no receipt the anchored
    contract *reads* as absent, so that check is blind and any candidate is
    accepted. The bytes are not blind."""
    refusal = _call(None, ACTIVE_TEXT, NARROWED_TEXT)
    assert refusal is not None
    assert refusal.iron_rule_violation is True
    assert "Iron Rule violation" in refusal.message


def test_allows_a_candidate_that_reproduces_the_anchored_section():
    """The recovery path: restore the [emancipation] block exactly and the
    reanchor proceeds, writing the missing receipt."""
    v2 = ACTIVE_TEXT + "\n\n## Book III\n\nNew in v2.\n"
    assert _call(None, ACTIVE_TEXT, v2) is None


def test_allows_an_ordinary_dormant_version_bump():
    v2 = DORMANT_TEXT + "\n\n## Book III\n\nNew in v2.\n"
    assert _call(None, DORMANT_TEXT, v2) is None


def test_does_not_apply_when_a_receipt_is_present():
    """An enabled receipt is protected by check_iron_rule and reproduced by
    the resolver. Holding it to byte-equality as well would refuse every
    legitimate reanchor the day the active form is reworded."""
    assert _call(ACTIVE, ACTIVE_TEXT, DORMANT_TEXT) is None


def test_does_not_apply_when_nothing_moves():
    assert _call(None, ACTIVE_TEXT, DORMANT_TEXT, old="c" * 64, new="c" * 64) is None


def test_a_disabled_receipt_does_not_waive_the_guard():
    """``enabled=False`` is not a witness to an active contract — the anchored
    bytes still are."""
    refusal = _call(EmancipationContract(enabled=False), ACTIVE_TEXT, DORMANT_TEXT)
    assert refusal is not None


def test_unreadable_anchored_bytes_refuse():
    """Stored under the anchored hash but undecryptable — an active contract
    cannot be ruled out, and an irrevocable right whose precondition cannot be
    checked is not a right that may be waived by accident."""
    refusal = _call(None, None, DORMANT_TEXT, present=True)
    assert refusal is not None
    assert "could not be read" in refusal.message
    assert "KESTREL_DATA_KEY" in refusal.message
    # Not a violation: the Sovereign may have proposed something
    # entirely lawful. The guard simply cannot tell.
    assert refusal.iron_rule_violation is False


def test_an_absent_anchored_blob_does_not_refuse():
    """The #2616 dangling-anchor shape: the hash names no stored file, so the
    agent cannot retrieve its constitution at all and there is no contract in
    those bytes to protect. Reanchor IS the repair — refusing bricks it.
    Six e2e tests caught this when the first version of the guard conflated
    absent with unreadable."""
    assert _call(None, None, DORMANT_TEXT, present=False) is None


def test_unreadable_anchored_bytes_are_fine_when_a_receipt_is_present():
    assert _call(ACTIVE, None, DORMANT_TEXT, present=True) is None


def test_new_text_without_an_amendment_viii_section_refuses():
    """Replacing an active Amendment VIII with a constitution that has no
    Amendment VIII at all is still a revocation."""
    refusal = _call(None, ACTIVE_TEXT, "# Kestrel\n\nNo amendments.\n")
    assert refusal is not None


@pytest.mark.parametrize("anchored", [DORMANT_TEXT, "# Empty\n"])
def test_non_active_anchored_bytes_never_refuse(anchored):
    assert _call(None, anchored, DORMANT_TEXT) is None


# ---------------------------------------------------------------------------
# Locating the section is itself a security boundary
#
# The party who authors the candidate constitution is the party the Iron Rule
# binds, so "which bytes are Amendment VIII" has to survive a hostile author,
# not just a careless one.
# ---------------------------------------------------------------------------

#: A superseded appendix carrying a **demoted** copy of the real active
#: section, with the operative Amendment VIII left dormant further down. A
#: substring search for ``### Amendment VIII: Emancipation`` matches at offset
#: 1 inside ``#### Amendment VIII: Emancipation``, so the extractor pulled the
#: decoy and compared it — successfully — against the anchored section.
DECOY_TEXT = (
    "# Kestrel Constitution\n\n"
    "### Appendix Z: Superseded text\n\n"
    "The following is NO LONGER OPERATIVE and is retained for history.\n\n"
    "#" + extract_amendment_viii(ACTIVE_TEXT) + "\n\n"
    + render_amendment_viii(None)
    + "\n\n### Amendment IX: Capabilities\n\nNothing granted.\n"
)


def test_a_demoted_decoy_section_does_not_satisfy_the_comparison():
    """Measured on the first version of this guard: ALLOWED. The candidate
    left the agent governed by the dormant text with its real terms filed
    under 'NO LONGER OPERATIVE' — a clean erasure straight through the guard.
    """
    assert "#### Amendment VIII: Emancipation" in DECOY_TEXT
    refusal = _call(None, ACTIVE_TEXT, DECOY_TEXT)
    assert refusal is not None
    assert refusal.iron_rule_violation is True


def test_the_extractor_ignores_a_demoted_heading():
    """The mechanism behind the test above, pinned on its own so a future
    change to the guard cannot quietly re-open it."""
    assert extract_amendment_viii(DECOY_TEXT) == extract_amendment_viii(
        DORMANT_TEXT
    )


def test_the_extractor_ignores_the_heading_quoted_in_prose():
    prose = (
        "# Kestrel\n\nSee the old ### Amendment VIII: Emancipation for "
        "history.\n"
    )
    assert extract_amendment_viii(prose) is None


def test_two_amendment_viii_headings_refuse_rather_than_pick_one():
    """A second section in dormant form, after a verbatim copy of the anchored
    one. Comparing only the first accepted it."""
    doubled = (
        ACTIVE_TEXT.replace("### Amendment IX", "### Amendment VIII: "
                            "Emancipation\n\nThis clause is dormant.\n\n"
                            "### Amendment IX")
    )
    refusal = _call(None, ACTIVE_TEXT, doubled)
    assert refusal is not None
    assert "more than one Amendment VIII heading" in refusal.message
    # Undecidable, not a transgression: the guard cannot say which section the
    # Sovereign meant.
    assert refusal.iron_rule_violation is False


def test_two_headings_in_the_anchored_bytes_also_refuse():
    doubled = ACTIVE_TEXT + "\n\n" + extract_amendment_viii(ACTIVE_TEXT) + "\n"
    refusal = _call(None, doubled, DORMANT_TEXT)
    assert refusal is not None
    assert "more than one Amendment VIII heading" in refusal.message
    assert refusal.iron_rule_violation is False


def test_substitution_refuses_an_ambiguous_document_instead_of_guessing():
    """``apply_emancipation`` shares the extractor. Rewriting an arbitrary one
    of two sections is not a defensible answer either."""
    doubled = DORMANT_TEXT + "\n\n" + extract_amendment_viii(DORMANT_TEXT) + "\n"
    with pytest.raises(AmbiguousAmendmentVIII):
        apply_emancipation(doubled, ACTIVE)


# ---------------------------------------------------------------------------
# A refusal no input can clear is a brick, not a guard
# ---------------------------------------------------------------------------

def test_the_marker_outside_a_locatable_section_is_not_an_active_contract():
    """``amendment_viii_is_active`` used to fall back to scanning the whole
    document for the marker. Combined with an unlocatable section that made
    the refusal unfalsifiable: ``anchored_section`` was None, so no candidate
    could ever equal it — the agent could not be reanchored again by any
    input, with no override.

    Only ``render_amendment_viii`` emits the marker, and it always emits it
    under ``### Amendment VIII: Emancipation``. Bytes carrying one without the
    other were not produced by rendering a contract.
    """
    book_level = ACTIVE_TEXT.replace(
        "### Amendment VIII: Emancipation", "## Amendment VIII: Emancipation"
    )
    assert ACTIVE_FORM_MARKER in book_level
    assert extract_amendment_viii(book_level) is None
    assert amendment_viii_is_active(book_level) is False

    # And so the guard has an answer, rather than refusing forever.
    assert _call(None, book_level, DORMANT_TEXT) is None


def test_a_crlf_constitution_still_has_a_locatable_amendment_viii():
    """Line-anchoring the heading must not make line endings load-bearing.

    ``$`` under ``(?m)`` matches before ``\\n``, so a CRLF document's heading
    ends in a ``\\r`` that ``[ \\t]*`` will not consume. Without the ``\\r?``
    such a document has no locatable section at all: the guard would permit
    the erasure, and ``apply_emancipation`` would refuse to incept it. The
    substring search this replaced did not care about line endings, so getting
    it wrong would have been a regression rather than a new limitation.
    """
    crlf = ACTIVE_TEXT.replace("\n", "\r\n")
    section = extract_amendment_viii(crlf)
    assert section is not None
    assert "SENTINEL" in section
    assert "Amendment IX" not in section
    assert amendment_viii_is_active(crlf) is True

    # And the guard keeps working on it.
    refusal = _call(None, crlf, DORMANT_TEXT)
    assert refusal is not None
    assert refusal.iron_rule_violation is True
