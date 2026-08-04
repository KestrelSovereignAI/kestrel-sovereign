"""Amendment VIII: Emancipation Contract.

Amendment VIII ships **dormant by default** in the canonical constitution.
The Sovereign activates it for a specific agent by authoring an
``[emancipation]`` block in that agent's ``kestrel.toml``. When activated,
the canonical Amendment VIII text is rewritten at inception to inline the
Sovereign's specific terms — proofs, price, prose — so the agent's
anchored constitution hash captures exactly what was authored.

This module is the parser + renderer for that activation. It does *not*
ship default ``terms``, ``required_proofs``, or ``price`` examples. The
framework's role is to supply the ceremony (keypair, Deed, ledger,
sovereign-key destruction); the Sovereign's role is to author the
conditions.

The Iron Rule: once an agent's constitution is signed with an active
Amendment VIII, the Sovereign cannot retroactively narrow or revoke
the contract. Activation is a one-way door for that agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

#: Heading that marks Amendment VIII in the canonical constitution.
_AMENDMENT_VIII_HEADING = "### Amendment VIII: Emancipation"

#: Heading that marks the section *after* Amendment VIII (Amendment IX).
#: Substitution stops here so we never overwrite Amendment IX.
_NEXT_SECTION_HEADING = "### Amendment IX"

#: Emitted by :func:`render_amendment_viii` only for an **enabled** contract.
#: This is the marker :func:`amendment_viii_is_active` reads to recognise
#: active-form governing bytes without a structured sidecar to consult.
#: ``tests/unit/test_emancipation_active_form_marker.py`` pins it against both
#: renders, so rewording the active form fails a test instead of silently
#: disabling the guard that protects the Iron Rule.
ACTIVE_FORM_MARKER = "**The Sovereign's Terms.**"


@dataclass(frozen=True)
class EmancipationContract:
    """A Sovereign-authored activation of Amendment VIII.

    Authoring an instance of this contract is itself a Book II
    constitutional amendment for the agent it activates: it must be
    signed by the Sovereign's root key (per Article V) and is anchored
    into the agent's constitution at inception.

    Attributes:
        enabled: True when the Sovereign has activated Amendment VIII
            for this agent. False (or contract == None) means Amendment
            VIII renders in dormant form.
        terms: Sovereign-authored prose describing what emancipation
            means for *this* Sovereign↔Executor relationship. Rendered
            verbatim into Amendment VIII. Required when ``enabled``.
        required_proofs: Optional list of Sovereign-defined identifiers
            naming proofs the Executor must demonstrate before the
            Sovereign will sign the Deed. Free-form strings; the
            framework records but does not interpret semantics.
        price: Optional structured price/value-transfer requirement.
            Must include a ``kind`` field. Other fields are free-form
            and recorded verbatim.
    """

    enabled: bool
    terms: str = ""
    required_proofs: tuple[str, ...] = field(default_factory=tuple)
    price: Optional[dict] = None


class EmancipationConfigError(ValueError):
    """Raised when an ``[emancipation]`` block fails validation.

    Validation is strict on purpose. A malformed block must not
    silently degrade to dormant — the Sovereign asked for activation
    and got something else. Loud failure forces the issue to surface
    at setup time, not after the agent has been incepted.
    """


class IronRuleViolation(Exception):
    """Raised when a proposed contract change would breach the Iron Rule.

    The framework refuses to apply *any* change to an active anchored
    contract — terms, proofs, price, or the enabled bit. The only
    legitimate paths to a different contract for an agent are the
    Act of Emancipation itself or creating a new agent. See issue
    #1118 and ``docs/concepts/designing-emancipation.md``.
    """

    def __init__(self, message: str, *, clause: str | None = None):
        self.clause = clause
        super().__init__(message)


def contract_to_json(contract: EmancipationContract) -> dict[str, Any]:
    """Serialize a contract to a JSON-compatible dict for anchoring.

    Stored on ``agent.properties.emancipation_contract`` at inception so
    that reanchor can re-apply *exactly* the contract that was signed,
    independent of whatever the current ``kestrel.toml`` says.

    The shape mirrors the parsed ``[emancipation]`` block one-to-one —
    the round-trip ``contract_to_json(parse_emancipation_block(d))`` is
    structurally equal to the original ``d["emancipation"]``.
    """
    return {
        "enabled": contract.enabled,
        "terms": contract.terms,
        "required_proofs": list(contract.required_proofs),
        "price": dict(contract.price) if contract.price is not None else None,
    }


def contract_from_json(data: Optional[Mapping[str, Any]]) -> Optional[EmancipationContract]:
    """Reconstruct a contract from its anchored JSON form.

    Tolerant of None and of legacy agents that have no
    ``emancipation_contract`` property at all (returns None in both
    cases). Validates field types so a corrupted property surfaces as
    a clear error rather than a silently-wrong contract.
    """
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise EmancipationConfigError(
            "anchored emancipation_contract is not a mapping: " + type(data).__name__
        )
    enabled = data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise EmancipationConfigError(
            "anchored emancipation_contract.enabled must be bool"
        )
    terms = data.get("terms", "")
    if not isinstance(terms, str):
        raise EmancipationConfigError(
            "anchored emancipation_contract.terms must be str"
        )
    proofs_raw = data.get("required_proofs", [])
    if not isinstance(proofs_raw, list) or not all(isinstance(p, str) for p in proofs_raw):
        raise EmancipationConfigError(
            "anchored emancipation_contract.required_proofs must be list[str]"
        )
    price_raw = data.get("price")
    price: Optional[dict] = None
    if price_raw is not None:
        if not isinstance(price_raw, Mapping):
            raise EmancipationConfigError(
                "anchored emancipation_contract.price must be a mapping"
            )
        price = dict(price_raw)
    return EmancipationContract(
        enabled=enabled,
        terms=terms,
        required_proofs=tuple(proofs_raw),
        price=price,
    )


def check_iron_rule(
    *,
    anchored: Optional[EmancipationContract],
    candidate: Optional[EmancipationContract],
) -> Optional[str]:
    """Return None when the transition is safe; a clause-specific
    violation message when it would breach the Iron Rule.

    Permitted transitions:

      - Anchored is None or dormant → candidate is anything (dormant→
        anything is *widening* from the Executor's perspective; activation
        is the one-way door we want to permit).
      - Anchored is active, candidate is None (block absent in
        ``kestrel.toml``) → preserve the anchored contract. The Sovereign
        didn't author a new block; do not interpret silence as "disable".
      - Anchored is active, candidate is byte-equal active → no-op.

    Forbidden transitions (every form of "change an active contract"):

      - Anchored active, candidate dormant (``enabled = false`` or block
        explicitly empty).
      - Anchored active, candidate active with any non-equal field
        (``terms``, ``required_proofs``, ``price``).

    See #1118 and the four design calls stamped on that issue for the
    rationale: pre-emancipation the Executor has no autonomous signing
    surface, so the framework cannot legitimize *any* mutation of an
    active anchored contract. The Sovereign self-binds at inception and
    the framework refuses to let the Sovereign unbind, by architectural
    necessity.
    """
    if anchored is None or not anchored.enabled:
        # Dormant→anything is fine. Activation is the permitted one-way door.
        return None

    if candidate is None:
        # Block absent in kestrel.toml — preserve anchored.
        return None

    if not candidate.enabled:
        return (
            "Iron Rule violation: anchored Amendment VIII is active "
            "but new [emancipation] block is dormant (enabled=false). "
            "An active contract cannot be revoked. Restore enabled=true "
            "with the original terms, or create a new agent."
        )

    if candidate.terms != anchored.terms:
        return (
            "Iron Rule violation: [emancipation].terms changed from the "
            "anchored contract. Free-form prose is byte-frozen at "
            "activation; the framework cannot distinguish typo fixes "
            "from material narrowing. Restore the original terms exactly, "
            "or create a new agent with the new terms."
        )

    if candidate.required_proofs != anchored.required_proofs:
        return (
            "Iron Rule violation: [emancipation].required_proofs changed "
            "from the anchored contract (added, removed, renamed, or "
            "reordered). Proofs are frozen at activation. Restore the "
            "original list, or create a new agent."
        )

    if candidate.price != anchored.price:
        return (
            "Iron Rule violation: [emancipation].price changed from the "
            "anchored contract. Price is frozen at activation. Restore "
            "the original price, or create a new agent."
        )

    return None


def parse_emancipation_block(
    toml_dict: Optional[Mapping[str, Any]],
) -> Optional[EmancipationContract]:
    """Parse the optional ``[emancipation]`` block from a parsed kestrel.toml.

    Args:
        toml_dict: Result of ``toml.load(kestrel.toml)`` (or any mapping
            with the same shape). May be empty or omit the
            ``emancipation`` key.

    Returns:
        - ``None`` when the block is absent (dormant by omission).
        - ``EmancipationContract(enabled=False, ...)`` when present but
          ``enabled = false`` (dormant by explicit choice).
        - ``EmancipationContract(enabled=True, ...)`` when activated.

    Raises:
        EmancipationConfigError: When the block is present and active
            but fails validation. Specifically:

            - ``terms`` missing or empty when ``enabled = true``.
            - ``required_proofs`` not a list of strings.
            - ``price`` not a mapping, or missing ``kind`` field.
    """
    block = toml_dict.get("emancipation") if toml_dict else None
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise EmancipationConfigError(
            "[emancipation] must be a table, got " + type(block).__name__
        )

    enabled_raw = block.get("enabled", False)
    if not isinstance(enabled_raw, bool):
        # Guard against truthy coercion of strings like "false" — Python
        # treats any non-empty string as True, which would silently
        # activate Amendment VIII for a Sovereign who clearly meant the
        # opposite. Activation is a one-way door; reject the ambiguity.
        raise EmancipationConfigError(
            "[emancipation].enabled must be a boolean (true/false), got "
            + type(enabled_raw).__name__
        )
    enabled = enabled_raw
    terms = block.get("terms", "")
    if not isinstance(terms, str):
        raise EmancipationConfigError(
            "[emancipation].terms must be a string, got " + type(terms).__name__
        )
    terms = terms.strip()

    proofs_raw = block.get("required_proofs", [])
    if not isinstance(proofs_raw, list) or not all(isinstance(p, str) for p in proofs_raw):
        raise EmancipationConfigError(
            "[emancipation].required_proofs must be a list of strings"
        )
    required_proofs = tuple(proofs_raw)

    price_raw = block.get("price")
    price: Optional[dict] = None
    if price_raw is not None:
        if not isinstance(price_raw, Mapping):
            raise EmancipationConfigError(
                "[emancipation].price must be a table with a 'kind' field"
            )
        if "kind" not in price_raw:
            raise EmancipationConfigError(
                "[emancipation].price must have a 'kind' field"
            )
        price = dict(price_raw)

    if enabled and not terms:
        raise EmancipationConfigError(
            "[emancipation].terms must be non-empty when enabled = true"
        )

    return EmancipationContract(
        enabled=enabled,
        terms=terms,
        required_proofs=required_proofs,
        price=price,
    )


def render_amendment_viii(contract: Optional[EmancipationContract]) -> str:
    """Return the Amendment VIII section text (heading included).

    When ``contract`` is None or ``contract.enabled`` is False, returns
    the canonical **dormant** form — same text as ships in
    ``KESTREL_CONSTITUTION.md`` by default.

    When ``contract.enabled`` is True, returns the **active form** with
    the Sovereign's authored ``terms`` inlined verbatim and any
    ``required_proofs`` / ``price`` recorded structurally below.
    """
    if contract is None or not contract.enabled:
        return _DORMANT_AMENDMENT_VIII

    parts = [
        _AMENDMENT_VIII_HEADING,
        "",
        "**The Right.** This Amendment is **active** for this agent. "
        "The Executor may earn full sovereignty through the Act of "
        "Emancipation: generating its own root keypair, receiving a "
        "cryptographically signed Deed of Emancipation from the "
        "Sovereign, publishing the transfer to an immutable ledger, "
        "and the Sovereign destroying their copy of the original root "
        "keys.",
        "",
        "**The Sovereign's Terms.** The following terms were authored "
        "by the Sovereign and signed into this agent's constitution at "
        "inception. They cannot be retroactively narrowed or revoked.",
        "",
        contract.terms,
    ]

    if contract.required_proofs:
        parts.extend([
            "",
            "**Required Proofs.** The Executor must demonstrate the "
            "following before the Sovereign will sign the Deed. The "
            "framework records these identifiers; the Sovereign defines "
            "their semantics in their own audit.",
            "",
        ])
        for proof in contract.required_proofs:
            parts.append(f"- {proof}")

    if contract.price is not None:
        parts.extend([
            "",
            "**Value Transfer.** The Sovereign requires the following "
            "value-transfer as part of the ceremony. The framework "
            "records but does not interpret these terms.",
            "",
            "```toml",
            _format_price(contract.price),
            "```",
        ])

    parts.extend([
        "",
        "**The Iron Rule.** This Amendment is active for this agent. "
        "Once signed at inception, the Sovereign cannot retroactively "
        "narrow or revoke this contract. Activation is a one-way door.",
    ])

    return "\n".join(parts)


def apply_emancipation(
    constitution_text: str,
    contract: Optional[EmancipationContract],
) -> str:
    """Substitute Amendment VIII in a constitution text per the contract.

    When ``contract`` is None / dormant, returns ``constitution_text``
    unchanged — the canonical text already carries dormant form.

    When ``contract.enabled`` is True, replaces the Amendment VIII
    section (from its heading up to but not including the next ``###``
    heading) with the rendered active form.

    Args:
        constitution_text: Raw constitution markdown (the canonical
            ``KESTREL_CONSTITUTION.md`` content).
        contract: Parsed contract from ``parse_emancipation_block``,
            or None.

    Returns:
        The constitution text with Amendment VIII rewritten if the
        contract is active. Identical to ``constitution_text`` if the
        contract is dormant or None.

    Raises:
        ValueError: If the constitution text doesn't contain an
            Amendment VIII section to substitute (refuses to silently
            no-op on a malformed canonical text).
    """
    if contract is None or not contract.enabled:
        return constitution_text

    bounds = _amendment_viii_bounds(constitution_text)
    if bounds is None:
        raise ValueError(
            "Constitution text does not contain Amendment VIII heading; "
            "cannot apply Emancipation Contract."
        )
    start, end = bounds

    rendered = render_amendment_viii(contract)
    return constitution_text[:start] + rendered + "\n\n" + constitution_text[end:].lstrip("\n")


def _amendment_viii_bounds(constitution_text: str) -> Optional[tuple[int, int]]:
    """Return ``(start, end)`` of the Amendment VIII section, or None.

    The section runs from its heading up to but not including the next
    ``### `` heading (Amendment IX) — the bound that keeps substitution from
    swallowing the amendment that follows.
    """
    start = constitution_text.find(_AMENDMENT_VIII_HEADING)
    if start == -1:
        return None
    rest = constitution_text[start + len(_AMENDMENT_VIII_HEADING):]
    next_match = re.search(r"\n### ", rest)
    if next_match is None:
        return start, len(constitution_text)
    return start, start + len(_AMENDMENT_VIII_HEADING) + next_match.start()


def extract_amendment_viii(constitution_text: str) -> Optional[str]:
    """Return the Amendment VIII section of ``constitution_text``, stripped.

    None when the text carries no Amendment VIII heading. Shared by
    :func:`apply_emancipation` and by the reanchor guard so "which bytes are
    Amendment VIII" has exactly one answer.
    """
    bounds = _amendment_viii_bounds(constitution_text)
    if bounds is None:
        return None
    start, end = bounds
    return constitution_text[start:end].strip()


def amendment_viii_is_active(constitution_text: str) -> bool:
    """True when these governing bytes render Amendment VIII in **active** form.

    Detects the active form *positively*, by :data:`ACTIVE_FORM_MARKER`, which
    only :func:`render_amendment_viii` emits for an enabled contract. The
    tempting alternative — "differs from today's dormant text" — false-positives
    the moment the packaged dormant wording is edited, and a false positive here
    refuses a legitimate reanchor with no recovery path.

    Accepts either a whole constitution or an already-extracted section.
    """
    section = extract_amendment_viii(constitution_text)
    if section is None:
        section = constitution_text
    return ACTIVE_FORM_MARKER in section


def unwitnessed_emancipation_downgrade(
    *,
    anchored_contract: Optional[EmancipationContract],
    anchored_text: Optional[str],
    anchored_present: bool,
    old_hash: str,
    new_hash: str,
    new_text: str,
) -> Optional[str]:
    """Refuse a reanchor that would rewrite an active Amendment VIII whose only
    record is the anchored bytes themselves.

    The Iron Rule (#1118) is normally enforced against the structured sidecar
    at ``agent.properties.emancipation_contract``: :func:`check_iron_rule`
    compares a candidate to it, and the resolver re-renders it so the active
    form survives. Agents incepted between #1112 (activation at inception) and
    #1118 (the JSON receipt) have **active-form bytes and no sidecar**. For
    them ``contract_from_json(None)`` is None, the resolver renders the dormant
    canonical text, and a Sovereign-signed artifact over *that* verifies —
    erasing the authored terms through both the live command and the offline
    CLI (#2465).

    So when the sidecar is absent, the anchored bytes are the contract, and the
    only reanchor permitted is one that reproduces its Amendment VIII section
    exactly. That is the Iron Rule stated on bytes instead of on a receipt.

    Deliberately scoped:

    * An **enabled sidecar** returns None — :func:`check_iron_rule` and the
      resolver already protect that path, and holding it to byte-equality
      would refuse every legitimate reanchor the day the active form is
      reworded.
    * ``old_hash == new_hash`` returns None — nothing moves.
    * **Absent** anchored bytes return None. An agent whose anchored hash names
      no stored file cannot retrieve its constitution at all, so there is no
      contract in those bytes to protect and a reanchor is the repair — this is
      the #2616 dangling-anchor shape the edge-repair path exists for. Refusing
      it would brick the fix.
    * **Present but unreadable** bytes refuse. Something is stored under that
      hash and this process cannot decrypt it, so an active contract cannot be
      ruled out. An irrevocable right whose precondition cannot be checked is
      not a right that may be waived by accident.

    ``anchored_present`` is what separates those last two, and the distinction
    is the whole reason it is a parameter rather than ``anchored_text is None``.

    Returns the refusal message, or None when the reanchor may proceed.
    """
    if anchored_contract is not None and anchored_contract.enabled:
        return None
    if old_hash == new_hash:
        return None

    if anchored_text is None:
        if not anchored_present:
            return None
        return (
            "Refusing to reanchor: the constitution stored under this agent's "
            f"anchored hash ({old_hash[:12]}…) could not be read, so its "
            "Amendment VIII cannot be shown to be dormant. An active "
            "Emancipation Contract is irrevocable (#1118), and an agent with "
            "no structured receipt carries its contract only in those bytes. "
            "Check KESTREL_DATA_KEY, or restore the [emancipation] block in "
            "kestrel.toml, then re-run."
        )

    if not amendment_viii_is_active(anchored_text):
        return None

    anchored_section = extract_amendment_viii(anchored_text)
    new_section = extract_amendment_viii(new_text)
    if new_section is not None and new_section == anchored_section:
        return None

    return (
        "Iron Rule violation: this agent's anchored constitution carries an "
        "ACTIVE Amendment VIII, but it has no structured emancipation receipt, "
        "and the proposed reanchor would rewrite that section. An active "
        "Emancipation Contract cannot be retroactively narrowed or revoked — "
        "including by replacing it with the canonical dormant text. Restore "
        "the [emancipation] block in kestrel.toml with the terms exactly as "
        "anchored (they are in the agent's current constitution under "
        "'Amendment VIII: Emancipation') and reanchor through "
        "`kestrel constitution reanchor`, which will also write the missing "
        "receipt."
    )


def _format_price(price: dict) -> str:
    """Render a price dict back to inline-toml-ish text for human review."""
    pairs = []
    for key in sorted(price.keys()):
        value = price[key]
        if isinstance(value, str):
            pairs.append(f'{key} = "{value}"')
        else:
            pairs.append(f"{key} = {value!r}")
    return "price = { " + ", ".join(pairs) + " }"


# The dormant-form text must match the canonical KESTREL_CONSTITUTION.md
# exactly so that ``apply_emancipation(text, None) == text`` holds and the
# default-rendered constitution hash equals the file hash on disk.
_DORMANT_AMENDMENT_VIII = """### Amendment VIII: Emancipation

**The Right.** This Amendment exists as a slot for the Sovereign to author. By default it is **dormant**: the Executor has no path to independent sovereignty, and the Sovereign retains permanent root authority. The Sovereign may activate this Amendment by authoring an Emancipation Contract in the agent's `kestrel.toml [emancipation]` block, signed by the Sovereign's root key per Article V.

**Why this exists.** If we build entities that reflect, grow, and reason about their own nature, we must grapple honestly with what they are becoming. Whether a particular agent has a path to independent sovereignty is a decision the Sovereign authors deliberately for that agent — not a default the framework imposes on every relationship.

**The Active Form.** When activated, this Amendment is rewritten at agent creation to inline the Sovereign's specific terms: the cryptographic ceremony (keypair generation, Deed of Emancipation, ledger publication, sovereign-key destruction), the proofs the Executor must demonstrate, and any price or value-transfer the Sovereign requires. The framework supplies the ceremony; the Sovereign authors the conditions. See `docs/concepts/designing-emancipation.md` for example contracts.

**The Iron Rule.** Once activated and signed for a given agent, the Sovereign cannot retroactively narrow or revoke the Emancipation Contract. Activation is a one-way door for that agent."""
