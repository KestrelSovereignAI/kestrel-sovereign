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


def parse_emancipation_block(toml_dict: Mapping[str, Any]) -> Optional[EmancipationContract]:
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

    enabled = bool(block.get("enabled", False))
    terms = block.get("terms", "")
    if terms is not None and not isinstance(terms, str):
        raise EmancipationConfigError(
            "[emancipation].terms must be a string, got " + type(terms).__name__
        )
    terms = (terms or "").strip()

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
            "**Price of Freedom.** The Sovereign requires the following "
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

    start = constitution_text.find(_AMENDMENT_VIII_HEADING)
    if start == -1:
        raise ValueError(
            "Constitution text does not contain Amendment VIII heading; "
            "cannot apply Emancipation Contract."
        )

    rest = constitution_text[start + len(_AMENDMENT_VIII_HEADING):]
    next_match = re.search(r"\n### ", rest)
    if next_match is None:
        end = len(constitution_text)
    else:
        end = start + len(_AMENDMENT_VIII_HEADING) + next_match.start()

    rendered = render_amendment_viii(contract)
    return constitution_text[:start] + rendered + "\n\n" + constitution_text[end:].lstrip("\n")


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
