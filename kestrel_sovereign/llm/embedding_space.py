"""Shared local/cloud embedding-space pins + parity gate (#2290).

Privacy modes fracture the embedding space: a ``force_local_only`` session
must embed locally (Ollama), other sessions may embed via a cloud route.
Because ``space_id`` defaults to ``"<provider>:<model>"`` (route-prefixed),
rows embedded on different routes land in different spaces and are mutually
invisible to kNN — even when recall should span them.

This module lets an operator declare a **shared embedding space** pinned to a
single open-weight model served on both sides (e.g. Qwen3-Embedding-0.6B on
Ollama locally AND ``qwen/qwen3-embedding-0.6b`` via OpenRouter — same weights,
same coordinate space). The pin keys ``space_id`` on the MODEL identity + dim
(``qwen3-embedding-0.6b@768``), not the serving route, so two member routes
serving the pinned model collapse into ONE space.

Aliasing is never assumed from the model name alone (a non-goal of #2290):
Ollama typically serves a quantized GGUF while cloud serves fp16, and
instruction-prefix / pooling / normalization conventions must match. Before two
routes are declared same-space, :func:`probe_parity` embeds K canary texts
through both and requires pairwise cosine ``>= parity_threshold`` (~0.98). The
alias is only applied to a pin that has passed this probe; a pin below threshold
is refused and its members keep their own route-scoped space ids.

Proprietary models (OpenAI ``text-embedding-3``) can never participate — they
are served on one side only, so there is nothing to share. This module only
merges routes an operator explicitly pins together and only after the probe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embedding_service import cosine_similarity

logger = logging.getLogger(__name__)

# Default pairwise-cosine floor two servings of the same weights must clear
# before their rows are treated as one space. 0.98 tolerates fp16↔quantized
# drift and minor pooling differences while still catching a genuinely
# divergent serving (wrong model, collapsed quantization, dim-truncation
# mismatch).
DEFAULT_PARITY_THRESHOLD = 0.98

# Fixed canary texts embedded through every member route to measure drift.
# Deliberately diverse (prose, code, numbers, non-English, clinical) so a
# serving that diverges on any register shows up as a low pairwise cosine on
# at least one text rather than being averaged away.
DEFAULT_PARITY_CANARIES: Tuple[str, ...] = (
    "The sovereign agent remembers across every conversation.",
    "def cosine(a, b): return dot(a, b) / (norm(a) * norm(b))",
    "Quarterly revenue rose 12% on strong international demand.",
    "El zorro marrón salta sobre el perro perezoso.",
    "Patient reports intermittent chest pain radiating to the left arm.",
)


class EmbeddingSpaceConfigError(ValueError):
    """Raised when an ``[llm.embedding_spaces]`` entry is malformed."""


@dataclass(frozen=True)
class EmbeddingSpacePin:
    """A declared shared embedding space pinned to one open-weight model.

    ``space_id`` is keyed on the MODEL identity + dim (``<model>@<dim>``), not
    the serving route, so every member route serving the pinned model produces
    the same profile id and their rows are mutually visible in cosine kNN.

    ``members`` are route selectors (``"<vendor>:<route>"`` or bare
    ``"<vendor>"``). The resolution order in ``resolve_embedding_provider``
    picks WHICH member route embeds a given session — a ``force_local_only``
    session uses the local member; others may use cloud — but both stamp the
    same pinned ``space_id`` so recall spans them.
    """

    name: str
    model: str
    dim: int
    members: Tuple[str, ...]
    normalized: bool = False
    parity_threshold: float = DEFAULT_PARITY_THRESHOLD

    @property
    def space_id(self) -> str:
        """The model-identity space key both members stamp: ``<model>@<dim>``.

        Matryoshka truncation makes the same model at a different dimension a
        different space, so the dim is part of the key.
        """
        return f"{self.model}@{int(self.dim)}"

    def covers(self, provider_name: Optional[str], vendor: Optional[str] = None) -> bool:
        """True iff a resolved provider is a member of this pinned space.

        A member selector with a colon (``"ollama:local"``) matches the
        provider's full ``"<vendor>:<route>"`` name; a bare selector
        (``"ollama"``) matches the vendor of any of its routes.
        """
        route_vendor = vendor or ((provider_name or "").split(":", 1)[0] or None)
        for selector in self.members:
            if ":" in selector:
                if provider_name == selector:
                    return True
            elif selector == route_vendor:
                return True
        return False


@dataclass
class ParityResult:
    """Outcome of a canary parity probe between two member routes."""

    passed: bool
    threshold: float
    min_cosine: float
    mean_cosine: float
    n: int
    per_text: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def drift(self) -> float:
        """Worst-case drift from perfect agreement (``1 - min_cosine``)."""
        return round(1.0 - self.min_cosine, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "threshold": self.threshold,
            "min_cosine": self.min_cosine,
            "mean_cosine": self.mean_cosine,
            "drift": self.drift,
            "n": self.n,
            "error": self.error,
        }


def parse_embedding_space_pins(llm_config: Any) -> List[EmbeddingSpacePin]:
    """Parse ``[llm.embedding_spaces]`` into validated :class:`EmbeddingSpacePin`s.

    Config shape (TOML)::

        [llm.embedding_spaces.qwen3]
        model = "qwen3-embedding-0.6b"
        dim = 768
        members = ["ollama:local", "openrouter:api"]
        normalized = false          # optional
        parity_threshold = 0.98     # optional

    A malformed entry raises :class:`EmbeddingSpaceConfigError` — a shared
    space is a deliberate, load-bearing declaration, so a typo must be loud
    rather than silently dropped. Callers wrap this to degrade gracefully at
    service init.
    """
    if not isinstance(llm_config, dict):
        return []
    raw = llm_config.get("embedding_spaces")
    if not raw:
        return []
    if not isinstance(raw, dict):
        raise EmbeddingSpaceConfigError(
            "[llm.embedding_spaces] must be a table of named spaces "
            f"(got {type(raw).__name__})"
        )

    pins: List[EmbeddingSpacePin] = []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} must be a table (got "
                f"{type(entry).__name__})"
            )
        model = entry.get("model")
        if not isinstance(model, str) or not model.strip():
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} requires a non-empty string 'model'"
            )
        dim = entry.get("dim")
        try:
            dim = int(dim)
        except (TypeError, ValueError):
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} requires an integer 'dim' "
                f"(got {dim!r})"
            )
        if dim <= 0:
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} 'dim' must be positive (got {dim})"
            )
        members = entry.get("members")
        if not isinstance(members, (list, tuple)) or len(members) < 2:
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} requires a 'members' list with at "
                "least two route selectors — a shared space needs at least two "
                "routes to be worth pinning"
            )
        member_tuple = tuple(str(m).strip() for m in members if str(m).strip())
        if len(member_tuple) < 2:
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} has fewer than two non-empty members"
            )
        threshold = entry.get("parity_threshold", DEFAULT_PARITY_THRESHOLD)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} 'parity_threshold' must be a number "
                f"(got {threshold!r})"
            )
        if not 0.0 < threshold <= 1.0:
            raise EmbeddingSpaceConfigError(
                f"embedding space {name!r} 'parity_threshold' must be in (0, 1] "
                f"(got {threshold})"
            )
        pins.append(
            EmbeddingSpacePin(
                name=str(name),
                model=model.strip(),
                dim=dim,
                members=member_tuple,
                normalized=bool(entry.get("normalized", False)),
                parity_threshold=threshold,
            )
        )
    return pins


async def probe_parity(
    service_a: Any,
    service_b: Any,
    *,
    threshold: float = DEFAULT_PARITY_THRESHOLD,
    canaries: Sequence[str] = DEFAULT_PARITY_CANARIES,
) -> ParityResult:
    """Embed canaries through both services and measure pairwise agreement.

    Same weights are NOT bit-identical across servings (quantization,
    instruction-prefix, pooling, normalization), so this is the mandatory gate
    before two routes are declared same-space. Returns a :class:`ParityResult`
    whose ``passed`` is True iff the WORST per-text cosine clears ``threshold``
    — the minimum, not the mean, so a single divergent register can't be
    averaged away.

    Any embedding failure (either service returns ``None`` or raises, or a
    dimension mismatch) yields ``passed=False`` with a populated ``error`` —
    never an exception. A pin that can't be probed must not be aliased.
    """
    texts = list(canaries)
    if not texts:
        return ParityResult(
            passed=False, threshold=threshold, min_cosine=0.0,
            mean_cosine=0.0, n=0, error="no canary texts",
        )
    try:
        vecs_a = await service_a.aembed_batch(texts)
        vecs_b = await service_b.aembed_batch(texts)
    except Exception as exc:  # a route that errors can't be proven same-space
        return ParityResult(
            passed=False, threshold=threshold, min_cosine=0.0,
            mean_cosine=0.0, n=0, error=f"embed failed: {exc}",
        )

    per_text: List[float] = []
    for i in range(len(texts)):
        va = vecs_a[i] if i < len(vecs_a) else None
        vb = vecs_b[i] if i < len(vecs_b) else None
        if not va or not vb:
            return ParityResult(
                passed=False, threshold=threshold, min_cosine=0.0,
                mean_cosine=0.0, n=len(per_text), per_text=per_text,
                error=f"missing embedding for canary #{i}",
            )
        if len(va) != len(vb):
            return ParityResult(
                passed=False, threshold=threshold, min_cosine=0.0,
                mean_cosine=0.0, n=len(per_text), per_text=per_text,
                error=(
                    f"dimension mismatch for canary #{i}: "
                    f"{len(va)} vs {len(vb)} — pin the SAME dims on both members"
                ),
            )
        per_text.append(cosine_similarity(va, vb))

    min_cosine = min(per_text)
    mean_cosine = sum(per_text) / len(per_text)
    return ParityResult(
        passed=min_cosine >= threshold,
        threshold=threshold,
        min_cosine=round(min_cosine, 6),
        mean_cosine=round(mean_cosine, 6),
        n=len(per_text),
        per_text=[round(c, 6) for c in per_text],
    )
