"""Permanent refusals for ownerless historical Runpod control paths.

Provider control belongs to kestrel-cloud-runpod, not a second Core SDK client.
A recorded Pod ID is reconciliation evidence, never fresh cleanup authority.
"""


class LegacyRunpodPathError(RuntimeError):
    """The obsolete path cannot provision or acknowledge resource cleanup."""

    def __init__(self, operation: str, *, resource_id: str | None = None) -> None:
        resource = "" if resource_id is None else f" Original resource {resource_id!r} remains unresolved."
        super().__init__(
            f"Legacy Runpod path {operation!r} is retired; no provider operation was performed."
            + resource
            + " Use the owned RESTv2 catalog/Kite acceptance and original-lease reconciliation"
            " in kestrel-cloud-runpod. Do not retry creation/start or infer ownership from a Pod ID."
            " See https://github.com/KestrelSovereignAI/kestrel-sovereign/issues/3828."
        )
