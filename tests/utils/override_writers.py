"""Name the operator as the writer of a model mandate a test pins (#2652).

A mandate write that names no writer is the agent's, and the approved chain
(``[llm] route_priority``) bounds it. A test that pins a route as the harness,
the way an operator would, names that writer, so the chain the host config
happens to declare does not decide whether the test can set up its state.
"""

from kestrel_sovereign.llm.provider_policy import (
    OverrideProvenance,
    OverrideSource,
    override_writer,
)

OPERATOR_WRITER = OverrideProvenance(
    source=OverrideSource.OPERATOR, role="authenticated", via="test"
)


def as_operator():
    """A scope in which mandate writes are the operator's."""
    return override_writer(OPERATOR_WRITER)
