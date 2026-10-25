"""Built-in signal source registrations.

Each module here exposes a `build_*_registration()` function that returns
a `SourceRegistration`. Features/runners call these at startup to register
their source with the agent's `SignalRegistry`. Keeping registrations in
one place makes default-deny enforcement legible (you can grep this
package to see every source the bird is reachable through).
"""


def a2a_cognition_consumers() -> tuple[tuple[str, str], ...]:
    """Return Core's native A2A consumer/source contracts in boot order."""
    from . import a2a, a2a_question_answered, a2a_task_submitted

    return tuple((source.DURABLE_COGNITION_CONSUMER_ID, source.SOURCE_NAME)
                 for source in (a2a_question_answered, a2a, a2a_task_submitted))


def core_cognition_consumers() -> tuple[tuple[str, str], ...]:
    """Bound fixed native cleanup to the same Core cognition contracts.

    An arbitrary durable subscription or dispatcher-like executor name does
    not acquire cognition cleanup authority. Source modules own their IDs;
    boot and both stores consume this inventory instead of copying them.
    """
    from . import channels

    return ((channels.DURABLE_COGNITION_CONSUMER_ID, channels.SOURCE_NAME),
            *a2a_cognition_consumers())
