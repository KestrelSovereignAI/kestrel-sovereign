"""Built-in signal source registrations.

Each module here exposes a `build_*_registration()` function that returns
a `SourceRegistration`. Features/runners call these at startup to register
their source with the agent's `SignalRegistry`. Keeping registrations in
one place makes default-deny enforcement legible (you can grep this
package to see every source the bird is reachable through).
"""
