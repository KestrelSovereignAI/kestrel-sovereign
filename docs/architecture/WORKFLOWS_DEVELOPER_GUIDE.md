# Workflows Developer Guide

## Compensation Alert Tiering

Workflow compensation has two operator-visible failure shapes. Keep them
separate in metrics, dashboards, and on-call routing:

| Condition | Metric / Rule | Operator Meaning | Paging |
|---|---|---|---|
| A real compensator returns failure | `KestrelWorkflowCompensateFailedPage` | The runner attempted rollback and it did not complete. Human intervention is required. | Yes |
| A run ends `cancelled_with_irreversible_residue` | `KestrelWorkflowIrreversibleResidueDashboardOnly` | An irreversible or `compensate_record_only` stage already committed. The runner recorded residue instead of pretending rollback was possible. | No |

The Prometheus rule file lives at
`docs/deployment/prometheus-workflows-alerts.yml`. The exported counters keep
the project-wide `kestrel_` prefix:

- `kestrel_workflow_compensate_failed_total{workflow_name,stage_name}`
- `kestrel_workflow_cancelled_with_irreversible_residue_total{workflow_name}`

The rule file provides unprefixed cumulative recording rules for dashboards
and seeds absent counters with `or vector(0)`. Alert expressions apply
`increase(...)` to the raw `kestrel_*` counters before summing, then add a
new-series term so first-ever events are not missed. The bounded expressions
keep one old counter increment from firing forever. Dashboards should show
both counters, but only `KestrelWorkflowCompensateFailedPage` should route to
a paging receiver.

When adding new compensation states, do not route residue-like states to the
paging alert unless the runner actually attempted a compensator and that
compensator failed.
