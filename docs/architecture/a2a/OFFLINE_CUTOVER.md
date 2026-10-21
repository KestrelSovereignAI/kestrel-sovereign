# Offline native A2A cutover

The native release replaces the draft protocol. The serving runtime never
translates draft requests or rewrites historical data at startup. A populated,
unversioned A2A store refuses startup until this offline cutover succeeds.

This command is part of the native cutover branch. Its availability does not
mean the full protocol cutover or a release has been completed.

## Prepare the old host

Drain legacy work while the old host can still communicate with its peers.
Every stored task must be terminal. Every outbound dispatch must have a known
terminal outcome, including notifications represented as tasks by the old
protocol. Resolve owed question wakes and confirm recipient settlement for
expired questions. An unknown send or an unconfirmed expiry is not evidence
that remote work stopped; the migration refuses those states.

Stop every process that can write the selected primary database, including
co-hosted agents and hosts on other machines. Create and verify a complete
backend backup. Keep the old release and backup available until the native
host passes acceptance. Do not restart old writers after conversion.

## Review provenance and preview

Create a JSON mapping for **every** historical task. Use the actual creator and
recipient identities from durable authority records, or independently reviewed
operator evidence when those records are absent. Sender claims in metadata
are never authority. Choose the configured native tenant explicitly; use an
empty string for the default tenant. Assign opaque context IDs consistently
for historical conversations within each creator, recipient, and tenant scope.

```json
{
  "historical-task-id": {
    "context_id": "reviewed-native-context-id",
    "tenant": "",
    "recipient_agent_id": "did:example:recipient",
    "creator_agent_id": "did:example:creator"
  }
}
```

For SQLite:

```sh
kestrel migrate-a2a preview --sqlite /path/to/agent.db --owners owners.json --output plan.json
```

For PostgreSQL, name an exported DSN variable so credentials are not placed in
command arguments or the plan:

```sh
kestrel migrate-a2a preview --postgres-dsn-env KESTREL_DATABASE_URL --owners owners.json --output plan.json
```

The preview writes a new private file. It contains the source fingerprint,
table counts, reviewed ownership, migration ID, and hashes of the exact native
conversions. It contains no message text or database credentials. Review this
artifact alongside the provenance evidence and backup. Preview does not create
native tables, alter task state, or settle questions.

## Apply the reviewed plan

```sh
kestrel migrate-a2a apply --sqlite /path/to/agent.db --plan plan.json
```

Use `--postgres-dsn-env KESTREL_DATABASE_URL` instead of `--sqlite` for
PostgreSQL. Apply refuses known local serving records; PostgreSQL apply also
requires all other client connections to the selected database to be closed.
Database locks are not a substitute for stopping writers that could reconnect.

Apply rechecks source data and schema under the database write lease. It
archives the original task, outbound, and question rows, retains the original
catalog in `a2a_legacy_migrations`, converts terminal task payloads in place,
reserves their reviewed native context ownership, and reinstalls task authority
and terminal fences. Historical timestamps and cancellation audit columns are
preserved. Missing native status timestamps remain absent.

Historical tasks are excluded from cognition wake recovery. Old truncated
outbound previews and settled question rows remain in archives; they do not
become invented native dispatch receipts or new sends. The protocol revision
is published last, in the same transaction. A failure rolls back conversion,
schema changes, archives, and revision publication together.

If the source or conversion differs from the reviewed plan, prepare and review
a new preview. A repeated apply after a committed cutover is refused. If a
connection fails around commit, inspect the journal by the plan's migration ID
and verify the protocol revision and archives before taking another action.
Never interpret a lost acknowledgement as permission to replay or reconvert.

Start the native host only after its release gates pass. Verify historical task
reads, context ownership, a fresh native exchange, and the absence of new wakes
for historical rows. Roll back through the verified whole-backend backup and
old release, rather than editing migrated payloads back into draft shapes.
