# Routing initialization control plan

## Question

Does tied sender–receiver routing help because the roles remain coupled while learning, or only because they happen to start from the same routing matrix?

## Frozen arms

- `routed`: independent sender and receiver routing, independently initialized.
- `sync_routed`: independent sender and receiver routing, initialized from the same matrix and then updated independently.
- `tied_routed`: one routing matrix shared by both roles and updated by the sum of sender and receiver gradients.

All arms keep separate atomic factor tables. The aligned population, hidden partner identity, two-slot/four-symbol channel, four-object scenes, nine seeds, 3,000 updates and support masks are unchanged from the shared-routing formal matrix. The only new intervention is the identical initialization in `sync_routed`.

## Predictions

1. `routed` should reproduce weak held-out-combination transfer.
2. `sync_routed` should improve only if starting alignment is sufficient; if independent learning breaks the alignment, its route mismatch and transfer should fall below `tied_routed`.
3. `tied_routed` should reproduce the previous positive transfer.
4. All arms should fail strict held-out-value recovery.

## Audit

The same stream hashes, parameter hashes, checkpoint replay and paired architecture/support checks are retained. The archive stores per-shard and combined replay audits.
