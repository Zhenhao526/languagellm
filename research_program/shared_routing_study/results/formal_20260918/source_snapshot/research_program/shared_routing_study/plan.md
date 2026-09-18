# Shared routing study plan

## Question

Does a shared sender–receiver routing coordinate turn a learned referential code into a transferable conjunction, even when the atomic factor tables remain role-specific?

## Frozen intervention

The world has two three-valued attributes, nine meanings, four-object scenes, and a two-slot four-symbol message. Parent policies train on every meaning. Child policies are freshly initialized and alternate sender/receiver roles with four partners.

- `holistic`: sender and receiver index complete meanings/messages.
- `factorized`: slot 0/1 receive fixed attribute 0/1 factor tables.
- `routed`: sender and receiver share atomic factor tables but learn independent row-softmax routing matrices.
- `tied_routed`: sender and receiver keep separate factor tables but share one learned row-softmax routing matrix. This is the only new intervention.
- `aligned` and `conflict`: the latter applies a fixed token permutation in community 1.
- `hidden` and `visible`: partner identity is unavailable or available to the fresh policy.
- `full`, `heldout_combo`, `heldout_value`: full support, one missing conjunction, or one missing atomic value.

Routing uses temperature 5 and initial logit standard deviation 0.5, frozen before the formal run. In `tied_routed`, sender and receiver route gradients are added to the same parameter tensor.

## Predictions

1. Fixed factorized children should preserve atomic slot mappings and recover held-out combinations.
2. Independent routed children should reproduce partial route alignment but unstable combination transfer.
3. Tied-routed children should reduce sender–receiver route mismatch and improve held-out-combination recovery if a shared coordinate is the missing mechanism.
4. Holistic children should retain ordinary communication effects but show weak held-out-combination recovery.
5. Community conflict should reduce transfer for all architectures; hidden identity should force a shared fresh code, while visible identity can permit partner-specific codes.
6. Strict held-out-value failure should delimit transfer even when held-out combinations succeed.

## Audit and storage

Every training row records world, goal, partner, role, message-uniform and action-uniform hashes plus parameter hashes. Independent replay checks every update and all paired streams before raw logs and checkpoints are deleted. The compact archive retains the frozen source snapshot, endpoint summaries, route matrices, seed-level analysis, audit and cleanup receipt.
