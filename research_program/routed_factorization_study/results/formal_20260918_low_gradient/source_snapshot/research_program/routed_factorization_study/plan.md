# Structured factor-sharing study plan

## Question

Does an explicit factor-sharing architecture turn a learned referential code into a transferable conjunction, while the matched holistic architecture remains unable to recover a held-out combination?

## Frozen intervention

The world has two three-valued attributes, nine meanings, four-object scenes, and a two-slot four-symbol message. Parent policies train on every meaning. Child policies are freshly initialized and alternate sender/receiver roles with four partners.

- `holistic`: sender and receiver index complete meanings/messages.
- `factorized`: slot 0 receives only attribute 0 and slot 1 only attribute 1; the receiver adds the two slot-local attribute scores.
- `aligned` and `conflict`: the latter applies a fixed token permutation in community 1.
- `hidden` and `visible`: partner identity is unavailable or available to the fresh policy.
- `full`, `heldout_combo`, `heldout_value`: full support, one missing conjunction, or one missing atomic value.

The architecture is the causal intervention. It is deliberately recorded as a structural prior rather than described as spontaneous language discovery.

## Predictions

1. Factorized children should preserve atomic slot mappings and recover held-out combinations when the parent has learned both factors.
2. Holistic children should show the same ordinary communication controls but weaker held-out-combination recovery.
3. Community conflict should reduce transfer for both architectures; hidden identity should force a shared fresh code, while visible identity can permit partner-specific codes.
4. Strict held-out-value failure would delimit transfer even when held-out combinations succeed.

## Audit and storage

Every training row records world, goal, partner, role, message-uniform and action-uniform hashes plus parameter hashes. Independent replay checks every update and all paired streams before raw logs and checkpoints are deleted.
