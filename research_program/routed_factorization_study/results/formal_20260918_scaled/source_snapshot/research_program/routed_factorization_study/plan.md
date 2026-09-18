# Structured factor-sharing study plan

## Question

Does atomic factor sharing turn a learned referential code into a transferable conjunction, and can a matched policy learn the slot-to-attribute alignment needed for that transfer?

## Frozen intervention

The world has two three-valued attributes, nine meanings, four-object scenes, and a two-slot four-symbol message. Parent policies train on every meaning. Child policies are freshly initialized and alternate sender/receiver roles with four partners.

- `holistic`: sender and receiver index complete meanings/messages.
- `factorized`: slot 0 receives only attribute 0 and slot 1 only attribute 1; the receiver adds the two slot-local attribute scores.
- `routed`: atomic factor tables are shared, but both sender and receiver learn a row-softmax routing matrix. Routing uses temperature 5 and initial logit standard deviation 0.5, frozen before the formal run.
- `aligned` and `conflict`: the latter applies a fixed token permutation in community 1.
- `hidden` and `visible`: partner identity is unavailable or available to the fresh policy.
- `full`, `heldout_combo`, `heldout_value`: full support, one missing conjunction, or one missing atomic value.

The architecture is the causal intervention. It is deliberately recorded as a structural prior rather than described as spontaneous language discovery.

## Predictions

1. Factorized children should preserve atomic slot mappings and recover held-out combinations when the parent has learned both factors.
2. Routed children should learn a near-permutation routing matrix and recover held-out combinations if alignment can emerge from the payoff alone.
3. Holistic children should show the same ordinary communication controls but weaker held-out-combination recovery.
4. Community conflict should reduce transfer for all architectures; hidden identity should force a shared fresh code, while visible identity can permit partner-specific codes.
5. Strict held-out-value failure would delimit transfer even when held-out combinations succeed.

## Audit and storage

Every training row records world, goal, partner, role, message-uniform and action-uniform hashes plus parameter hashes. Independent replay checks every update and all paired streams before raw logs and checkpoints are deleted.
