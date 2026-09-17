# Ecological factorization study plan

## Question

Does a factorized action ecology make a staged protocol more likely to support
zero-shot recovery of a novel conjunction by a new receiver, after controlling
for goal entropy, object inventory, message capacity, and partner rotation?

## Frozen intervention

Each episode has two stages. Each stage displays a random permutation of three
object types, so the receiver must use the message and the visible scene to
choose the matching object. The sender observes two private ternary factors.
The `factorized` task maps them directly to the two stage targets. The
`holistic` task applies a fixed permutation to the nine goal combinations
before producing the same two target values. The two tasks therefore have the
same marginal target and object distributions.

The parent trains sender and receiver on all nine combinations with a staged
message. A child replaces worker 0 while freezing the sender and receives
either all combinations or a leave-one-combination-out support. `tri3` has
two ternary slots; `mono9` has one nine-valued token. Both have nine complete
message states. The tri3 child is trained with either a joint-history state or
a slot-local state; mono9 has only the joint-history control.

## Predictions

1. Factorized tri3 with a slot-local receiver should have higher held-out
   recovery than factorized tri3 with joint history.
2. The factorized advantage should be absent or smaller for the holistic task
   and for mono9.
3. A factorized code should preserve raw-factor slot recombination, while a
   holistic code should preserve only recombination by the task's hidden target
   coordinates.
4. Full-support controls should remain higher than leave-one-out controls,
   making the zero-shot claim depend on the held-out intervention rather than
   endpoint performance alone.

## Prespecified endpoint

The zero-shot pass criterion is leave-one-out live held-out natural team
return at least 0.60. The four-action, four-reward-step ceiling is 0.6667.
All claims use seed-paired live/silent, leave-one-out/full, representation,
form, and task contrasts. A positive live endpoint without a silent or
permuted contrast is not treated as evidence of communication.

## Audit and storage

The runner records random-stream hashes, parameter hashes, checkpoint hashes,
and source hashes. The independent audit replays every parent and child update
and checks that live/silent pairs share world, goal, partner, message-uniform,
and action-uniform streams. Raw logs and checkpoints are deleted only after
the compact archive and audit have been verified.
