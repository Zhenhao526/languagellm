# Open-world expansion study

This study separates old-combination transfer from the formation of a symbol
for a value that the parent population has never observed. Parents train on a
closed 3x3 ontology encoded by a two-slot, four-symbol channel. Children face
a fourth value and are evaluated on an old held-out conjunction, a meaning with
one new attribute value, and the double-new meaning.

The three policies are `holistic`, fixed `factorized`, and `tied_routed`.
Children have three supports:

- `old_combo`: old meanings only, with one old conjunction absent from child training;
- `expanded_alternating`: the expanded world, but each episode has only one fresh role;
- `expanded_pair`: new meanings are handled by a fresh sender and fresh receiver together, while old meanings retain alternating incumbent exposure;
- `expanded_single_alternating`/`expanded_single_pair`: the double-new meaning is held out of child training, so its endpoint tests zero-shot composition of the learned single-new extension.

The pair condition is a mechanism intervention: it provides a joint payoff
window in which a new symbol can be negotiated, without giving either role a
semantic dictionary. The primary endpoints are old-combination recovery,
new-single recovery, new-double recovery, and cross-partner fresh-sender code
consistency.
