# Open-world expansion plan

## Question

When a community expands from a closed ontology to a new attribute value, can
an already compositional protocol extend its symbol system? Which part is
explained by a structural coordinate, and which part requires a fresh sender
and receiver to share an action window?

## Frozen design

Parents train for 3,000 updates on all nine meanings formed by values 0, 1 and
2. The world then exposes value 3 to a fresh child. The message remains two
slots of four symbols, so the expanded 4x4 meaning space is exactly capacity
matched to the message-state space.

- `holistic`: complete-meaning rows;
- `factorized`: fixed slot-to-attribute factor tables;
- `tied_routed`: role-specific factor tables with a shared learned routing matrix;
- `old_combo`: old-world child training with one old conjunction held out;
- `expanded_alternating`: new meanings appear, but the fresh policy alternates with an incumbent;
- `expanded_pair`: only new meanings use role 2, where both sender and receiver are fresh; old meanings retain alternating exposure.
- `expanded_single_alternating` and `expanded_single_pair`: the double-new meaning is excluded from child training; only single-new meanings are available, with either alternating or fresh–fresh interaction.

No value-3 episode appears in parent training. New-single means have exactly
one value 3; new-double is (3,3). The nine seed block, stream salts, learning
rate, route temperature, checkpoints and replay hashes are frozen before the
formal run.

## Predictions and interpretation

1. Factorized and tied-routed parents should recover an old held-out
   conjunction more often than holistic parents.
2. No parent should have reliable zero-shot new-value recovery; that is the
   open-world boundary, not a failure of old composition.
3. `expanded_alternating` should have difficulty extending a code when the
   incumbent has no representation for the new value.
4. `expanded_pair` should provide the payoff window needed to negotiate a new
   code. If factorized or tied-routed policies then produce more consistent
   new-single codes than holistic policies, this is evidence for a structural
   extension mechanism rather than a larger table alone.
5. In the single-new controls, new-double performance is a strict zero-shot
   composition test. A high endpoint there cannot be explained by direct
   memorization of the double-new meaning.

The experiment is a tabular mechanism test. It does not by itself establish
human-like language, open vocabulary, or cultural evolution.

## Audit

Every update records stream hashes and parameter hashes. Parent and child
replay audits verify the two fresh-role semantics, checkpoint hashes, and
architecture/support paired streams before raw trees are removed.
