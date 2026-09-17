# Leave-one-goal-out composition test

This package tests a narrow claim about compositional protocol recovery. A staged `dual2` parent sender is frozen, worker 0 is replaced by a new tabular worker, and one of the four binary goal combinations is removed from the child worker's training support. The child is evaluated on the missing combination, on the three seen combinations, and on the full balanced mixture.

The `full` arm trains on all four combinations. The `leave_one_out` arm changes only the child training support. Both arms have paired `live` and from-scratch `silent` channel controls. The held-out goal is balanced across seeds. Parent composability is determined before child training from the independent action-dependent experiment.

The zero-shot criterion is held-out natural team return at least 0.60, which is close to the four-step task's functional ceiling of 0.667. A pass is behavioral recovery of one missing combination under this finite protocol; it is not a claim about open vocabulary, syntax, or human language.
