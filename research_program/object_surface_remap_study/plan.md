# Binary object-surface remapping study

## Question

When a parent population has formed a shared protocol, what part of its
behavior can a fresh worker recover after the visible object labels are
changed? The intervention keeps the latent goals and semantic object scenes
paired, and changes only the mapping from surface labels to semantic types.

## Task and controls

Each episode has two action stages. At each stage the worker sees two objects,
one of semantic type 0 and one of type 1, in a random order. The sender sees a
private binary pair `(g0, g1)` and the worker receives a message before the
corresponding stage. A correct selection gives `+1`; an incorrect or exhausted
selection gives `-0.25`; the six-step horizon has four action steps and an
oracle team return of `0.6667`.

`dual2` sends two binary slots at times 1 and 3. `mono4` sends one four-valued
token at time 1. Both have four complete message states, so the form contrast
does not change raw capacity. `joint_history` exposes the accumulated message
state, while `slot_local` exposes only the current staged slot for `dual2`.
Four workers rotate behind a hidden partner identity.

The parent is trained on all four goal combinations with the identity mapping.
The child replaces worker 0 and keeps the parent sender frozen while learning a
fresh worker from reward. Child conditions cross identity versus a stable swap
of surface labels, full versus leave-one-goal-out support, live versus silent
channel, and the two message representations. The primary endpoint is
leave-one-out live held-out natural return `>= 0.60`; all evaluation streams are
balanced across workers and paired within seed.

## Interpretation boundary

The stable swap is learnable from reward because the scene always contains one
object of each semantic type. It tests local remapping of an existing protocol,
not an unidentifiable random mapping. A live-minus-silent or natural-minus-
permuted gap is evidence that the message affects behavior; it is not evidence
of human language, open vocabulary, or syntax.
