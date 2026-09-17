# New-receiver joint-combination holdout

This experiment tests whether communication helps a fresh participant respond
to unseen combinations of already familiar need values.

- The 1560 two-alternative-partner needs are partitioned by a deterministic
  semantic transformation orbit. Six complete orbits (312 joint needs) are
  held out; 1248 needs remain for adaptation. Every individual need value
  appears in both partitions for every actor, so the holdout is joint rather
  than a new-token test.
- A is replaced by a fresh learner; B/C are frozen at the audited generation-1
  live full-C endpoint. The fresh A trains only on the 1248 seen combinations.
- Live and silent channels share the same parent, initialization, worlds,
  messages and rematching streams. The audited generation-2 replace-A runs
  from the same parent and streams provide an all-needs control.
- Primary evaluation is heldout-orbit Q live−silent at six checkpoints and at
  heldout layouts. Seen-orbit new-layout Q and heldout conditional metrics are
  secondary controls.

The measured object is joint task-protocol transfer. It does not establish
words, lexical meanings, compositional grammar, or human language origin.
All policies are local NumPy float64 tanh MLPs; no LLM, vision model, external
API or network service is used.
