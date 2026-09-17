# Two-generation protocol chain

## Question

Does a short learner-replacement chain retain a task protocol, and does
cross-agent packet routing change adaptation and retention?  The experiment
keeps the task and architecture fixed while replacing one role at a time.

## Chain

- Generation 1 is the audited live full-C endpoint from
  `triadic_new_receiver_transmission_study/results/transmission_002`.
- Generation 2 replaces agent A with a fresh three-module learner.  B and C
  are frozen, and only A is updated under live or silent routing.
- Generation 3 replaces agent B with another fresh learner.  A and C are
  copied from the generation-2 **live** endpoint and frozen; only B is
  updated under live or silent routing.
- The generation-2 silent counterfactual is not propagated.  Both generation-3
  channels therefore start from the same generation-2 live parent.

Each generation uses eight prespecified seeds, static or rematched private
need schedules, 6000 updates, batch size 256, and the same PL factorized
neutral/engage environment as the transmission reference.  Live and silent
runs share parent parameters, fresh learner initialization, world uniforms,
batch indices, packet uniforms and rematching assignments.  Frozen modules
receive exact zero gradients and are checked by parameter hashes.

## Measurements

The primary contrasts are live minus silent heldout Q at the final layouts and
centered monitor-Q time AUC, reported separately for generations 2 and 3 with
paired Student-t intervals over the eight seeds.  Retention is the paired
change in live heldout Q from generation 1 to generation 2 and from generation
2 to generation 3.  Conditional Q, physical execution, target-pair legality,
proposal legality and engagement are secondary diagnostics.

This is a test of short-horizon task-protocol transmission.  It does not by
itself establish words, compositionality, conventionality, or the origin of
human language.  All policies are local NumPy float64 tanh MLPs; no LLM,
pretrained model, vision model, external API or network service is used.
