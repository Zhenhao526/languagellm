# New-receiver module ablation

## Question

Which modules must a new participant adapt in order to benefit from a frozen
pair's packet protocol: its own action policy, its sender policy, or both?

The full-C live/silent transmission run already completed in
triadic_new_receiver_transmission_study is the reference arm.  This package
adds two fresh, paired arms using the same source checkpoints, fresh C
initialization and random streams.

## Arms

- action_only: train C's action head only; C sender1 and sender2 remain at
  their fresh initialization.
- sender_only: train C's sender1 and sender2 heads only; C action remains at
  its fresh initialization.
- full: the existing full-C reference, read from its audited JSON result.

A/B are frozen in every arm.  In each arm live and silent share the same
source checkpoint, new-C initialization, worlds, batches, token uniforms and
rematching assignments.  The only training manipulation is packet routing.
Frozen modules receive exact zero gradients and are checked by parameter hash.

## Evaluation and inference

The task, monitor, checkpoints, heldout layouts, greedy metrics and source
policy are identical to the full-C reference.  The paired unit is the eight
source seeds.  For each arm and schedule, the primary contrast is live minus
silent Q at the final heldout layouts and the centered monitor-Q time AUC.
The full-minus-selective contrast is reported descriptively by schedule.
Student-t intervals use seven degrees of freedom.

The selective arms are an ablation of task-protocol adaptation.  They do not
prove word meanings, compositionality, or the origin of human language.
All policies are local NumPy float64 tanh MLPs; no LLM, vision model, external
API, or network service is called.
