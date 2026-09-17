# Fixed versus rotating partners: a population signaling control

This experiment asks whether one sender can establish a token convention that
remains useful when the receiving partner changes. It is a deliberately small
mechanistic control, not a claim about human language.

One scout observes a persistent binary resource type and sends one token at
round 0. A worker receives the token one round later and repeatedly chooses
one of two sites. Each worker has a private two-site type permutation (the
observed type of site 0 determines the opposite type at site 1) and private
inventory; a correct available choice gives +1, a wrong or depleted choice
gives −0.25, and waiting gives 0. The sender never sees the worker's local
site map. In `fixed`, worker 0 is always selected. In `rotating`, one of four
independent tabular workers is selected uniformly on every episode. `hidden`
keeps the partner identity out of the sender state; `visible` allows the
sender to condition its token on identity and is a control for partner
specific conventions. `silent` removes the token during training and
`live` trains through the token channel. The recurrent worker retains the
round-0 token; no language model, teacher, or token dictionary is supplied.

The primary causal signature is the per-worker natural-minus-closed return and
the natural-minus-within-worker-permuted return. The population signature is
whether different workers converge on the same token for each target and
whether the live advantage survives when evaluated one worker at a time. A
centralized finite-horizon oracle supplies an upper bound. For each seed and
update, worlds, target bits, partner uniforms and action/message uniforms are
shared across partner, visibility, channel and capacity arms; only the partner
assignment interpretation changes. Training-support and held-out streams are
both evaluated.

The study crosses partner stability (`fixed`/`rotating`), sender partner
visibility (`hidden`/`visible`), channel (`silent`/`live`), and capacity
(`scarce`/`abundant`) over eight seeds. The visible sender arm is an explicit
diagnostic: if rotating-hidden loses a convention while rotating-visible
retains one, partner identity may be substituting for a shared symbol. If
rotating-hidden retains per-worker causal gains and token alignment, the
result is evidence for a population-level common code in this task family. The
visible sender rows are initialized by copying the two hidden rows, so the
visibility comparison does not start from different token preferences.
