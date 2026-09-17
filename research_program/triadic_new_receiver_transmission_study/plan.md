# New-receiver transmission study

## Scientific question

Can a freshly initialized participant recover a frozen pair's task-coordination
protocol from environmental reward, and does routing the pair's discrete
packets change the recovery trajectory?

This is the next layer after the source study's coordination, rematching,
recoding, masking, and single-slot probes.  It makes the transmission unit
explicit: two source agents are kept fixed and the third participant is
replaced by a new policy.

## Frozen task and population

- Source: research_program/triadic_factorized_neutral_altpartner_study/results/altpair_001.
- Source agents A/B/C were jointly trained on PL observations, strict mutual
  settlement, and worlds with exactly two full-success plans using different
  partner pairs.
- Source policies are local NumPy float64 tanh MLPs.  No LLM, vision model,
  external API, or pretrained language model is called.
- Eight source seeds (66701–66708) are fixed before execution.  The source
  checkpoint is the final live policy for the matching static/rematched
  schedule.
- A and B are frozen byte-for-byte.  C's three modules are discarded and
  replaced by fresh random networks with the same architecture.

## Manipulation

For each source seed and schedule, the new C is trained for 6000 updates:

1. live: all agents see all agents' two four-token windows.
2. silent: each agent sees only its own packets; the same packet samples are
   still drawn.

The two arms share source checkpoint, new-C initialization, world uniforms,
batch indices, packet uniforms, and rematching assignments.  Only the routing
flag differs.  Zero gradients are supplied for every A/B module at every
update; their parameter hashes are checked at each checkpoint.

## Evaluation

- A fixed monitor uses all 1560 source needs, the first two source training
  layouts, and all six owner assignments (18,720 worlds).
- Monitor evaluation is greedy and is saved at updates 0, 100, 500, 1500,
  3000, and 6000.
- The final evaluation uses all six held-out layouts (56,160 worlds).
- Primary endpoint: live minus silent Q-rate on held-out layouts.
- Primary trajectory: centered time-AUC of live minus silent monitor Q-rate.
- Secondary metrics: Q conditional on physical execution, physical execution,
  target-pair legality, proposal legality, engagement, and neutral rate.

The paired statistical unit is the source seed.  Static and rematched gains are
reported separately and averaged within seed for the overall estimate.  The
eight-seed intervals are Student-t intervals with seven degrees of freedom;
they are a pilot estimate rather than a claim of population-level convergence.

## Interpretation boundary

A positive routed-packet advantage would support task-protocol transmission
into a new receiver.  It would not by itself establish words, compositionality,
lexical semantics, or the origin of human language.  A null result would mean
that this frozen pair and training signal did not make transmission measurable
under the specified channel and budget.
