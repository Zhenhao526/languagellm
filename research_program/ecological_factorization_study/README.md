# Ecological factorization and novel-combination transfer

This study asks whether a protocol formed in a structured object ecology can
be recovered by a new worker on a goal combination that was absent from the
worker's training support. It keeps nine equally likely combinations of two
three-valued private factors, randomizes the visible order of the three object
types, and compares two target ecologies:

- `factorized`: stage 0 and stage 1 use the two raw private factors;
- `holistic`: the same nine goal combinations are mapped through a frozen
  permutation to the same nine stage-target pairs.

`tri3` exposes two staged three-valued slots and `mono9` exposes one staged
nine-valued token. They have equal complete-message capacity. After a parent
sender has trained on all combinations, worker 0 is replaced and trained with
either full support or one combination removed. The tri3 child is compared
under joint-history and slot-local receiver representations; mono9 is the
capacity-matched joint-history control.

The primary endpoint is leave-one-out live held-out return. Raw-factor slot
recombination and task-target recombination are secondary interventions. This
is a finite protocol-transfer experiment, not a claim that the agents have
acquired open vocabulary or human language.

Raw execution trees are temporary. The formal archive stores frozen sources,
compact results, replay audit, and the cleanup receipt.

The formal 2026-09-18 matrix contains 36 parent and 216 child runs. Three
independent shard replays passed with zero numerical error. Live communication
improved full-support performance, but all nine seeds failed the pre-registered
0.60 leave-one-combination-out endpoint; the detailed Chinese analysis is in
[`ecological_factorization_formal_结果与下一步.md`](ecological_factorization_formal_结果与下一步.md),
and compact results are in [`results/formal_20260918`](results/formal_20260918/).
