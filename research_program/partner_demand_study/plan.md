# Private-demand cooperation study

## Core question

The preceding temporal scarcity pilot allowed each agent to satisfy its own
request. It learned a useful action policy, but communication was often
dispensable. This experiment places the information asymmetry directly in the
division of labour: agent `a` privately knows the material it requests, while
agent `1-a` is the actor whose collection is scored against that request.
The minimal candidate for a shared symbol is therefore a token convention
that lets a partner recover a private demand.

## Environment

There are two independent resource patches and six rounds. A sees patch 0 and
B sees patch 1. Each patch has one of two material types and a persistent
inventory. A request is binary (material 0 or 1), private to its owner. At a
round, agent `a` may wait, take patch 0, or take patch 1. Its reward is +1 if
the selected patch matches the **partner's** current request and has inventory;
an incorrect or depleted take is −1 and waiting is 0. Same-patch requests
share inventory, with the lower-index agent winning the last unit. Team return
is the mean reward over both agents and all six rounds.

The channel has eight meaningless tokens. Tokens can be sent at rounds 0 and
1 and are delivered before the next round; rounds 2–5 are blackout. The
natural evaluation preserves sender–receiver pairing. `closed` replaces the
tokens by the null symbol. `permuted` reassigns tokens within the evaluation
batch, preserving token frequencies while breaking their episode-specific
causal link.

## Factors

The frozen grid crosses 8 seeds with:

* `memory`: stateless versus a 32-unit recurrent policy;
* `scarcity`: capacity 2 versus capacity 6 per patch;
* `information`: PI masks partner request, remote type and remote inventory;
  FI exposes them;
* `channel`: silent versus live;
* `task`: `persistent` request (one private bit repeated for six rounds) versus
  `switching` request (four train patterns and four held-out patterns).

The two memory arms share every initial parameter except whether `W_h` is
enabled. Scarcity arms share sites, requests, and score uniforms because
capacity is excluded from the episode RNG key. Live/silent arms share all
exogenous streams. No token meanings, language model, teacher, pretrained
language knowledge, or researcher-only state enters a policy observation.

## Evaluation and claims

Every final policy is evaluated on both the training-support distribution and
the held-out switching distribution. A centralized finite-horizon oracle sees
both requests and both patches and supplies an upper bound. We report raw and
oracle-normalized team return, regret, positive-episode rate, token entropy,
token mutual information with the sender's request and local patch, and the
natural-minus-closed/permuted causal contrasts.

We call a convention operationally shared only if (i) live/natural improves
over closed, (ii) permutation removes that improvement, and (iii) token
statistics are stable across seeds and task realizations. A drop in entropy or
an isolated performance gain is insufficient. The expected interaction is a
larger live effect in PI than FI, in scarce than abundant worlds, and for
switching than persistent requests; recurrent policies should retain a token
meaning after blackout.

## Execution stages

1. Run one seed and four representative cells for 20 updates to check all
   invariants and the oracle bound.
2. Run two seeds over all 32 cells for 200 updates to detect learning and
   tune only the pre-registered training budget if necessary.
3. Run all 8 seeds and 32 cells for 2000 updates with checkpoints at
   0/500/1000/2000 and an independent audit of hashes and paired streams.
4. If the live effect survives controls, add one pressure at a time (longer
   blackout, partner rotation or a third agent) while retaining this grid as
   the fixed baseline.

This is a controlled model of the conditions for a shared code, not a claim
to reproduce the historical origin of human language. Scaling the same
environment to an open-source pretrained visual model is a later transfer
test, not part of this causal baseline.
