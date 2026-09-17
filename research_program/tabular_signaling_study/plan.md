# Tabular signaling control

The neural pilots established the task interfaces but often collapsed to
waiting or to a no-message policy. Before adding more environmental detail,
this control asks whether the proposed causal signature is learnable at all by
a minimal agent with low-variance parameters.

Agent 0 is a fixed scout and agent 1 is a worker. The scout privately observes
a binary persistent request or a six-round switching pattern. It sends one
token from an eight-token alphabet at the communication-only round 0. The
worker receives that token at round 1 and chooses wait, patch 0 or patch 1 in
rounds 1–5. In the recurrent condition the last non-null token is retained;
in the stateless condition only the current received token is available. A
correct available collection gives +1, a wrong or depleted collection gives
−0.25, and waiting gives 0. Patch types are independent and inventory is 2 in
the scarce arm or 6 in the abundant arm. PI hides the request from the worker;
FI exposes it. No language model, teacher, or pretrained language knowledge is
used.

The sender and worker are tabular softmax policies. Sender logits are indexed
by the private request context and scout-local patch type. Worker logits are
indexed by received/memorized token, time, local patch type, local inventory,
and (in FI) the visible request. A state-group baseline and exact paired score
uniforms reduce gradient variance; all message and action updates remain
self-play reward updates.

The 32-cell factor grid crosses memory, scarcity, information, channel
(`silent`/`live`) and task (`persistent`/`switching`) over eight seeds. Every
cell is evaluated on training-support and held-out switching contexts with
natural, closed and token-permuted controls. A centralized finite-horizon
oracle gives the upper bound. A shared code is counted only when natural live
performance exceeds closed, permutation removes the gain, and token/request
mutual information is stable across seeds.

This is a positive-control stage, not the final claim about human language. If
the tabular control cannot learn the code, the neural environment should not be
made more complex. If it can, the learned code becomes a reference for the
later stateless/recurrent neural and visual-model comparisons.
