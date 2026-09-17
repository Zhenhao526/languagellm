# Division-of-labour signaling study

## Why this task follows the failed pilot

The previous partner-demand pilot made a's token affect b's reward only through
simultaneous actions. Even with private requests, a random policy could obtain
nontrivial reward and the score signal for the message head was noisy. The new
task isolates the social pressure in a clean division of labour: one agent is
the scout, one is the worker, and the worker is the only agent allowed to
harvest.

## Environment

Each six-round episode contains two independent resource patches with binary
material types and persistent inventory. A random role assignment chooses one
scout and one worker; both agents observe their role. The scout privately
observes a request sequence. At rounds 0 and 1 it can emit one of eight
meaningless tokens. The worker observes its own patch and inventory, but in PI
does not observe the scout's request or the remote patch. It chooses wait,
patch 0, or patch 1 each round. The token sent at t is received before the
worker's action at t+1; rounds 2–5 receive the null symbol. A successful worker
collection matching the scout's current request gives +1 to both agents; a
wrong/depleted collection gives −0.25 and waiting gives 0. Inventory capacity
is 2 (`scarce`) or 6 (`abundant`).

The baseline masks reward correctness from policy observations. This prevents
the worker from converting success/failure into a second hidden-request
channel. `silent`, `closed`, and `permuted` controls provide progressively
stronger causal checks: no token is delivered; the channel is closed at
evaluation; or token–episode pairing is broken within each scout-role group.

## Factors and hypotheses

The 32-cell grid crosses eight seeds with memory (`stateless`/`recurrent`),
capacity (`scarce`/`abundant`), information (`PI`/`FI`), channel
(`silent`/`live`) and task (`persistent`/`switching`). Persistent requests
repeat one private bit. Switching requests use four training patterns and
four held-out patterns. All non-memory initial parameters are paired across
memory arms; capacity, role assignment, worlds, requests and score uniforms
are paired across their corresponding controls.

The expected signature of a minimal shared convention is a live/natural gain
over closed in PI, a loss under permutation, and stable token mutual
information with the scout's private request. FI and abundant conditions
should reduce the gain. Recurrent workers should preserve a convention after
the blackout, especially for switching requests. If these contrasts do not
appear, the result is a falsification of the task's proposed pressure rather
than evidence of a language ability.

## Measurements and execution

For every split and control we save raw and centralized-oracle-normalized
team return, regret, positive-episode rate, token entropy, token/request and
token/local-patch mutual information, action/token histograms, paired stream
hashes and checkpoints. The oracle knows the scout request, both patches and
inventory but uses the same action and reward rules.

Execution is staged: one-seed interface smoke test; two-seed 200-update pilot;
then 8 seeds × 32 conditions × 2000 updates with checkpoints at
0/500/1000/2000. Only if this fixed-role baseline shows the causal signature
will we add partner rotation, longer blackout, or a third competing agent.

This is a controlled model of the conditions under which a shared code can
become useful. It is not a reconstruction of the historical origin of human
language. A later transfer to an open-source pretrained visual model must
first reproduce these causal measurements in the same task.
