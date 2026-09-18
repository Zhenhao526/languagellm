# Learned-routing factor-sharing formal result

This is the frozen 9-seed, 3,000-update formal matrix for the `holistic`, fixed `factorized`, and learnably `routed` policies. It keeps the four-object referential game, two three-valued attributes, two-slot/four-symbol channel, aligned/conflict communities, hidden/visible partner identity, and full/held-out-combination/held-out-value support unchanged.

The routed policy shares atomic factor tables but learns sender and receiver slot-to-attribute routing. The frozen optimization setting is route-logit initial standard deviation 0.5 and routing softmax temperature 5. These values are part of `prepared.json`; a separate low-gradient baseline with the original near-uniform routing setting is archived beside this directory.

The matrix has 108 parent runs and 324 child runs. The independent replay audit passed: 324,000 parent and 972,000 child training-log rows, 432 and 1,296 checkpoints, all paired architecture/visibility/population/support streams, and maximum replay error 0.0.

In aligned/hidden held-out-combination evaluation, fixed factorized children averaged 1.000 (9/9 functional), holistic children −0.156 (0/9), and routed children 0.245 (1/9). Routed sender and receiver best slot-permutation alignment averaged 0.717 and 0.728, but alignment was unstable across seeds and did not reliably produce compositional transfer. Held-out-value return for routed was −0.020, preserving the boundary that an unseen atomic value cannot be recovered.

Raw execution trees are intentionally excluded from the repository after audit; the compact endpoint results, frozen source snapshot, plan, hashes, and audit are retained here.
