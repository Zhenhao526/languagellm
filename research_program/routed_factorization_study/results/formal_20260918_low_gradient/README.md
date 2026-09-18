# Low-gradient learned-routing baseline

This is the matched 9-seed, 3,000-update routing baseline run before the routing optimization setting was strengthened. Its source snapshot is frozen under `source_snapshot/`; the original route logits were near-uniform and the route temperature was 1 (the older prepared configuration predates explicit route hyperparameter fields).

The matrix has 108 parent runs and 324 child runs. It is retained as an optimization-sensitivity control, not as the primary routed estimate. In aligned/hidden held-out-combination evaluation, routed children averaged −0.016 and route alignment stayed near 0.50, while the fixed factorized arm averaged 1.000. The result shows that a nominally learnable routing parameterization can remain in a low-gradient basin.

Raw execution trees are excluded after replay audit.
