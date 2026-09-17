# v0.43 permutation formation

This batch trains the three-resource, two-token grounded coordination task with two formation factors:

- all six permutations of the resource input columns (`012`, `021`, `102`, `120`, `201`, `210`);
- static resource-role order versus deterministic per-team/per-update random role order;
- fixed A, rotating A/B, and random A/B/C partner topologies.

The formal factorial contains 432 chains (4 seeds × 3 partitions × 6 assignments × 2 role modes × 3 topologies), 600 updates per chain, and checkpoints 0/100/300/600. Communication heads are freshly initialized; private visual frontends and the reward machinery are inherited and frozen from the earlier controlled batches.

## Reproduce the analysis

From this directory, with the bundled environment:

```bash
../.venv/bin/python permutation_analysis.py --out results/permutation_001
../.venv/bin/python permutation_statistics.py --out results/permutation_001
../.venv/bin/python permutation_audit.py --out results/permutation_001
../.venv/bin/python plot_permutation.py --out results/permutation_001
../.venv/bin/python visual_qa.py --out results/permutation_001
../.venv/bin/python build_permutation_report.py --out results/permutation_001
../.venv/bin/python build_review.py --out .
```

The formal training invocation is recorded in `results/permutation_001/invocation.json`; it should not be rerun into the existing non-empty result directory. The endpoint metrics are in `permutation_analysis.json`, paired bootstrap summaries in `permutation_statistics.json`, and the independent trace replay in `permutation_audit.json`.

## Main artifacts

- [Chinese research report](results/permutation_001/三资源排列与角色随机化形成研究报告.md)
- [analysis JSON](results/permutation_001/permutation_analysis.json)
- [statistics JSON](results/permutation_001/permutation_statistics.json)
- [audit JSON](results/permutation_001/permutation_audit.json)
- [figure](results/permutation_001/figures/01_permutation_formation.png)
- [review](结果审查.md)

The task remains a finite grounded protocol assay, not an open-ended language benchmark. Role-equivalent scores are synchronized target-axis counterfactuals; they should be read together with literal scores, role spread, held-out compositions and future partner-replacement tests.
