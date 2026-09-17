# v0.44 partner transfer endpoint

This batch tests whether independently formed v0.43 communication conventions are directly compatible when a sender is replaced. Within each visual group, recipient and donor cultures share the same frozen visual frontends (same seed, partition and resource-column assignment), so the endpoint assay isolates communication compatibility.

- 72 visual groups: 4 seeds × 3 partitions × 6 resource-column assignments;
- six cultures per group: static/random role order × fixed A, rotating A/B, random A/B/C formation;
- A/B/C evaluation schedules, six role permutations, four team slots;
- replacement modes: none, one sender slot (0/1/2), or all three senders;
- no post-replacement adaptation training.

## Reproduce

From this directory:

```bash
../.venv/bin/python partner_transfer_train.py --out results/partner_transfer_001
../.venv/bin/python partner_transfer_analysis.py --out results/partner_transfer_001
../.venv/bin/python partner_transfer_audit.py --out results/partner_transfer_001
../.venv/bin/python plot_partner_transfer.py --out results/partner_transfer_001
../.venv/bin/python visual_qa.py --out results/partner_transfer_001
../.venv/bin/python build_partner_transfer_report.py --out results/partner_transfer_001
../.venv/bin/python build_review.py --out .
```

The formal endpoint inference is bound to the sealed v0.43 completion manifest in `results/partner_transfer_001/invocation.json`. Do not rerun it into the existing non-empty result directory.

## Main artifacts

- [Chinese research report](results/partner_transfer_001/双token伙伴替换端点迁移研究报告.md)
- [analysis JSON](results/partner_transfer_001/partner_transfer_analysis.json)
- [audit JSON](results/partner_transfer_001/partner_transfer_audit.json)
- [figure](results/partner_transfer_001/figures/01_partner_transfer.png)
- [review](结果审查.md)

The assay is a zero-adaptation compatibility baseline. Low cross-culture scores mean that independent groups formed different codebooks; they do not measure how quickly a newcomer can relearn a resident convention. That requires the next formation/transfer experiment with a controlled adaptation budget.
