# Frozen plan: incumbent-worker surface transfer

## Question

After a sender and a worker have formed a protocol, where is the convention
stored? Does a worker transfer the convention when visible object labels are
renamed, and can reward-only interaction repair the mismatch? A fresh-worker
control separates a remapping cost from ordinary child learning.

## Intervention

The parent trains sender and four workers for 3,000 updates on the identity
surface. The sender codebook and all parent parameters are frozen in a final
checkpoint. The target child is worker 0. `incumbent` copies that worker's
parameters; `fresh` samples a new worker with the same initialization rule.
The sender remains frozen in every child. The child mapping is either
`identity` or a stable swap of surface labels 0 and 1. The live/silent pair
holds all latent streams and initial parameters fixed.

## Environment

Each episode contains two subtasks and two site object types. A two-symbol,
two-slot message arrives at steps 1 and 3. The worker acts at steps 2–5 and is
rewarded for taking the requested semantic object at each site. Partners are
rotating and hidden. All four goal pairs occur in a complete block for each
partner in evaluation, so the message permutation control crosses goals rather
than rotating within a constant goal class.

## Design and analysis

There are 9 fixed seeds, 1 parent condition, and 8 child conditions
(`fresh/incumbent × identity/swap × live/silent`), for 81 runs. Each run uses
3,000 updates, batch size 512, checkpoints at 0/500/1000/2000/3000, and the
same deterministic NumPy streams. The primary endpoint is full-support live
held-out natural return at initialization and after training. Secondary
endpoints are repair gain, all-goal `natural−permuted`, live−silent, codebook
separation, and paired swap contrasts with 95% t intervals over seeds.

The preparation manifest snapshots every source file and freezes its hashes.
The independent audit replays every episode, gradient, parameter hash,
checkpoint, initial incumbent copy, sender freeze, and paired stream. Raw
training logs and checkpoints are deleted only after the audit; compact results
and receipts remain in the archive.
